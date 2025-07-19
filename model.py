import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from typing import Dict, Tuple, Optional, List, NamedTuple, Any, Callable
import logging
import math

# Configure pretty tensor printing for debugging
import lovely_tensors as lt
lt.monkey_patch()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======================================================================================
# 1. Configuration & Helper Classes
# ======================================================================================

class TEMConfig:
    """
    Stores all hyperparameters and architectural choices for the TEM model,
    acting as a single source of truth for the model's structure.
    """
    def __init__(self):
        # --- Core Architecture ---
        self.n_streams: int = 5
        self.g_dims: List[int] = [30, 30, 24, 18, 18]     # g-space (MEC) dimensions per stream
        self.g_downsampled_dims: List[int] = [10, 10, 8, 6, 6] # g-space dimension when projected to p-space
        self.sensory_dim: int = 10                        # Dimension of the compressed sensory vector (x^c)
        self.n_sensory_objects: int = 45                  # Number of unique sensory objects (size of one-hot x)
        self.n_actions: int = 4                           # Number of possible actions (e.g., NESW)
        self.use_stochastic: bool = True                  # Use reparameterization trick for g
        self.bptt_len: int = 75                           # Backpropagation through time length

        # --- Derived Dimensions ---
        # p-space (Hippocampus) is a conjunction of sensory and downsampled g-space
        self.p_dims: List[int] = [d * self.sensory_dim for d in self.g_downsampled_dims]
        self.total_p_dim: int = sum(self.p_dims)
        self.total_g_dim: int = sum(self.g_dims)

        # --- Hierarchical Memory Parameters (Hebbian System) ---
        self.use_p_inf: bool = True                       # Use inference memory to correct path integration
        self.common_memory: bool = False                  # Use separate generative and inference memories
        self.memory_iterations: int = 5                   # Attractor network steps (tau)
        self.memory_decay: float = 0.8                    # Decay term in attractor dynamics (kappa)
        self.hebbian_learning_rate: float = 0.5           # Hebbian learning rate (eta)
        self.hebbian_forget_rate: float = 0.9999          # Hebbian forgetting rate (lambda)

        # --- Hierarchy & Stream Specialization ---
        # Temporal filtering coefficients (alpha^f)
        self.temporal_frequencies: List[float] = [0.01, 0.7, 0.91, 0.97, 0.99]

        # Defines asymmetric connectivity for hierarchical transitions
        self.g_connections: torch.Tensor = torch.zeros(self.n_streams, self.n_streams)
        for f_from in range(self.n_streams):
            for f_to in range(self.n_streams):
                # Connection allowed if source stream is slower (or same) as target stream
                if self.temporal_frequencies[f_from] <= self.temporal_frequencies[f_to]:
                    self.g_connections[f_to, f_from] = 1.0

        # --- Optional Sensory Encoding ---
        self.use_two_hot: bool = False                    # Use fixed two-hot encoding for sensory input
        self.two_hot_mat: Optional[torch.Tensor] = None
        self.load_two_hot()

    def load_two_hot(self):
        """Pre-computes the fixed two-hot encoding matrix if enabled."""
        if self.use_two_hot:
            mat = make_two_hot_matrix(self.n_sensory_objects, self.sensory_dim)
            self.two_hot_mat = torch.from_numpy(mat)

class VariationalOutput(NamedTuple):
    """A standard container for the output of a variational layer."""
    mean: torch.Tensor
    log_var: torch.Tensor
    sample: torch.Tensor

@torch.jit.script
def _attractor_update_step(h: torch.Tensor, memory: torch.Tensor, decay: float) -> torch.Tensor:
    """
    A JIT-scripted function for a single attractor network update step.
    Equation: h_tau = f(kappa * h_{tau-1} + M * h_{tau-1})

    Args:
        h: The current state of the attractor network (batch, p_dim).
        memory: The Hebbian memory matrix M (batch, p_dim, p_dim).
        decay: The scalar decay term kappa.

    Returns:
        The updated state of the attractor network.
    """
    mv = torch.bmm(memory, h.unsqueeze(-1)).squeeze(-1)
    update = decay * h + mv
    return F.leaky_relu(torch.clamp(update, -1, 1))

@torch.jit.script
def scale_log_sigma(raw: torch.Tensor, temp: float = 1.0, LOGSIG_RATIO: float = 6.0, LOGSIG_OFFSET: float = -2.0) -> torch.Tensor:
    """
    Scales the raw output of an MLP to a bounded log-variance. This prevents pathologically
    large or small variances during training.

    Args:
        raw: Unconstrained output of an MLP.
        temp: A curriculum scalar in [0, 1] (0 => fixed sigma, 1 => full range).

    Returns:
        A tensor of log variances (log sigma^2).
    """
    bounded = torch.tanh(raw / LOGSIG_RATIO) * LOGSIG_RATIO
    return LOGSIG_OFFSET + temp * bounded

def make_two_hot_matrix(n_sensory: int, two_hot_dim: int) -> np.ndarray:
    """
    Builds a fixed two-hot encoding matrix of shape (n_sensory, two_hot_dim),
    where each sensory index maps to exactly two '1's in a vector of length `two_hot_dim`.
    This is an alternative to a learnable sensory compressor.
    """
    combos = list(itertools.combinations(range(two_hot_dim), 2))
    if len(combos) < n_sensory:
        raise ValueError(f"Need at least {n_sensory} distinct two-hot codes; only {len(combos)} available.")
    mat = np.zeros((n_sensory, two_hot_dim), dtype=np.float32)
    for row_idx, (i, j) in enumerate(combos[:n_sensory]):
        mat[row_idx, i] = 1.0
        mat[row_idx, j] = 1.0
    return mat

# ======================================================================================
# 2. Hierarchical Memory System
# ======================================================================================

class HierarchicalMemory(nn.Module):
    """
    Implements the fast-learning Hebbian memory system with attractor dynamics.
    This module handles memory initialization, Hebbian updates, and hierarchical retrieval.
    The hierarchy is enforced via masking to ensure larger scales (gist) influence
    smaller scales (details) during retrieval.
    """
    def __init__(self, config: TEMConfig):
        super().__init__()
        self.config = config
        self.n_streams = config.n_streams
        self.p_dims = config.p_dims
        self.total_p_dim = config.total_p_dim

        # Register learning/forgetting rates as non-trainable buffers
        self.register_buffer("hebbian_learning_rate", torch.tensor(float(config.hebbian_learning_rate)))
        self.register_buffer("hebbian_forget_rate", torch.tensor(float(config.hebbian_forget_rate)))

        self._build_stream_indices()
        self._build_and_register_masks()

    def _build_stream_indices(self):
        """Defines the start and end indices for each stream's data in concatenated vectors."""
        self.stream_indices = []
        start = 0
        for p_dim in self.p_dims:
            self.stream_indices.append((start, start + p_dim))
            start += p_dim

    def _build_and_register_masks(self):
        """Creates and registers masks for hierarchical memory dynamics."""
        # 1. Hierarchical Connectivity Mask (`hier_mask`)
        # Allows larger scales (higher index f) to influence smaller scales (lower index f)
        # but not vice-versa, creating an upper triangular block matrix.
        hier_mask = torch.zeros(self.total_p_dim, self.total_p_dim)
        for i in range(self.n_streams): # Target stream
            for j in range(i, self.n_streams): # Source stream
                s_i, e_i = self.stream_indices[i]
                s_j, e_j = self.stream_indices[j]
                hier_mask[s_i:e_i, s_j:e_j] = 1.0
        self.register_buffer("hier_mask", hier_mask)

        # 2. Staggered Retrieval Masks (`retrieval_masks`)
        # Implements staggered retrieval where higher-level streams run for fewer
        # attractor iterations (e.g., [5, 4, 3, 2, 1] iterations for 5 streams).
        masks = []
        if isinstance(self.config.memory_iterations, int):
            max_mem_iters = [self.config.memory_iterations - f for f in range(self.n_streams)]
        else:
            max_mem_iters = self.config.memory_iterations

        for it in range(max(max_mem_iters)):
            m = torch.zeros(self.total_p_dim)
            for s, remaining_iters in enumerate(max_mem_iters):
                if it < remaining_iters:
                    st, en = self.stream_indices[s]
                    m[st:en] = 1.0
            masks.append(m)
        self.register_buffer("retrieval_masks", torch.stack(masks), persistent=False)

    def create_empty_memory(self, batch_size: int, device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        """Initializes empty (zero) memory matrices for a new environment."""
        P = self.total_p_dim
        M_gen = torch.zeros(batch_size, P, P, device=device)
        M_inf = None
        if self.config.use_p_inf and not self.config.common_memory:
            M_inf = torch.zeros_like(M_gen)
        return {"generative": M_gen, "inference": M_inf}

    def hebbian_update(
        self,
        M_prev: torch.Tensor,
        p_inferred: torch.Tensor,
        p_generated: torch.Tensor,
        memory_type: str = "generative",
    ) -> torch.Tensor:
        """
        Performs a Hebbian update on the memory matrix.
        Equation: M_t = lambda * M_{t-1} + eta * (p_t - p_hat_t) * (p_t + p_hat_t)^T

        Args:
            M_prev: The memory matrix from the previous timestep.
            p_inferred: The inferred hippocampal state p_t.
            p_generated: The predicted hippocampal state p_hat_t.
            memory_type: 'generative' or 'inference' to apply appropriate masks.
        """
        # Ensure tensors have a batch dimension
        if p_inferred.dim() == 1: p_inferred = p_inferred.unsqueeze(0)
        if p_generated.dim() == 1: p_generated = p_generated.unsqueeze(0)

        term1 = (p_inferred - p_generated).unsqueeze(-1)  # (p - p_hat), Shape: (B, P, 1)
        term2 = (p_inferred + p_generated).unsqueeze(1)   # (p + p_hat)^T, Shape: (B, 1, P)

        # Fused multiply-add for efficiency: M_t = alpha*T1*T2 + beta*M_{t-1}
        hebbian_change = torch.baddbmm(
            M_prev, term1, term2,
            beta=float(self.hebbian_forget_rate),   # lambda
            alpha=float(self.hebbian_learning_rate) # eta
        ).clamp_(-1, 1)

        if memory_type == "generative":
            hebbian_change.mul_(self.hier_mask) # Apply hierarchical mask

        return hebbian_change

    def hierarchical_retrieve(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        max_iterations: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Retrieves a memory from the attractor network with hierarchical dynamics.
        Equation: h_tau = f(kappa * h_{tau-1} + M * h_{tau-1})

        Args:
            query: The initial state of the attractor network, h_0.
            memory: The memory matrix M to use for retrieval.
            max_iterations: Number of attractor steps. Defaults to config.
        """
        if max_iterations is None:
            max_mem_iters_cfg = self.config.memory_iterations
            max_iterations = max(max_mem_iters_cfg) if isinstance(max_mem_iters_cfg, list) else max_mem_iters_cfg

        if query.dim() == 1:
            query = query.unsqueeze(0)

        h = F.leaky_relu(torch.clamp(query, -1, 1))
        masks = self.retrieval_masks

        for it in range(max_iterations):
            active_mask = masks[it].unsqueeze(0)
            update = _attractor_update_step(h, memory, self.config.memory_decay)
            # Only update the streams that are active at this iteration
            h = (1 - active_mask) * h + active_mask * update

        return h

    def split_by_stream(self, p_concat: torch.Tensor, dim: int = 1) -> List[torch.Tensor]:
        """Splits a concatenated p-vector back into a list of tensors, one for each stream."""
        return list(torch.split(p_concat, self.p_dims, dim=dim))

# ======================================================================================
# 3. Loss Calculation
# ======================================================================================

class CompleteLoss(nn.Module):
    """
    Computes the complete TEM loss, combining reconstruction, latent alignment,
    and regularization terms, as described in the STAR methods.
    """
    def __init__(self, config: TEMConfig):
        super().__init__()
        self.cfg = config
        # Default weights. These are updated by the Curriculum scheduler during training.
        self.weights = {
            # Reconstruction losses
            "x_nll_p": 1.0,
            "x_nll_g": 1.0,
            "x_nll_gt": 1.0,

            # Latent alignment (KL divergence surrogates)
            "g_kl": 0.0,
            "p_align": 0.0,
            "p_inf_align": 0.0,

            # Regularization
            "g_reg": 0.01, # L2 regularization on g
            "p_reg": 0.02, # L1 regularization on p
        }

    @staticmethod
    def _gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
        """
        Analytic KL divergence D_KL(q || p) between two diagonal Gaussians.
        Returns KL per sample in the batch (B,).
        """
        var_q, var_p = logvar_q.exp(), logvar_p.exp()
        kl = 0.5 * (
            logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0
        )
        return kl.sum(-1)

    def forward(self, predictions: dict, targets: dict, reps: dict) -> dict:
        """
        Calculates all components of the loss function.

        Args:
            predictions: Dictionary of predicted logits for x.
            targets: Dictionary with the ground truth sensory observation x.
            reps: Dictionary with all inferred and generated latent variables (g, p).
        """
        losses = {}
        x_target = targets["x"].argmax(dim=-1) # Ground truth class indices

        # --- 1. Reconstruction Loss (Cross-Entropy) ---
        # Compares predicted sensory observation with the actual observation.
        # The paper describes three prediction pathways.
        losses["x_nll_p"] = F.cross_entropy(predictions["x_logits_p"], x_target, reduction="none")
        losses["x_nll_g"] = F.cross_entropy(predictions["x_logits_g"], x_target, reduction="none")
        losses["x_nll_gt"] = F.cross_entropy(predictions["x_logits_gt"], x_target, reduction="none")

        # --- 2. Latent Alignment (KL Divergence or MSE surrogate) ---
        # This term aligns the inference model's posteriors with the generative model's priors.

        # g-space alignment: D_KL(q(g_t|...) || p(g_t|g_{t-1}, a_t))
        # g_kl = [
        #     self._gaussian_kl(
        #         g_inf.mean, g_inf.log_var, g_pr.mean, g_pr.log_var
        #     )
        #     for g_inf, g_pr in zip(reps["g_inf"], reps["g_prior"])
        # ]
        # losses["g_kl"] = torch.stack(g_kl, dim=0).sum(0)
        g_align_mse = [
            0.5 * (g_inf.sample - g_pr.mean).pow(2).sum(-1)
            for g_inf, g_pr in zip(reps["g_inf"], reps["g_prior"])
        ]
        losses["g_kl"] = torch.stack(g_align_mse, dim=0).sum(0)

        # p-space alignment: D_KL(q(p_t|...) || p(p_t|g_t, M_{t-1}))
        p_align_mse = [
            0.5 * (p_inf - p_gen).pow(2).sum(-1)
            for p_inf, p_gen in zip(reps["p_inf"], reps["p_gen"])
        ]
        losses["p_align"] = torch.stack(p_align_mse, dim=0).sum(0)

        # Optional memory-cued alignment for the inference path
        if "p_inf_x" in reps:
            p_inf_align_mse = [
                0.5 * (p_inf - p_mem).pow(2).sum(-1)
                for p_inf, p_mem in zip(reps["p_inf"], reps["p_inf_x"])
            ]
            losses["p_inf_align"] = torch.stack(p_inf_align_mse, dim=0).sum(0)
        else:
            losses["p_inf_align"] = torch.zeros_like(losses["g_kl"])

        # --- 3. Regularization ---
        # Encourages sparse and reasonably-normed representations.
        g_samples = torch.cat([g.mean for g in reps["g_inf"]], dim=-1)
        p_samples = torch.cat(reps["p_inf"], dim=-1)
        losses["g_reg"] = g_samples.pow(2).sum(-1) # L2 penalty
        losses["p_reg"] = p_samples.abs().sum(-1)  # L1 penalty

        # --- 4. Weighted Total Loss ---
        total = torch.zeros_like(losses["g_kl"])
        for name, val in losses.items():
            w = self.weights.get(name, 0.0)
            if w > 0:
                total = total + w * val
        losses["total_loss"] = total
        return losses

    def update_weights(self, new: dict):
        """Adjusts loss weighting dict on-the-fly from a curriculum scheduler."""
        self.weights.update(new)

# ======================================================================================
# 4. Core Model Components
# ======================================================================================

class TEMStream(nn.Module):
    """
    Encapsulates the fixed, non-trainable parameters for a single hierarchical stream,
    specifically the projection matrices W_repeat and W_tile. These matrices are crucial
    for implementing the conjunctive binding of g and x into p.
    """
    def __init__(self, stream_idx: int, config: TEMConfig):
        super().__init__()
        self.config = config
        self.stream_idx = stream_idx
        p_dim = config.p_dims[stream_idx]
        g_down_dim = config.g_downsampled_dims[stream_idx]
        sensory_dim = config.sensory_dim
        self._init_fixed_matrices(p_dim, g_down_dim, sensory_dim)

    def _init_fixed_matrices(self, p_dim, g_down_dim, s_dim):
        """
        Initializes the fixed projection matrices for conjunctive binding.
        - W_repeat: Expands g into p-space by repeating it for each sensory dimension.
        - W_tile: Expands x into p-space by tiling it for each g dimension.
        The element-wise product of these expanded vectors forms p.
        """
        # W_repeat: (p_dim, g_down_dim)
        W_repeat = torch.zeros(p_dim, g_down_dim)
        block = torch.eye(g_down_dim)
        for k in range(s_dim):
            W_repeat[k*g_down_dim:(k+1)*g_down_dim, :] = block
        self.register_buffer('W_repeat', W_repeat)

        # W_tile: (p_dim, s_dim)
        W_tile = torch.zeros(p_dim, s_dim)
        for k in range(s_dim):
            W_tile[k * g_down_dim : (k + 1) * g_down_dim, k] = 1.0
        self.register_buffer('W_tile', W_tile)

@torch.jit.script
def combine_distributions_precision_weighted(mu1, log_var1, mu2, log_var2):
    """
    Combines two Gaussian distributions using precision-weighted averaging.
    This is the Bayesian way to fuse two independent estimates of the same variable.
    Here, it fuses the path integration prior with the sensory-cued memory estimate.

    Returns:
        The mean and log-variance of the combined Gaussian distribution.
    """
    inv_sigma_sq1 = torch.exp(-log_var1)
    inv_sigma_sq2 = torch.exp(-log_var2)

    # New variance: 1 / (1/sigma1^2 + 1/sigma2^2) -> New log-variance: -log(1/sigma1^2 + 1/sigma2^2)
    log_var_comb = -torch.log(inv_sigma_sq1 + inv_sigma_sq2)
    var_comb = torch.exp(log_var_comb)

    # New mean: sigma_comb^2 * (mu1/sigma1^2 + mu2/sigma2^2)
    mu_comb = var_comb * (mu1 * inv_sigma_sq1 + mu2 * inv_sigma_sq2)

    return mu_comb, log_var_comb

class InferenceNetwork(nn.Module):
    """
    Implements the full TEM inference path (the "recognition model" q_phi).
    It infers the latent states g and p from the sensory observations x.
    Flow: x_t -> x_filtered -> p_inf_x (from memory) -> g_inf -> p_inf.
    """
    def __init__(self, config: TEMConfig):
        super().__init__()
        self.cfg = config
        S, g_dims, sens_dim = config.n_streams, config.g_dims, config.sensory_dim

        # (1) Sensory Compressor: Projects one-hot sensory input to a dense vector.
        self.sensory_compressor = nn.Sequential(
            nn.Linear(config.n_sensory_objects, 20 * sens_dim),
            nn.ELU(),
            nn.Linear(20 * sens_dim, sens_dim)
        )

        # (2) Temporal Filter: Learns a separate filtering coefficient gamma_f for each stream.
        # This allows streams to specialize to different temporal scales.
        freqs = torch.tensor(config.temporal_frequencies)
        init_gamma_logits = torch.log(freqs / (1 - freqs)) # Inverse sigmoid
        self.gamma_logit = nn.Parameter(init_gamma_logits)

        # (3) Place-cell Gain: Learns a scalar gain w_p[f] for each stream's sensory input.
        self.w_p = nn.Parameter(torch.ones(S))

        # (4) Memory-to-Grid Heads (p -> g):
        # These MLPs map the memory-retrieved p-state back to a distribution over g,
        # providing a sensory-based correction to the path integration.
        self.p2g_mu = nn.ModuleList()
        self.p2g_logsig = nn.ModuleList()
        for f, gd in enumerate(g_dims):
            n_phase = self.cfg.g_downsampled_dims[f]
            self.p2g_mu.append(nn.Sequential(
                nn.Linear(n_phase, 2*gd), nn.ELU(), nn.Linear(2*gd, gd)
            ))
            # Input to logsig MLP is [g_norm, err_from_mem]
            self.p2g_logsig.append(nn.Sequential(
                nn.Linear(2, 2*gd), nn.ELU(), nn.Linear(2*gd, gd), nn.Tanh()
            ))

        # Curriculum scalar p2g_use(t) ramps from 0 to 1 to gate this pathway.
        self.register_buffer("p2g_use", torch.tensor(0.0))
        if config.use_two_hot:
            self.register_buffer('two_hot_mat', config.two_hot_mat)

    def _temporal_filter(self, x_comp: torch.Tensor, x_prev: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Applies exponential temporal smoothing to the compressed sensory input,
        with a different learned rate for each stream.
        x'_t = alpha * x'_{t-1} + (1-alpha) * x^c_t
        """
        a = torch.sigmoid(self.gamma_logit)
        return [(a[f] * x_old + (1 - a[f]) * x_comp) for f, x_old in enumerate(x_prev)]

    def _normalize_sensory(self, x_filt_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Applies f_n normalization from paper: demean, ReLU, L2 normalize."""
        x_normed = []
        for x_filt in x_filt_list:
            x_demean = F.relu(x_filt - x_filt.mean(dim=1, keepdim=True))
            x_norm = F.normalize(x_demean, p=2, dim=1)
            x_normed.append(x_norm)
        return x_normed

    def forward(self,
                g_prior_mean: torch.Tensor, g_prior_log_var: torch.Tensor,
                x_t: torch.Tensor,
                x_filtered_prev: List[torch.Tensor],
                fused_p_fn: Callable,
                p_inf_x_concat: Optional[torch.Tensor] = None,
                p2g_scale_offset: float = 0.0,
                err_from_mem: Optional[torch.Tensor] = None) -> Dict[str, Any]:

        # --- 1. Sensory Processing ---
        x_comp = self.sensory_compressor(x_t)
        x_filt_list = self._temporal_filter(x_comp, x_filtered_prev)
        x_normed_list = self._normalize_sensory(x_filt_list)

        # --- 2. Memory-Driven Correction for g ---
        if p_inf_x_concat is not None and err_from_mem is not None:
            p_inf_x_split = torch.split(p_inf_x_concat, self.cfg.p_dims, dim=1)
            mu_mem_list, logvar_mem_list = [], []

            for f in range(self.cfg.n_streams):
                # Reshape p to (B, n_phase, sens) then average over sensory dimension
                B, n_phase, sens = p_inf_x_split[f].size(0), self.cfg.g_downsampled_dims[f], self.cfg.sensory_dim
                mu_attr = p_inf_x_split[f].view(B, n_phase, sens).mean(-1)

                # Get mean and log-variance from memory-to-grid heads
                mu_mem_f = self.p2g_mu[f](mu_attr)
                g_norm = mu_mem_f.pow(2).sum(1, keepdim=True).detach()
                logsig_input = torch.cat([g_norm, err_from_mem], 1)
                logvar_f = scale_log_sigma(self.p2g_logsig[f](logsig_input), temp=float(self.p2g_use))

                # Add curriculum-controlled noise floor
                logvar_f = logvar_f + (1 - float(self.p2g_use)) * p2g_scale_offset

                mu_mem_list.append(mu_mem_f)
                logvar_mem_list.append(logvar_f)

            mu_mem = torch.cat(mu_mem_list, dim=1)
            logvar_mem = torch.cat(logvar_mem_list, dim=1)

            # Fuse path integration prior with memory-based estimate
            g_inf_mu, g_inf_log_var = combine_distributions_precision_weighted(
                g_prior_mean, g_prior_log_var, mu_mem, logvar_mem
            )
        else:
            # If no memory pathway, posterior is just the prior
            g_inf_mu, g_inf_log_var = g_prior_mean, g_prior_log_var

        # --- 3. Reparameterization Trick ---
        if self.cfg.use_stochastic:
            eps = torch.randn_like(g_inf_mu)
            g_smp = g_inf_mu + torch.exp(0.5 * g_inf_log_var) * eps
        else:
            g_smp = g_inf_mu

        g_smp = torch.clamp(g_smp, -1, 1)
        g_inf_mu = torch.clamp(g_inf_mu, -1, 1)

        # --- 4. Split by Stream and Form `p` ---
        g_dims_split = self.cfg.g_dims
        g_means, g_logvars, g_samples = torch.split(g_inf_mu, g_dims_split, 1), torch.split(g_inf_log_var, g_dims_split, 1), torch.split(g_smp, g_dims_split, 1)
        g_inf_list = [VariationalOutput(m, v, s) for m, v, s in zip(g_means, g_logvars, g_samples)]

        x_scaled = [torch.sigmoid(self.w_p[f]) * xf for f, xf in enumerate(x_normed_list)]
        p_inf_list = fused_p_fn(g_samples, x_scaled)

        return {
            "g_inferred": g_inf_list,
            "p_inferred": p_inf_list,
            "x_filtered_current": x_filt_list
        }

class GenerativeNetwork(nn.Module):
    """
    Implements the generative model `p_theta`, which decodes a hippocampal state `p`
    back into a prediction over sensory observations `x`.
    """
    def __init__(self, config: TEMConfig, W_tile_stream0: torch.Tensor):
        super().__init__()
        self.config = config
        # This fixed matrix projects p back to the compressed sensory space
        self.register_buffer("W_tile", W_tile_stream0)
        self.w_x = nn.Parameter(torch.tensor(1.0))
        self.b_x = nn.Parameter(torch.zeros(config.sensory_dim))
        # This MLP decompresses the sensory vector to logits over all possible objects
        self.f_d_mlp = nn.Sequential(
            nn.Linear(config.sensory_dim, 20 * config.sensory_dim),
            nn.ELU(),
            nn.Linear(20 * config.sensory_dim, config.n_sensory_objects)
        )

    def forward_decode(self, p_t_stream0: torch.Tensor) -> torch.Tensor:
        """Decodes the lowest-frequency stream of p to predict x."""
        # Project p back to sensory space: x_s = w_x * (p @ W_tile) + b_x
        x_s = self.w_x * (p_t_stream0 @ self.W_tile) + self.b_x
        return self.f_d_mlp(x_s) # Decompress to logits

# ======================================================================================
# 5. Main TEM Class
# ======================================================================================

@torch.jit.script
def _scriptable_fused_form_p(g_samples_list: List[torch.Tensor], 
                             x_filtered_list: List[torch.Tensor], 
                             g_downsampled_dims: List[int],
                             W_repeat_all: torch.Tensor,
                             W_tile_all: torch.Tensor) -> torch.Tensor:
    """JIT-scriptable version of forming the p-vector."""
    g_down_list = [g[:, :d] for g, d in zip(g_samples_list, g_downsampled_dims)]
    g_down_all = torch.cat(g_down_list, dim=1)
    x_filtered_all = torch.cat(x_filtered_list, dim=1)

    g_expanded = g_down_all @ W_repeat_all.T
    x_expanded = x_filtered_all @ W_tile_all.T

    return F.leaky_relu(torch.clamp(g_expanded * x_expanded, -1, 1))

@torch.jit.script
def _scriptable_transform_g_to_p(g_list: List[torch.Tensor], 
                                 g_downsampled_dims: List[int],
                                 W_repeat_all: torch.Tensor) -> torch.Tensor:
    """JIT-scriptable version of transforming g to p-space."""
    g_down_list = [g[:, :d] for g, d in zip(g_list, g_downsampled_dims)]
    g_down = torch.cat(g_down_list, dim=1)
    return F.leaky_relu(torch.clamp(g_down @ W_repeat_all.T, -1, 1))

class TolmanEichenbaumMachine(nn.Module):
    """
    The main TEM model, integrating the inference, generative, and memory components.
    This class orchestrates the entire computational flow for a single timestep,
    as depicted in Figures 2D and S2.
    """
    def __init__(self, config: TEMConfig):
        super().__init__()
        self.config = config
        self.streams = nn.ModuleList([TEMStream(i, config) for i in range(config.n_streams)])
        self.inference_net = InferenceNetwork(config)
        self.generative_net = GenerativeNetwork(config, W_tile_stream0=self.streams[0].W_tile)
        self.memory_system = HierarchicalMemory(config)
        self.loss_fn = CompleteLoss(config)

        # --- Learnable Initial State g_0 ---
        g_init_std = 0.5
        g_init_params = []
        for g_dim in config.g_dims:
            tensor = torch.empty(g_dim)
            torch.nn.init.trunc_normal_(tensor, mean=0.0, std=g_init_std, a=-2*g_init_std, b=2*g_init_std)
            g_init_params.append(nn.Parameter(tensor))
        self.g_init = nn.ParameterList(g_init_params)

        # --- Learnable Transition Model ---
        # A separate MLP for each stream generates the action-dependent transition matrix.
        g_conn = config.g_connections
        g_dims_tensor = torch.tensor(config.g_dims, dtype=torch.float)
        in_dims_per_module = [(g_conn[f_to, :] @ g_dims_tensor).int().item() for f_to in range(config.n_streams)]
        mlp_out_dims = [in_d * out_d for in_d, out_d in zip(in_dims_per_module, config.g_dims)]

        modules = []
        for f in range(config.n_streams):
            module = nn.Sequential(
                nn.Linear(config.n_actions, 20, bias=True),
                nn.Tanh(),
                nn.Linear(20, mlp_out_dims[f], bias=False)
            )
            nn.init.zeros_(module[0].bias) # Start with zero bias
            nn.init.zeros_(module[2].weight) # Start with identity-like transition
            modules.append(module)
        self.transition_model = nn.ModuleList(modules)

        # --- Learnable Prior Variance MLP ---
        self.g_prior_log_var_mlp = nn.Sequential(
            nn.Linear(config.total_g_dim, config.total_g_dim * 2),
            nn.LeakyReLU(0.01),
            nn.Linear(config.total_g_dim * 2, config.total_g_dim)
        )

        # --- Fixed Projection Matrices (fused) ---
        self.register_buffer("W_repeat_all", torch.block_diag(*[s.W_repeat for s in self.streams]))
        self.register_buffer("W_tile_all", torch.block_diag(*[s.W_tile for s in self.streams]))

        # --- Curriculum-controlled Buffers ---
        self.register_buffer("p2g_scale_offset", torch.tensor(0.0))
        self.register_buffer("temp", torch.tensor(1.0))

    def set_p2g_scale_offset(self, value: float):
        """Sets the variance offset from the curriculum for the g-inference pathway."""
        self.p2g_scale_offset.fill_(value)

    def set_prior_temp(self, value: float):
        """Sets the temperature for scaling the learned prior variance."""
        self.temp.fill_(value)

    @property
    def loss_weights(self):
        return self.loss_fn.weights

    @loss_weights.setter
    def loss_weights(self, weights: dict):
        """Updates loss weights from a curriculum scheduler."""
        if hasattr(self, 'loss_fn') and weights is not None:
            self.loss_fn.update_weights(weights)

    def _prepare_sensory_for_memory_query(self, x_t: torch.Tensor, x_filtered_prev_list: List[torch.Tensor]) -> torch.Tensor:
        """Prepares raw sensory input to be a valid query for the hierarchical memory system."""
        x_comp = self.inference_net.sensory_compressor(x_t)
        x_filt_list = self.inference_net._temporal_filter(x_comp, x_filtered_prev_list)
        x_normed_list = self.inference_net._normalize_sensory(x_filt_list)
        x_scaled = [torch.sigmoid(self.inference_net.w_p[f]) * xf for f, xf in enumerate(x_normed_list)]

        # Project into p-space using the fixed W_tile matrix
        x_scaled_concat = torch.cat(x_scaled, dim=1)
        x_query = x_scaled_concat @ self.W_tile_all.T
        return F.leaky_relu(torch.clamp(x_query, -1, 1))

    def forward(self, x_t: torch.Tensor, a_t: torch.Tensor, prev_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        A full forward pass for a single timestep, following the paper's logic.
        """
        # --- Unpack Previous State ---
        g_prev_list = prev_state['g_states']
        M_prev = prev_state['memory']
        x_filtered_prev_list = prev_state.get('x_filtered_states')
        B = x_t.shape[0]

        if x_filtered_prev_list is None: # Initialize if first step
            x_filtered_prev_list = [torch.zeros(B, self.config.sensory_dim, device=x_t.device) for _ in range(self.config.n_streams)]

        # --- STEP 1: Path Integration (Generative Prior for g_t) ---
        # g_prior ~ p(g_t | g_{t-1}, a_t)
        a_onehot = F.one_hot(a_t, self.config.n_actions).float()
        g_prev_concat = torch.cat(g_prev_list, dim=1)

        D_a_flat_list = [module(a_onehot) for module in self.transition_model]
        all_g_updates = []
        for f_to in range(self.config.n_streams):
            connected_indices = self.config.g_connections[f_to].nonzero().flatten()
            g_in_list = [g_prev_list[i] for i in connected_indices]
            g_in_concat = torch.cat(g_in_list, dim=1)
            in_dim, out_dim = g_in_concat.shape[1], self.config.g_dims[f_to]
            D_a_batch = D_a_flat_list[f_to].view(B, out_dim, in_dim)
            delta_f = torch.bmm(D_a_batch, g_in_concat.unsqueeze(2)).squeeze(2)
            all_g_updates.append(delta_f)

        g_prior_mean_list = [(g_prev + delta) for g_prev, delta in zip(g_prev_list, all_g_updates)]
        g_prior_mean = torch.clamp(torch.cat(g_prior_mean_list, dim=1), -1, 1)

        # Get generative prior (fixed variance) and inference prior (learned variance)
        g_prior_log_var_gen = torch.full_like(g_prior_mean, 0.0)
        raw_log_var_inf = self.g_prior_log_var_mlp(g_prev_concat)
        g_prior_log_var_inf = scale_log_sigma(raw_log_var_inf, temp=float(self.temp))

        g_priors_gen = [VariationalOutput(m, v, m) for m, v in zip(self._split_g(g_prior_mean), self._split_g(g_prior_log_var_gen))]
        g_priors_inf = [VariationalOutput(m, v, m) for m, v in zip(self._split_g(g_prior_mean), self._split_g(g_prior_log_var_inf))]

        # --- STEP 2: Sensory-Cued Memory Retrieval (for Inference) ---
        p_inf_x_concat, p_inf_x_list, err_from_mem = None, None, None
        if self.config.use_p_inf:
            x_query = self._prepare_sensory_for_memory_query(x_t, x_filtered_prev_list)
            mem_key = 'inference' if not self.config.common_memory else 'generative'
            if M_prev.get(mem_key) is not None:
                p_inf_x_concat = self.memory_system.hierarchical_retrieve(x_query, M_prev[mem_key])
                p_inf_x_list = self.memory_system.split_by_stream(p_inf_x_concat, dim=1)

                # Calculate reconstruction error from this memory-cued `p`
                with torch.no_grad():
                    x_hat_from_mem = self.generative_net.forward_decode(p_inf_x_list[0])
                    err_from_mem = (x_t - x_hat_from_mem).pow(2).sum(1, keepdim=True)

        # --- STEP 3: Full Inference ---
        # g_inf ~ q(g_t|...), p_inf ~ q(p_t|...)
        inf_results = self.inference_net(
            g_prior_mean, g_prior_log_var_inf, x_t, x_filtered_prev_list,
            self._fused_form_p_vectorized, p_inf_x_concat,
            float(self.p2g_scale_offset), err_from_mem
        )
        g_inf_list, p_inf_list, x_filt_curr = inf_results.values()

        # --- STEP 4: Generation (Decoding from Latents) ---
        # Three prediction pathways are generated to be used in the loss function.
        # 1. From p_inf (pure inference path)
        x_logits_p = self.generative_net.forward_decode(p_inf_list[0])

        # 2. From g_inf (hybrid path: inferred g -> generative memory retrieval -> decode)
        p_query_g_inf = self._transform_g_to_p_space_fused([g.sample for g in g_inf_list])
        p_gen_g_inf = self.memory_system.hierarchical_retrieve(p_query_g_inf, M_prev['generative'])
        p_gen_list = self.memory_system.split_by_stream(p_gen_g_inf, dim=1)
        x_logits_g = self.generative_net.forward_decode(p_gen_list[0])

        # 3. From g_prior (pure generative path: prior g -> generative memory retrieval -> decode)
        p_query_g_prior = self._transform_g_to_p_space_fused([g.sample for g in g_priors_gen])
        p_gen_g_prior = self.memory_system.hierarchical_retrieve(p_query_g_prior, M_prev['generative'])
        p_gen_prior_list = self.memory_system.split_by_stream(p_gen_g_prior, dim=1)
        x_logits_gt = self.generative_net.forward_decode(p_gen_prior_list[0])

        # --- STEP 5: Loss Calculation ---
        predictions_dict = {'x_logits_p': x_logits_p, 'x_logits_g': x_logits_g, 'x_logits_gt': x_logits_gt}
        targets_dict = {'x': x_t}
        reps_dict = {'g_inf': g_inf_list, 'g_prior': g_priors_gen, 'p_inf': p_inf_list, 'p_gen': p_gen_list}
        if p_inf_x_list: reps_dict['p_inf_x'] = p_inf_x_list
        losses = self.loss_fn(predictions_dict, targets_dict, reps_dict)

        # --- STEP 6: Hebbian Memory Update ---
        p_inf_flat = torch.cat(p_inf_list, dim=1)
        p_gen_flat = torch.cat(p_gen_list, dim=1)
        M_new = {'generative': self.memory_system.hebbian_update(M_prev['generative'], p_inf_flat, p_gen_flat, 'generative')}

        if self.config.use_p_inf and not self.config.common_memory:
            p_inf_x_flat = torch.cat(p_inf_x_list, dim=1)
            M_new['inference'] = self.memory_system.hebbian_update(M_prev['inference'], p_inf_flat, p_inf_x_flat, 'inference')
        else:
            M_new['inference'] = M_prev['inference']
            
        # --- STEP 7: Gather Diagnostics for Logging ---
        diagnostics = {}
        with torch.no_grad():
            # (a) saturating fraction of g samples
            g_inf_samples = torch.cat([g.sample.abs() for g in g_inf_list], dim=-1)
            diagnostics['health/g_saturation'] = (g_inf_samples > 0.995).float().mean().item()

            # (b) cosine between inferred p and memory-retrieved p
            mem_cos = F.cosine_similarity(p_inf_flat, p_gen_flat, dim=-1).mean().item()
            diagnostics['health/memory_retrieval_cosine'] = mem_cos
            
            # (c) g-space variance
            g_vars = torch.cat([g.log_var.exp() for g in g_inf_list], dim=1)
            diagnostics["health/g_variance_mean"] = g_vars.mean().item()
            
             # (d) Representation Health (Sparsity & Norms)
            p_inf_samples = p_inf_flat
            diagnostics['health/g_sparsity'] = (g_inf_samples.abs() < 1e-3).float().mean().item()
            diagnostics['health/p_sparsity'] = (p_inf_samples.abs() < 1e-3).float().mean().item()
            diagnostics['health/g_norm'] = g_inf_samples.norm(dim=1).mean().item()
            diagnostics['health/p_norm'] = p_inf_samples.norm(dim=1).mean().item()
            
            # (e) Learned Parameter Tracking
            alphas = torch.sigmoid(self.inference_net.gamma_logit)
            w_ps = torch.sigmoid(self.inference_net.w_p)
            for f in range(self.config.n_streams):
                diagnostics[f'params/alpha_stream_{f}'] = alphas[f].item()
                diagnostics[f'params/w_p_stream_{f}'] = w_ps[f].item()
                
             # (f) Transition and Memory Stats
            g_update_norm = torch.cat(all_g_updates, dim=1).norm(dim=1).mean().item()
            diagnostics['health/g_update_norm'] = g_update_norm
            
            M_gen_new = M_new['generative']
            diagnostics['health/memory_max_abs_gen'] = M_gen_new.abs().max().item()
            diagnostics['health/memory_mean_abs_gen'] = M_gen_new.abs().mean().item()
            
            # (g) Generative Pathway Divergence Diagnostics
            g_inf_mean = torch.cat([g.mean for g in g_inf_list], dim=-1)

            # 1. How different is the predicted location from the "correct" sensory-informed location?
            g_divergence = (g_prior_mean - g_inf_mean).norm(dim=1).mean().item()
            diagnostics['divergence/g_space_L2'] = g_divergence

            # 2. How different are the retrieved memories when using the prior vs. the inferred g?
            p_retrieval_cos = F.cosine_similarity(p_gen_g_prior, p_gen_g_inf, dim=-1).mean().item()
            diagnostics['divergence/p_retrieval_cosine'] = p_retrieval_cos

            # 3. What is the final impact on the output logits?
            logit_cos = F.cosine_similarity(x_logits_gt, x_logits_g, dim=-1).mean().item()
            diagnostics['divergence/logit_cosine'] = logit_cos
            
            if self.config.use_p_inf and not self.config.common_memory and M_new.get('inference') is not None:
                 M_inf_new = M_new['inference']
                 diagnostics['health/memory_max_abs_inf'] = M_inf_new.abs().max().item()
                 diagnostics['health/memory_mean_abs_inf'] = M_inf_new.abs().mean().item()

        # --- STEP 8: Package State and Outputs ---
        new_state = {
            'g_states': [g.mean for g in g_inf_list],
            'p_inf': p_inf_list,
            'x_filtered_states': x_filt_curr,
            'memory': {k: (v.detach() if v is not None else None) for k, v in M_new.items()},
        }

        return {
            'losses': losses,
            'predictions': {
                'x_p': F.softmax(x_logits_p, dim=-1),
                'x_g': F.softmax(x_logits_g, dim=-1),
                'x_gt': F.softmax(x_logits_gt, dim=-1),
            },
            'new_state': new_state,
            'diagnostics': diagnostics,
            'reps_dict': reps_dict
        }

    def create_empty_memory(self, batch_size, device: torch.device) -> Dict[str, Any]:
        """Creates an initial empty state dictionary for the start of an episode."""
        initial_g_states = [g_init.unsqueeze(0).repeat(batch_size, 1) for g_init in self.g_init]
        return {
            'g_states': initial_g_states,
            'memory': self.memory_system.create_empty_memory(batch_size, device),
            'x_filtered_states': [torch.zeros(batch_size, self.config.sensory_dim, device=device) for _ in range(self.config.n_streams)],
            'p_inf': [torch.zeros(batch_size, p_dim, device=device) for p_dim in self.config.p_dims]
        }

    def _fused_form_p_vectorized(self, g_samples_list: List[torch.Tensor], x_filtered_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forms the conjunctive p-representation p = g elementwise* x in a single vectorized operation."""
        p_concat_fused = _scriptable_fused_form_p(
            g_samples_list, 
            x_filtered_list, 
            self.config.g_downsampled_dims,
            self.W_repeat_all,
            self.W_tile_all
        )
        return self._split_p(p_concat_fused)

    def _transform_g_to_p_space_fused(self, g_list: List[torch.Tensor]) -> torch.Tensor:
        """Transforms g-states into a p-space query for memory retrieval."""
        return _scriptable_transform_g_to_p(
            g_list,
            self.config.g_downsampled_dims,
            self.W_repeat_all
        )

    # --- Helper methods for splitting and stacking tensors by stream ---
    def _split_g(self, t: torch.Tensor, dim: int = -1) -> List[torch.Tensor]:
        return list(torch.split(t, self.config.g_dims, dim=dim))

    def _split_p(self, t: torch.Tensor, dim: int = -1) -> List[torch.Tensor]:
        return list(torch.split(t, self.config.p_dims, dim=dim))