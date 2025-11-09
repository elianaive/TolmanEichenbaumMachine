"""
Tolman-Eichenbaum Machine (TEM) - PyTorch Implementation

This module implements the TEM architecture, a hierarchical neural model for cognitive
mapping that factorizes structural knowledge (grid cells) from sensory knowledge (sensory
inputs) and combines them through conjunctive coding (place cells).

Core Architecture:
    - Factorization: g (grid/MEC) represents reusable spatial structure
    - Sensory processing: x (sensory/LEC) encodes observation-specific information
    - Conjunction: p (place/HC) = f(g, x) binds structure to sensory context
    - Hebbian memory: Fast learning with attractor dynamics for retrieval
    - Hierarchical streams: 5 temporal frequencies with asymmetric connectivity

Three Computational Pathways:
    1. Generative (g_t -> p_gen -> x_gt): Pure path integration + memory
    2. Inference (x_t -> p_inf -> g_inf): Sensory-driven state estimation
    3. Hybrid (g_inf -> p_gen -> x_g): Corrected path integration + memory

References:
    Whittington et al. (2020). The Tolman-Eichenbaum Machine.
"""

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


# ==============================================================================
# Configuration
# ==============================================================================

class TEMConfig:
    """
    Configuration for TEM architecture. Acts as single source of truth for all
    hyperparameters and architectural choices.
    """
    def __init__(self):
        # Core architecture
        self.n_streams: int = 5
        self.n_recurs = self.n_streams
        self.max_attractor_its_gen = [self.n_recurs - f for f in range(self.n_streams)]
        self.max_attractor_its_inv = [self.n_recurs for _ in range(self.n_streams)]

        # Dimensionality per stream
        self.g_dims: List[int] = [30, 30, 24, 18, 18]           # g-space (MEC)
        self.g_downsampled_dims: List[int] = [10, 10, 8, 6, 6] # g projected to p-space
        self.sensory_dim: int = 10                              # Compressed sensory x^c
        self.n_sensory_objects: int = 45                        # One-hot x vocabulary
        self.n_actions: int = 4                                 # Action space size
        self.use_stochastic: bool = False                       # Reparameterization trick
        self.bptt_len: int = 75                                 # BPTT chunk length

        # Derived dimensions
        # p-space is conjunction of downsampled g and compressed x
        self.p_dims: List[int] = [d * self.sensory_dim for d in self.g_downsampled_dims]
        self.total_p_dim: int = sum(self.p_dims)
        self.total_g_dim: int = sum(self.g_dims)

        # Hebbian memory parameters
        self.use_p_inf: bool = True                    # Use inference memory pathway
        self.common_memory: bool = False               # Separate gen/inf memories
        self.memory_iterations: int = 5                # Attractor iterations (tau)
        self.memory_decay: float = 0.8                 # Attractor decay (kappa)
        self.hebbian_learning_rate: float = 0.5        # Memory update rate (eta)
        self.hebbian_forget_rate: float = 0.9999       # Memory decay (lambda)

        # Hierarchical temporal specialization
        # Temporal filtering coefficients alpha^f (higher = slower)
        self.temporal_frequencies: List[float] = [0.01, 0.7, 0.91, 0.97, 0.99]

        # Asymmetric connectivity: slower streams influence faster ones
        self.g_connections: torch.Tensor = torch.zeros(self.n_streams, self.n_streams)
        for f_from in range(self.n_streams):
            for f_to in range(self.n_streams):
                # Connection allowed if source >= target in temporal frequency
                if self.temporal_frequencies[f_from] >= self.temporal_frequencies[f_to]:
                    self.g_connections[f_to, f_from] = 1.0

        # Optional sensory encoding
        self.use_two_hot: bool = True
        self.two_hot_mat: Optional[torch.Tensor] = None
        self.load_two_hot()

    def load_two_hot(self):
        """Precomputes fixed two-hot encoding matrix if enabled."""
        if self.use_two_hot:
            mat = make_two_hot_matrix(self.n_sensory_objects, self.sensory_dim)
            self.two_hot_mat = torch.from_numpy(mat)


class VariationalOutput(NamedTuple):
    """Container for variational layer outputs."""
    mean: torch.Tensor       # [B, D] posterior mean
    log_var: torch.Tensor    # [B, D] posterior log-variance
    sample: torch.Tensor     # [B, D] reparameterized sample


# ==============================================================================
# Helper Functions
# ==============================================================================

@torch.jit.script
def _attractor_update_step(h: torch.Tensor, memory: torch.Tensor, decay: float) -> torch.Tensor:
    """
    Single step of attractor network dynamics.

    Equation: h_tau = f(kappa * h_{tau-1} + M * h_{tau-1})

    Args:
        h: Current attractor state [B, P]
        memory: Hebbian memory matrix [B, P, P]
        decay: Scalar decay term kappa

    Returns:
        Updated attractor state [B, P]
    """
    mv = torch.bmm(memory, h.unsqueeze(-1)).squeeze(-1)  # [B, P]
    update = decay * h + mv
    return F.leaky_relu(torch.clamp(update, -1, 1))


@torch.jit.script
def scale_log_sigma(raw: torch.Tensor, temp: float = 1.0, LOGSIG_RATIO: float = 6.0,
                   LOGSIG_OFFSET: float = -2.0) -> torch.Tensor:
    """
    Scales raw MLP output to bounded log-variance, preventing pathological variances.

    Args:
        raw: Unconstrained MLP output [B, D]
        temp: Curriculum scalar in [0,1] (0=fixed sigma, 1=full range)

    Returns:
        Bounded log variance [B, D]
    """
    bounded = torch.tanh(raw / LOGSIG_RATIO) * LOGSIG_RATIO
    return LOGSIG_OFFSET + temp * bounded


def make_two_hot_matrix(n_sensory: int, two_hot_dim: int) -> np.ndarray:
    """
    Creates fixed two-hot encoding matrix where each sensory index maps to
    exactly two '1's in a vector of length two_hot_dim.

    Args:
        n_sensory: Number of unique sensory objects
        two_hot_dim: Dimension of two-hot vectors

    Returns:
        Encoding matrix [n_sensory, two_hot_dim]
    """
    combos = list(itertools.combinations(range(two_hot_dim), 2))
    if len(combos) < n_sensory:
        raise ValueError(f"Need at least {n_sensory} distinct two-hot codes; "
                        f"only {len(combos)} available.")
    mat = np.zeros((n_sensory, two_hot_dim), dtype=np.float32)
    for row_idx, (i, j) in enumerate(combos[:n_sensory]):
        mat[row_idx, i] = 1.0
        mat[row_idx, j] = 1.0
    return mat


@torch.jit.script
def combine_distributions_precision_weighted(mu1, log_var1, mu2, log_var2):
    """
    Bayesian fusion of two Gaussian estimates using precision weighting.
    Used to combine path integration prior with sensory-cued memory estimate.

    Equations:
        sigma_comb^2 = 1 / (1/sigma1^2 + 1/sigma2^2)
        mu_comb = sigma_comb^2 * (mu1/sigma1^2 + mu2/sigma2^2)

    Args:
        mu1, log_var1: First distribution parameters [B, D]
        mu2, log_var2: Second distribution parameters [B, D]

    Returns:
        Combined (mu, log_var) [B, D] each
    """
    inv_sigma_sq1 = torch.exp(-log_var1)
    inv_sigma_sq2 = torch.exp(-log_var2)

    log_var_comb = -torch.log(inv_sigma_sq1 + inv_sigma_sq2)
    var_comb = torch.exp(log_var_comb)

    mu_comb = var_comb * (mu1 * inv_sigma_sq1 + mu2 * inv_sigma_sq2)

    return mu_comb, log_var_comb


# ==============================================================================
# Hierarchical Memory System
# ==============================================================================

class HierarchicalMemory(nn.Module):
    """
    Fast-learning Hebbian memory with attractor dynamics and hierarchical retrieval.

    The hierarchy ensures larger temporal scales (gist) influence smaller scales
    (details) during retrieval via asymmetric connectivity masks.

    Key Operations:
        - Hebbian update: M_t = lambda * M_{t-1} + eta * (p - p_hat)(p + p_hat)^T
        - Attractor retrieval: h_tau = f(kappa * h_{tau-1} + M * h_{tau-1})
        - Staggered retrieval: Different streams run different iteration counts
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
        """Defines start/end indices for each stream in concatenated vectors."""
        self.stream_indices = []
        start = 0
        for p_dim in self.p_dims:
            self.stream_indices.append((start, start + p_dim))
            start += p_dim

    def _build_and_register_masks(self):
        """Creates hierarchical connectivity and staggered retrieval masks."""
        # Hierarchical connectivity mask: slower (higher freq index) -> faster
        # Creates upper triangular block structure
        hier_mask = torch.zeros(self.total_p_dim, self.total_p_dim)
        for tgt in range(self.n_streams):
            s_i, e_i = self.stream_indices[tgt]
            for src in range(self.n_streams):
                s_j, e_j = self.stream_indices[src]
                allow = self.config.temporal_frequencies[src] >= self.config.temporal_frequencies[tgt]
                if allow:
                    hier_mask[s_i:e_i, s_j:e_j] = 1.0
        self.register_buffer("hier_mask", hier_mask)

        # Staggered retrieval masks: higher-level streams run fewer iterations
        # e.g., [5, 4, 3, 2, 1] iterations for 5 streams
        def _build_masks(max_its):
            masks = []
            for it in range(max(max_its)):
                m = torch.zeros(self.total_p_dim)
                for s, rem in enumerate(max_its):
                    if it < rem:
                        st, en = self.stream_indices[s]
                        m[st:en] = 1.0
                masks.append(m)
            return torch.stack(masks)

        self.register_buffer("retrieval_masks_gen", _build_masks(self.config.max_attractor_its_gen),
                           persistent=False)
        self.register_buffer("retrieval_masks_inv", _build_masks(self.config.max_attractor_its_inv),
                           persistent=False)

    def create_empty_memory(self, batch_size: int, device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        """
        Initializes empty memory matrices for new environment.

        Returns:
            Dictionary with 'generative' and optionally 'inference' memory matrices [B, P, P]
        """
        P = self.total_p_dim
        M_gen = torch.zeros(batch_size, P, P, device=device)
        M_inf = None
        if self.config.use_p_inf and not self.config.common_memory:
            M_inf = torch.zeros_like(M_gen)
        return {"generative": M_gen, "inference": M_inf}

    def hebbian_update(
        self,
        M_prev: torch.Tensor,           # [B, P, P]
        p_inferred: torch.Tensor,       # [B, P]
        p_generated: torch.Tensor,      # [B, P]
        memory_type: str = "generative",
    ) -> torch.Tensor:
        """
        Hebbian memory update with hierarchical masking.

        Equation: M_t = lambda * M_{t-1} + eta * (p - p_hat)(p + p_hat)^T

        Args:
            M_prev: Previous memory matrix [B, P, P]
            p_inferred: Inferred place state p_t [B, P]
            p_generated: Predicted place state p_hat_t [B, P]
            memory_type: 'generative' or 'inference' for mask selection

        Returns:
            Updated memory matrix [B, P, P]
        """
        # Ensure batch dimension
        if p_inferred.dim() == 1: p_inferred = p_inferred.unsqueeze(0)
        if p_generated.dim() == 1: p_generated = p_generated.unsqueeze(0)

        term1 = (p_inferred - p_generated).unsqueeze(-1)  # [B, P, 1]
        term2 = (p_inferred + p_generated).unsqueeze(1)   # [B, 1, P]

        # Hebbian outer product
        M_new = torch.bmm(term1, term2)  # [B, P, P]

        # Apply hierarchical mask to M_new BEFORE adding to M_prev
        if memory_type == "generative":
            M_new = M_new * self.hier_mask

        # Update: M_t = lambda * M_{t-1} + eta * M_new, then clamp
        lambda_val = float(self.hebbian_forget_rate)
        eta_val = float(self.hebbian_learning_rate)
        hebbian_change = (lambda_val * M_prev + eta_val * M_new).clamp_(-1, 1)

        return hebbian_change

    def hierarchical_retrieve(
        self,
        query: torch.Tensor,     # [B, P]
        memory: torch.Tensor,    # [B, P, P]
        which: str = "gen"
    ) -> torch.Tensor:
        """
        Retrieves memory via attractor dynamics with staggered stream updates.

        Equation: h_tau = f(kappa * h_{tau-1} + M * h_{tau-1})

        Args:
            query: Initial attractor state h_0 [B, P]
            memory: Memory matrix M [B, P, P]
            which: 'gen' or 'inv' for retrieval mask selection

        Returns:
            Retrieved state after convergence [B, P]
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)

        h = F.leaky_relu(torch.clamp(query, -1, 1))
        masks = self.retrieval_masks_gen if which == "gen" else self.retrieval_masks_inv

        for it in range(masks.size(0)):
            active_mask = masks[it].unsqueeze(0)  # [1, P]
            update = _attractor_update_step(h, memory, self.config.memory_decay)
            # Only update active streams at this iteration
            h = (1 - active_mask) * h + active_mask * update

        return h

    def split_by_stream(self, p_concat: torch.Tensor, dim: int = 1) -> List[torch.Tensor]:
        """Splits concatenated p-vector into per-stream tensors."""
        return list(torch.split(p_concat, self.p_dims, dim=dim))

    def _block_ranges(self, dims: List[int]) -> List[Tuple[int, int]]:
        """Helper to compute block indices for block-diagonal analysis."""
        rs, s = [], 0
        for d in dims:
            rs.append((s, s + d))
            s += d
        return rs

    @torch.no_grad()
    def block_energy(self, M: torch.Tensor, dims: Optional[List[int]] = None) -> torch.Tensor:
        """
        Computes per-block mean |M| for visualization. Averages over batch if 3D.

        Args:
            M: Memory matrix [B, P, P] or [P, P]
            dims: Block dimensions (defaults to p_dims)

        Returns:
            Block energy matrix [S, S]
        """
        dims = self.p_dims if dims is None else dims
        ranges = self._block_ranges(dims)
        X = M.abs()
        if X.dim() == 3:
            X = X.mean(0)  # [P, P]
        S = len(ranges)
        out = X.new_zeros(S, S)
        for i, (si, ei) in enumerate(ranges):
            for j, (sj, ej) in enumerate(ranges):
                out[i, j] = X[si:ei, sj:ej].mean()
        return out.detach().cpu()

    @torch.no_grad()
    def leakage_ratio(self, M: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Energy in forbidden blocks / energy in allowed blocks.
        Measures hierarchical mask violation.

        Args:
            M: Memory matrix [B, P, P] or [P, P]
            mask: Hierarchical mask [P, P]

        Returns:
            Leakage ratio scalar
        """
        X = M.abs()
        if X.dim() == 3:
            X = X.mean(0)
        allowed = (mask > 0.5).float()
        forb = 1.0 - allowed
        num = (X * forb).sum()
        den = (X * allowed).sum() + 1e-8
        return float((num / den).item())

    @torch.no_grad()
    def convergence_trace(self, query: torch.Tensor, memory: torch.Tensor,
                         which: str = "gen") -> List[float]:
        """
        Per-iteration ||h_{t+1} - h_t||_2 to verify attractor convergence.

        Args:
            query: Initial state [B, P]
            memory: Memory matrix [B, P, P]
            which: 'gen' or 'inv'

        Returns:
            List of per-iteration delta norms
        """
        h = F.leaky_relu(torch.clamp(query, -1, 1))
        masks = self.retrieval_masks_gen if which == "gen" else self.retrieval_masks_inv
        trace: List[float] = []
        for it in range(masks.size(0)):
            h_new = _attractor_update_step(h, memory, float(self.config.memory_decay))
            active_mask = masks[it].unsqueeze(0)
            h_next = (1 - active_mask) * h + active_mask * h_new
            delta = (h_next - h).norm(dim=1).mean().item()
            trace.append(delta)
            h = h_next
        return trace


# ==============================================================================
# Loss Function
# ==============================================================================

class CompleteLoss(nn.Module):
    """
    Complete TEM loss combining reconstruction, latent alignment, and regularization.

    Components:
        1. Reconstruction: Cross-entropy for three prediction pathways (x_p, x_g, x_gt)
        2. Latent alignment: KL divergence surrogates for g and p spaces
        3. Regularization: L2 on g, L1 on p for sparsity
    """
    def __init__(self, config: TEMConfig):
        super().__init__()
        self.cfg = config
        # Default weights (updated by curriculum during training)
        self.weights = {
            # Reconstruction losses
            "x_nll_p": 1.0,
            "x_nll_g": 1.0,
            "x_nll_gt": 1.0,

            # Latent alignment
            "g_kl": 0.0,
            "p_align": 0.0,
            "p_inf_align": 0.0,

            # Regularization
            "g_reg": 0.01,
            "p_reg": 0.02,
        }

    @staticmethod
    def _gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
        """
        Analytic KL divergence D_KL(q || p) between diagonal Gaussians.

        Returns:
            KL per sample [B]
        """
        var_q, var_p = logvar_q.exp(), logvar_p.exp()
        kl = 0.5 * (
            logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0
        )
        return kl.sum(-1)

    def forward(self, predictions: dict, targets: dict, reps: dict) -> dict:
        """
        Calculates all loss components.

        Args:
            predictions: Dict with x_logits_{p,g,gt} [B, n_sensory]
            targets: Dict with ground truth x [B, n_sensory] (one-hot)
            reps: Dict with latent representations (g_inf, g_prior, p_inf, p_gen)

        Returns:
            Dictionary of losses including 'total_loss' [B]
        """
        losses = {}
        x_target = targets["x"].argmax(dim=-1)  # [B]

        # Reconstruction loss (cross-entropy) for three pathways
        losses["x_nll_p"] = F.cross_entropy(predictions["x_logits_p"], x_target, reduction="none")
        losses["x_nll_g"] = F.cross_entropy(predictions["x_logits_g"], x_target, reduction="none")
        losses["x_nll_gt"] = F.cross_entropy(predictions["x_logits_gt"], x_target, reduction="none")

        # Latent alignment: align inference posterior with generative prior
        # g-space: D_KL(q(g_t|...) || p(g_t|g_{t-1}, a_t))
        # Using MSE surrogate instead of full KL for stability
        g_align_mse = [
            0.5 * (g_inf.sample - g_pr.mean).pow(2).sum(-1)
            for g_inf, g_pr in zip(reps["g_inf"], reps["g_prior"])
        ]
        losses["g_kl"] = torch.stack(g_align_mse, dim=0).sum(0)

        # p-space: D_KL(q(p_t|...) || p(p_t|g_t, M_{t-1}))
        p_align_mse = [
            0.5 * (p_inf - p_gen).pow(2).sum(-1)
            for p_inf, p_gen in zip(reps["p_inf"], reps["p_gen"])
        ]
        losses["p_align"] = torch.stack(p_align_mse, dim=0).sum(0)

        # Optional: memory-cued alignment for inference path
        if "p_inf_x" in reps:
            p_inf_align_mse = [
                0.5 * (p_inf - p_mem).pow(2).sum(-1)
                for p_inf, p_mem in zip(reps["p_inf"], reps["p_inf_x"])
            ]
            losses["p_inf_align"] = torch.stack(p_inf_align_mse, dim=0).sum(0)
        else:
            losses["p_inf_align"] = torch.zeros_like(losses["g_kl"])

        # Regularization for sparse, bounded representations
        g_samples = torch.cat([g.mean for g in reps["g_inf"]], dim=-1)
        p_samples = torch.cat(reps["p_inf"], dim=-1)
        losses["g_reg"] = g_samples.pow(2).sum(-1)  # L2
        losses["p_reg"] = p_samples.abs().sum(-1)   # L1

        # Weighted total loss
        total = torch.zeros_like(losses["g_kl"])
        for name, val in losses.items():
            w = self.weights.get(name, 0.0)
            if w > 0:
                total = total + w * val
        losses["total_loss"] = total
        return losses

    def update_weights(self, new: dict):
        """Updates loss weights from curriculum scheduler."""
        self.weights.update(new)


# ==============================================================================
# Core Model Components
# ==============================================================================

class TEMStream(nn.Module):
    """
    Fixed projection matrices for a single hierarchical stream.

    Implements conjunctive binding p = g (elementwise)* x via:
        - W_repeat: Expands g to p-space by repeating
        - W_tile: Expands x to p-space by tiling
        - p = (W_repeat @ g) * (W_tile @ x)
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
        Initializes fixed projection matrices for conjunctive binding.

        W_repeat: [p_dim, g_down_dim] - repeats g for each sensory dimension
        W_tile: [p_dim, s_dim] - tiles x for each g dimension
        """
        # W_repeat: block-diagonal identity repeated s_dim times
        W_repeat = torch.zeros(p_dim, g_down_dim)
        block = torch.eye(g_down_dim)
        for k in range(s_dim):
            W_repeat[k*g_down_dim:(k+1)*g_down_dim, :] = block
        self.register_buffer('W_repeat', W_repeat)

        # W_tile: tiling pattern for sensory input
        W_tile = torch.zeros(p_dim, s_dim)
        for k in range(s_dim):
            W_tile[k * g_down_dim : (k + 1) * g_down_dim, k] = 1.0
        self.register_buffer('W_tile', W_tile)


class InferenceNetwork(nn.Module):
    """
    TEM inference model q_phi: infers latent states g and p from observations x.

    Flow:
        x_t -> compress -> temporal filter -> p_inf_x (memory retrieval)
            -> g_inf (fusion with path integration) -> p_inf (conjunctive binding)
    """
    def __init__(self, config: TEMConfig):
        super().__init__()
        self.cfg = config
        S, g_dims, sens_dim = config.n_streams, config.g_dims, config.sensory_dim

        # Sensory compressor: one-hot -> dense
        self.sensory_compressor = nn.Sequential(
            nn.Linear(config.n_sensory_objects, 20 * sens_dim),
            nn.ELU(),
            nn.Linear(20 * sens_dim, sens_dim)
        )

        # Temporal filter: learnable per-stream filtering coefficient gamma_f
        # x_filt^f = alpha^f * x_filt^f_{t-1} + (1 - alpha^f) * x_comp
        freqs = torch.tensor(config.temporal_frequencies)
        init_gamma_logits = torch.log(freqs / (1 - freqs))  # Inverse sigmoid
        self.gamma_logit = nn.Parameter(init_gamma_logits)

        # Place-cell gain: learnable scalar gain w_p[f] per stream
        self.w_p = nn.Parameter(torch.ones(S))

        # Memory-to-grid heads: p -> g distribution
        # Provides sensory-based correction to path integration
        self.p2g_mu = nn.ModuleList()
        self.p2g_logsig = nn.ModuleList()
        for f, gd in enumerate(g_dims):
            n_phase = self.cfg.g_downsampled_dims[f]
            self.p2g_mu.append(nn.Sequential(
                nn.Linear(n_phase, 2*gd), nn.ELU(), nn.Linear(2*gd, gd)
            ))
            # Input: [g_norm, err_from_mem]
            self.p2g_logsig.append(nn.Sequential(
                nn.Linear(2, 2*gd), nn.ELU(), nn.Linear(2*gd, gd), nn.Tanh()
            ))

        # Curriculum scalar gates memory pathway
        self.register_buffer("p2g_use", torch.tensor(0.0))
        if config.use_two_hot:
            self.register_buffer('two_hot_mat', config.two_hot_mat)

    def _temporal_filter(self, x_comp: torch.Tensor, x_prev: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Exponential temporal smoothing with per-stream learned rate.

        Equation: x_filt^f_t = alpha^f * x_filt^f_{t-1} + (1 - alpha^f) * x_comp_t

        Args:
            x_comp: Compressed sensory input [B, sens_dim]
            x_prev: Previous filtered states, list of [B, sens_dim]

        Returns:
            Updated filtered states, list of [B, sens_dim]
        """
        a = torch.sigmoid(self.gamma_logit)  # [S]
        return [(a[f] * x_old + (1 - a[f]) * x_comp) for f, x_old in enumerate(x_prev)]

    def _normalize_sensory(self, x_filt_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Normalization: demean -> ReLU -> L2 normalize.

        Args:
            x_filt_list: List of filtered sensory inputs [B, sens_dim]

        Returns:
            Normalized inputs, list of [B, sens_dim]
        """
        x_normed = []
        for x_filt in x_filt_list:
            x_demean = F.relu(x_filt - x_filt.mean(dim=1, keepdim=True))
            x_norm = F.normalize(x_demean, p=2, dim=1)
            x_normed.append(x_norm)
        return x_normed

    def forward(self,
                g_prior_mean: torch.Tensor,            # [B, total_g_dim]
                g_prior_log_var: torch.Tensor,         # [B, total_g_dim]
                x_t: torch.Tensor,                     # [B, n_sensory]
                x_filtered_prev: List[torch.Tensor],   # List[S] of [B, sens_dim]
                fused_p_fn: Callable,
                p_inf_x_concat: Optional[torch.Tensor] = None,  # [B, total_p_dim]
                p2g_scale_offset: float = 0.0,
                err_from_mem: Optional[torch.Tensor] = None) -> Dict[str, Any]:  # [B, 1]
        """
        Full inference forward pass.

        Returns:
            Dict with:
                'g_inferred': List[S] of VariationalOutput
                'p_inferred': List[S] of [B, p_dim]
                'x_filtered_current': List[S] of [B, sens_dim]
        """
        # Sensory processing
        if self.cfg.use_two_hot:
            x_comp = x_t @ self.two_hot_mat  # [B, sens_dim]
        else:
            x_comp = self.sensory_compressor(x_t)
        x_filt_list = self._temporal_filter(x_comp, x_filtered_prev)
        x_normed_list = self._normalize_sensory(x_filt_list)

        # Memory-driven correction for g (if available)
        if p_inf_x_concat is not None and err_from_mem is not None:
            p_inf_x_split = torch.split(p_inf_x_concat, self.cfg.p_dims, dim=1)
            mu_mem_list, logvar_mem_list = [], []

            for f in range(self.cfg.n_streams):
                # Reshape p to [B, n_phase, sens] and sum over sensory dimension
                B, n_phase, sens = p_inf_x_split[f].size(0), self.cfg.g_downsampled_dims[f], self.cfg.sensory_dim
                mu_attr = p_inf_x_split[f].view(B, n_phase, sens).sum(-1)  # [B, n_phase]

                # Memory-to-grid mean and variance
                mu_mem_f = self.p2g_mu[f](mu_attr)  # [B, g_dim]
                g_norm = mu_mem_f.pow(2).sum(1, keepdim=True)  # [B, 1]
                logsig_input = torch.cat([g_norm, err_from_mem], 1)  # [B, 2]

                # Scale variance with curriculum
                logvar_f = scale_log_sigma(self.p2g_logsig[f](logsig_input), temp=float(self.p2g_use))
                sigma_f = torch.exp(0.5 * logvar_f)
                sigma_f = sigma_f + p2g_scale_offset  # Add noise floor
                logvar_f = 2.0 * torch.log(sigma_f.clamp(min=1e-6))

                mu_mem_list.append(mu_mem_f)
                logvar_mem_list.append(logvar_f)

            mu_mem = torch.cat(mu_mem_list, dim=1)        # [B, total_g_dim]
            logvar_mem = torch.cat(logvar_mem_list, dim=1)

            # Precision-weighted fusion of path integration and memory
            g_inf_mu, g_inf_log_var = combine_distributions_precision_weighted(
                g_prior_mean, g_prior_log_var, mu_mem, logvar_mem
            )
        else:
            # No memory pathway: posterior equals prior
            g_inf_mu, g_inf_log_var = g_prior_mean, g_prior_log_var

        # Reparameterization trick
        if self.cfg.use_stochastic:
            eps = torch.randn_like(g_inf_mu)
            g_smp = g_inf_mu + torch.exp(0.5 * g_inf_log_var) * eps
        else:
            g_smp = g_inf_mu

        g_smp = torch.clamp(g_smp, -1, 1)
        g_inf_mu = torch.clamp(g_inf_mu, -1, 1)

        # Split by stream and form conjunctive p
        g_dims_split = self.cfg.g_dims
        g_means, g_logvars, g_samples = (
            torch.split(g_inf_mu, g_dims_split, 1),
            torch.split(g_inf_log_var, g_dims_split, 1),
            torch.split(g_smp, g_dims_split, 1)
        )
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
    Generative model p_theta: decodes hippocampal state p back to sensory observation x.

    Flow: p -> sensory space projection -> decompressor -> x_logits
    """
    def __init__(self, config: TEMConfig, W_tile_stream0: torch.Tensor):
        super().__init__()
        self.config = config
        # Fixed projection from p back to sensory space
        self.register_buffer("W_tile", W_tile_stream0)
        self.w_x = nn.Parameter(torch.tensor(1.0))
        self.b_x = nn.Parameter(torch.zeros(config.sensory_dim))
        # Sensory decompressor: dense -> logits over objects
        self.f_d_mlp = nn.Sequential(
            nn.Linear(config.sensory_dim, 20 * config.sensory_dim),
            nn.ELU(),
            nn.Linear(20 * config.sensory_dim, config.n_sensory_objects)
        )

    def forward_decode(self, p_t_stream0: torch.Tensor) -> torch.Tensor:
        """
        Decodes lowest-frequency stream of p to predict x.

        Args:
            p_t_stream0: Place state from stream 0 [B, p_dim[0]]

        Returns:
            Logits over sensory objects [B, n_sensory]
        """
        # Project to sensory space: x_s = w_x * (p @ W_tile) + b_x
        x_s = self.w_x * (p_t_stream0 @ self.W_tile) + self.b_x  # [B, sens_dim]
        return self.f_d_mlp(x_s)  # [B, n_sensory]


# ==============================================================================
# Main TEM Model
# ==============================================================================

@torch.jit.script
def _scriptable_fused_form_p(g_samples_list: List[torch.Tensor],     # List[S] of [B, g_dim]
                             x_filtered_list: List[torch.Tensor],    # List[S] of [B, sens_dim]
                             g_downsampled_dims: List[int],
                             W_repeat_all: torch.Tensor,             # [total_p_dim, sum(g_down)]
                             W_tile_all: torch.Tensor) -> torch.Tensor:  # [total_p_dim, S*sens_dim]
    """
    Vectorized conjunctive binding: p = (W_repeat @ g_down) * (W_tile @ x).

    Returns:
        Concatenated p vector [B, total_p_dim]
    """
    g_down_list = [g[:, :d] for g, d in zip(g_samples_list, g_downsampled_dims)]
    g_down_all = torch.cat(g_down_list, dim=1)        # [B, sum(g_down)]
    x_filtered_all = torch.cat(x_filtered_list, dim=1)  # [B, S*sens_dim]

    g_expanded = g_down_all @ W_repeat_all.T  # [B, total_p_dim]
    x_expanded = x_filtered_all @ W_tile_all.T  # [B, total_p_dim]

    return F.leaky_relu(torch.clamp(g_expanded * x_expanded, -1, 1))


@torch.jit.script
def _scriptable_transform_g_to_p(g_list: List[torch.Tensor],      # List[S] of [B, g_dim]
                                 g_downsampled_dims: List[int],
                                 W_repeat_all: torch.Tensor) -> torch.Tensor:  # [total_p_dim, sum(g_down)]
    """
    Transforms g to p-space for memory query: p_query = W_repeat @ g_down.

    Returns:
        p-space query [B, total_p_dim]
    """
    g_down_list = [g[:, :d] for g, d in zip(g_list, g_downsampled_dims)]
    g_down = torch.cat(g_down_list, dim=1)  # [B, sum(g_down)]
    return g_down @ W_repeat_all.T


class TolmanEichenbaumMachine(nn.Module):
    """
    Tolman-Eichenbaum Machine: Hierarchical cognitive mapping model.

    Integrates:
        - Path integration (transition model)
        - Sensory inference (inference network)
        - Memory retrieval (hierarchical memory)
        - Sensory prediction (generative network)

    Three computational pathways:
        1. Generative: g_t -> M_gen -> p_gen -> x_gt
        2. Inference: x_t -> M_inf -> g_inf -> p_inf -> x_p
        3. Hybrid: g_inf -> M_gen -> p_gen -> x_g
    """
    def __init__(self, config: TEMConfig):
        super().__init__()
        self.config = config
        self.streams = nn.ModuleList([TEMStream(i, config) for i in range(config.n_streams)])
        self.inference_net = InferenceNetwork(config)
        self.generative_net = GenerativeNetwork(config, W_tile_stream0=self.streams[0].W_tile)
        self.memory_system = HierarchicalMemory(config)
        self.loss_fn = CompleteLoss(config)

        # Learnable initial state g_0
        g_init_std = 0.5
        g_init_params = []
        for g_dim in config.g_dims:
            tensor = torch.empty(g_dim)
            torch.nn.init.trunc_normal_(tensor, mean=0.0, std=g_init_std, a=-2*g_init_std, b=2*g_init_std)
            g_init_params.append(nn.Parameter(tensor))
        self.g_init = nn.ParameterList(g_init_params)

        # Learnable transition model
        # Equation: g_t = g_{t-1} + Delta_a(g_{t-1})
        # Delta_a implemented as action-conditioned bilinear transform
        g_conn = config.g_connections
        g_dims_tensor = torch.tensor(config.g_dims, dtype=torch.float)
        in_dims_per_module = [(g_conn[f_to, :] @ g_dims_tensor).int().item() for f_to in range(config.n_streams)]
        mlp_out_dims = [in_d * out_d for in_d, out_d in zip(in_dims_per_module, config.g_dims)]

        modules = []
        for f in range(config.n_streams):
            module = nn.Sequential(
                nn.Linear(config.n_actions, 20, bias=True),
                nn.Tanh(),
                nn.Linear(20, mlp_out_dims[f], bias=False)  # No bias on output layer
            )
            nn.init.zeros_(module[0].bias)   # Start with zero bias on hidden layer
            nn.init.zeros_(module[2].weight)  # Zero weights for identity transition
            modules.append(module)
        self.transition_model = nn.ModuleList(modules)

        # Learnable prior variance heads (per-stream)
        self.g2g_logsig_inf = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gd, 2 * gd),
                nn.ELU(),
                nn.Linear(2 * gd, gd),
                nn.Tanh()
            ) for gd in config.g_dims
        ])

        # Fixed projection matrices (block-diagonal fusion)
        self.register_buffer("W_repeat_all", torch.block_diag(*[s.W_repeat for s in self.streams]))
        self.register_buffer("W_tile_all", torch.block_diag(*[s.W_tile for s in self.streams]))

        # Curriculum-controlled buffers
        self.register_buffer("p2g_scale_offset", torch.tensor(0.0))
        self.register_buffer("temp", torch.tensor(1.0))

    def set_p2g_scale_offset(self, value: float):
        """Sets variance offset for g-inference pathway (curriculum-controlled)."""
        self.p2g_scale_offset.fill_(value)

    def set_prior_temp(self, value: float):
        """Sets temperature for scaling learned prior variance (curriculum-controlled)."""
        self.temp.fill_(value)

    @property
    def loss_weights(self):
        return self.loss_fn.weights

    @loss_weights.setter
    def loss_weights(self, weights: dict):
        """Updates loss weights from curriculum scheduler."""
        if hasattr(self, 'loss_fn') and weights is not None:
            self.loss_fn.update_weights(weights)

    def _prepare_sensory_for_memory_query(self, x_t: torch.Tensor,           # [B, n_sensory]
                                          x_filtered_prev_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Prepares raw sensory input as valid query for hierarchical memory.

        Returns:
            Query vector in p-space [B, total_p_dim]
        """
        if self.config.use_two_hot:
            x_comp = x_t @ self.inference_net.two_hot_mat
        else:
            x_comp = self.inference_net.sensory_compressor(x_t)
        x_filt_list = self.inference_net._temporal_filter(x_comp, x_filtered_prev_list)
        x_normed_list = self.inference_net._normalize_sensory(x_filt_list)
        x_scaled = [torch.sigmoid(self.inference_net.w_p[f]) * xf for f, xf in enumerate(x_normed_list)]

        # Project into p-space using fixed W_tile matrix
        x_scaled_concat = torch.cat(x_scaled, dim=1)
        x_query = x_scaled_concat @ self.W_tile_all.T
        return x_query

    def forward(self, x_seq: torch.Tensor,      # [T, B]
                a_seq: torch.Tensor,             # [T, B]
                prev_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Forward pass for entire BPTT sequence.

        Returns:
            List of per-timestep outputs
        """
        T = x_seq.shape[0]
        all_outputs = []
        state = prev_state

        for t in range(T):
            outputs = self.forward_single_step(x_seq[t], a_seq[t], state)
            all_outputs.append(outputs)
            state = outputs['new_state']

        return all_outputs

    def forward_single_step(self, x_t: torch.Tensor,        # [B, n_sensory]
                           a_t: torch.Tensor,               # [B]
                           prev_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single-timestep forward pass following paper's computational flow.

        Steps:
            1. Path integration: g_prior ~ p(g_t | g_{t-1}, a_t)
            2. Sensory-cued retrieval: p_inf_x from memory
            3. Full inference: g_inf, p_inf ~ q(g_t, p_t | x_t, ...)
            4. Generation: Three prediction pathways
            5. Loss calculation
            6. Hebbian memory update
            7. Diagnostics

        Returns:
            Dict with losses, predictions, new_state, diagnostics, reps_dict
        """
        # Unpack previous state
        g_prev_list = prev_state['g_states']
        M_prev = prev_state['memory']
        x_filtered_prev_list = prev_state.get('x_filtered_states')
        B = x_t.shape[0]

        if x_filtered_prev_list is None:
            x_filtered_prev_list = [torch.zeros(B, self.config.sensory_dim, device=x_t.device)
                                   for _ in range(self.config.n_streams)]

        # STEP 1: Path Integration (Generative Prior)
        # Equation: g_t = g_{t-1} + Delta_a(g_{t-1})
        a_onehot = F.one_hot(a_t, self.config.n_actions).float()
        a_prev_onehot = prev_state['prev_action_onehot']
        g_prev_concat = torch.cat(g_prev_list, dim=1)

        D_a_flat_list = [module(a_prev_onehot) for module in self.transition_model]
        all_g_updates = []
        for f_to in range(self.config.n_streams):
            connected_indices = self.config.g_connections[f_to].nonzero().flatten()
            g_in_list = [g_prev_list[i] for i in connected_indices]
            g_in_concat = torch.cat(g_in_list, dim=1)
            in_dim, out_dim = g_in_concat.shape[1], self.config.g_dims[f_to]
            # Reshape to [B, in_dim, out_dim] for bilinear transform
            D_a_batch = D_a_flat_list[f_to].view(B, in_dim, out_dim)
            # Delta: [B, 1, in_dim] @ [B, in_dim, out_dim] = [B, 1, out_dim]
            delta_f = torch.bmm(g_in_concat.unsqueeze(1), D_a_batch).squeeze(1)
            all_g_updates.append(delta_f)

        delta = torch.cat(all_g_updates, dim=1)  # [B, total_g_dim]

        g_prior_mean_list = [(g_prev + delta) for g_prev, delta in zip(g_prev_list, all_g_updates)]
        g_prior_mean = torch.clamp(torch.cat(g_prior_mean_list, dim=1), -1, 1)

        # Per-stream inference variances
        logv_inf_pieces = []
        for f, (head, g_prev_f) in enumerate(zip(self.g2g_logsig_inf, g_prev_list)):
            raw = head(g_prev_f)
            logv_f = scale_log_sigma(raw, temp=float(self.temp))
            logv_inf_pieces.append(logv_f)
        g_prior_log_var_inf = torch.cat(logv_inf_pieces, dim=1)

        # Fixed variance for generative pathway (eliminates gradient conflicts)
        g_prior_log_var_gen = torch.zeros_like(g_prior_mean)

        g_priors_gen = [VariationalOutput(m, v, m) for m, v in zip(self._split_g(g_prior_mean),
                                                                    self._split_g(g_prior_log_var_gen))]
        g_priors_inf = [VariationalOutput(m, v, m) for m, v in zip(self._split_g(g_prior_mean),
                                                                    self._split_g(g_prior_log_var_inf))]

        # STEP 2: Sensory-Cued Memory Retrieval
        p_inf_x_concat, p_inf_x_list, err_from_mem = None, None, None
        if self.config.use_p_inf:
            x_query = self._prepare_sensory_for_memory_query(x_t, x_filtered_prev_list)
            mem_key = 'inference' if not self.config.common_memory else 'generative'
            if M_prev.get(mem_key) is not None:
                p_inf_x_concat = self.memory_system.hierarchical_retrieve(x_query, M_prev[mem_key], which="inv")
                p_inf_x_list = self.memory_system.split_by_stream(p_inf_x_concat, dim=1)

                # Reconstruction error from memory-cued p
                with torch.no_grad():
                    x_hat_logits = self.generative_net.forward_decode(p_inf_x_list[0])
                    x_hat_probs = torch.softmax(x_hat_logits, dim=-1)
                    err_from_mem = (x_t - x_hat_probs).pow(2).sum(1, keepdim=True)

        # STEP 3: Full Inference
        inf_results = self.inference_net(
            g_prior_mean, g_prior_log_var_inf, x_t, x_filtered_prev_list,
            self._fused_form_p_vectorized, p_inf_x_concat,
            float(self.p2g_scale_offset), err_from_mem
        )
        g_inf_list, p_inf_list, x_filt_curr = inf_results.values()

        # STEP 4: Generation (Three Prediction Pathways)
        # 1. Pure inference: p_inf -> x_p
        x_logits_p = self.generative_net.forward_decode(p_inf_list[0])

        # 2. Hybrid: g_inf -> memory retrieval -> p_gen -> x_g
        p_query_g_inf = self._transform_g_to_p_space_fused([g.sample for g in g_inf_list])
        p_gen_g_inf = self.memory_system.hierarchical_retrieve(p_query_g_inf, M_prev['generative'], which="gen")
        p_gen_list = self.memory_system.split_by_stream(p_gen_g_inf, dim=1)
        x_logits_g = self.generative_net.forward_decode(p_gen_list[0])

        # 3. Pure generative: g_prior -> memory retrieval -> p_gen -> x_gt
        p_query_g_prior = self._transform_g_to_p_space_fused([g.sample for g in g_priors_gen])
        p_gen_g_prior = self.memory_system.hierarchical_retrieve(p_query_g_prior, M_prev['generative'], which="gen")
        p_gen_prior_list = self.memory_system.split_by_stream(p_gen_g_prior, dim=1)
        x_logits_gt = self.generative_net.forward_decode(p_gen_prior_list[0])

        # STEP 5: Loss Calculation
        predictions_dict = {'x_logits_p': x_logits_p, 'x_logits_g': x_logits_g, 'x_logits_gt': x_logits_gt}
        targets_dict = {'x': x_t}
        reps_dict = {'g_inf': g_inf_list, 'g_prior': g_priors_gen, 'p_inf': p_inf_list, 'p_gen': p_gen_list}
        if p_inf_x_list: reps_dict['p_inf_x'] = p_inf_x_list
        losses = self.loss_fn(predictions_dict, targets_dict, reps_dict)

        # STEP 6: Hebbian Memory Update
        # Equation: M_t = lambda * M_{t-1} + eta * (p_inf - p_gen)(p_inf + p_gen)^T
        p_inf_flat = torch.cat(p_inf_list, dim=1)
        p_gen_flat = torch.cat(p_gen_list, dim=1)
        M_new = {'generative': self.memory_system.hebbian_update(M_prev['generative'], p_inf_flat, p_gen_flat,
                                                                 'generative')}

        if self.config.use_p_inf and not self.config.common_memory:
            p_inf_x_flat = torch.cat(p_inf_x_list, dim=1)
            M_new['inference'] = self.memory_system.hebbian_update(M_prev['inference'], p_inf_flat, p_inf_x_flat,
                                                                   'inference')
        else:
            M_new['inference'] = M_prev['inference']

        # STEP 7: Diagnostics
        diagnostics = {}
        with torch.no_grad():
            # g-space health metrics
            g_inf_samples = torch.cat([g.sample.abs() for g in g_inf_list], dim=-1)
            diagnostics['health/g_saturation'] = (g_inf_samples > 0.995).float().mean().item()
            diagnostics['health/g_sparsity'] = (g_inf_samples.abs() < 1e-3).float().mean().item()
            diagnostics['health/g_norm'] = g_inf_samples.norm(dim=1).mean().item()

            # p-space health metrics
            p_inf_flat = torch.cat(p_inf_list, dim=1)
            diagnostics['health/p_sparsity'] = (p_inf_flat.abs() < 1e-3).float().mean().item()
            diagnostics['health/p_norm'] = p_inf_flat.norm(dim=1).mean().item()

            # Memory-retrieval alignment
            p_gen_flat = torch.cat(p_gen_list, dim=1)
            mem_cos = F.cosine_similarity(p_inf_flat, p_gen_flat, dim=-1).mean().item()
            diagnostics['health/memory_retrieval_cosine'] = mem_cos

            # g variance
            g_vars = torch.cat([g.log_var.exp() for g in g_inf_list], dim=1)
            diagnostics["health/g_variance_mean"] = g_vars.mean().item()

            # Learned parameters
            alphas = torch.sigmoid(self.inference_net.gamma_logit)
            w_ps = torch.sigmoid(self.inference_net.w_p)
            for f in range(self.config.n_streams):
                diagnostics[f'params/alpha_stream_{f}'] = float(alphas[f].item())
                diagnostics[f'params/w_p_stream_{f}'] = float(w_ps[f].item())

            # Transition diagnostics
            diagnostics['transition/update_norm'] = delta.norm(dim=1).mean().item()

            # Memory diagnostics
            M_gen_new = M_new['generative']
            diagnostics['memory/gen_max_abs'] = M_gen_new.abs().max().item()
            diagnostics['memory/gen_mean_abs'] = M_gen_new.abs().mean().item()
            gen_block = self.memory_system.block_energy(M_gen_new)
            diagnostics['memory/gen_block_energy'] = gen_block.tolist()
            diagnostics['memory/gen_leakage_ratio'] = self.memory_system.leakage_ratio(
                M_gen_new, self.memory_system.hier_mask
            )

            if self.config.use_p_inf and not self.config.common_memory and M_new.get('inference') is not None:
                M_inf_new = M_new['inference']
                diagnostics['memory/inf_max_abs'] = M_inf_new.abs().max().item()
                diagnostics['memory/inf_mean_abs'] = M_inf_new.abs().mean().item()
                inf_block = self.memory_system.block_energy(M_inf_new)
                diagnostics['memory/inf_block_energy'] = inf_block.tolist()

            # Attractor convergence traces
            trace_gen = self.memory_system.convergence_trace(
                p_query_g_inf.detach(), M_prev['generative'], which="gen"
            )
            diagnostics['attractor/delta_trace_gen'] = trace_gen

            if (self.config.use_p_inf and not self.config.common_memory
                    and p_inf_x_concat is not None and (M_prev.get('inference') is not None)):
                trace_inv = self.memory_system.convergence_trace(
                    p_inf_x_concat.detach(), M_prev['inference'], which="inv"
                )
                diagnostics['attractor/delta_trace_inv'] = trace_inv
            else:
                diagnostics['attractor/delta_trace_inv'] = None

            # Decoder-side phase energy
            n_phase0 = self.config.g_downsampled_dims[0]
            sens = self.config.sensory_dim
            p0 = p_inf_list[0].reshape(p_inf_list[0].size(0), n_phase0, sens)
            phase_energy = p0.abs().mean(dim=(0, 2))
            diagnostics['decoder/phase0_energy_distribution'] = phase_energy.detach().cpu().tolist()

            # Divergence metrics
            g_inf_mean = torch.cat([g.mean for g in g_inf_list], dim=-1)
            g_divergence = (g_prior_mean - g_inf_mean).norm(dim=1).mean().item()
            diagnostics['divergence/g_space_L2'] = g_divergence

            p_retrieval_cos = F.cosine_similarity(p_gen_g_prior, p_gen_g_inf, dim=-1).mean().item()
            diagnostics['divergence/p_retrieval_cosine'] = p_retrieval_cos

            logit_cos = F.cosine_similarity(x_logits_gt, x_logits_g, dim=-1).mean().item()
            diagnostics['divergence/logit_cosine'] = logit_cos

            # Memory precision contribution
            g_post_logvar = torch.cat([g.log_var for g in g_inf_list], dim=1)
            prior_prec = torch.exp(-g_prior_log_var_inf)
            post_prec = torch.exp(-g_post_logvar)
            mem_prec = (post_prec - prior_prec).clamp_min(0.0)
            diagnostics['health/mem_to_prior_precision'] = float(
                mem_prec.mean().item() / (prior_prec.mean().item() + 1e-8)
            )

        # STEP 8: Package State and Outputs
        a_curr_onehot = F.one_hot(a_t, self.config.n_actions).float()
        new_state = {
            'g_states': [g.mean for g in g_inf_list],
            'p_inf': p_inf_list,
            'x_filtered_states': x_filt_curr,
            'memory': M_new,
            'prev_action_onehot': a_curr_onehot,
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
        """
        Creates initial empty state dictionary for start of episode.

        Returns:
            Dict with g_states, memory, x_filtered_states, p_inf, prev_action_onehot
        """
        initial_g_states = [g_init.unsqueeze(0).repeat(batch_size, 1) for g_init in self.g_init]
        return {
            'g_states': initial_g_states,
            'memory': self.memory_system.create_empty_memory(batch_size, device),
            'x_filtered_states': [torch.zeros(batch_size, self.config.sensory_dim, device=device)
                                 for _ in range(self.config.n_streams)],
            'p_inf': [torch.zeros(batch_size, p_dim, device=device) for p_dim in self.config.p_dims],
            'prev_action_onehot': torch.zeros(batch_size, self.config.n_actions, device=device)
        }

    def _fused_form_p_vectorized(self, g_samples_list: List[torch.Tensor],
                                x_filtered_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forms conjunctive p-representation via vectorized operation.

        Equation: p = (W_repeat @ g) * (W_tile @ x)

        Returns:
            List[S] of per-stream p tensors
        """
        p_concat_fused = _scriptable_fused_form_p(
            g_samples_list,
            x_filtered_list,
            self.config.g_downsampled_dims,
            self.W_repeat_all,
            self.W_tile_all
        )
        return self._split_p(p_concat_fused)

    def _transform_g_to_p_space_fused(self, g_list: List[torch.Tensor]) -> torch.Tensor:
        """Transforms g-states into p-space query for memory retrieval."""
        return _scriptable_transform_g_to_p(
            g_list,
            self.config.g_downsampled_dims,
            self.W_repeat_all
        )

    def _split_g(self, t: torch.Tensor, dim: int = -1) -> List[torch.Tensor]:
        """Splits concatenated g-vector into per-stream tensors."""
        return list(torch.split(t, self.config.g_dims, dim=dim))

    def _split_p(self, t: torch.Tensor, dim: int = -1) -> List[torch.Tensor]:
        """Splits concatenated p-vector into per-stream tensors."""
        return list(torch.split(t, self.config.p_dims, dim=dim))
