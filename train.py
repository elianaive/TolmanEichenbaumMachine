import torch
import torch.optim as optim
import numpy as np
import argparse
import os
import sys
import math
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from torch.cuda.amp import autocast, GradScaler
import wandb
import lovely_tensors as lt

# ---
# Project Imports & Setup
# ---
lt.monkey_patch()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TolmanEichenbaumMachine, TEMConfig
from environment import (
    Grid2DEnvironment, GraphEnvironment, TrajectoryDataset,
    create_line_graph_dict, create_tree_graph_dict
)
from experiments.generalization_analysis import run_generalization_analysis

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_histogram(data, name="data"):
    """Creates a wandb.Histogram while handling potential edge cases like collapsed variance."""
    if data is None or len(data) == 0:
        logger.warning(f"Empty data for histogram '{name}'. Skipping.")
        return None
    
    clean_data = data[np.isfinite(data)]
    if len(clean_data) <= 1:
        return clean_data.mean() if len(clean_data) > 0 else 0.0

    if (clean_data.max() - clean_data.min()) < 1e-9:
        return clean_data.mean()

    try:
        return wandb.Histogram(clean_data)
    except Exception as e:
        logger.error(f"Could not create wandb.Histogram for '{name}': {e}")
        return clean_data.mean()

# ======================================================================================
# 1. Curriculum Manager
# ======================================================================================

class Curriculum:
    """Manages the annealing schedules for learning rates, loss weights, and other parameters."""
    def __init__(self, args):
        self.args = args

    def get(self, step: int):
        """Calculates all curriculum-dependent parameters for a given training step."""
        # Learning Rate Schedule (Exponential Decay)
        lr = max(
            self.args.lr_min,
            self.args.lr_min + (self.args.lr - self.args.lr_min) *
            (self.args.lr_decay_rate ** (step / self.args.lr_decay_steps))
        )

        # Progress Ratios for Annealing
        g_kl_progress = min((step + 1) / self.args.g_kl_anneal_steps, 1.0)
        p_align_progress = min((step + 1) / self.args.p_align_anneal_steps, 1.0)
        p_reg_progress = 1.0 - min((step + 1) / self.args.p_reg_anneal_steps, 1.0)
        g_reg_progress = 1.0 - min((step + 1) / self.args.g_reg_anneal_steps, 1.0)
        p2g_sigmoid = 1.0 / (1.0 + np.exp(-(step - self.args.p2g_half_it) / self.args.p2g_scale_it))

        # Loss Weights Dictionary
        loss_weights = {
            'x_nll_p': 1.0, 'x_nll_g': 1.0, 'x_nll_gt': 1.0,
            'g_kl': self.args.g_kl_weight * g_kl_progress,
            'p_align': self.args.p_align_weight * p_align_progress,
            'p_inf_align': self.args.p_inf_align_weight * p2g_sigmoid,
            'g_reg': self.args.g_reg_weight * g_reg_progress,
            'p_reg': self.args.p_reg_weight * p_reg_progress,
        }

        # Hebbian and Other Parameters
        hebb_eta = min((step + 1) / self.args.hebb_learn_it, 1.0) * self.args.hebbian_learning_rate
        lambda_eff = min((step + 1) / self.args.lambda_it, 1.0) * self.args.hebbian_forget_rate
        p2g_scale_offset = (1.0 - p2g_sigmoid) * self.args.p2g_scale_val
        temp = min((step + 1) / 2000.0, 1.0)

        return lr, loss_weights, p2g_scale_offset, temp, hebb_eta, lambda_eff, p2g_sigmoid

# ======================================================================================
# 2. Main Trainer Class
# ======================================================================================

class TEMTrainer:
    """Handles the stateful training, validation, and logging loop for the TEM model."""
    def __init__(self, model, config, args, wandb_run):
        self.model = model
        self.config = config
        self.args = args
        self.wandb_run = wandb_run
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_envs = None # Will be set in the train method

        self.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scaler = GradScaler(enabled=(self.device == 'cuda' and args.autocast))
        self.curriculum = Curriculum(args)
        
        os.makedirs(args.save_dir, exist_ok=True)
        logger.info(f"Trainer initialized on device: {self.device}. Results saved to '{args.save_dir}'")

    def _set_learning_rate(self, lr):
        """Updates the learning rate for the optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _detach_state(self, state: dict):
        """Recursively detaches all tensors in the model's state dictionary to break BPTT connections."""
        detached = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                detached[k] = v.detach()
            elif isinstance(v, list):
                detached[k] = [item.detach() if isinstance(item, torch.Tensor) else item for item in v]
            elif isinstance(v, dict):
                detached[k] = self._detach_state(v)
            else:
                detached[k] = v
        return detached

    def train(self, train_envs, val_envs):
        """The main training loop."""
        logger.info(f"Starting Training for {self.args.train_it} iterations.")
        
        # --- Initialization ---
        self.train_envs = train_envs # FIX: Assign train_envs to the instance
        assert len(self.train_envs) == self.args.batch_size, "Number of training environments must equal batch size."
        
        active_datasets = [
            TrajectoryDataset(env.generate_trajectory(2000), self.config.n_sensory_objects, self.device)
            for env in self.train_envs
        ]
        
        persistent_state = self.model.create_empty_memory(self.args.batch_size, self.device)
        trajectory_positions = [0] * self.args.batch_size
        start_iteration, best_val_acc = 0, -1.0
        
        if hasattr(self.args, '_checkpoint_start_iteration'):
            start_iteration = self.args._checkpoint_start_iteration
            best_val_acc = self.args._checkpoint_best_val_acc
            logger.info(f"Resuming training from iteration {start_iteration}")

        pbar = tqdm(range(start_iteration, self.args.train_it), desc="Training", initial=start_iteration)
        for i in pbar:
            self.model.train()
            
            # 1. Update curriculum-dependent parameters
            lr, loss_weights, p2g_offset, temp, hebb_eta, lambda_eff, p2g_gate = self.curriculum.get(i)
            self._set_learning_rate(lr)
            self.model.loss_weights = loss_weights
            self.model.set_p2g_scale_offset(p2g_offset)
            self.model.set_prior_temp(temp)
            self.model.inference_net.p2g_use.fill_(p2g_gate)
            self.model.memory_system.hebbian_learning_rate.fill_(hebb_eta)
            self.model.memory_system.hebbian_forget_rate.fill_(lambda_eff)
            
            # 2. Prepare batch of trajectory segments
            x_bptt, a_bptt, persistent_state, active_datasets, trajectory_positions = self._prepare_batch_data(
                active_datasets, trajectory_positions, persistent_state, i
            )
            
            # 3. Forward and Backward Pass through BPTT window
            self.optimizer.zero_grad(set_to_none=True)
            torch.compiler.cudagraph_mark_step_begin()
            bptt_state = self._detach_state(persistent_state)
            
            total_loss = 0.0
            all_losses_for_log = defaultdict(float) 
            diagnostics_for_log = defaultdict(list)

            with autocast(enabled=self.scaler.is_enabled()):
                for t in range(self.args.bptt_len - 1):
                    outputs = self.model(x_bptt[t], a_bptt[t], bptt_state)
                    total_loss += outputs['losses']['total_loss'].mean()
                    bptt_state = outputs['new_state']
                    
                    for k, v in outputs['losses'].items():
                        all_losses_for_log[f"train_loss/{k}"] += v.mean().item()

                    if self.args.log_debug_metrics and 'diagnostics' in outputs:
                        for k, v in outputs['diagnostics'].items():
                            diagnostics_for_log[k].append(v)
            
            avg_loss = total_loss / (self.args.bptt_len - 1)
            
            if torch.isfinite(avg_loss):
                self.scaler.scale(avg_loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            persistent_state = self._detach_state(bptt_state)
            
            # 4. Logging and Checkpointing
            if (i + 1) % self.args.log_interval == 0:
                val_loss, val_acc = self.run_validation(val_envs[0:2])
                avg_diagnostics = {
                    k: np.mean([t for t in v]) 
                    for k, v in diagnostics_for_log.items()
                }
                
                hist_data = {}
                if self.args.log_debug_metrics:
                    with torch.no_grad():
                        g_inf_list = bptt_state.get('g_states', [])
                        p_inf_list = bptt_state.get('p_inf', [])
                        
                        last_g_inf_list = outputs.get('new_state', {}).get('g_states', [])
                        if isinstance(outputs['new_state']['g_states'][0], torch.Tensor):
                             last_g_inf_list_full = outputs.get('reps_dict', {}).get('g_inf', [])
                             if last_g_inf_list_full:
                                g_vars_list = [g.log_var.exp() for g in last_g_inf_list_full]
                                g_var_data = torch.cat(g_vars_list, dim=-1).flatten().cpu().numpy()
                                hist_data['hist/g_var'] = g_var_data

                        if p_inf_list:
                            p_inf_data = torch.cat(p_inf_list, dim=-1).abs().flatten().cpu().numpy()
                            hist_data['hist/p_abs'] = p_inf_data

                self._log_training_metrics(i, avg_loss, all_losses_for_log, avg_diagnostics, val_loss, val_acc, grad_norm)
                pbar.set_postfix({'Loss': avg_loss.item(), 'Val Acc (gt)': f"{val_acc['gt']:.3f}"})
                
                if val_acc['gt'] > best_val_acc:
                    best_val_acc = val_acc['gt']
                    self._save_checkpoint(i + 1, best_val_acc, is_best=True)
            
            if (i + 1) % self.args.save_interval == 0:
                self._save_checkpoint(i + 1, best_val_acc)

        logger.info("Training complete.")

    def _prepare_batch_data(self, active_datasets, trajectory_positions, persistent_state, step):
        """Prepares a BPTT segment, resetting environments and states as needed."""
        x_bptt = torch.empty(self.args.bptt_len, self.args.batch_size, self.config.n_sensory_objects, device=self.device)
        a_bptt = torch.empty(self.args.bptt_len, self.args.batch_size, device=self.device, dtype=torch.long)
        
        for b in range(self.args.batch_size):
            start_idx, end_idx = trajectory_positions[b], trajectory_positions[b] + self.args.bptt_len
            if end_idx >= len(active_datasets[b]):
                env = self.train_envs[b] # Uses self.train_envs
                env.reset_sensory_map()
                traj_len = self._get_curriculum_trajectory_length(env, step)
                active_datasets[b] = TrajectoryDataset(env.generate_trajectory(traj_len), self.config.n_sensory_objects, self.device)
                
                fresh_state = self.model.create_empty_memory(1, self.device)
                for key, val in fresh_state.items():
                    if isinstance(val, torch.Tensor):
                        persistent_state[key][b].copy_(val[0])
                    elif isinstance(val, list):
                        for s_idx in range(len(val)):
                            persistent_state[key][s_idx][b].copy_(val[s_idx][0])
                    elif isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            if sub_val is not None:
                                persistent_state[key][sub_key][b].copy_(sub_val[0])
                start_idx, end_idx = 0, self.args.bptt_len

            ds = active_datasets[b]
            x_bptt[:, b].copy_(ds.one_hot_observations[start_idx:end_idx])
            a_bptt[:, b].copy_(ds.actions[start_idx:end_idx])
            trajectory_positions[b] = end_idx
        return x_bptt, a_bptt, persistent_state, active_datasets, trajectory_positions

    def _get_curriculum_trajectory_length(self, env, step):
        """Calculates trajectory length based on a curriculum that shortens trajectories over time."""
        progress = min(step / self.args.curriculum_decay_steps, 1.0)
        n_restart = self.args.restart_max - progress * (self.args.restart_max - self.args.restart_min)
        jitter = np.random.randint(0, self.args.seq_jitter)
        walk_len = int((n_restart + jitter) * env.n_nodes)
        return max(self.args.bptt_len, (walk_len // self.args.bptt_len) * self.args.bptt_len)

    def _log_training_metrics(self, step, train_loss, individual_losses, diagnostics, val_loss, val_acc, grad_norm, hist_data=None):
        """Logs all training and validation metrics to Wandb in an organized way."""
        log_data = {
            'validation/loss': val_loss,
            'validation/accuracy_p': val_acc['p'],
            'validation/accuracy_g': val_acc['g'],
            'validation/accuracy_gt': val_acc['gt'],
        }
        
        for k, v in individual_losses.items():
            log_data[k] = v / (self.args.bptt_len - 1)

        lr, loss_weights, _, temp, hebb_eta, _, p2g_gate = self.curriculum.get(step)
        log_data.update({
            'curriculum/learning_rate': lr,
            'curriculum/g_kl_weight': loss_weights['g_kl'],
            'curriculum/p2g_use_gate': p2g_gate,
            'curriculum/hebb_eta': hebb_eta,
            'curriculum/temp': temp,
        })

        if self.args.log_debug_metrics:
            log_data.update(diagnostics)
            if grad_norm is not None:
                log_data['grads/total_norm'] = grad_norm.item()
                
            # Log per-module gradient norms
            self._log_module_grad_norms(log_data)

            # Log histograms
            if hist_data:
                if 'hist/g_var' in hist_data and hist_data['hist/g_var'].size > 1:
                    log_data['hist/g_var'] = safe_histogram(hist_data['hist/g_var'], "g_var")
                if 'hist/p_abs' in hist_data and hist_data['hist/p_abs'].size > 1:
                    log_data['hist/p_abs'] = safe_histogram(hist_data['hist/p_abs'], "p_abs")
        
        self.wandb_run.log(log_data, step=step)
        
    def _log_module_grad_norms(self, log_data: dict):
        """Calculates and logs the L2 norm of gradients for each major model component."""
        if not self.args.log_debug_metrics:
            return
            
        module_norms = defaultdict(float)
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            
            norm_sq = p.grad.norm().pow(2).item()
            if "inference_net" in name:
                module_norms["grads/inference_net_norm"] += norm_sq
            elif "transition_model" in name:
                module_norms["grads/transition_model_norm"] += norm_sq
            elif "generative_net" in name:
                module_norms["grads/generative_net_norm"] += norm_sq
            elif "g_init" in name:
                module_norms["grads/g_init_norm"] += norm_sq
        
        for k, v in module_norms.items():
            log_data[k] = math.sqrt(v)

    def run_validation(self, val_envs):
        """Runs the model on validation environments to assess performance."""
        self.model.eval()
        total_loss = 0.0
        total_acc = defaultdict(float)
        n_samples = min(len(val_envs), self.args.n_val_samples)

        with torch.no_grad():
            for i in range(n_samples):
                dataset = TrajectoryDataset(
                    val_envs[i].generate_trajectory(self.args.val_trajectory_len),
                    self.config.n_sensory_objects, self.device
                )
                val_state = self.model.create_empty_memory(1, self.device)
                
                segment_loss = 0.0
                segment_correct = defaultdict(int)
                val_steps = len(dataset) - 1
                
                if val_steps <= 0:
                    continue

                for t in range(val_steps):
                    outputs = self.model(
                        dataset[t]['x_t'].unsqueeze(0),
                        torch.tensor([dataset[t]['a_t']], device=self.device),
                        val_state
                    )
                    
                    if torch.isfinite(outputs['losses']['total_loss']):
                        segment_loss += outputs['losses']['total_loss'].item()

                    true_pred_obs_idx = dataset[t]['x_t'].argmax().item()
                    for pred_type in ['p', 'g', 'gt']:
                        if outputs['predictions'][f'x_{pred_type}'].argmax().item() == true_pred_obs_idx:
                            segment_correct[pred_type] += 1
                    val_state = outputs['new_state']
                
                total_loss += segment_loss / val_steps
                for pred_type in ['p', 'g', 'gt']:
                    total_acc[pred_type] += segment_correct[pred_type] / val_steps if val_steps > 0 else 0

        avg_loss = total_loss / n_samples if n_samples > 0 else 0
        avg_acc = {k: v / n_samples for k, v in total_acc.items()}
        return avg_loss, avg_acc

    def _save_checkpoint(self, step, best_val_acc, is_best=False):
        """Saves a model checkpoint."""
        state = {
            "model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(), "step": step,
            "best_val_acc": best_val_acc, "args": self.args, "config": self.config,
        }
        filename = 'best_model.pt' if is_best else f"ckpt_{step:07d}.pt"
        ckpt_path = os.path.join(self.args.save_dir, filename)
        torch.save(state, ckpt_path)
        logger.info(f"Saved checkpoint -> {ckpt_path}")

# ======================================================================================
# 3. Main Execution Block
# ======================================================================================

def create_environments(n_envs, env_type, min_size, max_size, n_sensory):
    """Factory function to create a list of environments."""
    envs = []
    for _ in range(n_envs):
        size = np.random.randint(min_size, max_size + 1)
        if env_type == 'grid': envs.append(Grid2DEnvironment(width=size, height=size, n_sensory=n_sensory))
        elif env_type == 'line': envs.append(GraphEnvironment(create_line_graph_dict(length=size, n_sensory=n_sensory)))
        elif env_type == 'tree':
            depth = np.random.randint(max(2, min_size // 2), max(3, max_size // 2) + 1)
            envs.append(GraphEnvironment(create_tree_graph_dict(depth=depth, n_sensory=n_sensory)))
    return envs

def load_checkpoint(checkpoint_path, model, optimizer, scaler, device):
    """Loads a checkpoint and returns training state."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if 'scaler_state' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state'])
    return checkpoint.get('step', 0), checkpoint.get('best_val_acc', -1.0)

def main():
    parser = argparse.ArgumentParser(description="Train a Tolman-Eichenbaum Machine model.")
    
    # --- Group: Core Settings ---
    core_group = parser.add_argument_group('Core Settings')
    core_group.add_argument('--env_type', type=str, default='grid', choices=['grid', 'line', 'tree'])
    core_group.add_argument('--save_dir', type=str, default='results')
    core_group.add_argument('--seed', type=int, default=42)
    core_group.add_argument('--continue_from', type=str, default=None, help='Path to checkpoint to continue from.')

    # --- Group: Wandb Logging ---
    wandb_group = parser.add_argument_group('Wandb Logging')
    wandb_group.add_argument('--project_name', type=str, default="TEM")
    wandb_group.add_argument('--run_name', type=str, default=None)
    wandb_group.add_argument('--log_interval', type=int, default=100)
    wandb_group.add_argument('--save_interval', type=int, default=10000)
    wandb_group.add_argument('--log_debug_metrics', action='store_true', help='Log detailed debug metrics and histograms.')

    # --- Group: Training Hyperparameters ---
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--train_it', type=int, default=50000)
    train_group.add_argument('--batch_size', type=int, default=16)
    train_group.add_argument('--bptt_len', type=int, default=75)
    train_group.add_argument('--lr', type=float, default=9e-4)
    train_group.add_argument('--lr_min', type=float, default=8e-5)
    train_group.add_argument('--lr_decay_steps', type=int, default=4000)
    train_group.add_argument('--lr_decay_rate', type=float, default=0.5)
    train_group.add_argument('--weight_decay', type=float, default=1e-4)
    train_group.add_argument('--clip_grad_norm', type=float, default=2.0)
    train_group.add_argument('--autocast', action='store_true', help='Enable Automatic Mixed Precision (AMP).')
    train_group.add_argument('--full_graph_compile', action='store_true', help='Use full graph torch.compile mode (slower startup).')

    # --- Group: Dataset Settings ---
    data_group = parser.add_argument_group('Dataset Settings')
    data_group.add_argument('--n_val_samples', type=int, default=20)
    data_group.add_argument('--val_trajectory_len', type=int, default=200)
    data_group.add_argument('--min_size', type=int, default=8)
    data_group.add_argument('--max_size', type=int, default=12)

    # --- Group: Curriculum Schedules ---
    curriculum_group = parser.add_argument_group('Curriculum Schedules')
    curriculum_group.add_argument('--g_kl_anneal_steps', type=int, default=15000)
    curriculum_group.add_argument('--p_align_anneal_steps', type=int, default=2000)
    curriculum_group.add_argument('--p_reg_anneal_steps', type=int, default=4000)
    curriculum_group.add_argument('--g_reg_anneal_steps', type=int, default=40000)
    curriculum_group.add_argument('--p2g_half_it', type=int, default=0)
    curriculum_group.add_argument('--p2g_scale_it', type=int, default=200)
    curriculum_group.add_argument('--p2g_scale_val', type=float, default=10000.0)
    curriculum_group.add_argument('--hebb_learn_it', type=int, default=16000)
    curriculum_group.add_argument('--lambda_it', type=int, default=200)
    curriculum_group.add_argument('--curriculum_decay_steps', type=int, default=40000)
    curriculum_group.add_argument('--restart_max', type=int, default=40)
    curriculum_group.add_argument('--restart_min', type=int, default=5)
    curriculum_group.add_argument('--seq_jitter', type=int, default=30)
    
    # --- Group: Loss Weights ---
    loss_group = parser.add_argument_group('Loss Weights')
    loss_group.add_argument('--g_kl_weight', type=float, default=1.0)
    loss_group.add_argument('--p_align_weight', type=float, default=1.0)
    loss_group.add_argument('--p_inf_align_weight', type=float, default=1.0)
    loss_group.add_argument('--g_reg_weight', type=float, default=0.01)
    loss_group.add_argument('--p_reg_weight', type=float, default=0.02)
    loss_group.add_argument('--hebbian_learning_rate', type=float, default=0.5)
    loss_group.add_argument('--hebbian_forget_rate', type=float, default=0.9999)

    args = parser.parse_args()

    # --- Initialization ---
    if args.run_name is None:
        args.run_name = f"tem_{args.env_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    run = wandb.init(project=args.project_name, name=args.run_name, config=args)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    
    config = TEMConfig()
    if args.env_type == 'line': config.n_actions = 2
    elif args.env_type == 'tree': config.n_actions = 3
    else: config.n_actions = 4
    config.bptt_len = args.bptt_len

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TolmanEichenbaumMachine(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(device == 'cuda' and args.autocast))

    if args.continue_from:
        start_iter, best_acc = load_checkpoint(args.continue_from, model, optimizer, scaler, device)
        args._checkpoint_start_iteration, args._checkpoint_best_val_acc = start_iter, best_acc
        wandb.config.update({"continued_from": args.continue_from})
        
    # --- Conditional torch.compile ---
    if device == 'cuda' and hasattr(torch, 'compile'):
        if args.full_graph_compile:
            logger.info("Compiling model with 'max-autotune' mode.")
            model = torch.compile(model, mode="max-autotune")
        else:
            logger.info("Compiling model with 'reduce-overhead' mode.")
            model = torch.compile(
                model,
                mode="reduce-overhead",
                disable="cudagraphs",
                dynamic=True,
                fullgraph=True
            )

    train_envs = create_environments(args.batch_size, args.env_type, args.min_size, args.max_size, config.n_sensory_objects)
    val_envs = create_environments(50, args.env_type, args.min_size, args.max_size, config.n_sensory_objects)

    trainer = TEMTrainer(model, config, args, run)
    trainer.train(train_envs, val_envs)
    
    run.finish()

if __name__ == "__main__":
    main()