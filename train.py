"""
TEM Training Script - Curriculum Learning with BPTT

This script implements the training loop for the Tolman-Eichenbaum Machine with:
    - Curriculum scheduling for learning rates, loss weights, and trajectory lengths
    - Truncated backpropagation through time (BPTT) with state detachment
    - Loss masking: only compute loss on revisits to previously visited nodes
    - Multi-pathway validation (inference, hybrid, generative)
    - Comprehensive diagnostics and tensor inspection logging

Key Training Features:
    - Annealing schedules for all hyperparameters (lr, loss weights, Hebbian rates)
    - Trajectory length curriculum: long episodes early, shorter later
    - Memory pathway gating: gradual activation of sensory-cued memory
    - Gradient clipping and optional mixed precision (AMP)
    - WandB logging with CSV fallback

Usage:
    python train.py --env_type grid --train_it 50000 --batch_size 64
    python train.py --continue_from results/best_model.pt
"""

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
import csv

# Project imports and setup
lt.monkey_patch()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TolmanEichenbaumMachine, TEMConfig
from environment import (
    Grid2DEnvironment, GraphEnvironment, TrajectoryDataset,
    create_line_graph_dict, create_tree_graph_dict
)
from experiments.generalization_analysis import run_generalization_analysis

# PyTorch performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# Helper Functions
# ==============================================================================

def safe_histogram(data, name="data"):
    """
    Creates WandB histogram with robust error handling for edge cases.

    Args:
        data: Numpy array to histogram
        name: Name for logging

    Returns:
        wandb.Histogram or scalar mean if histogram creation fails
    """
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


# ==============================================================================
# Curriculum Manager
# ==============================================================================

class Curriculum:
    """
    Manages all annealing schedules for curriculum learning.

    Schedules include:
        - Learning rate: Exponential decay
        - Loss weights: Linear ramp-up for alignment losses, ramp-down for regularization
        - Memory pathway: Sigmoid activation for p2g correction
        - Hebbian rates: Linear ramp-up for learning, forgetting
        - Trajectory length: Linear decay for environment complexity
    """
    def __init__(self, args):
        self.args = args

    def get(self, step: int):
        """
        Calculates all curriculum-dependent parameters for a given training step.

        Args:
            step: Current training iteration

        Returns:
            Tuple of (lr, loss_weights, p2g_offset, temp, hebb_eta, lambda_eff, p2g_gate)
        """
        # Learning rate: exponential decay
        # lr(t) = lr_min + (lr_0 - lr_min) * decay_rate^(t / decay_steps)
        lr = max(
            self.args.lr_min,
            self.args.lr_min + (self.args.lr - self.args.lr_min) *
            (self.args.lr_decay_rate ** (step / self.args.lr_decay_steps))
        )

        # Progress ratios for annealing (linear ramps)
        g_kl_progress = min((step + 1) / self.args.g_kl_anneal_steps, 1.0)
        p_align_progress = min((step + 1) / self.args.p_align_anneal_steps, 1.0)
        p_reg_progress = 1.0 - min((step + 1) / self.args.p_reg_anneal_steps, 1.0)
        g_reg_progress = 1.0 - min((step + 1) / self.args.g_reg_anneal_steps, 1.0)

        # Memory pathway gating: sigmoid activation
        # p2g_gate(t) = sigmoid((t - half_it) / scale_it)
        p2g_sigmoid = 1.0 / (1.0 + np.exp(-(step - self.args.p2g_half_it) / self.args.p2g_scale_it))

        # Loss weights dictionary
        loss_weights = {
            'x_nll_p': 1.0,
            'x_nll_g': 1.0,
            'x_nll_gt': 1.0,
            'g_kl': self.args.g_kl_weight * g_kl_progress,
            'p_align': self.args.p_align_weight * p_align_progress,
            'p_inf_align': self.args.p_inf_align_weight * p_align_progress * p2g_sigmoid,
            'g_reg': self.args.g_reg_weight * g_reg_progress,
            'p_reg': self.args.p_reg_weight * p_reg_progress,
        }

        # Hebbian parameters
        hebb_eta = min((step + 1) / self.args.hebb_learn_it, 1.0) * self.args.hebbian_learning_rate
        lambda_eff = min((step + 1) / self.args.lambda_it, 1.0) * self.args.hebbian_forget_rate

        # Memory variance offset: high early (ignore memory), low late (trust memory)
        p2g_scale_offset = (1.0 - p2g_sigmoid) * self.args.p2g_scale_val

        # Prior temperature: ramps from 0 to 1 over 2000 steps
        temp = min((step + 1) / 2000.0, 1.0)

        return lr, loss_weights, p2g_scale_offset, temp, hebb_eta, lambda_eff, p2g_sigmoid


# ==============================================================================
# Main Trainer Class
# ==============================================================================

class TEMTrainer:
    """
    Handles stateful training, validation, and logging loop for TEM.

    Key responsibilities:
        - BPTT with state detachment between chunks
        - Loss masking (only on revisited nodes)
        - Curriculum parameter updates
        - Gradient clipping and optimization
        - Validation and checkpointing
        - Comprehensive logging (WandB + CSV + tensor inspection)
    """
    def __init__(self, model, config, args, wandb_run):
        self.model = model
        self.config = config
        self.args = args
        self.wandb_run = wandb_run
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_envs = None  # Set in train method

        self.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scaler = GradScaler(enabled=(self.device == 'cuda' and args.autocast))
        self.curriculum = Curriculum(args)

        self.visited_sensory = None  # [batch_size, n_sensory]

        os.makedirs(args.save_dir, exist_ok=True)

        # Initialize CSV logging
        self.csv_path = os.path.join(args.save_dir, 'training_metrics.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = None
        self.csv_header_written = False

        # Initialize tensor inspection logging
        self.tensor_log_path = os.path.join(args.save_dir, 'tensor_inspection.log')
        self.tensor_log_file = open(self.tensor_log_path, 'w')

        logger.info(f"Trainer initialized on device: {self.device}. Results saved to '{args.save_dir}'")
        logger.info(f"CSV metrics will be logged to: {self.csv_path}")
        logger.info(f"Tensor inspection will be logged to: {self.tensor_log_path}")

    def _set_learning_rate(self, lr):
        """Updates learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _detach_state(self, state: dict):
        """
        Recursively detaches all tensors in state dictionary to break BPTT connections.

        Critical for truncated BPTT: prevents gradients from flowing across chunk boundaries.

        Args:
            state: Model state dictionary

        Returns:
            Detached state dictionary
        """
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
        """
        Main training loop with curriculum scheduling and BPTT.

        Args:
            train_envs: List of training environments (one per batch item)
            val_envs: List of validation environments
        """
        logger.info(f"Starting Training for {self.args.train_it} iterations.")

        # Initialization
        self.train_envs = train_envs

        # Determine max number of nodes for visited tensor
        max_n_nodes = 0
        if self.train_envs:
            max_n_nodes = max(env.n_nodes for env in self.train_envs)

        # Initialize visited state tracking for each environment in batch
        # visited[b, node_id] = True if node has been visited in current episode
        self.visited = torch.zeros(
            self.args.batch_size, max_n_nodes,
            dtype=torch.bool, device=self.device
        )

        self.visited_sensory = torch.zeros(
            self.args.batch_size, self.config.n_sensory_objects,
            dtype=torch.bool, device=self.device
        )

        active_datasets = [None] * self.args.batch_size

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

            # Update curriculum-dependent parameters
            lr, loss_weights, p2g_offset, temp, hebb_eta, lambda_eff, p2g_gate = self.curriculum.get(i)
            self._set_learning_rate(lr)
            self.model.loss_weights = loss_weights
            self.model.set_p2g_scale_offset(p2g_offset)
            self.model.set_prior_temp(temp)
            self.model.inference_net.p2g_use.fill_(p2g_gate)
            self.model.memory_system.hebbian_learning_rate.fill_(hebb_eta)
            self.model.memory_system.hebbian_forget_rate.fill_(lambda_eff)

            # Prepare batch of trajectory segments
            x_bptt, a_bptt, node_bptt, persistent_state, active_datasets, trajectory_positions = self._prepare_batch_data(
                active_datasets, trajectory_positions, persistent_state, i
            )

            # Forward and backward pass through BPTT window
            self.optimizer.zero_grad(set_to_none=True)
            torch.compiler.cudagraph_mark_step_begin()
            bptt_state = self._detach_state(persistent_state)

            # Compute loss mask: only calculate loss on revisits
            is_revisit_mask = self._compute_visit_mask(node_bptt).float()  # [T, B]

            total_loss = 0.0
            all_losses_for_log = defaultdict(float)
            diagnostics_for_log = defaultdict(list)
            total_active_samples = 0.0

            with autocast(enabled=self.scaler.is_enabled()):
                # Call model once for entire sequence
                all_outputs = self.model(x_bptt, a_bptt, bptt_state)

                # Accumulate losses across timesteps
                for t, outputs in enumerate(all_outputs):
                    per_sample_loss = outputs['losses']['total_loss']  # [B]

                    mask_t = is_revisit_mask[t]  # [B]
                    num_active_samples_t = mask_t.sum()

                    if num_active_samples_t > 0:
                        # Masked loss: only backprop through revisited states
                        timestep_loss = (per_sample_loss * mask_t).sum() / num_active_samples_t
                        total_loss += timestep_loss
                        total_active_samples += 1

                    # Log all loss components
                    for k, v in outputs['losses'].items():
                        all_losses_for_log[f"train_loss/{k}"] += v.mean().item()

                    # Collect diagnostics
                    if self.args.log_debug_metrics and 'diagnostics' in outputs:
                        for k, v in outputs['diagnostics'].items():
                            if isinstance(v, torch.Tensor):
                                diagnostics_for_log[k].append(v.detach().cpu())
                            else:
                                diagnostics_for_log[k].append(v)

                # Save final state for next chunk
                bptt_state = all_outputs[-1]['new_state']

            # Average loss over active timesteps (prevents gradient explosion)
            avg_loss = total_loss / total_active_samples if total_active_samples > 0 else total_loss
            grad_norm = torch.tensor(0.0)
            module_grad_norms = None

            if total_active_samples > 0 and torch.isfinite(avg_loss):
                self.scaler.scale(avg_loss).backward()
                self.scaler.unscale_(self.optimizer)
                # Compute per-module norms BEFORE clipping
                module_grad_norms = self._compute_module_grad_norms()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            persistent_state = self._detach_state(bptt_state)

            # Detailed tensor inspection on step 0 and every 500 steps
            if i == 0 or (i + 1) % 500 == 0:
                try:
                    self._print_detailed_tensors(i + 1, x_bptt, a_bptt, outputs, bptt_state)
                except Exception as e:
                    logger.error(f"Error in tensor inspection at step {i+1}: {e}")
                    import traceback
                    traceback.print_exc()

            # Logging and checkpointing
            if (i + 1) % self.args.log_interval == 0:
                val_loss, val_acc = self.run_validation(val_envs)
                avg_diagnostics = {
                    k: np.mean([t for t in v])
                    for k, v in diagnostics_for_log.items()
                }

                hist_data = {}
                if self.args.log_debug_metrics:
                    with torch.no_grad():
                        p_inf_list = bptt_state.get('p_inf', [])

                        if isinstance(outputs['new_state']['g_states'][0], torch.Tensor):
                            last_g_inf_list_full = outputs.get('reps_dict', {}).get('g_inf', [])
                            if last_g_inf_list_full:
                                g_vars_list = [g.log_var.exp() for g in last_g_inf_list_full]
                                g_var_data = torch.cat(g_vars_list, dim=-1).flatten().cpu().numpy()
                                hist_data['hist/g_var'] = g_var_data

                        if p_inf_list:
                            p_inf_data = torch.cat(p_inf_list, dim=-1).abs().flatten().cpu().numpy()
                            hist_data['hist/p_abs'] = p_inf_data

                self._log_training_metrics(i, avg_loss, all_losses_for_log, avg_diagnostics, val_loss, val_acc,
                                          grad_norm, module_grad_norms)
                pbar.set_postfix({'Loss': avg_loss.item(), 'Val Acc (gt)': f"{val_acc['revisit']['gt']:.3f}"})

                if val_acc['revisit']['gt'] > best_val_acc:
                    best_val_acc = val_acc['revisit']['gt']
                    self._save_checkpoint(i + 1, best_val_acc, is_best=True)

            if (i + 1) % self.args.save_interval == 0:
                self._save_checkpoint(i + 1, best_val_acc)

        # Close log files
        if self.csv_file:
            self.csv_file.close()
            logger.info(f"CSV metrics saved to: {self.csv_path}")

        if self.tensor_log_file:
            self.tensor_log_file.close()
            logger.info(f"Tensor inspection saved to: {self.tensor_log_path}")

        logger.info("Training complete.")

    def _prepare_batch_data(self, active_datasets, trajectory_positions, persistent_state, step):
        """
        Prepares BPTT segment, resetting environments and states as needed.

        For each batch item:
            - If trajectory exhausted, generate new trajectory and reset state
            - Otherwise, continue from current position
            - Apply curriculum trajectory length schedule

        Args:
            active_datasets: List of current datasets [batch_size]
            trajectory_positions: List of current positions [batch_size]
            persistent_state: Model state dict
            step: Current training iteration

        Returns:
            Tuple of (x_bptt, a_bptt, node_bptt, persistent_state, active_datasets, trajectory_positions)
            where x_bptt: [T, B, n_sensory], a_bptt: [T, B], node_bptt: [T, B]
        """
        x_bptt = torch.empty(self.args.bptt_len, self.args.batch_size, self.config.n_sensory_objects,
                            device=self.device)
        a_bptt = torch.empty(self.args.bptt_len, self.args.batch_size, device=self.device, dtype=torch.long)
        node_bptt = torch.empty(self.args.bptt_len, self.args.batch_size, device=self.device, dtype=torch.long)

        for b in range(self.args.batch_size):
            start_idx = trajectory_positions[b]
            end_idx = start_idx + self.args.bptt_len

            # Check if we need new trajectory
            need_new = (active_datasets[b] is None) or (end_idx >= len(active_datasets[b]))
            if need_new:
                # Reset visited tracking for this batch item
                self.visited[b].zero_()
                self.visited_sensory[b].zero_()

                # Generate new trajectory with curriculum length
                env = self.train_envs[b]
                env.reset_sensory_map()
                traj_len = self._get_curriculum_trajectory_length(env, step)
                active_datasets[b] = TrajectoryDataset(
                    env.generate_trajectory(traj_len),
                    self.config.n_sensory_objects, self.device
                )

                # Reset model state for this batch item
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

            # Copy BPTT segment
            ds = active_datasets[b]
            x_bptt[:, b].copy_(ds.one_hot_observations[start_idx:end_idx])
            a_bptt[:, b].copy_(ds.actions[start_idx:end_idx])
            node_bptt[:, b].copy_(ds.nodes[start_idx:end_idx])
            trajectory_positions[b] = end_idx

        return x_bptt, a_bptt, node_bptt, persistent_state, active_datasets, trajectory_positions

    def _get_curriculum_trajectory_length(self, env, step):
        """
        Calculates trajectory length based on curriculum that shortens episodes over time.

        Early training: long episodes (many revisits, easier credit assignment)
        Late training: shorter episodes (harder credit assignment, better generalization)

        Args:
            env: Environment instance
            step: Current training iteration

        Returns:
            Trajectory length in steps
        """
        progress = min(step / self.args.curriculum_decay_steps, 1.0)
        n_restart = self.args.restart_max - progress * (self.args.restart_max - self.args.restart_min)
        jitter = np.random.randint(0, self.args.seq_jitter)
        walk_len = int((n_restart + jitter) * env.n_nodes)
        return max(self.args.bptt_len, (walk_len // self.args.bptt_len) * self.args.bptt_len)

    def _compute_visit_mask(self, node_bptt: torch.Tensor) -> torch.Tensor:
        """
        Computes boolean mask indicating which timesteps are revisits to a node.

        Critical design choice: Loss only calculated on revisits, not first visits.
        This ensures the model learns to predict based on memory, not just encode.

        Updates self.visited as side effect to track which nodes have been seen.

        Args:
            node_bptt: Node indices [T, B]

        Returns:
            Boolean mask [T, B] where True indicates revisit
        """
        mask_bool = torch.empty_like(node_bptt, dtype=torch.bool)

        for t in range(node_bptt.shape[0]):
            nodes_t = node_bptt[t]  # [B]

            # Check if current nodes have been visited
            visited_t = self.visited.gather(1, nodes_t.unsqueeze(1)).squeeze(1)  # [B]
            mask_bool[t] = visited_t

            # Update visited tracker
            self.visited.scatter_(1, nodes_t.unsqueeze(1), True)

        return mask_bool

    def _compute_visit_mask_from_x(self, x_bptt: torch.Tensor) -> torch.Tensor:
        """
        Alternative mask: revisit based on sensory observation rather than node.

        Returns mask[t, b] == True if sensory category at (t, b) has been seen
        earlier in current environment for that batch item.

        Args:
            x_bptt: One-hot observations [T, B, n_sensory]

        Returns:
            Boolean mask [T, B]
        """
        T, B, _ = x_bptt.shape
        mask = torch.empty(T, B, dtype=torch.bool, device=x_bptt.device)
        for t in range(T):
            ids = x_bptt[t].argmax(dim=1)  # [B]
            seen = self.visited_sensory.gather(1, ids[:, None]).squeeze(1)
            mask[t] = seen
            self.visited_sensory.scatter_(1, ids[:, None], True)
        return mask

    def _log_training_metrics(self, step, train_loss, individual_losses, diagnostics, val_loss, val_acc,
                             grad_norm, module_grad_norms=None, hist_data=None):
        """
        Logs all training and validation metrics to WandB and CSV.

        Args:
            step: Current iteration
            train_loss: Average training loss
            individual_losses: Dict of loss components
            diagnostics: Dict of model diagnostics
            val_loss: Validation loss
            val_acc: Dict with 'overall' and 'revisit' accuracy dicts
            grad_norm: Total gradient norm
            module_grad_norms: Per-module gradient norms
            hist_data: Optional histogram data
        """
        log_data = {
            'validation/loss': val_loss,
            # Overall accuracy
            'validation/accuracy_overall_p': val_acc['overall']['p'],
            'validation/accuracy_overall_g': val_acc['overall']['g'],
            'validation/accuracy_overall_gt': val_acc['overall']['gt'],
            # Revisit accuracy
            'validation/accuracy_revisit_p': val_acc['revisit']['p'],
            'validation/accuracy_revisit_g': val_acc['revisit']['g'],
            'validation/accuracy_revisit_gt': val_acc['revisit']['gt'],
        }

        # Average losses over BPTT window
        for k, v in individual_losses.items():
            log_data[k] = v / (self.args.bptt_len - 1)

        # Curriculum parameters
        lr, loss_weights, _, temp, hebb_eta, _, p2g_gate = self.curriculum.get(step)
        log_data.update({
            'curriculum/learning_rate': lr,
            'curriculum/g_kl_weight': loss_weights['g_kl'],
            'curriculum/p2g_use_gate': p2g_gate,
            'curriculum/hebb_eta': hebb_eta,
            'curriculum/temp': temp,
        })

        log_data['curriculum/trajectory_length'] = self._get_curriculum_trajectory_length(self.train_envs[0], step)

        # Debug metrics
        if self.args.log_debug_metrics:
            log_data.update(diagnostics)
            if grad_norm is not None:
                log_data['grads/total_norm'] = grad_norm.item()

            # Per-module gradient norms
            if module_grad_norms is not None:
                log_data.update(module_grad_norms)

            # Histograms
            if hist_data:
                if 'hist/g_var' in hist_data and hist_data['hist/g_var'].size > 1:
                    log_data['hist/g_var'] = safe_histogram(hist_data['hist/g_var'], "g_var")
                if 'hist/p_abs' in hist_data and hist_data['hist/p_abs'].size > 1:
                    log_data['hist/p_abs'] = safe_histogram(hist_data['hist/p_abs'], "p_abs")

        self.wandb_run.log(log_data, step=step)

        # Write to CSV (only scalar values, skip histograms)
        csv_row = {'step': step}
        for key, value in log_data.items():
            if isinstance(value, (int, float, np.number)):
                csv_row[key] = value

        # Write CSV header on first log
        if not self.csv_header_written:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=sorted(csv_row.keys()))
            self.csv_writer.writeheader()
            self.csv_header_written = True

        # Write row
        self.csv_writer.writerow(csv_row)
        self.csv_file.flush()

    def _compute_module_grad_norms(self):
        """
        Computes L2 norm of gradients for each major model component BEFORE clipping.

        Returns:
            Dict mapping component names to gradient norms
        """
        # Initialize all keys to ensure consistent CSV columns
        module_norms = {
            "grads/inference_net_norm": 0.0,
            "grads/transition_model_norm": 0.0,
            "grads/generative_net_norm": 0.0,
            "grads/g2g_logsig_inf_norm": 0.0,
            "grads/g_init_norm": 0.0,
            "grads/streams_norm": 0.0,
            "grads/memory_system_norm": 0.0,
            "grads/other_norm": 0.0,
        }

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
            elif "g2g_logsig_inf" in name:
                module_norms["grads/g2g_logsig_inf_norm"] += norm_sq
            elif "g_init" in name:
                module_norms["grads/g_init_norm"] += norm_sq
            elif "streams" in name:
                module_norms["grads/streams_norm"] += norm_sq
            elif "memory_system" in name:
                module_norms["grads/memory_system_norm"] += norm_sq
            else:
                module_norms["grads/other_norm"] += norm_sq

        # Take square root to get norms
        result = {k: math.sqrt(v) for k, v in module_norms.items()}

        # Debug print on first step
        if not hasattr(self, '_debug_grad_printed'):
            self._debug_grad_printed = True
            print("\n=== GRADIENT BREAKDOWN (first step, PRE-CLIP) ===")
            for k in sorted(result.keys()):
                print(f"{k}: {result[k]:.6f}")
            print(f"Sum from all modules: {sum(result.values()):.6f}")
            print("=" * 50 + "\n")

        return result

    def _print_detailed_tensors(self, step, x_bptt, a_bptt, outputs, state):
        """
        Logs detailed tensor values to file for first 3 batch items for debugging.

        Inspects:
            - Curriculum state
            - Input observations and actions
            - Predictions from three pathways
            - Latent states (g_prior, g_inf, p_inf, p_gen)
            - Memory matrices
            - Losses with pathway breakdown
            - Sanity checks for NaN/Inf

        Args:
            step: Current iteration
            x_bptt: Observations [T, B, n_sensory]
            a_bptt: Actions [T, B]
            outputs: Model outputs from last timestep
            state: Model state
        """
        f = self.tensor_log_file

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"DETAILED TENSOR INSPECTION - Step {step}\n")
        f.write("=" * 80 + "\n")

        # Curriculum state
        _, loss_weights, p2g_offset, temp, hebb_eta, lambda_eff, p2g_gate = self.curriculum.get(step - 1)
        f.write("\n--- CURRICULUM STATE ---\n")
        f.write(f"  p2g_gate: {p2g_gate:.4f} (memory activity level)\n")
        f.write(f"  p2g_scale_offset: {p2g_offset:.4f} (variance inflation)\n")
        f.write(f"  temp: {temp:.4f} (temperature annealing)\n")
        f.write(f"  hebb_eta: {hebb_eta:.4f} (memory learning rate)\n")
        f.write(f"  loss_weights['g_kl']: {loss_weights['g_kl']:.4f}\n")
        f.write(f"  loss_weights['p_inf_align']: {loss_weights['p_inf_align']:.4f}\n")

        n_inspect = min(3, x_bptt.shape[1])

        # Inputs
        f.write("\n--- INPUTS (first 3 batch items) ---\n")
        for b in range(n_inspect):
            f.write(f"\nBatch item {b}:\n")
            f.write(f"  x_t (observation): {x_bptt[-1, b].detach().cpu().numpy()}\n")
            f.write(f"  a_t (action): {a_bptt[-1, b].item()}\n")

            obs_idx = x_bptt[-1, b].argmax().item()
            is_revisit = self.visited_sensory[b, obs_idx].item()
            f.write(f"  Is revisit: {is_revisit}\n")

        # Predictions
        if 'predictions' in outputs:
            f.write("\n--- PREDICTIONS (last timestep, first 3 batch items) ---\n")
            for b in range(n_inspect):
                f.write(f"\nBatch item {b}:\n")
                true_class = x_bptt[-1, b].argmax().item()
                for pred_type in ['x_p', 'x_g', 'x_gt']:
                    if pred_type in outputs['predictions']:
                        pred = outputs['predictions'][pred_type][b].detach().cpu().numpy()
                        pred_class = pred.argmax()
                        confidence = pred[pred_class]
                        correct = "[CORRECT]" if pred_class == true_class else "[WRONG]"
                        f.write(f"  {pred_type}: class={pred_class} {correct}, confidence={confidence:.3f}\n")

        # G-states with divergence analysis
        if 'reps_dict' in outputs:
            reps = outputs['reps_dict']

            if 'g_prior' in reps and reps['g_prior']:
                f.write("\n--- G_PRIOR (generative pathway, first 3 batch items) ---\n")
                for b in range(n_inspect):
                    f.write(f"\nBatch item {b}:\n")
                    for f_idx, g_prior_f in enumerate(reps['g_prior']):
                        if hasattr(g_prior_f, 'mean'):
                            mean = g_prior_f.mean[b].detach().cpu().numpy()
                            var = g_prior_f.log_var.exp()[b].detach().cpu().numpy()
                            f.write(f"  Freq {f_idx}: mean norm={np.linalg.norm(mean):.4f}, "
                                  f"var mean={var.mean():.4f}\n")
                            if var.max() > 1000:
                                f.write(f"    [WARNING] LARGE VARIANCE: {var.max():.1f}\n")

            if 'g_inf' in reps and reps['g_inf']:
                f.write("\n--- G_INF (inference pathway, first 3 batch items) ---\n")
                for b in range(n_inspect):
                    f.write(f"\nBatch item {b}:\n")
                    for f_idx, g_inf_f in enumerate(reps['g_inf']):
                        if hasattr(g_inf_f, 'mean'):
                            mean = g_inf_f.mean[b].detach().cpu().numpy()
                            var = g_inf_f.log_var.exp()[b].detach().cpu().numpy()
                            f.write(f"  Freq {f_idx}: mean norm={np.linalg.norm(mean):.4f}, "
                                  f"var mean={var.mean():.4f}\n")

            # G-space divergence
            if 'g_prior' in reps and 'g_inf' in reps and reps['g_prior'] and reps['g_inf']:
                f.write("\n--- G-SPACE DIVERGENCE (g_prior vs g_inf, first 3 batch items) ---\n")
                for b in range(n_inspect):
                    f.write(f"\nBatch item {b}:\n")
                    total_divergence = 0.0
                    for f_idx, (g_prior_f, g_inf_f) in enumerate(zip(reps['g_prior'], reps['g_inf'])):
                        if hasattr(g_prior_f, 'mean') and hasattr(g_inf_f, 'mean'):
                            mean_diff = (g_prior_f.mean[b] - g_inf_f.mean[b]).detach().cpu().numpy()
                            l2_dist = np.linalg.norm(mean_diff)
                            total_divergence += l2_dist
                            f.write(f"  Freq {f_idx}: L2 distance = {l2_dist:.4f}\n")
                    f.write(f"  TOTAL g-space L2 divergence: {total_divergence:.4f}\n")
                    if total_divergence > 10.0:
                        f.write(f"    [WARNING] LARGE DIVERGENCE between g_prior and g_inf\n")

        # P-states
        if 'p_inf' in state and state['p_inf']:
            f.write("\n--- P_INF (place cells from inference, first 3 batch items) ---\n")
            for b in range(n_inspect):
                f.write(f"\nBatch item {b}:\n")
                for s_idx, p_stream in enumerate(state['p_inf']):
                    p_vals = p_stream[b].detach().cpu().numpy()
                    f.write(f"  Stream {s_idx}: norm={np.linalg.norm(p_vals):.4f}\n")

        # P-space divergence (p_inf vs p_gen from memory)
        if 'p_gen' in state and state['p_gen'] and 'p_inf' in state and state['p_inf']:
            f.write("\n--- P-SPACE DIVERGENCE (p_inf vs p_gen from memory) ---\n")
            for b in range(n_inspect):
                f.write(f"\nBatch item {b}:\n")
                total_divergence = 0.0
                for s_idx, (p_inf_stream, p_gen_stream) in enumerate(zip(state['p_inf'], state['p_gen'])):
                    p_inf_vals = p_inf_stream[b].detach().cpu().numpy()
                    p_gen_vals = p_gen_stream[b].detach().cpu().numpy()
                    cosine_sim = np.dot(p_inf_vals, p_gen_vals) / (
                        np.linalg.norm(p_inf_vals) * np.linalg.norm(p_gen_vals) + 1e-8)
                    l2_dist = np.linalg.norm(p_inf_vals - p_gen_vals)
                    total_divergence += l2_dist
                    f.write(f"  Stream {s_idx}: cosine_sim={cosine_sim:.4f}, L2={l2_dist:.4f}\n")
                f.write(f"  TOTAL p-space L2 divergence: {total_divergence:.4f}\n")
                if total_divergence > 5.0:
                    f.write(f"    [WARNING] LARGE DIVERGENCE - memory vs inference mismatch\n")

        # Memory states
        if 'memory' in state:
            mem = state['memory']
            f.write("\n--- MEMORY STATE (first 3 batch items) ---\n")
            for b in range(n_inspect):
                f.write(f"\nBatch item {b}:\n")
                if 'generative' in mem and mem['generative'] is not None:
                    m_gen = mem['generative'][b].detach().cpu().numpy()
                    f.write(f"  M_generative: mean={m_gen.mean():.4f}, std={m_gen.std():.4f}, "
                          f"max={m_gen.max():.4f}\n")
                if 'inference' in mem and mem['inference'] is not None:
                    m_inf = mem['inference'][b].detach().cpu().numpy()
                    f.write(f"  M_inference: mean={m_inf.mean():.4f}, std={m_inf.std():.4f}, "
                          f"max={m_inf.max():.4f}\n")

        # Losses with pathway breakdown
        if 'losses' in outputs:
            f.write("\n--- LOSSES (first 3 batch items) ---\n")

            # Group losses by pathway
            pathway_losses = {'x_p': [], 'x_g': [], 'x_gt': [], 'alignment': [], 'other': []}
            for loss_name in outputs['losses'].keys():
                if 'x_nll_p' in loss_name:
                    pathway_losses['x_p'].append(loss_name)
                elif 'x_nll_g' in loss_name and 'gt' not in loss_name:
                    pathway_losses['x_g'].append(loss_name)
                elif 'x_nll_gt' in loss_name:
                    pathway_losses['x_gt'].append(loss_name)
                elif 'kl' in loss_name or 'align' in loss_name:
                    pathway_losses['alignment'].append(loss_name)
                else:
                    pathway_losses['other'].append(loss_name)

            for b in range(n_inspect):
                f.write(f"\nBatch item {b}:\n")

                # Pathway reconstruction losses
                for pathway, loss_names in [('x_p (inference)', pathway_losses['x_p']),
                                           ('x_g (hybrid)', pathway_losses['x_g']),
                                           ('x_gt (generative)', pathway_losses['x_gt'])]:
                    for loss_name in loss_names:
                        loss_val = outputs['losses'][loss_name]
                        val = loss_val[b].item() if loss_val.numel() > 1 else loss_val.item()
                        f.write(f"  {pathway}: {loss_name} = {val:.4f}\n")

                # Alignment losses
                f.write(f"  Alignment losses:\n")
                for loss_name in pathway_losses['alignment']:
                    loss_val = outputs['losses'][loss_name]
                    val = loss_val[b].item() if loss_val.numel() > 1 else loss_val.item()
                    f.write(f"    {loss_name} = {val:.4f}\n")

                # Other losses
                for loss_name in pathway_losses['other']:
                    loss_val = outputs['losses'][loss_name]
                    val = loss_val[b].item() if loss_val.numel() > 1 else loss_val.item()
                    f.write(f"  {loss_name}: {val:.4f}\n")

        # Sanity checks for NaN/Inf
        f.write("\n--- SANITY CHECKS ---\n")
        issues_found = []

        # Check predictions
        if 'predictions' in outputs:
            for pred_type in ['x_p', 'x_g', 'x_gt']:
                if pred_type in outputs['predictions']:
                    pred_tensor = outputs['predictions'][pred_type]
                    if torch.isnan(pred_tensor).any():
                        issues_found.append(f"NaN in {pred_type} predictions")
                    if torch.isinf(pred_tensor).any():
                        issues_found.append(f"Inf in {pred_type} predictions")

        # Check g-states
        if 'reps_dict' in outputs:
            for g_type in ['g_prior', 'g_inf']:
                if g_type in outputs['reps_dict'] and outputs['reps_dict'][g_type]:
                    for f_idx, g_state in enumerate(outputs['reps_dict'][g_type]):
                        if hasattr(g_state, 'mean'):
                            if torch.isnan(g_state.mean).any():
                                issues_found.append(f"NaN in {g_type}[{f_idx}].mean")
                            if torch.isnan(g_state.log_var).any():
                                issues_found.append(f"NaN in {g_type}[{f_idx}].log_var")
                            if torch.isinf(g_state.log_var).any():
                                issues_found.append(f"Inf in {g_type}[{f_idx}].log_var")

        # Check memory
        if 'memory' in state:
            for mem_type in ['generative', 'inference']:
                if mem_type in state['memory'] and state['memory'][mem_type] is not None:
                    mem_tensor = state['memory'][mem_type]
                    if torch.isnan(mem_tensor).any():
                        issues_found.append(f"NaN in memory['{mem_type}']")
                    if torch.isinf(mem_tensor).any():
                        issues_found.append(f"Inf in memory['{mem_type}']")

        # Check losses
        if 'losses' in outputs:
            for loss_name, loss_val in outputs['losses'].items():
                if torch.isnan(loss_val).any():
                    issues_found.append(f"NaN in loss '{loss_name}'")
                if torch.isinf(loss_val).any():
                    issues_found.append(f"Inf in loss '{loss_name}'")

        if issues_found:
            f.write("[WARNING] CRITICAL ISSUES DETECTED:\n")
            for issue in issues_found:
                f.write(f"  - {issue}\n")
        else:
            f.write("[OK] No NaN or Inf values detected\n")

        f.write("\n" + "=" * 80 + "\n\n")
        f.flush()

    def run_validation(self, val_envs):
        """
        Runs model on validation environments to assess performance.

        Evaluates:
            - Overall accuracy (all timesteps)
            - Revisit accuracy (only on previously seen observations)
            - Separate metrics for three pathways (p, g, gt)

        Args:
            val_envs: List of validation environments

        Returns:
            Tuple of (avg_loss, accuracy_dict)
            where accuracy_dict has 'overall' and 'revisit' subdicts
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = defaultdict(float)
        total_acc_revisit = defaultdict(float)
        n_samples = min(len(val_envs), self.args.n_val_samples)

        with torch.no_grad():
            for i in range(n_samples):
                dataset = TrajectoryDataset(
                    val_envs[i].generate_trajectory(self.args.val_trajectory_len),
                    self.config.n_sensory_objects, self.device
                )
                val_state = self.model.create_empty_memory(1, self.device)

                visited_sensory = torch.zeros(
                    1, self.config.n_sensory_objects,
                    dtype=torch.bool, device=self.device
                )

                segment_loss = 0.0
                segment_correct = defaultdict(int)
                segment_correct_revisit = defaultdict(int)
                revisit_count = 0
                val_steps = len(dataset) - 1

                if val_steps <= 0:
                    continue

                for t in range(val_steps):
                    outputs = self.model.forward_single_step(
                        dataset[t]['x_t'].unsqueeze(0),
                        torch.tensor([dataset[t]['a_t']], device=self.device),
                        val_state
                    )

                    true_obs_idx = dataset[t]['x_t'].argmax().item()
                    is_revisit = visited_sensory[0, true_obs_idx].item()

                    if is_revisit and torch.isfinite(outputs['losses']['total_loss']):
                        segment_loss += outputs['losses']['total_loss'].item()

                    # Overall accuracy
                    for pred_type in ['p', 'g', 'gt']:
                        if outputs['predictions'][f'x_{pred_type}'].argmax().item() == true_obs_idx:
                            segment_correct[pred_type] += 1

                    # Revisit accuracy
                    if is_revisit:
                        revisit_count += 1
                        for pred_type in ['p', 'g', 'gt']:
                            if outputs['predictions'][f'x_{pred_type}'].argmax().item() == true_obs_idx:
                                segment_correct_revisit[pred_type] += 1

                    visited_sensory[0, true_obs_idx] = True
                    val_state = outputs['new_state']

                total_loss += segment_loss / revisit_count if revisit_count > 0 else 0

                for pred_type in ['p', 'g', 'gt']:
                    total_acc[pred_type] += segment_correct[pred_type] / val_steps if val_steps > 0 else 0

                for pred_type in ['p', 'g', 'gt']:
                    total_acc_revisit[pred_type] += (
                        segment_correct_revisit[pred_type] / revisit_count if revisit_count > 0 else 0
                    )

        avg_loss = total_loss / n_samples if n_samples > 0 else 0
        avg_acc_overall = {k: v / n_samples for k, v in total_acc.items()}
        avg_acc_revisit = {k: v / n_samples for k, v in total_acc_revisit.items()}

        return avg_loss, {
            'overall': avg_acc_overall,
            'revisit': avg_acc_revisit
        }

    def _save_checkpoint(self, step, best_val_acc, is_best=False):
        """
        Saves model checkpoint.

        Args:
            step: Current iteration
            best_val_acc: Best validation accuracy so far
            is_best: Whether this is the best model so far
        """
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "step": step,
            "best_val_acc": best_val_acc,
            "args": self.args,
            "config": self.config,
        }
        filename = 'best_model.pt' if is_best else f"ckpt_{step:07d}.pt"
        ckpt_path = os.path.join(self.args.save_dir, filename)
        torch.save(state, ckpt_path)
        logger.info(f"Saved checkpoint -> {ckpt_path}")


# ==============================================================================
# Utility Functions
# ==============================================================================

def create_environments(n_envs, env_type, min_size, max_size, n_sensory):
    """
    Factory function to create list of environments.

    Args:
        n_envs: Number of environments to create
        env_type: 'grid', 'line', or 'tree'
        min_size: Minimum environment size
        max_size: Maximum environment size
        n_sensory: Number of sensory objects

    Returns:
        List of environment instances
    """
    envs = []
    for _ in range(n_envs):
        size = np.random.randint(min_size, max_size + 1)
        if env_type == 'grid':
            envs.append(Grid2DEnvironment(width=size, height=size, n_sensory=n_sensory))
        elif env_type == 'line':
            envs.append(GraphEnvironment(create_line_graph_dict(length=size, n_sensory=n_sensory)))
        elif env_type == 'tree':
            depth = np.random.randint(max(2, min_size // 2), max(3, max_size // 2) + 1)
            envs.append(GraphEnvironment(create_tree_graph_dict(depth=depth, n_sensory=n_sensory)))
    return envs


def load_checkpoint(checkpoint_path, model, optimizer, scaler, device):
    """
    Loads checkpoint and returns training state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance
        optimizer: Optimizer instance
        scaler: GradScaler instance
        device: Device to map checkpoint to

    Returns:
        Tuple of (step, best_val_acc)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if 'scaler_state' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state'])
    return checkpoint.get('step', 0), checkpoint.get('best_val_acc', -1.0)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """
    Main training script entry point.

    Parses arguments, initializes model and environments, runs training loop.
    """
    parser = argparse.ArgumentParser(description="Train a Tolman-Eichenbaum Machine model.")

    # Core settings
    core_group = parser.add_argument_group('Core Settings')
    core_group.add_argument('--env_type', type=str, default='grid', choices=['grid', 'line', 'tree'])
    core_group.add_argument('--save_dir', type=str, default='results')
    core_group.add_argument('--seed', type=int, default=973)
    core_group.add_argument('--continue_from', type=str, default=None, help='Path to checkpoint to continue from.')

    # WandB logging
    wandb_group = parser.add_argument_group('Wandb Logging')
    wandb_group.add_argument('--project_name', type=str, default="TEM")
    wandb_group.add_argument('--run_name', type=str, default=None)
    wandb_group.add_argument('--log_interval', type=int, default=100)
    wandb_group.add_argument('--save_interval', type=int, default=10000)
    wandb_group.add_argument('--log_debug_metrics', action='store_true',
                            help='Log detailed debug metrics and histograms.')

    # Training hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--train_it', type=int, default=50000)
    train_group.add_argument('--batch_size', type=int, default=64)
    train_group.add_argument('--bptt_len', type=int, default=75)
    train_group.add_argument('--lr', type=float, default=9e-4)
    train_group.add_argument('--lr_min', type=float, default=8e-5)
    train_group.add_argument('--lr_decay_steps', type=int, default=4000)
    train_group.add_argument('--lr_decay_rate', type=float, default=0.5)
    train_group.add_argument('--weight_decay', type=float, default=1e-4)
    train_group.add_argument('--clip_grad_norm', type=float, default=2.0)
    train_group.add_argument('--autocast', action='store_true', help='Enable Automatic Mixed Precision (AMP).')
    train_group.add_argument('--full_graph_compile', action='store_true',
                            help='Use full graph torch.compile mode (slower startup).')

    # Dataset settings
    data_group = parser.add_argument_group('Dataset Settings')
    data_group.add_argument('--n_val_samples', type=int, default=3)
    data_group.add_argument('--val_trajectory_len', type=int, default=200)
    data_group.add_argument('--min_size', type=int, default=8)
    data_group.add_argument('--max_size', type=int, default=12)

    # Curriculum schedules
    curriculum_group = parser.add_argument_group('Curriculum Schedules')
    curriculum_group.add_argument('--g_kl_anneal_steps', type=int, default=2000)
    curriculum_group.add_argument('--p_align_anneal_steps', type=int, default=2000)
    curriculum_group.add_argument('--p_reg_anneal_steps', type=int, default=4000)
    curriculum_group.add_argument('--g_reg_anneal_steps', type=int, default=40000000)
    curriculum_group.add_argument('--p2g_half_it', type=int, default=400)
    curriculum_group.add_argument('--p2g_scale_it', type=int, default=200)
    curriculum_group.add_argument('--p2g_scale_val', type=float, default=10000.0)
    curriculum_group.add_argument('--hebb_learn_it', type=int, default=16000)
    curriculum_group.add_argument('--lambda_it', type=int, default=200)
    curriculum_group.add_argument('--curriculum_decay_steps', type=int, default=40000)
    curriculum_group.add_argument('--restart_max', type=int, default=40)
    curriculum_group.add_argument('--restart_min', type=int, default=5)
    curriculum_group.add_argument('--seq_jitter', type=int, default=30)

    # Loss weights
    loss_group = parser.add_argument_group('Loss Weights')
    loss_group.add_argument('--g_kl_weight', type=float, default=1.0)
    loss_group.add_argument('--p_align_weight', type=float, default=1.0)
    loss_group.add_argument('--p_inf_align_weight', type=float, default=1.0)
    loss_group.add_argument('--g_reg_weight', type=float, default=0.01)
    loss_group.add_argument('--p_reg_weight', type=float, default=0.02)
    loss_group.add_argument('--hebbian_learning_rate', type=float, default=0.5)
    loss_group.add_argument('--hebbian_forget_rate', type=float, default=0.9999)

    args = parser.parse_args()

    # Initialization
    if args.run_name is None:
        args.run_name = f"tem_{args.env_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    run = wandb.init(project=args.project_name, name=args.run_name, config=args)

    # Log source code as artifact
    code_artifact = wandb.Artifact(
        "source-code", type="source-code",
        description="The source code for this run."
    )
    code_artifact.add_file("model.py")
    code_artifact.add_file("train.py")
    code_artifact.add_file("environment.py")
    run.log_artifact(code_artifact)

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create config
    config = TEMConfig()
    if args.env_type == 'line':
        config.n_actions = 2
    elif args.env_type == 'tree':
        config.n_actions = 3
    else:
        config.n_actions = 4
    config.bptt_len = args.bptt_len

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TolmanEichenbaumMachine(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(device == 'cuda' and args.autocast))

    # Load checkpoint if continuing
    if args.continue_from:
        start_iter, best_acc = load_checkpoint(args.continue_from, model, optimizer, scaler, device)
        args._checkpoint_start_iteration, args._checkpoint_best_val_acc = start_iter, best_acc
        wandb.config.update({"continued_from": args.continue_from})

    # Conditional torch.compile for performance
    if device == 'cuda' and hasattr(torch, 'compile'):
        if args.full_graph_compile:
            logger.info("Compiling model with 'max-autotune' mode.")
            model = torch.compile(model, mode="max-autotune")
        else:
            logger.info("Compiling model with 'max-autotune-no-cudagraphs' mode.")
            model = torch.compile(
                model,
                mode="max-autotune-no-cudagraphs",
                disable="cudagraphs",
                dynamic=True,
                fullgraph=True
            )

    # Create environments
    train_envs = create_environments(args.batch_size, args.env_type, args.min_size, args.max_size,
                                     config.n_sensory_objects)
    val_envs = create_environments(50, args.env_type, args.min_size, args.max_size, config.n_sensory_objects)

    # Train
    trainer = TEMTrainer(model, config, args, run)
    trainer.train(train_envs, val_envs)

    run.finish()


if __name__ == "__main__":
    main()
