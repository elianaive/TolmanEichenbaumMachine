import torch
import numpy as np
import argparse
import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TolmanEichenbaumMachine, TEMConfig
from environment import (Grid2DEnvironment, GraphEnvironment, TrajectoryDataset, 
                         create_line_graph_dict, create_tree_graph_dict)


def run_generalization_analysis(model, env, config, trajectory_len=10000, device='cuda', 
                               save_dir=None, show_plots=False, verbose=True):
    """
    Run generalization analysis on a trained TEM model.
    
    Args:
        model: Trained TolmanEichenbaumMachine
        env: Environment to test in
        config: TEMConfig
        trajectory_len: Length of trajectory to analyze
        device: Device to run on
        save_dir: Directory to save results (creates timestamped subdir if provided)
        show_plots: Whether to display plots
        verbose: Whether to print progress
        
    Returns:
        dict: Analysis results including accuracies and statistics
    """
    model.eval()
    
    # Generate trajectory
    if verbose:
        print(f"Generating trajectory of length {trajectory_len} in {env.__class__.__name__}...")
    dataset = TrajectoryDataset(env.generate_trajectory(trajectory_len), config.n_sensory_objects, device)
    
    # Initialize tracking
    history = defaultdict(list)
    visited_nodes, visited_edges = set(), set()
    correct_preds = {'p': 0, 'g': 0, 'gt': 0}
    total_preds = 0
    
    # Track inference opportunities (visiting known node via new edge)
    inference_opportunities = 0
    correct_inferences = {'p': 0, 'g': 0, 'gt': 0}
    
    prev_state = model.create_empty_memory(1, device)
    
    # Run analysis
    with torch.no_grad():
        iterator = tqdm(range(len(dataset) - 1), desc="Analyzing") if verbose else range(len(dataset) - 1)
        
        for t in iterator:
            current_step, next_step = dataset[t], dataset[t+1]
            
            # Get model predictions for CURRENT observation
            outputs = model(
                current_step['x_t'].unsqueeze(0),
                torch.tensor([current_step['a_t']], device=device),
                prev_state
            )
            prev_state = outputs['new_state']
            
            # Build edge representation: edge that brought us TO current position
            if t > 0:
                prev_step = dataset[t-1]
                edge = (prev_step['node'], current_step['node'], prev_step['a_t'])
            
            # Test predictions only on previously visited nodes
            if current_step['node'] in visited_nodes:
                true_current = current_step['x_t'].argmax().item()
                
                # Check if this is an inference opportunity (new edge to known node)
                is_new_edge = False
                if t > 0:
                    is_new_edge = edge not in visited_edges
                    if is_new_edge:
                        inference_opportunities += 1
                
                # Test all three prediction pathways
                for pred_type in ['p', 'g', 'gt']:
                    pred_current = outputs['predictions'][f'x_{pred_type}'].argmax(dim=-1)[0].item()
                    if pred_current == true_current:
                        correct_preds[pred_type] += 1
                        if is_new_edge:
                            correct_inferences[pred_type] += 1
                
                total_preds += 1
            
            # Update visited sets AFTER testing
            visited_nodes.add(current_step['node'])
            if t > 0:
                visited_edges.add(edge)
            
            # Record history
            history['steps'].append(t)
            history['visited_nodes'].append(len(visited_nodes))
            history['visited_edges'].append(len(visited_edges))
            history['prop_nodes_visited'].append(len(visited_nodes) / env.n_nodes)
            
            if hasattr(env, 'edges') and env.edges:
                history['prop_edges_visited'].append(len(visited_edges) / len(env.edges))
            else:
                # Estimate total edges for grid environments
                total_edges_estimate = env.n_nodes * 4  # Rough estimate for 4-connected
                history['prop_edges_visited'].append(len(visited_edges) / total_edges_estimate)
            
            # Record accuracies
            for pred_type in ['p', 'g', 'gt']:
                acc_key = f'accuracy_{pred_type}'
                history[acc_key].append(
                    correct_preds[pred_type] / total_preds if total_preds > 0 else 0.0
                )
    
    # Calculate final statistics
    final_results = {
        'environment': {
            'type': env.__class__.__name__,
            'n_nodes': env.n_nodes,
            'n_edges_estimate': len(env.edges) if hasattr(env, 'edges') else env.n_nodes * 4,
        },
        'coverage': {
            'nodes_visited': len(visited_nodes),
            'edges_visited': len(visited_edges),
            'prop_nodes_visited': len(visited_nodes) / env.n_nodes,
            'prop_edges_visited': history['prop_edges_visited'][-1],
        },
        'accuracy': {
            'final': {k: v / total_preds if total_preds > 0 else 0.0 
                     for k, v in correct_preds.items()},
            'total_predictions': total_preds,
        },
        'generalization': {
            'inference_opportunities': inference_opportunities,
            'inference_accuracy': {k: v / inference_opportunities if inference_opportunities > 0 else 0.0 
                                  for k, v in correct_inferences.items()},
        },
        'history': dict(history),
    }
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Accuracy plot
    ax1.plot(history['steps'], history['accuracy_gt'], label='Generative (x_gt)', 
             color='red', linewidth=2)
    ax1.plot(history['steps'], history['accuracy_g'], label='Hybrid (x_g)', 
             color='orange', linewidth=2, alpha=0.7)
    ax1.plot(history['steps'], history['accuracy_p'], label='Inference (x_p)', 
             color='purple', linewidth=2, alpha=0.7)
    ax1.plot(history['steps'], history['prop_nodes_visited'], 
             label='Prop. Nodes Visited', color='blue', linestyle='--')
    ax1.plot(history['steps'], history['prop_edges_visited'], 
             label='Prop. Edges Visited', color='green', linestyle=':')
    
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Accuracy / Proportion')
    ax1.set_title('TEM Generalization Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Coverage vs accuracy plot
    ax2.scatter(history['prop_nodes_visited'], history['accuracy_gt'], 
                alpha=0.5, s=1, color='red', label='vs Nodes (x_gt)')
    ax2.scatter(history['prop_edges_visited'], history['accuracy_gt'], 
                alpha=0.5, s=1, color='green', label='vs Edges (x_gt)')
    
    # Fit lines
    if len(history['prop_nodes_visited']) > 10:
        z_nodes = np.polyfit(history['prop_nodes_visited'][-1000:], 
                             history['accuracy_gt'][-1000:], 1)
        z_edges = np.polyfit(history['prop_edges_visited'][-1000:], 
                            history['accuracy_gt'][-1000:], 1)
        p_nodes = np.poly1d(z_nodes)
        p_edges = np.poly1d(z_edges)
        
        x_nodes = np.linspace(0, 1, 100)
        ax2.plot(x_nodes, p_nodes(x_nodes), 'r-', alpha=0.8, 
                label=f'Node fit (slope={z_nodes[0]:.2f})')
        ax2.plot(x_nodes, p_edges(x_nodes), 'g-', alpha=0.8, 
                label=f'Edge fit (slope={z_edges[0]:.2f})')
    
    ax2.set_xlabel('Proportion Visited')
    ax2.set_ylabel('Accuracy (x_gt)')
    ax2.set_title('Accuracy vs Coverage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    # Save results if requested
    if save_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_subdir = os.path.join(save_dir, f'generalization_{timestamp}')
        os.makedirs(save_subdir, exist_ok=True)
        
        # Save plot
        plt.savefig(os.path.join(save_subdir, 'generalization_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        
        # Save results as JSON
        results_json = final_results.copy()
        results_json['history'] = {k: [float(v) for v in vals] 
                                  for k, vals in results_json['history'].items()}
        
        with open(os.path.join(save_subdir, 'results.json'), 'w') as f:
            json.dump(results_json, f, indent=2)
        
        if verbose:
            print(f"\nResults saved to: {save_subdir}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    if verbose:
        print("\n" + "="*50)
        print("GENERALIZATION ANALYSIS SUMMARY")
        print("="*50)
        print(f"Environment: {final_results['environment']['type']}")
        print(f"Nodes: {len(visited_nodes)}/{env.n_nodes} visited "
              f"({final_results['coverage']['prop_nodes_visited']:.1%})")
        print(f"Edges: {len(visited_edges)} visited")
        print(f"\nFinal Accuracies:")
        for k, v in final_results['accuracy']['final'].items():
            print(f"  x_{k}: {v:.3f}")
        print(f"\nGeneralization Performance:")
        print(f"  Inference opportunities: {inference_opportunities}")
        if inference_opportunities > 0:
            print(f"  Inference accuracy (x_gt): "
                  f"{final_results['generalization']['inference_accuracy']['gt']:.3f}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Run generalization analysis on a trained TEM model')
    
    # Model arguments
    parser.add_argument('model_path', type=str, help='Path to saved model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    # Environment arguments
    parser.add_argument('--env_type', type=str, default='grid', 
                       choices=['grid', 'line', 'tree'], help='Type of environment')
    parser.add_argument('--env_size', type=int, default=10, 
                       help='Size of environment (width for grid, length for line, depth for tree)')
    parser.add_argument('--n_sensory', type=int, default=45, 
                       help='Number of sensory objects')
    
    # Analysis arguments
    parser.add_argument('--trajectory_len', type=int, default=10000, 
                       help='Length of trajectory to analyze')
    parser.add_argument('--save_dir', type=str, default='./analysis_results',
                       help='Directory to save results')
    parser.add_argument('--no_save', action='store_true', 
                       help='Do not save results')
    parser.add_argument('--show_plots', action='store_true', 
                       help='Display plots')
    parser.add_argument('--quiet', action='store_true', 
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Create config
    config = TEMConfig()
    config.n_sensory_objects = args.n_sensory
    
    # Set number of actions based on environment
    if args.env_type == 'line':
        config.n_actions = 2
    elif args.env_type == 'tree':
        config.n_actions = 3
    else:
        config.n_actions = 4
    
    # Load model
    if not args.quiet:
        print(f"Loading model from: {args.model_path}")
    
    model = TolmanEichenbaumMachine(config).to(args.device)
    #torch.serialization.add_safe_globals([TEMConfig, argparse.Namespace])
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create environment
    if args.env_type == 'grid':
        env = Grid2DEnvironment(width=args.env_size, height=args.env_size, 
                               n_sensory=args.n_sensory)
    elif args.env_type == 'line':
        env = GraphEnvironment(create_line_graph_dict(length=args.env_size, 
                                                     n_sensory=args.n_sensory))
    else:  # tree
        env = GraphEnvironment(create_tree_graph_dict(depth=args.env_size, 
                                                     n_sensory=args.n_sensory))
    
    # Run analysis
    results = run_generalization_analysis(
        model=model,
        env=env,
        config=config,
        trajectory_len=args.trajectory_len,
        device=args.device,
        save_dir=None if args.no_save else args.save_dir,
        show_plots=args.show_plots,
        verbose=not args.quiet
    )
    
    return results


if __name__ == '__main__':
    main()