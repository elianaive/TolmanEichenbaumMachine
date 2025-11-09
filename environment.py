import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset
import torch.nn.functional as f
from typing import Dict, List
import heapq
from collections import deque

class BaseEnvironment:
    """Abstract base class for all environments."""
    def __init__(self):
        self.n_nodes = 0
        self.n_actions = 0
        self.n_sensory = 0

    def generate_trajectory(self, num_steps: int) -> dict:
        raise NotImplementedError


class Grid2DEnvironment(BaseEnvironment):
    """
    Manages a 2D grid world with multiple behavioral policies including angle-based movement.
    """
    def __init__(self, width: int, height: int, n_sensory: int, n_objects: int = 0):
        super().__init__()
        self.width, self.height = width, height
        self.n_nodes = width * height
        self.n_actions = 4  # 0:N, 1:E, 2:S, 3:W
        self.n_sensory = n_sensory
        self.node_observations = np.random.randint(0, n_sensory, self.n_nodes)
        self.transitions = self._build_transitions()
        
        self.edges = set()
        for i in range(self.height):
            for j in range(self.width):
                node_idx = i * self.width + j
                # Add edge to the right neighbor (action 1)
                if j < self.width - 1:
                    neighbor_idx = self.transitions[node_idx, 1]
                    # Add edge as a sorted tuple to ensure uniqueness
                    self.edges.add(tuple(sorted((node_idx, neighbor_idx))))
                # Add edge to the bottom neighbor (action 2)
                if i < self.height - 1:
                    neighbor_idx = self.transitions[node_idx, 2]
                    self.edges.add(tuple(sorted((node_idx, neighbor_idx))))
        
        # Place objects/landmarks for the object policy
        self.objects = []
        if n_objects > 0:
            self.objects = np.random.choice(self.n_nodes, size=n_objects, replace=False)
            
    def reset_sensory_map(self):
        """Re-randomizes the mapping of observations to nodes."""
        self.node_observations = np.random.randint(0, self.n_sensory, self.n_nodes)

    def _build_transitions(self) -> np.ndarray:
        transitions = np.full((self.n_nodes, self.n_actions), -1, dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                node_idx = i * self.width + j
                if i > 0: transitions[node_idx, 0] = (i - 1) * self.width + j  # North
                if j < self.width - 1: transitions[node_idx, 1] = i * self.width + (j + 1)  # East
                if i < self.height - 1: transitions[node_idx, 2] = (i + 1) * self.width + j  # South
                if j > 0: transitions[node_idx, 3] = i * self.width + (j - 1)  # West
        return transitions

    def _get_angle_between_nodes(self, node1: int, node2: int) -> float:
        """Calculate angle from node1 to node2 in radians (-pi to pi)."""
        y1, x1 = divmod(node1, self.width)
        y2, x2 = divmod(node2, self.width)
        
        # Grid coordinates: +x is right, +y is down
        # Convert to standard coordinates where +y is up
        return np.arctan2(-(y2 - y1), x2 - x1)
    
    def _get_action_angle(self, action: int) -> float:
        """Get the angle corresponding to an action."""
        # Actions: 0:N, 1:E, 2:S, 3:W
        angles = {
            0: np.pi/2,    # North (up)
            1: 0,          # East (right)
            2: -np.pi/2,   # South (down)
            3: np.pi       # West (left)
        }
        return angles[action]
    
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """Compute the minimum angular difference between two angles."""
        diff = angle2 - angle1
        # Normalize to [-pi, pi]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return abs(diff)
    
    def _get_angle_based_action(self, current_node: int, current_angle: float, 
                            available_actions: np.ndarray, angle_threshold: float = np.pi/4, angle_change = 0.4) -> tuple:
        """
        Select action based on angle preference, matching author's implementation.
        Returns (action, new_angle).
        """
        # Calculate angle difference for each available action
        angle_diffs = []
        for action in available_actions:
            action_angle = self._get_action_angle(action)
            diff = self._angle_difference(current_angle, action_angle)
            angle_diffs.append((diff, action))
        
        if not angle_diffs:
            # No valid moves (shouldn't happen in a proper grid)
            return np.random.choice(available_actions), current_angle
        
        # Sort by angle difference
        angle_diffs.sort(key=lambda x: x[0])
        min_diff, best_action = angle_diffs[0]
        
        # Decide whether to keep going straight or pick new direction
        if min_diff < angle_threshold:
            # Continue in same direction
            chosen_action = best_action
            new_angle = current_angle  # Keep the same angle
        else:
            # Hit a wall or corner - choose randomly and update angle
            valid_actions = [action for _, action in angle_diffs]
            chosen_action = np.random.choice(valid_actions)
            new_angle = self._get_action_angle(chosen_action)  # Update to new direction
        
        # Add noise to angle (random walk)
        new_angle += np.random.uniform(-angle_change, angle_change)
        # Normalize to [-pi, pi]
        new_angle = np.arctan2(np.sin(new_angle), np.cos(new_angle))
        
        return chosen_action, new_angle

    def _get_policy_for_node(self, node: int, policy_type: str, policy_beta: float, 
                            current_angle: float = None) -> np.ndarray:
        """Get action probabilities for a given policy type."""
        probs = np.ones(self.n_actions)
        y, x = divmod(node, self.width)
        center_y, center_x = (self.height-1)/2.0, (self.width-1)/2.0

        if policy_type == 'angle':
            # This is handled separately in generate_trajectory
            # Return uniform for compatibility
            pass
            
        elif policy_type == 'border':
            scores = []
            for action in range(self.n_actions):
                next_node = self.transitions[node, action]
                if next_node != -1:
                    next_y, next_x = divmod(next_node, self.width)
                    scores.append(abs(next_y - center_y) + abs(next_x - center_x))
                else:
                    scores.append(-np.inf)
            
            exp_scores = np.exp(policy_beta * np.array(scores))
            if np.sum(exp_scores) > 0: probs = exp_scores / np.sum(exp_scores)

        elif policy_type == 'object' and len(self.objects) > 0:
            scores = []
            target_obj_y, target_obj_x = divmod(self.objects[0], self.width)
            for action in range(self.n_actions):
                next_node = self.transitions[node, action]
                if next_node != -1:
                    next_y, next_x = divmod(next_node, self.width)
                    dist_to_obj = abs(next_y - target_obj_y) + abs(next_x - target_obj_x)
                    scores.append(-dist_to_obj)
                else:
                    scores.append(-np.inf)
            
            exp_scores = np.exp(policy_beta * np.array(scores))
            if np.sum(exp_scores) > 0: probs = exp_scores / np.sum(exp_scores)

        valid_moves = (self.transitions[node] != -1)
        probs = probs * valid_moves
        if np.sum(probs) > 0:
            probs /= np.sum(probs)
        else:
            probs = valid_moves.astype(float) / valid_moves.sum()
        return probs

    def generate_trajectory(self, num_steps: int, policy_type: str = 'random',
                      policy_beta: float = 1.5, direc_bias: float = 0.25) -> dict:
        """
        Generate trajectory with various policies including angle-based movement.
        
        Args:
            policy_type: 'random', 'border', 'object', or 'angle'
            policy_beta: Temperature parameter for softmax policies
            direc_bias: For angle policy, probability of following current angle (default 0.25 from author)
        """
        trajectory = {'observations': [], 'actions': [], 'nodes': []}
        current_node = np.random.randint(0, self.n_nodes)
        
        # Initialize angle for angle-based policy
        current_angle = np.random.uniform(-np.pi, np.pi) if policy_type == 'angle' else None
        
        for _ in range(num_steps):
            trajectory['observations'].append(self.node_observations[current_node])
            trajectory['nodes'].append(current_node)
            
            # Get valid actions
            valid_actions = np.where(self.transitions[current_node] != -1)[0]
            
            if len(valid_actions) == 0:
                # Dead end (shouldn't happen in grid)
                action = 0
                next_node = current_node
            elif policy_type == 'angle':
                action, proposed_angle = self._get_angle_based_action(
                    current_node, current_angle, valid_actions
                )

                if np.random.rand() > direc_bias:
                    action = np.random.choice(valid_actions)
                    current_angle = self._get_action_angle(action)
                else:
                    current_angle = proposed_angle

            else:
                # Other policies (random, border, object)
                probs = self._get_policy_for_node(current_node, policy_type, policy_beta)
                action = np.random.choice(self.n_actions, p=probs)
            
            trajectory['actions'].append(action)
            next_node = self.transitions[current_node, action]
            if next_node == -1:
                next_node = current_node
            current_node = next_node
        
        return trajectory

class GraphEnvironment(BaseEnvironment):
    def __init__(self, graph_dict: Dict, n_objects: int = 0):
        super().__init__()
        self.n_nodes = graph_dict['n_nodes']
        self.n_actions = graph_dict['n_actions']
        self.n_sensory = graph_dict['n_sensory']
        self.transitions = np.array(graph_dict['transitions'], dtype=int)
        self.edges = graph_dict.get('edges', [])
        self.node_observations = np.random.randint(0, self.n_sensory, self.n_nodes)
        
        # Pre-calculate adjacency list and node degrees for policies
        self.adj = [[] for _ in range(self.n_nodes)]
        self.node_degrees = np.zeros(self.n_nodes, dtype=int)
        for node in range(self.n_nodes):
            for action in range(self.n_actions):
                neighbor = self.transitions[node, action]
                if neighbor != -1:
                    self.adj[node].append((neighbor, action))
            self.node_degrees[node] = len(self.adj[node])
            
        self.objects = []
        if n_objects > 0 and self.n_nodes > 0:
            self.objects = np.random.choice(self.n_nodes, size=min(n_objects, self.n_nodes), replace=False)
            
    def reset_sensory_map(self):
        """Re-randomizes the mapping of observations to nodes."""
        self.node_observations = np.random.randint(0, self.n_sensory, self.n_nodes)

    def _bfs_shortest_path_action(self, start_node, target_nodes):
        """Finds the first action on the shortest path to any of the target nodes."""
        if start_node in target_nodes: return None
        
        q = deque([(start_node, [])]) # (current_node, path_of_actions)
        visited = {start_node}

        while q:
            current, path = q.popleft()
            for neighbor, action in self.adj[current]:
                if neighbor in target_nodes:
                    return path[0] if path else action
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, path + [action]))
        return None

    def generate_trajectory(self, num_steps: int, policy_type: str = 'random', policy_beta: float = 1.5) -> dict:
        """Generates a trajectory using a specified policy for abstract graphs."""
        trajectory = {'observations': [], 'actions': [], 'nodes': []}
        current_node = np.random.randint(0, self.n_nodes)
        
        for _ in range(num_steps):
            trajectory['observations'].append(self.node_observations[current_node])
            trajectory['nodes'].append(current_node)
            
            valid_actions = [action for _, action in self.adj[current_node]]
            if not valid_actions: # Node is a dead end
                action, next_node = 0, current_node
            
            elif policy_type == 'object' and len(self.objects) > 0 and np.random.rand() < policy_beta:
                # Policy: Move towards the nearest object via shortest path
                path_action = self._bfs_shortest_path_action(current_node, self.objects)
                if path_action is not None:
                    action = path_action
                else: # Already at an object or no path exists, act randomly
                    action = np.random.choice(valid_actions)

            elif policy_type == 'border' and np.random.rand() < policy_beta:
                # Policy: Prefer to move to neighbors with a lower degree (more "border-like")
                neighbor_degrees = [self.node_degrees[n] for n, a in self.adj[current_node]]
                # Invert degrees so lower degree gets higher score
                scores = -np.array(neighbor_degrees, dtype=float)
                probs = np.exp(scores - np.max(scores)) # Softmax for stability
                probs /= probs.sum()
                
                # Get the action corresponding to the chosen neighbor
                neighbor_actions = [a for n, a in self.adj[current_node]]
                action = np.random.choice(neighbor_actions, p=probs)
                
            else: # Default 'random' policy
                action = np.random.choice(valid_actions)
                
            next_node = self.transitions[current_node, action]
            trajectory['actions'].append(action)
            current_node = next_node
            
        return trajectory

#  Graph Factory Functions 
def create_line_graph_dict(length: int, n_sensory: int) -> Dict:
    """Creates a graph dictionary for a transitive inference task (line graph)."""
    n_nodes = length
    n_actions = 2  # 0: Forward (+1), 1: Backward (-1)
    transitions = np.full((n_nodes, n_actions), -1, dtype=int)
    for i in range(n_nodes):
        if i < n_nodes - 1: transitions[i, 0] = i + 1
        if i > 0: transitions[i, 1] = i - 1
    return {'n_nodes': n_nodes, 'n_actions': n_actions, 'n_sensory': n_sensory, 'transitions': transitions.tolist()}

def create_tree_graph_dict(depth: int, n_sensory: int) -> Dict:
    """Creates a graph dictionary for a social hierarchy task (binary tree)."""
    n_nodes = 2**depth - 1
    n_actions = 3  # 0: Parent, 1: Child_L, 2: Child_R
    transitions = np.full((n_nodes, n_actions), -1, dtype=int)
    for i in range(n_nodes):
        if i > 0: transitions[i, 0] = (i - 1) // 2
        if (2 * i + 1) < n_nodes: transitions[i, 1] = 2 * i + 1
        if (2 * i + 2) < n_nodes: transitions[i, 2] = 2 * i + 2
    return {'n_nodes': n_nodes, 'n_actions': n_actions, 'n_sensory': n_sensory, 'transitions': transitions.tolist()}

class TrajectoryDataset(Dataset):
    """
    An efficient PyTorch Dataset that wraps a trajectory and pre-computes tensors.
    """
    def __init__(self, trajectory: dict, n_sensory_objects: int, device: str):
        self.device = device
        observations = torch.tensor(trajectory['observations'], dtype=torch.long)
        self.actions = torch.tensor(trajectory['actions'], dtype=torch.long)
        self.nodes = torch.tensor(trajectory['nodes'], dtype=torch.long)
        self.one_hot_observations = f.one_hot(observations, num_classes=n_sensory_objects).float().to(device)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx: int) -> dict:
        """Returns all necessary data for a single timestep."""
        return {
            'x_t': self.one_hot_observations[idx],
            'a_t': self.actions[idx].item(),
            'node': self.nodes[idx].item()
        }
