import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import deque
import random

class DreamerReplayBuffer:
    """
    Replay buffer for storing real environment transitions for Dreamer-style training.
    """
    
    def __init__(self, capacity: int = 100000, obs_dim: int = 147, action_dim: int = 7):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Circular buffers for transitions
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)  # one-hot
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=bool)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
    def add(self, obs: np.ndarray, action: int, reward: float, done: bool, next_obs: np.ndarray):
        """Add a transition to the buffer."""
        # Convert action to one-hot
        action_onehot = np.zeros(self.action_dim, dtype=np.float32)
        action_onehot[action] = 1.0
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action_onehot
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.next_observations[self.ptr] = next_obs
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'observations': torch.tensor(self.observations[indices], dtype=torch.float32),
            'actions': torch.tensor(self.actions[indices], dtype=torch.float32),
            'rewards': torch.tensor(self.rewards[indices], dtype=torch.float32),
            'dones': torch.tensor(self.dones[indices], dtype=torch.bool),
            'next_observations': torch.tensor(self.next_observations[indices], dtype=torch.float32)
        }
    
    def sample_sequences(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """Sample sequences for RSSM training."""
        # Sample starting indices that allow for full sequences
        max_start = self.size - seq_len
        if max_start <= 0:
            raise ValueError(f"Buffer too small for sequences of length {seq_len}")
        
        start_indices = np.random.choice(max_start, batch_size, replace=True)
        
        obs_seqs = []
        action_seqs = []
        reward_seqs = []
        done_seqs = []
        
        for start_idx in start_indices:
            end_idx = start_idx + seq_len
            obs_seqs.append(self.observations[start_idx:end_idx])
            action_seqs.append(self.actions[start_idx:end_idx])
            reward_seqs.append(self.rewards[start_idx:end_idx])
            done_seqs.append(self.dones[start_idx:end_idx])
        
        return {
            'observations': torch.tensor(np.array(obs_seqs), dtype=torch.float32),
            'actions': torch.tensor(np.array(action_seqs), dtype=torch.float32),
            'rewards': torch.tensor(np.array(reward_seqs), dtype=torch.float32),
            'dones': torch.tensor(np.array(done_seqs), dtype=torch.bool)
        }
    
    def sample_latent_states(self, agent, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample latent states for imagination starting points."""
        indices = np.random.choice(self.size, batch_size, replace=True)
        obs_batch = torch.tensor(self.observations[indices], dtype=torch.float32, device=device)
        
        # Get latent states from RSSM encoder
        h_batch = agent.init_hidden(batch_size=batch_size, device=device)
        
        with torch.no_grad():
            z_means, z_stds = agent.rssm.encode(obs_batch, h_batch)
            z_batch = agent.rssm.sample_latent(z_means, z_stds)
        
        return z_batch, h_batch
    
    def __len__(self):
        return self.size
    
    def is_ready(self, min_size: int = 1000) -> bool:
        """Check if buffer has enough data for training."""
        return self.size >= min_size 