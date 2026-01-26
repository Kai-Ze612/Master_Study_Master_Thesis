"""
Replay Buffer
-------------
Stores transitions in CPU RAM and samples batches to GPU to prevent VRAM exhaustion.
"""

import torch
import numpy as np
from typing import Dict

class ReplayBuffer:
    def __init__(
        self, 
        capacity: int, 
        obs_dim: int, 
        action_dim: int, 
        training_device: torch.device
    ):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.training_device = training_device
        self.storage_device = torch.device("cpu")
        
        # Pre-allocate memory on CPU
        print(f"[Buffer] Initializing Replay Buffer (Cap: {capacity}) on CPU...")
        self.obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.storage_device)
        self.next_obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.storage_device)
        self.act_buf = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.storage_device)
        self.rew_buf = torch.zeros(capacity, dtype=torch.float32, device=self.storage_device)
        self.done_buf = torch.zeros(capacity, dtype=torch.float32, device=self.storage_device)
    
    def add(self, obs, action, reward, next_obs, done):
        """Add a single transition."""
        self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.storage_device)
        self.next_obs_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.storage_device)
        self.act_buf[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.storage_device)
        self.rew_buf[self.ptr] = float(reward)
        self.done_buf[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def add_batch(self, obs, action, reward, next_obs, done):
        """Add a batch of transitions (from vectorized envs)."""
        N = obs.shape[0]
        indices = np.arange(self.ptr, self.ptr + N) % self.capacity
        
        self.obs_buf[indices] = torch.as_tensor(obs, dtype=torch.float32, device=self.storage_device)
        self.next_obs_buf[indices] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.storage_device)
        self.act_buf[indices] = torch.as_tensor(action, dtype=torch.float32, device=self.storage_device)
        self.rew_buf[indices] = torch.as_tensor(reward, dtype=torch.float32, device=self.storage_device)
        self.done_buf[indices] = torch.as_tensor(done, dtype=torch.float32, device=self.storage_device)
        
        self.ptr = (self.ptr + N) % self.capacity
        self.size = min(self.size + N, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch and move to GPU."""
        idxs = torch.randint(0, self.size, (batch_size,))
        return {
            'obs': self.obs_buf[idxs].to(self.training_device),
            'actions': self.act_buf[idxs].to(self.training_device),
            'rewards': self.rew_buf[idxs].unsqueeze(1).to(self.training_device),
            'next_obs': self.next_obs_buf[idxs].to(self.training_device),
            'dones': self.done_buf[idxs].unsqueeze(1).to(self.training_device)
        }