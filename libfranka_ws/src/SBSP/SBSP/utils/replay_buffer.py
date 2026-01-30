import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device
        self.storage_device = torch.device("cpu")
        
        self.obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.storage_device)
        self.next_obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.storage_device)
        self.act_buf = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.storage_device)
        self.rew_buf = torch.zeros(capacity, dtype=torch.float32, device=self.storage_device)
        self.done_buf = torch.zeros(capacity, dtype=torch.float32, device=self.storage_device)
        
        # [NEW] Store True State (7 dim joint pos)
        self.true_state_buf = torch.zeros((capacity, 7), dtype=torch.float32, device=self.storage_device)

    # Updated signature to accept true_state
    def add(self, obs, action, reward, next_obs, done, true_state):
        self.obs_buf[self.ptr] = torch.as_tensor(obs, device=self.storage_device)
        self.next_obs_buf[self.ptr] = torch.as_tensor(next_obs, device=self.storage_device)
        self.act_buf[self.ptr] = torch.as_tensor(action, device=self.storage_device)
        self.rew_buf[self.ptr] = float(reward)
        self.done_buf[self.ptr] = float(done)
        self.true_state_buf[self.ptr] = torch.as_tensor(true_state, device=self.storage_device)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # Updated signature to accept true_states
    def add_batch(self, obs, action, reward, next_obs, done, true_states):
        n = len(obs)
        idxs = np.arange(self.ptr, self.ptr + n) % self.capacity
        
        self.obs_buf[idxs] = torch.as_tensor(obs, device=self.storage_device)
        self.next_obs_buf[idxs] = torch.as_tensor(next_obs, device=self.storage_device)
        self.act_buf[idxs] = torch.as_tensor(action, device=self.storage_device)
        self.rew_buf[idxs] = torch.as_tensor(reward, device=self.storage_device)
        self.done_buf[idxs] = torch.as_tensor(done, device=self.storage_device)
        self.true_state_buf[idxs] = torch.as_tensor(true_states, device=self.storage_device)
        
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,))
        return {
            'obs': self.obs_buf[idxs].to(self.device),
            'actions': self.act_buf[idxs].to(self.device),
            'rewards': self.rew_buf[idxs].unsqueeze(1).to(self.device),
            'next_obs': self.next_obs_buf[idxs].to(self.device),
            'dones': self.done_buf[idxs].unsqueeze(1).to(self.device),
            'true_state': self.true_state_buf[idxs].to(self.device)
        }