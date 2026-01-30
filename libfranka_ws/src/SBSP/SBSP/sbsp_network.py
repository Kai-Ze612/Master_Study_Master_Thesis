"""
SBSP Network: Standard SAC Actor + Independent DCNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import SBSP.config.robot_config as cfg

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# [NEW] The separate Dynamics Model from 'delay_correcting_nn.py'
class DCNN(nn.Module):
    def __init__(self, input_dim, action_dim, layer_size=128, n_layers=2):
        super(DCNN, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim # For Leader trajectory, this effectively acts as dummy or 0 if autonomous
        
        self.fc1 = nn.Linear(self.input_dim + self.action_dim, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, self.input_dim)
        
        self.n_layers = n_layers
        self.apply(weight_init)

    def forward(self, state, action):
        # Concatenate State + Action
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        if self.n_layers > 1:
            x = F.relu(self.fc2(x))
            if self.n_layers > 2:
                x = F.relu(self.fc3(x))
        return self.fc4(x) # Output is Next State (delta or absolute depends on training, usually absolute)

class SBSPActor(nn.Module):
    """
    Standard SAC Actor (No Auxiliary Head)
    """
    def __init__(self, input_dim=None):
        super().__init__()
        self.input_dim = input_dim if input_dim is not None else cfg.ROBOT.RL_OBS_DIM
        self.action_scale = 1.0 
        
        layers = []
        in_dim = self.input_dim
        for h_dim in cfg.ROBOT.ACTOR_HIDDEN_DIMS:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        
        self.mu_layer = nn.Linear(in_dim, cfg.ROBOT.ACTION_DIM)
        self.log_std_layer = nn.Linear(in_dim, cfg.ROBOT.ACTION_DIM)
        
        self.log_std_min = cfg.ROBOT.LOG_STD_MIN
        self.log_std_max = cfg.ROBOT.LOG_STD_MAX
        self.apply(weight_init)

    def forward(self, obs):
        latent = self.encoder(obs)
        mu = self.mu_layer(latent)
        log_std = torch.clamp(self.log_std_layer(latent), self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob

class SBSPCritic(nn.Module):
    def __init__(self, obs_dim=None):
        super().__init__()
        self.obs_dim = obs_dim if obs_dim is not None else cfg.ROBOT.RL_OBS_DIM
        self.action_dim = cfg.ROBOT.ACTION_DIM
        self.input_dim = self.obs_dim + self.action_dim
        
        hidden_dims = cfg.ROBOT.CRITIC_HIDDEN_DIMS
        self.q1_net = self._build_net(self.input_dim, hidden_dims)
        self.q2_net = self._build_net(self.input_dim, hidden_dims)
        self.apply(weight_init)

    def _build_net(self, in_dim, hidden_dims):
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2