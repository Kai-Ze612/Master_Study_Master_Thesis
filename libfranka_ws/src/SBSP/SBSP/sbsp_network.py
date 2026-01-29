"""
SBSP Network: Actor with Auxiliary State Predictor
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

class SBSPActor(nn.Module):
    """
    Inputs: Stacked Observation (History)
    Outputs: 
      1. PD Gains (Policy)
      2. Predicted Current Leader State (Auxiliary)
    """
    def __init__(self, input_dim=None):
        super().__init__()
        self.input_dim = input_dim if input_dim is not None else cfg.ROBOT.RL_OBS_DIM
        self.action_scale = 1.0 
        
        # Shared Encoder
        layers = []
        in_dim = self.input_dim
        # First few layers serve as shared feature extractor
        for h_dim in cfg.ROBOT.ACTOR_HIDDEN_DIMS:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        
        # 1. Policy Head (PD Gains)
        self.mu_layer = nn.Linear(in_dim, cfg.ROBOT.ACTION_DIM)
        self.log_std_layer = nn.Linear(in_dim, cfg.ROBOT.ACTION_DIM)
        
        # 2. Prediction Head (Current Leader Joint Pos 7 dim)
        self.pred_layer = nn.Linear(in_dim, 7)
        
        self.log_std_min = cfg.ROBOT.LOG_STD_MIN
        self.log_std_max = cfg.ROBOT.LOG_STD_MAX
        self.apply(weight_init)

    def forward(self, obs):
        latent = self.encoder(obs)
        
        # Policy
        mu = self.mu_layer(latent)
        log_std = torch.clamp(self.log_std_layer(latent), self.log_std_min, self.log_std_max)
        
        # Prediction
        pred_state = self.pred_layer(latent)
        
        return mu, log_std, pred_state

    def sample(self, obs):
        mu, log_std, pred_state = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        
        # Tanh squashing
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob, pred_state

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