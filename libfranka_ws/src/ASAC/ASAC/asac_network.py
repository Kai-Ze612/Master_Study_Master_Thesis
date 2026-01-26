"""
A-SAC Network Architecture for PD Gain Tuning
----------------------------------------------
Actor outputs 14-dim action: Kp[7] + Kd[7]
Normalized to [-1, 1], then scaled to gain ranges in environment.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import ASAC.config.robot_config as cfg


def build_mlp(input_dim: int, hidden_dims: list, output_dim: int = None, output_activation=None) -> nn.Sequential:
    """Helper to construct a standard MLP."""
    layers = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.ReLU())
        in_dim = h_dim
    
    if output_dim is not None:
        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation:
            layers.append(output_activation)
            
    return nn.Sequential(*layers)


class GainTuningActor(nn.Module):
    """
    Actor network for PD gain tuning.
    
    Outputs 14-dim action:
        - action[0:7]  -> Kp gains (normalized to [-1, 1])
        - action[7:14] -> Kd gains (normalized to [-1, 1])
    
    The environment scales these to actual gain ranges.
    """
    
    def __init__(self, input_dim: int = None):
        super().__init__()
        
        # Input dimension
        self.input_dim = input_dim if input_dim is not None else cfg.ROBOT.RL_OBS_DIM
        self.hidden_dims = cfg.ROBOT.ASAC_HIDDEN_DIMS
        
        # Output dimension: 14 (Kp[7] + Kd[7])
        self.action_dim = cfg.ROBOT.N_JOINTS * 2  # 14
        
        # Network backbone
        self.backbone = build_mlp(self.input_dim, self.hidden_dims)
        
        # Output heads (Mean and LogStd for each action dimension)
        last_hidden = self.hidden_dims[-1]
        self.mu_head = nn.Linear(last_hidden, self.action_dim)
        self.log_std_head = nn.Linear(last_hidden, self.action_dim)
        
        # Action scale: normalized to [-1, 1] (scaling happens in env)
        self.register_buffer("action_scale", torch.ones(self.action_dim, dtype=torch.float32))
        
        self.log_std_min = cfg.ROBOT.LOG_STD_MIN
        self.log_std_max = cfg.ROBOT.LOG_STD_MAX
        
        # Initialize output layers to output near-zero (base gains)
        self._init_output_layers()

    def _init_output_layers(self):
        """Initialize output layers to output near-base gains."""
        # Initialize mu_head to output zeros (which maps to base gains in env)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.01)
        
        # Initialize log_std to reasonable exploration
        nn.init.constant_(self.log_std_head.bias, -1.0)
        nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        """Forward pass returning mean and log_std."""
        x = self.backbone(obs)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        """Sample action with reparameterization trick."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        
        # Reparameterization trick
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        
        # Scale to [-1, 1] (env does the rest)
        final_action = y_t * self.action_scale
        
        # Log probability with tanh correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return final_action, log_prob, mu


class GainTuningCritic(nn.Module):
    """
    Critic network for PD gain tuning.
    
    Input: observation + action (14-dim gains)
    Output: Q-value (scalar)
    """
    
    def __init__(self, obs_dim: int = None):
        super().__init__()
        
        # Critic input: observation + action
        base_dim = obs_dim if obs_dim is not None else cfg.ROBOT.RL_OBS_DIM
        action_dim = cfg.ROBOT.N_JOINTS * 2  # 14
        self.input_dim = base_dim + action_dim
        
        self.hidden_dims = cfg.ROBOT.ASAC_HIDDEN_DIMS
        
        # Double Q-learning architecture
        self.q1_net = build_mlp(self.input_dim, self.hidden_dims, output_dim=1)
        self.q2_net = build_mlp(self.input_dim, self.hidden_dims, output_dim=1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Forward pass returning two Q-values."""
        xu = torch.cat([obs, action], dim=1)
        return self.q1_net(xu), self.q2_net(xu)


# =============================================================================
# Aliases for backward compatibility
# =============================================================================
AugmentedActor = GainTuningActor
AugmentedCritic = GainTuningCritic