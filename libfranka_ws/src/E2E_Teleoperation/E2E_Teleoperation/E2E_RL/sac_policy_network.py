"""
This module defines the policy network architecture used in the
End-to-End Teleoperation system based on Soft Actor-Critic (SAC).

Three main components are implemented:
1. Autoregressive LSTM Encoder: Estimates current leader state from delayed history.
2. Joint Actor Network: Outputs torque actions (Initialized via Inverse Dynamics).
3. Joint Critic Network: Evaluates state-action pairs for SAC training.

The training pipeline consists of two phases:
- Stage 1 (Behavioral Cloning): The LSTM Encoder is trained for state prediction, 
  and the Actor is pre-trained via Supervised Learning to clone the 'Teacher' (Inverse Dynamics).
- Stage 2 (E2E SAC): The Critic is introduced, and the entire system (Encoder + Actor) 
  is fine-tuned end-to-end using Reinforcement Learning with delays enabled.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import E2E_Teleoperation.config.robot_config as cfg

class LSTM(nn.Module):
    """
    Stage 1 Component: Temporal Encoder
    Input: Delayed History Sequence
    Output: Predicted State (14-dim: 7 Pos + 7 Vel)
    """
    def __init__(self):
        super().__init__()
        
        # 1. LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=cfg.ROBOT.ESTIMATOR_INPUT_DIM, # 15
            hidden_size=cfg.ROBOT.RNN_HIDDEN_DIM,     # 256
            num_layers=cfg.ROBOT.RNN_NUM_LAYERS,      # 3
            batch_first=True
        )
        
        # 2. State Predictor Head (MLP)
        self.predictor = nn.Sequential(
            nn.Linear(cfg.ROBOT.RNN_HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, cfg.ROBOT.ROBOT_STATE_DIM) # 14
        )

    def forward(self, history, hidden=None):
        out, hidden = self.lstm(history, hidden)
        feat = out[:, -1, :] 
        pred_state = self.predictor(feat)
        return feat, pred_state, hidden


class JointActor(nn.Module):
    """
    Stage 2 & 3 Component: Inverse Dynamics Policy
    Input: [Remote_State(14), Predicted_Leader(14), Prev_Action(7)] = 35 dims
    Output: Torque(7)
    """
    LOG_STD_MIN = -10.0
    LOG_STD_MAX = 2.0
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
        # 14 (Remote) + 14 (Pred Leader) + 7 (Prev Action) = 35
        self.input_dim = 35 
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        
        self.mu = nn.Linear(256, cfg.ROBOT.N_JOINTS)
        self.log_std = nn.Linear(256, cfg.ROBOT.N_JOINTS)
        
        # Move scale to buffer so it moves with device automatically
        self.register_buffer("scale", torch.tensor(cfg.ROBOT.TORQUE_LIMITS))

    def forward(self, obs, hidden=None):
        # 1. Parse Observation Components
        idx_rem = cfg.ROBOT.ROBOT_STATE_DIM      # 14
        
        # Assumes obs structure: [Remote(14), RemoteHist(...), TargetHist(...), PrevAction(7)]
        remote_state = obs[:, :idx_rem]          
        prev_action = obs[:, -7:]                
        
        # Extract Leader History for Encoder 
        target_seq_len = cfg.ROBOT.RNN_SEQ_LEN * cfg.ROBOT.ESTIMATOR_INPUT_DIM
        target_hist = obs[:, -7 - target_seq_len : -7]
        target_seq = target_hist.view(-1, cfg.ROBOT.RNN_SEQ_LEN, cfg.ROBOT.ESTIMATOR_INPUT_DIM)

        # 2. Get Predicted State from Encoder
        feat, pred_state, next_hidden = self.encoder(target_seq, hidden)

        # 3. Concatenate [Remote, Predicted, PrevAction]
        x = torch.cat([remote_state, pred_state, prev_action], dim=1)
        
        x = self.net(x)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mu, log_std, pred_state, next_hidden, feat

    def sample(self, obs, hidden=None):
        """
        Samples an action from the policy distribution (with reparameterization).
        Required for SAC training.
        """
        mu, log_std, pred_state, next_hidden, feat = self.forward(obs, hidden)
        
        std = log_std.exp()
        normal = Normal(mu, std)
        
        # Reparameterization trick (rsample)
        x_t = normal.rsample() 
        y_t = torch.tanh(x_t)
        
        # Scale to torque limits
        action = y_t * self.scale
        
        # Calculate Log Prob (with Tanh correction)
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound (Tanh correction formula)
        # log_prob -= log(1 - tanh(x)^2 + epsilon)
        log_prob -= torch.log(self.scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        # Return same structure as forward, but swapping mu/log_std for action/log_prob
        return action, log_prob, pred_state, next_hidden, feat


class JointCritic(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
        # Critic Input: Pred State (14) + Action (7) = 21
        self.input_dim = cfg.ROBOT.ROBOT_STATE_DIM + cfg.ROBOT.N_JOINTS 
        
        self.q1 = nn.Sequential(nn.Linear(self.input_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(self.input_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, pred_state, action):
        xu = torch.cat([pred_state, action], dim=1)
        return self.q1(xu), self.q2(xu)