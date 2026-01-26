"""
E2E Networks with Fixed Critic Architecture

Key Changes:
1. Critic uses a COMPACT state representation, not the full observation
2. Added LayerNorm for stability
3. Proper weight initialization
4. Option to use the actor's latent representation in critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import E2E_Teleoperation.config.robot_config as cfg


def weight_init(m):
    """Xavier uniform initialization for better gradient flow."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class LSTM(nn.Module):
    """LSTM-based state estimator - unchanged from original."""
    def __init__(self):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(
            input_size=cfg.ROBOT.ESTIMATOR_INPUT_DIM, 
            hidden_size=cfg.ROBOT.RNN_HIDDEN_DIM
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(cfg.ROBOT.RNN_HIDDEN_DIM, cfg.ROBOT.LSTM_PRED_HEAD_DIM),
            nn.ReLU(),
            nn.Linear(cfg.ROBOT.LSTM_PRED_HEAD_DIM, 7)
        )
        
        self.delay_norm_factor = cfg.ROBOT.DELAY_INPUT_NORM_FACTOR
        self.max_rollout = cfg.ROBOT.MAX_PREDICTION_ROLLOUT_STEPS
        self.dt = 1.0 / cfg.ROBOT.CONTROL_FREQ
        
        self.register_buffer('pos_std', torch.tensor(cfg.ROBOT.Q_STD, dtype=torch.float32))
        self.register_buffer('vel_std', torch.tensor(cfg.ROBOT.QD_STD, dtype=torch.float32))
        self.register_buffer('vel_mean', torch.tensor(cfg.ROBOT.QD_MEAN, dtype=torch.float32))
        
        dt_scale_val = (self.vel_std / self.pos_std) * self.dt
        self.register_buffer('dt_scale', dt_scale_val)

    def forward(self, history, hidden=None):
        batch_size, seq_len, _ = history.size()
        device = history.device
        
        if hidden is None:
            h = torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM, device=device)
            c = torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM, device=device)
        else:
            h, c = hidden
        
        for t in range(seq_len):
            h, c = self.lstm_cell(history[:, t, :], (h, c))
        
        anchor_state = history[:, -1, :14] 
        initial_delay = history[:, -1, 14] 
        
        steps_to_predict = (initial_delay * self.delay_norm_factor).ceil().long()
        
        current_pos = anchor_state[:, :7].clone()
        current_vel = anchor_state[:, 7:14].clone()
        current_delay = initial_delay.clone()
        
        for step_i in range(self.max_rollout):
            mask = (step_i < steps_to_predict).float().unsqueeze(1)
            
            clamped_pos = torch.clamp(current_pos, -5.0, 5.0)
            clamped_vel = torch.clamp(current_vel, -5.0, 5.0)
            
            recur_input = torch.cat([clamped_pos, clamped_vel, current_delay.unsqueeze(1)], dim=1)
            h_next, c_next = self.lstm_cell(recur_input, (h, c))
            
            delta_vel = self.predictor(h_next) 
            
            vel_next = current_vel + delta_vel
            pos_next = current_pos + (current_vel * self.dt_scale)
            
            h = mask * h_next + (1 - mask) * h
            c = mask * c_next + (1 - mask) * c
            
            current_pos = mask * pos_next + (1 - mask) * current_pos
            current_vel = mask * vel_next + (1 - mask) * current_vel
            
            current_delay = current_delay - self.dt
            current_delay = torch.clamp(current_delay, min=0.0)
        
        pred_state = torch.cat([current_pos, current_vel], dim=1)
        
        return h, pred_state, (h, c)


class JointActor(nn.Module):
    """Actor network - returns latent for critic use."""
    def __init__(self):
        super().__init__()
        self.base_encoder = LSTM()
        self.aux_head = self.base_encoder.predictor
        
        self.state_dim = cfg.ROBOT.ROBOT_STATE_DIM      
        self.target_seq_len = cfg.ROBOT.TARGET_HISTORY_DIM 
        self.action_hist_len = cfg.ROBOT.ACTION_HISTORY_LEN * cfg.ROBOT.N_JOINTS
        self.prev_action_dim = cfg.ROBOT.N_JOINTS       
        self.input_dim = cfg.ROBOT.ACTOR_INPUT_DIM 
        
        layers = []
        in_dim = self.input_dim
        for h_dim in cfg.ROBOT.ACTOR_HIDDEN_DIMS:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.residual_net = nn.Sequential(*layers)
        
        self.res_mu = nn.Linear(in_dim, cfg.ROBOT.N_JOINTS)
        self.res_log_std = nn.Linear(in_dim, cfg.ROBOT.N_JOINTS)
        
        self.register_buffer("action_scale", torch.tensor(cfg.ROBOT.MAX_ACTION_TORQUE, dtype=torch.float32))
        self.log_std_min = cfg.ROBOT.LOG_STD_MIN
        self.log_std_max = cfg.ROBOT.LOG_STD_MAX
        
        # Store dimensions for external use
        self.latent_dim = cfg.ROBOT.RNN_HIDDEN_DIM

    def forward(self, obs, hidden=None):
        real_follower_state = obs[:, :14]
        idx_state_end = 14
        idx_target_end = idx_state_end + self.target_seq_len
        idx_act_hist_end = idx_target_end + self.action_hist_len
        target_hist_flat = obs[:, idx_state_end:idx_target_end]
        action_hist_flat = obs[:, idx_target_end:idx_act_hist_end]
        prev_action = obs[:, idx_act_hist_end:]

        full_seq = target_hist_flat.view(-1, cfg.ROBOT.RNN_SEQ_LEN, 36)
        leader_seq = full_seq[:, :, :15]
        
        latent_vector, pred_leader, next_hidden = self.base_encoder(leader_seq, hidden)
        
        rl_input = torch.cat([real_follower_state, latent_vector, action_hist_flat, prev_action], dim=1)
        x = self.residual_net(rl_input)
        mu = self.res_mu(x)
        log_std = torch.clamp(self.res_log_std(x), self.log_std_min, self.log_std_max)
        
        return mu, log_std, pred_leader, next_hidden, latent_vector  # [CHANGED] Also return latent

    def sample(self, obs, hidden=None):
        mu, log_std, pred_leader, next_hidden, latent = self.forward(obs, hidden)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        final_action = y_t * self.action_scale
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return final_action, log_prob, pred_leader, next_hidden, latent  # [CHANGED]


class JointCritic(nn.Module):
    """
    FIXED Critic Network
    
    Key Changes:
    1. Uses COMPACT inputs: follower_state (14) + latent (256) + pred_state (14) + action (7)
       Total: ~291 dims instead of 500+ dims
    2. Added LayerNorm for stability
    3. Proper weight initialization
    4. Smaller, more focused network
    """
    def __init__(self, use_layer_norm=True):
        super().__init__()
        
        # Compact input: follower_state + latent + pred_state + action
        # 14 + RNN_HIDDEN_DIM + 14 + 7
        self.follower_state_dim = 14
        self.latent_dim = cfg.ROBOT.RNN_HIDDEN_DIM  # 256 typically
        self.pred_state_dim = 14
        self.action_dim = cfg.ROBOT.N_JOINTS  # 7
        
        self.input_dim = self.follower_state_dim + self.latent_dim + self.pred_state_dim + self.action_dim
        
        # Smaller, more stable architecture
        hidden_dims = [256, 256]  # Simpler than potentially huge CRITIC_HIDDEN_DIMS
        
        self.use_layer_norm = use_layer_norm
        
        # Q1 Network
        self.q1_layers = nn.ModuleList()
        self.q1_norms = nn.ModuleList() if use_layer_norm else None
        in_dim = self.input_dim
        for h_dim in hidden_dims:
            self.q1_layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                self.q1_norms.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
        self.q1_out = nn.Linear(in_dim, 1)
        
        # Q2 Network
        self.q2_layers = nn.ModuleList()
        self.q2_norms = nn.ModuleList() if use_layer_norm else None
        in_dim = self.input_dim
        for h_dim in hidden_dims:
            self.q2_layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                self.q2_norms.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
        self.q2_out = nn.Linear(in_dim, 1)
        
        # Apply initialization
        self.apply(weight_init)
        
        # Initialize output layers with smaller weights for stability
        nn.init.uniform_(self.q1_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q1_out.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.q2_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q2_out.bias, -3e-3, 3e-3)

    def forward(self, follower_state, latent, pred_state, action):
        """
        Args:
            follower_state: [B, 14] - Current follower joint pos/vel (normalized)
            latent: [B, 256] - LSTM latent from actor
            pred_state: [B, 14] - Predicted leader state
            action: [B, 7] - Action taken
        """
        x = torch.cat([follower_state, latent, pred_state, action], dim=1)
        
        # Q1
        q1 = x
        for i, layer in enumerate(self.q1_layers):
            q1 = layer(q1)
            if self.use_layer_norm:
                q1 = self.q1_norms[i](q1)
            q1 = F.relu(q1)
        q1 = self.q1_out(q1)
        
        # Q2
        q2 = x
        for i, layer in enumerate(self.q2_layers):
            q2 = layer(q2)
            if self.use_layer_norm:
                q2 = self.q2_norms[i](q2)
            q2 = F.relu(q2)
        q2 = self.q2_out(q2)
        
        return q1, q2
    
    def q1_forward(self, follower_state, latent, pred_state, action):
        """Forward only Q1 for efficiency in some cases."""
        x = torch.cat([follower_state, latent, pred_state, action], dim=1)
        
        q1 = x
        for i, layer in enumerate(self.q1_layers):
            q1 = layer(q1)
            if self.use_layer_norm:
                q1 = self.q1_norms[i](q1)
            q1 = F.relu(q1)
        return self.q1_out(q1)


# Backward-compatible wrapper that extracts compact features from full obs
class JointCriticWrapper(nn.Module):
    """
    Wrapper to maintain API compatibility with original code.
    Extracts compact features from full observation.
    """
    def __init__(self):
        super().__init__()
        self.critic = JointCritic(use_layer_norm=True)
        
    def forward(self, obs, action, pred_state, latent=None):
        """
        If latent is not provided, we need to extract follower_state from obs.
        pred_state is already provided.
        
        For full compatibility, you should pass latent from actor.
        """
        # Extract follower state (first 14 dims of obs)
        follower_state = obs[:, :14]
        
        if latent is None:
            # Fallback: use zeros (not recommended, but maintains compatibility)
            latent = torch.zeros(obs.size(0), cfg.ROBOT.RNN_HIDDEN_DIM, device=obs.device)
        
        return self.critic(follower_state, latent, pred_state, action)