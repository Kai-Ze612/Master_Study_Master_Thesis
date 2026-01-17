"""
E2E model:

1. 

"""


import torch
import torch.nn as nn
from torch.distributions import Normal
import E2E_Teleoperation.config.robot_config as cfg

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(
            input_size=cfg.ROBOT.ESTIMATOR_INPUT_DIM, 
            hidden_size=cfg.ROBOT.RNN_HIDDEN_DIM
        )
        
        # Predicts Acceleration (or Velocity Correction) instead of generic state delta
        self.predictor = nn.Sequential(
            nn.Linear(cfg.ROBOT.RNN_HIDDEN_DIM, cfg.ROBOT.LSTM_PRED_HEAD_DIM),
            nn.ReLU(),
            nn.Linear(cfg.ROBOT.LSTM_PRED_HEAD_DIM, 7) # Output dim = 7 (Velocity/Accel only)
        )
        
        self.delay_norm_factor = cfg.ROBOT.DELAY_INPUT_NORM_FACTOR
        self.max_rollout = cfg.ROBOT.MAX_PREDICTION_ROLLOUT_STEPS
        
        # Physics Constants for Unrolling
        self.dt = 1.0 / cfg.ROBOT.CONTROL_FREQ
        
        # Pre-calculate scaling factors to handle Normalized Space integration
        # Pos_norm += Vel_norm * (Vel_std / Pos_std) * dt
        self.register_buffer('pos_std', torch.tensor(cfg.ROBOT.Q_STD, dtype=torch.float32))
        self.register_buffer('vel_std', torch.tensor(cfg.ROBOT.QD_STD, dtype=torch.float32))
        self.register_buffer('vel_mean', torch.tensor(cfg.ROBOT.QD_MEAN, dtype=torch.float32))
        
        # [FIX] Register dt_scale as a buffer so it moves to GPU with the model
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
        
        # 1. Process History
        for t in range(seq_len):
            h, c = self.lstm_cell(history[:, t, :], (h, c))
        
        # 2. Setup Autoregressive Rollout
        # anchor_state is [Batch, 14] -> (7 pos, 7 vel)
        anchor_state = history[:, -1, :14] 
        initial_delay = history[:, -1, 14] 
        
        steps_to_predict = (initial_delay * self.delay_norm_factor).ceil().long()
        
        current_pos = anchor_state[:, :7].clone()
        current_vel = anchor_state[:, 7:14].clone()
        current_delay = initial_delay.clone()
        
        for step_i in range(self.max_rollout):
            # Create mask for batch elements that still need prediction
            mask = (step_i < steps_to_predict).float().unsqueeze(1)
            
            # Clamp for stability in normalized space
            clamped_pos = torch.clamp(current_pos, -5.0, 5.0)
            clamped_vel = torch.clamp(current_vel, -5.0, 5.0)
            
            # Input to LSTM Cell
            recur_input = torch.cat([clamped_pos, clamped_vel, current_delay.unsqueeze(1)], dim=1)
            h_next, c_next = self.lstm_cell(recur_input, (h, c))
            
            # Predict Velocity Delta (Acceleration-like term)
            delta_vel = self.predictor(h_next) 
            
            # --- PHYSICS INTEGRATION (Euler) ---
            # 1. Update Velocity: v_new = v_old + delta
            vel_next = current_vel + delta_vel
            
            # 2. Update Position: p_new = p_old + v_old * dt
            # Normalized math: p_norm += v_norm * dt_scale
            # [FIX] self.dt_scale is now on the correct device
            pos_next = current_pos + (current_vel * self.dt_scale)
            # -----------------------------------
            
            # Apply Mask (Stop updating if we reached real-time)
            h = mask * h_next + (1 - mask) * h
            c = mask * c_next + (1 - mask) * c
            
            current_pos = mask * pos_next + (1 - mask) * current_pos
            current_vel = mask * vel_next + (1 - mask) * current_vel
            
            current_delay = current_delay - self.dt
            current_delay = torch.clamp(current_delay, min=0.0)
        
        # Recombine
        pred_state = torch.cat([current_pos, current_vel], dim=1)
        
        return h, pred_state, (h, c)

class JointActor(nn.Module):
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
        
        return mu, log_std, pred_leader, next_hidden

    def sample(self, obs, hidden=None):
        mu, log_std, pred_leader, next_hidden = self.forward(obs, hidden)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        final_action = y_t * self.action_scale
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return final_action, log_prob, pred_leader, next_hidden

class JointCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = cfg.ROBOT.CRITIC_INPUT_DIM
        layers_q1 = []
        in_dim = self.input_dim
        for h_dim in cfg.ROBOT.CRITIC_HIDDEN_DIMS:
            layers_q1.append(nn.Linear(in_dim, h_dim))
            layers_q1.append(nn.ReLU())
            in_dim = h_dim
        layers_q1.append(nn.Linear(in_dim, 1))
        self.q1 = nn.Sequential(*layers_q1)
        layers_q2 = []
        in_dim = self.input_dim
        for h_dim in cfg.ROBOT.CRITIC_HIDDEN_DIMS:
            layers_q2.append(nn.Linear(in_dim, h_dim))
            layers_q2.append(nn.ReLU())
            in_dim = h_dim
        layers_q2.append(nn.Linear(in_dim, 1))
        self.q2 = nn.Sequential(*layers_q2)

    def forward(self, obs, action, pred_state):
        xu = torch.cat([obs, pred_state, action], dim=1)
        return self.q1(xu), self.q2(xu)