"""
This script defines the network architecture and forward pass used in the End-to-End training.

Three main components are implemented:
1. Autoregressive LSTM Encoder: Estimates current leader state from delayed history states.
2. Joint Actor Network: Outputs torque actions.
3. Joint Critic Network: Evaluates state-action pairs for SAC training.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import E2E_Teleoperation.config.robot_config as cfg

class LSTM(nn.Module):
    """
    Predicts CURRENT LEADER from DELAYED LEADER History.
    Input: [Leader(15)]
    Output: Latent, PredLeader(14)
    """
    def __init__(self):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(
            input_size=cfg.ROBOT.ESTIMATOR_INPUT_DIM, # 15
            hidden_size=cfg.ROBOT.RNN_HIDDEN_DIM
        )
        self.predictor = nn.Sequential(
            nn.Linear(cfg.ROBOT.RNN_HIDDEN_DIM, cfg.ROBOT.LSTM_PRED_HEAD_DIM),
            nn.ReLU(),
            nn.Linear(cfg.ROBOT.LSTM_PRED_HEAD_DIM, cfg.ROBOT.ESTIMATOR_OUTPUT_DIM) # 14
        )
        self.delay_norm_factor = cfg.DELAY_INPUT_NORM_FACTOR

        # Ensure gradients flow (Safety check)
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, history, hidden=None):
        # history: [Batch, Seq_Len, 15] (Sliced to only include Leader)
        batch_size, seq_len, _ = history.size()
        device = history.device
        if hidden is None:
            h = torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM, device=device)
            c = torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM, device=device)
        else:
            h, c = hidden
        
        # 1. Process History Window
        h_win, c_win = h, c
        for t in range(seq_len):
            h_win, c_win = self.lstm_cell(history[:, t, :], (h_win, c_win))
        
        # 2. Autoregressive Rollout (Bridge the Delay)
        anchor_pred_state = self.predictor(h_win)
        
        # Delay is at index 14
        norm_delay_tensor = history[:, -1, 14] 
        steps_to_predict = (norm_delay_tensor * self.delay_norm_factor).ceil().long()
        max_steps = int(torch.clamp(steps_to_predict.max(), max=cfg.ROBOT.MAX_PREDICTION_ROLLOUT_STEPS).item())
        
        current_state = anchor_pred_state # [Batch, 14]
        h, c = h_win, c_win
        
        # --- FIX: Use ACTUAL delay for conditioning, not zeros ---
        # This fixes the "Frozen Output" by making physics consistent
        dummy_delay = norm_delay_tensor.unsqueeze(1)
        
        for step_i in range(max_steps):
            mask = (step_i < steps_to_predict).float().unsqueeze(1)
            
            # Recur Input: [PredLeader(14), Delay(1)]
            recur_input = torch.cat([current_state, dummy_delay], dim=1)
            
            h_next, c_next = self.lstm_cell(recur_input, (h, c))
            state_next = self.predictor(h_next)
            
            h = mask * h_next + (1 - mask) * h
            c = mask * c_next + (1 - mask) * c
            current_state = mask * state_next + (1 - mask) * current_state
        
        return h, current_state, (h, c)

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
        # Slice Obs
        # 0-14: Real Follower State
        real_follower_state = obs[:, :14]
        
        idx_state_end = 14
        idx_target_end = idx_state_end + self.target_seq_len
        idx_act_hist_end = idx_target_end + self.action_hist_len
        
        target_hist_flat = obs[:, idx_state_end:idx_target_end]
        action_hist_flat = obs[:, idx_target_end:idx_act_hist_end]
        prev_action = obs[:, idx_act_hist_end:]

        # Reshape Full History: [Batch, 50, 36]
        full_seq = target_hist_flat.view(-1, cfg.ROBOT.RNN_SEQ_LEN, 36)
        
        # EXTRACT LEADER ONLY: [Leader(14), qd(7), Delay(1)] -> Indices 0-15
        leader_seq = full_seq[:, :, :15]
        
        # LSTM predicts Leader
        latent_vector, pred_leader, next_hidden = self.base_encoder(leader_seq, hidden)
            
        # Actor Input: [RealFollower, Latent, ActionHist, PrevAct]
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