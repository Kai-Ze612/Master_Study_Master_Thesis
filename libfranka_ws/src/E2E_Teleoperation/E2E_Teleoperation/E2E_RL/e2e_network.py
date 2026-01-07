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
    Autoregressive LSTM Encoder for Leader State Estimation.
    Input: Sequence of delayed observations [q(7), qd(7), norm_delay(1)] = 15 dims
    Output:
        - h: Hidden state (inertial memory feature)
        - predicted_state: Predicted current leader state [q(7), qd(7)] = 14 dims
        - (h, c): Tuple of hidden and cell states for next step
    """
    def __init__(self):
        super().__init__()

        # LSTM Cell
        # Why LSTMCell and not nn.LSTM?
        # Standard nn.LSTM is like a "Video Player": It requires the entire film 
        # (the full sequence of inputs) to be ready BEFORE you press play.
        #
        # However, we are doing "Autoregressive Prediction" (Time Travel):
        # 1. We predict step t+1.
        # 2. We use that prediction as the INPUT for step t+2.
        #
        # Since the inputs for t+2, t+3... don't exist yet, we cannot use nn.LSTM.
        # We must use LSTMCell to manually do prediction step by step.
        
        self.lstm_cell = nn.LSTMCell(
            input_size=cfg.ROBOT.ESTIMATOR_INPUT_DIM, 
            hidden_size=cfg.ROBOT.RNN_HIDDEN_DIM
        )
       
        # Decoder to map hidden state to robot state prediction
        self.predictor = nn.Sequential(
            nn.Linear(cfg.ROBOT.RNN_HIDDEN_DIM, cfg.ROBOT.LSTM_PRED_HEAD_DIM),
            nn.ReLU(),
            nn.Linear(cfg.ROBOT.LSTM_PRED_HEAD_DIM, cfg.ROBOT.ROBOT_STATE_DIM) 
        )
        
        self.delay_norm_factor = cfg.DELAY_INPUT_NORM_FACTOR

    def forward(self, history, hidden=None):
        batch_size, seq_len, _ = history.size()
        device = history.device
        if hidden is None:
            h = torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM, device=device)
            c = torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM, device=device)
        else:
            h, c = hidden
        
        # 1. Faster Window Processing
        # Process the history sequence using the LSTM cell
        h_win, c_win = h, c
        for t in range(seq_len):
            h_win, c_win = self.lstm_cell(history[:, t, :], (h_win, c_win))
        
        # 2. Optimized Autoregressive Rollout
        anchor_pred_state = self.predictor(h_win)
        norm_delay_tensor = history[:, -1, -1]
        steps_to_predict = (norm_delay_tensor * self.delay_norm_factor).ceil().long()
        
        # SAFETY CAP: Prevents infinite or extreme loops in HIGH_VARIANCE config
        max_steps = int(torch.clamp(steps_to_predict.max(), max=60).item())
        
        current_state = anchor_pred_state
        h, c = h_win, c_win
        
        # Pre-allocate dummy delay to avoid repeated tensor creation in loop
        dummy_delay = torch.zeros(batch_size, 1, device=device)
        
        for step_i in range(max_steps):
            mask = (step_i < steps_to_predict).float().unsqueeze(1)
            recur_input = torch.cat([current_state, dummy_delay], dim=1)
            h_next, c_next = self.lstm_cell(recur_input, (h, c))
            state_next = self.predictor(h_next)
            
            # Use masking to only update relevant batch elements
            h = mask * h_next + (1 - mask) * h
            c = mask * c_next + (1 - mask) * c
            current_state = mask * state_next + (1 - mask) * current_state
        
        return h, current_state, (h, c)

class JointActor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Base Policy (Frozen LSTM)
        self.base_encoder = LSTM()
        for param in self.base_encoder.parameters():
            param.requires_grad = False
            
        # 2. Policy Network
        self.input_dim = cfg.ROBOT.ACTOR_INPUT_DIM 
        layers = []
        in_dim = self.input_dim
        for h_dim in cfg.ROBOT.ACTOR_HIDDEN_DIMS:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.residual_net = nn.Sequential(*layers)
        
        # 3. Output Heads
        self.res_mu = nn.Linear(in_dim, cfg.ROBOT.N_JOINTS)
        self.res_log_std = nn.Linear(in_dim, cfg.ROBOT.N_JOINTS)
        
        # Initialize small to start near zero, but NOT strictly capped
        nn.init.uniform_(self.res_mu.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.res_mu.bias)
        nn.init.constant_(self.res_log_std.bias, -2.0)
        
        # Use scaled action output based on ID ourput
        self.register_buffer("action_scale", torch.tensor(cfg.ROBOT.TORQUE_LIMITS, dtype=torch.float32))
        
        self.log_std_min = cfg.ROBOT.LOG_STD_MIN
        self.log_std_max = cfg.ROBOT.LOG_STD_MAX

    def forward(self, obs, hidden=None):
        # A. Base Policy (No Grad)
        with torch.no_grad():
            target_seq_len = cfg.ROBOT.RNN_SEQ_LEN * cfg.ROBOT.ESTIMATOR_INPUT_DIM
            target_hist = obs[:, -7 - target_seq_len : -7]
            target_seq = target_hist.view(-1, cfg.ROBOT.RNN_SEQ_LEN, cfg.ROBOT.ESTIMATOR_INPUT_DIM)
            _, pred_state, next_hidden = self.base_encoder(target_seq, hidden)
            
        # B. Residual Policy (Trainable)
        idx_rem = cfg.ROBOT.ROBOT_STATE_DIM      
        remote_state = obs[:, :idx_rem]      
        prev_action = obs[:, -7:]      
        
        # Detach pred_state to stop gradients flowing into LSTM from Actor Loss
        rl_input = torch.cat([remote_state, pred_state.detach(), prev_action], dim=1)
        
        x = self.residual_net(rl_input)
        
        # Raw Mean (No Tanh here, we Tanh at the end)
        mu = self.res_mu(x)
        log_std = torch.clamp(self.res_log_std(x), self.log_std_min, self.log_std_max)
        
        return mu, log_std, pred_state, next_hidden

    def sample(self, obs, hidden=None):
        """
        Samples actions from the ouput(distribution) of the actor network.
        """
        mu, log_std, pred_state, next_hidden = self.forward(obs, hidden)
        std = log_std.exp()
        dist = Normal(mu, std)
        
        # 1. Sample from Normal
        x_t = dist.rsample()
        
        # 2. Squash to [-1, 1]
        y_t = torch.tanh(x_t)
        
        # 3. Scale to [Min_Torque, Max_Torque] (e.g., [-87, 87])
        final_action = y_t * self.action_scale
        
        # 4. Correct Log Prob (Jacobian Correction)
        log_prob = dist.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return final_action, log_prob, pred_state, next_hidden


class JointCritic(nn.Module):
    """
    Standard Twin Critic
    Input: Observation + Action + Predicted State
    Output: Q1, Q2 values
    """
    def __init__(self):
        super().__init__()
        
        self.input_dim = cfg.ROBOT.CRITIC_INPUT_DIM 
        
        # Q1
        layers_q1 = []
        in_dim = self.input_dim
        for h_dim in cfg.ROBOT.CRITIC_HIDDEN_DIMS:
            layers_q1.append(nn.Linear(in_dim, h_dim))
            layers_q1.append(nn.ReLU())
            in_dim = h_dim
        layers_q1.append(nn.Linear(in_dim, 1))
        self.q1 = nn.Sequential(*layers_q1)

        # Q2
        layers_q2 = []
        in_dim = self.input_dim
        for h_dim in cfg.ROBOT.CRITIC_HIDDEN_DIMS:
            layers_q2.append(nn.Linear(in_dim, h_dim))
            layers_q2.append(nn.ReLU())
            in_dim = h_dim
        layers_q2.append(nn.Linear(in_dim, 1))
        self.q2 = nn.Sequential(*layers_q2)

    def forward(self, obs, action, pred_state):
        
        idx_rem = cfg.ROBOT.ROBOT_STATE_DIM
        remote_state = obs[:, :idx_rem]
        prev_action = obs[:, -7:]
        
        # Construct full state-action pair
        # Shape: [Batch, 14 + 14 + 7 + 7] = 42 dims
        xu = torch.cat([remote_state, pred_state, prev_action, action], dim=1)
        
        return self.q1(xu), self.q2(xu)