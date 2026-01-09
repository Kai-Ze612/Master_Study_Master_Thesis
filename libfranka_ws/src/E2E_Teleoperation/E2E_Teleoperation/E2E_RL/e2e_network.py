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
    Autoregressive LSTM Encoder.
    Returns: 
        - h: The hidden state (Latent Vector)
        - predicted_state: The physical state prediction (Auxiliary)
        - (h, c): Next step states
    """
    def __init__(self):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(
            input_size=cfg.ROBOT.ESTIMATOR_INPUT_DIM, 
            hidden_size=cfg.ROBOT.RNN_HIDDEN_DIM
        )
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
        
        # 1. Process History Window
        h_win, c_win = h, c
        for t in range(seq_len):
            h_win, c_win = self.lstm_cell(history[:, t, :], (h_win, c_win))
        
        # 2. Autoregressive Rollout (To bridge delay gap)
        # We start rollout from the state "h_win" (state at t_now)
        anchor_pred_state = self.predictor(h_win)
        norm_delay_tensor = history[:, -1, -1]
        steps_to_predict = (norm_delay_tensor * self.delay_norm_factor).ceil().long()
        max_steps = int(torch.clamp(steps_to_predict.max(), max=60).item())
        
        current_state = anchor_pred_state
        h, c = h_win, c_win
        dummy_delay = torch.zeros(batch_size, 1, device=device)
        
        for step_i in range(max_steps):
            mask = (step_i < steps_to_predict).float().unsqueeze(1)
            recur_input = torch.cat([current_state, dummy_delay], dim=1)
            h_next, c_next = self.lstm_cell(recur_input, (h, c))
            state_next = self.predictor(h_next)
            
            # Mask updates
            h = mask * h_next + (1 - mask) * h
            c = mask * c_next + (1 - mask) * c
            current_state = mask * state_next + (1 - mask) * current_state
        
        # Return h (Latent Embedding at predicted time) and current_state (Physical Prediction)
        return h, current_state, (h, c)

class JointActor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Base Encoder (LSTM)
        self.base_encoder = LSTM()
        for param in self.base_encoder.parameters():
            param.requires_grad = True
        
        # --- FIX: EXPOSE AUX_HEAD FOR OPTIMIZER ---
        # The trainer expects 'self.actor.aux_head'.
        # Since LSTM already has a predictor, we alias it here.
        self.aux_head = self.base_encoder.predictor
            
        # 2. Policy Network (Latent Actor)
        # Input: Remote(14) + Latent(256) + PrevAction(7)
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
        
        # Init
        nn.init.xavier_uniform_(self.res_mu.weight, gain=1.0)
        nn.init.zeros_(self.res_mu.bias)
        nn.init.constant_(self.res_log_std.bias, -2.0)
        
        self.register_buffer("action_scale", torch.tensor(cfg.ROBOT.TORQUE_LIMITS, dtype=torch.float32))
        self.log_std_min = cfg.ROBOT.LOG_STD_MIN
        self.log_std_max = cfg.ROBOT.LOG_STD_MAX

    def forward(self, obs, hidden=None):
        # A. Get Latent Embedding
        target_seq_len = cfg.ROBOT.RNN_SEQ_LEN * cfg.ROBOT.ESTIMATOR_INPUT_DIM
        target_hist = obs[:, -7 - target_seq_len : -7]
        target_seq = target_hist.view(-1, cfg.ROBOT.RNN_SEQ_LEN, cfg.ROBOT.ESTIMATOR_INPUT_DIM)
        
        # LSTM returns: latent_vector(h), pred_state, next_hidden
        latent_vector, pred_state, next_hidden = self.base_encoder(target_seq, hidden)
            
        # B. Policy Forward
        idx_rem = cfg.ROBOT.ROBOT_STATE_DIM      
        remote_state = obs[:, :idx_rem]      
        prev_action = obs[:, -7:]      
        
        # REAL E2E INPUT: Remote + LATENT + PrevAction
        rl_input = torch.cat([remote_state, latent_vector, prev_action], dim=1)
        
        x = self.residual_net(rl_input)
        mu = self.res_mu(x)
        log_std = torch.clamp(self.res_log_std(x), self.log_std_min, self.log_std_max)
        
        return mu, log_std, pred_state, next_hidden

    def sample(self, obs, hidden=None):
        mu, log_std, pred_state, next_hidden = self.forward(obs, hidden)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        final_action = y_t * self.action_scale
        
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return final_action, log_prob, pred_state, next_hidden


class JointCritic(nn.Module):
    """
    JointCritic
    Input: Remote(14) + Predicted_State(14) + PrevAction(7) + CurrAction(7)
    """
    def __init__(self):
        super().__init__()
        
        # Dimensions must match robot_config.py: CRITIC_INPUT_DIM = 14 + 14 + 7 + 7
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
        
        # We concatenate [Remote, Pred, PrevAction, CurrAction]
        xu = torch.cat([remote_state, pred_state, prev_action, action], dim=1)
        
        return self.q1(xu), self.q2(xu)