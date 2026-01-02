"""
This module defines the policy network architecture used in the
End-to-End Teleoperation system based on Soft Actor-Critic (SAC).

Three main components are implemented:
1. Autoregressive LSTM Encoder: Estimates current leader state from delayed history.
2. Joint Actor Network: Outputs torque actions (Initialized via Inverse Dynamics).
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
        """
        Forward pass through the LSTM Encoder with Autoregressive Rollout.
        Input: history (Batch, Seq_Len, 15) - Sequence of DELAYED observations.
        Hidden state: (h, c) tuple or None for zero initialization.
        Output:
            - h: Hidden state (inertial memory feature) for decoder use
            - current_state: Autoregressively predicted current leader state (Batch, 14), for 
            - (h, c): Tuple of hidden and cell states for next step
        """
        
        # History sequence batch
        batch_size, seq_len, _ = history.size()
        device = history.device
       
       # Initialize hidden state if not provided
        if hidden is None:
            h = torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM).to(device)
            c = torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM).to(device)
        else:
            h, c = hidden
        
        # h and c window size
        h_win, c_win = (torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM).to(device),
                        torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM).to(device))
        
        # process the entire history sequence to update window hidden state                
        for t in range(seq_len):
            input_t = history[:, t, :]
            h_win, c_win = self.lstm_cell(input_t, (h_win, c_win))
        
        # Update main hidden state to match the result of the window processing
        h, c = h_win, c_win

        # Current 'Anchor' Prediction based on delayed data
        anchor_pred_state = self.predictor(h) 
        
        # 1. Extract the delay magnitude from the last anchor input
        # Input format is [q, qd, norm_delay], so delay is at index -1
        norm_delay_tensor = history[:, -1, -1]
        
        # Convert normalized delay back to approximate integer steps
        # We use .ceil() to ensure we cover the full time gap
        steps_to_predict = (norm_delay_tensor * self.delay_norm_factor).ceil().long()
        
        # We need the maximum delay in this batch to vectorise the loop
        max_steps = steps_to_predict.max().item()
        
        # Initialize current input with the Anchor State
        current_state = anchor_pred_state
        
        # Loop forward in time: t -> t + max_delay
        for step_i in range(max_steps):
            # Create a mask for batch elements that still need predicting
            # (i.e., those where step_i < their specific delay)
            mask = (step_i < steps_to_predict).float().unsqueeze(1) # (Batch, 1)
            
            # 2. Formulate Input for the "Imagined" Step
            # The input needs to be 15 dims: [Pred_q, Pred_qd, Remaining_Delay]
            # We approximate remaining delay as 0.0 for the rollout (or decaying)
            dummy_delay = torch.zeros(batch_size, 1).to(device)
            recur_input = torch.cat([current_state, dummy_delay], dim=1)
            
            # 3. Recursive LSTM Step (Eq 4 in paper)
            h_next, c_next = self.lstm_cell(recur_input, (h, c))
            state_next = self.predictor(h_next)
            
            # 4. Update State (Soft Update based on mask)
            # If this batch item still has delay gap, update h, c, and current_state.
            # If gap is closed, keep the old values (don't overshoot).
            h = mask * h_next + (1 - mask) * h
            c = mask * c_next + (1 - mask) * c
            current_state = mask * state_next + (1 - mask) * current_state

        return h, current_state, (h, c)

class JointActor(nn.Module):
    """
    Actor Network: Joint torque action
    Input: [Remote State (14), Predicted Leader State (14), Prev Action (7)] = 35 dims
    Output: Mean and Log Std of action distribution
    """
    LOG_STD_MIN = cfg.ROBOT.LOG_STD_MIN
    LOG_STD_MAX = cfg.ROBOT.LOG_STD_MAX
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
        # 14 (Remote) + 14 (Pred Leader) + 7 (Prev Action) = 35
        self.input_dim = 35 
        
        # --- Build MLP Explicitly ---
        layers = []
        in_dim = self.input_dim
        
        # Iterate through config list [512, 256]
        for h_dim in cfg.ROBOT.ACTOR_HIDDEN_DIMS:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.net = nn.Sequential(*layers)
        
        # --- Output Heads ---
        self.mu = nn.Linear(in_dim, cfg.ROBOT.N_JOINTS)
        self.log_std = nn.Linear(in_dim, cfg.ROBOT.N_JOINTS)
        
        self.register_buffer("scale", torch.tensor(cfg.ROBOT.TORQUE_LIMITS))

    def forward(self, obs, hidden=None):
        # 1. Parse Observation
        idx_rem = cfg.ROBOT.ROBOT_STATE_DIM      
        remote_state = obs[:, :idx_rem]          
        prev_action = obs[:, -7:]                
        
        target_seq_len = cfg.ROBOT.RNN_SEQ_LEN * cfg.ROBOT.ESTIMATOR_INPUT_DIM
        target_hist = obs[:, -7 - target_seq_len : -7]
        target_seq = target_hist.view(-1, cfg.ROBOT.RNN_SEQ_LEN, cfg.ROBOT.ESTIMATOR_INPUT_DIM)

        # 2. Get Prediction
        feat, pred_state, next_hidden = self.encoder(target_seq, hidden)

        # 3. Main Network
        x = torch.cat([remote_state, pred_state, prev_action], dim=1)
        x = self.net(x)
        
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mu, log_std, pred_state, next_hidden, feat

    def sample(self, obs, hidden=None):
        mu, log_std, pred_state, next_hidden, feat = self.forward(obs, hidden)
        
        std = log_std.exp()
        normal = Normal(mu, std)
        
        x_t = normal.rsample() 
        y_t = torch.tanh(x_t)
        action = y_t * self.scale
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob, pred_state, next_hidden, feat


class JointCritic(nn.Module):
    """
    Critic Network: Twin Q-Networks for SAC training.
    Input: [Predicted Leader State (14), Action (7)] = 21 dims]
    """
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        
        self.input_dim = cfg.ROBOT.ROBOT_STATE_DIM + cfg.ROBOT.N_JOINTS 

        # First Q-Network
        self.q1 = nn.Sequential(
            nn.Linear(self.input_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Second Q-Network
        self.q2 = nn.Sequential(
            nn.Linear(self.input_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, pred_state, action):
        # Concatenate predicted state and action
        xu = torch.cat([pred_state, action], dim=1)
       
        # Compute Q-values from both networks
        return self.q1(xu), self.q2(xu)
    
class JointCritic(nn.Module):
    """
    Critic Network: Twin Q-Networks
    Input: [Predicted Leader State (14), Action (7)] = 21 dims]
    Output: Two Q-Value estimates
    Goal: Evaluate state-action pairs for SAC training.
    """
    
    def __init__(self, encoder):
        super().__init__()
        
        self.encoder = encoder
        self.input_dim = cfg.ROBOT.ROBOT_STATE_DIM + cfg.ROBOT.N_JOINTS 

        # First Q-Network
        layers_q1 = []
        in_dim = self.input_dim
        for h_dim in cfg.CRITIC_HIDDEN_DIMS:
            layers_q1.append(nn.Linear(in_dim, h_dim))
            layers_q1.append(nn.ReLU())
            in_dim = h_dim
        layers_q1.append(nn.Linear(in_dim, 1)) # Final Output layer 
        self.q1 = nn.Sequential(*layers_q1) # Shape: [512, 256, 1]

        # Second Q-Network
        layers_q2 = []
        in_dim = self.input_dim
        for h_dim in cfg.CRITIC_HIDDEN_DIMS:
            layers_q2.append(nn.Linear(in_dim, h_dim))
            layers_q2.append(nn.ReLU())
            in_dim = h_dim
        layers_q2.append(nn.Linear(in_dim, 1)) # Final Output layer
        self.q2 = nn.Sequential(*layers_q2) # Shape: [512, 256, 1]

    def forward(self, pred_state, action):
        # Concatenate predicted state and action
        xu = torch.cat([pred_state, action], dim=1)
        # Feed through both Q-networks
        return self.q1(xu), self.q2(xu) 