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

        # Determining the starting line.
        # This is the last known truth state (Anchor State)
        anchor_pred_state = self.predictor(h) 
        
        # Extract the delay magnitude from the last anchor input
        norm_delay_tensor = history[:, -1, -1]
        
        # Checking the timestamps.
        steps_to_predict = (norm_delay_tensor * self.delay_norm_factor).ceil().long()
        max_steps = steps_to_predict.max().item()
        
        # Initialize current input with the Anchor State
        current_state = anchor_pred_state
        
        # Autoregressive Prediction Loop
        for step_i in range(max_steps):
            
            # Create a mask for batch elements that still need predicting
            mask = (step_i < steps_to_predict).float().unsqueeze(1) # (Batch, 1)  # The individualized Stop Sign.
            dummy_delay = torch.zeros(batch_size, 1).to(device)
            
            # Self-Feeding
            recur_input = torch.cat([current_state, dummy_delay], dim=1)
            
            # Recursive LSTM Step
            h_next, c_next = self.lstm_cell(recur_input, (h, c))
            state_next = self.predictor(h_next)
            
            # Update State
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
       
    def __init__(self):
        super().__init__()
        self.encoder = LSTM()
                
        self.LOG_STD_MIN = cfg.ROBOT.LOG_STD_MIN
        self.LOG_STD_MAX = cfg.ROBOT.LOG_STD_MAX
        
        self.input_dim = cfg.ACTOR_INPUT_DIM  # 35 dims
        layers = []
       
        # Iterate through config list [512, 256]
        for h_dim in cfg.ROBOT.ACTOR_HIDDEN_DIMS:
            layers.append(nn.Linear(self.input_dim, h_dim))
            layers.append(nn.ReLU())
            self.input_dim = h_dim
            
        self.net = nn.Sequential(*layers)
        
        # Decoder layers for mean and log std
        self.mu = nn.Linear(self.input_dim, cfg.ROBOT.N_JOINTS)  # mean action 
        self.log_std = nn.Linear(self.input_dim, cfg.ROBOT.N_JOINTS) # log std

        # Scale factor for action output (torque limits)
        self.register_buffer("scale", torch.tensor(cfg.ROBOT.TORQUE_LIMITS))

    def forward(self, obs, hidden=None):
        
        # Extract components from observation
        idx_rem = cfg.ROBOT.ROBOT_STATE_DIM      
        remote_state = obs[:, :idx_rem]      
        prev_action = obs[:, -7:]                
        
        # Extract components for LSTM Encoder
        target_seq_len = cfg.ROBOT.RNN_SEQ_LEN * cfg.ROBOT.ESTIMATOR_INPUT_DIM
        target_hist = obs[:, -7 - target_seq_len : -7]
        target_seq = target_hist.view(-1, cfg.ROBOT.RNN_SEQ_LEN, cfg.ROBOT.ESTIMATOR_INPUT_DIM)

        # LSTM output: h,current state,next_hidden
        feat, pred_state, next_hidden = self.encoder(target_seq, hidden)

        # Main Network
        x = torch.cat([remote_state, pred_state, prev_action], dim=1) # RL observation space
        
        # Get action distribution
        x = self.net(x) # Feed into action network
        mu = self.mu(x) # Get mean 
        log_std = torch.clamp(self.log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX) # Get log std
        
        return mu, log_std, pred_state, next_hidden, feat

    def sample(self, obs, hidden=None):
        """
        Get physical action by sampling from the policy distribution.
        Uses reparameterization trick for backpropagation.
        output:
            - action: Sampled action after tanh squashing and scaling
            - log_prob: Log probability of the sampled action
        """
        
        # Get action distribution parameters
        mu, log_std, pred_state, next_hidden, feat = self.forward(obs, hidden)
        
        # Reparameterization trick to sample action
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
    Critic Network: Twin Q-Networks
    Input: [Predicted Leader State (14), Action (7)] = 21 dims]
    Output: Two Q-Value estimates
    Goal: Evaluate state-action pairs for SAC training.
    """
    
    def __init__(self):
        super().__init__()
        
        self.input_dim = cfg.CRITIC_INPUT_DIM  # 21 dims

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