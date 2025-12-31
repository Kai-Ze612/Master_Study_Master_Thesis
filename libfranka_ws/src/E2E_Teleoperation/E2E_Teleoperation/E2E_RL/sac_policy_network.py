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
    Implements 'Inertial Memory Learning' and 'Autoregressive State Estimation'
    as described in the paper.
    """
    def __init__(self):
        super().__init__()
        
        # 1. LSTM Core (Inertial Memory)
        # Input: [q(7), qd(7), delay(1)] = 15
        self.lstm_cell = nn.LSTMCell(
            input_size=cfg.ROBOT.ESTIMATOR_INPUT_DIM, 
            hidden_size=cfg.ROBOT.RNN_HIDDEN_DIM
        )
        
        # 2. State Predictor Head (Decodes hidden state -> robot state)
        self.predictor = nn.Sequential(
            nn.Linear(cfg.ROBOT.RNN_HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, cfg.ROBOT.ROBOT_STATE_DIM) # Outputs 14: [q, qd]
        )
        
        # Internal params for de-normalizing delay inside the network
        self.delay_norm_factor = cfg.DELAY_INPUT_NORM_FACTOR

    def forward(self, history, hidden=None):
        """
        Args:
            history: (Batch, Seq_Len, 15) - Sequence of DELAYED observations.
                     The last element history[:, -1, :] is the 'Anchor Observation'.
            hidden:  Tuple (h, c) from previous step (optional)
        """
        batch_size, seq_len, _ = history.size()
        device = history.device
        
        # Initialize hidden state if not provided
        if hidden is None:
            h = torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM).to(device)
            c = torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM).to(device)
        else:
            h, c = hidden
        
        # --- A. Process the 'Anchor' History (Inertial Memory Learning) ---
        # We process the delayed sequence to build up the 'inertial memory' (hidden state)
        # equivalent to Eq (2) and (3) in your paper.
        
        # Note: Ideally, we should initialize with the PASSED 'hidden' state 
        # to maintain memory across standard steps, but for the 'Anchor' processing
        # of a full history window, we typically re-roll from scratch or use the 
        # previous state as the initial condition for the window.
        # Given your 'training_env' sends a full window every time, we roll the window.
        
        # Reset for the window rollout to ensure stability (Sliding Window logic)
        # But if you want true recurrence, you can try starting with 'h, c'.
        # For now, we stick to the stable "History -> Anchor" encoding:
        h_win, c_win = (torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM).to(device),
                        torch.zeros(batch_size, cfg.ROBOT.RNN_HIDDEN_DIM).to(device))
                        
        for t in range(seq_len):
            input_t = history[:, t, :]
            h_win, c_win = self.lstm_cell(input_t, (h_win, c_win))
        
        # Update our main hidden state to match the result of the window processing
        h, c = h_win, c_win

        # Current 'Anchor' Prediction based on delayed data
        anchor_pred_state = self.predictor(h) 
        
        # --- B. Autoregressive Rollout (Gap Filling) ---
        # This matches "Iterative recursive rollouts" in Sec 0.0.2
        
        # 1. Extract the delay magnitude from the last anchor input
        # Input format is [q, qd, norm_delay], so delay is at index -1
        norm_delay_tensor = history[:, -1, -1] # (Batch,)
        
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

        # Return:
        # h: The hidden state (inertial memory feature)
        # current_state: The autoregressively predicted state
        # (h, c): The tuple for the next step (if needed)
        return h, current_state, (h, c)


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
        # The LSTM now returns: h (feat), current_state, (h, c) (next_hidden)
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
        # log_prob -= log(1 - tanh(x)^2 + epsilon)
        log_prob = normal.log_prob(x_t)
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