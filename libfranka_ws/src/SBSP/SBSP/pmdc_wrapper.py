"""
PMDC Wrapper for SBSP: Uses DCNN to predict undelayed state
"""
import gymnasium as gym
import torch
import numpy as np
from collections import deque
import SBSP.config.robot_config as cfg
from SBSP.sbsp_network import DCNN

class PMDCWrapper(gym.Wrapper):
    def __init__(self, env, dcnn_model: DCNN, device):
        super().__init__(env)
        self.env = env
        self.dcnn = dcnn_model
        self.device = device
        self.dcnn.eval()
        self.dcnn.to(device)
        
        # Dimensions for Leader State (Pos 7 + Vel 7 + Delay 1 = 15)
        # We only predict Pos(7) + Vel(7) = 14 dims
        self.leader_dim = 14 
        
        # Buffer to perform recursive prediction
        self.prediction_buffer = deque(maxlen=cfg.ROBOT.FRAME_STACK)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # On reset, we don't have history to roll forward significantly, 
        # or we assume start state is known.
        # We just pass the obs through for the first step, or perform 0-step prediction.
        return self._predict_and_replace(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get the delay estimate from the last observation frame
        # Obs structure: [Leader(15), Follower(14), Action(14)] repeated FRAME_STACK times
        # The last element of Leader vec is delay.
        
        predicted_obs = self._predict_and_replace(obs)
        return predicted_obs, reward, terminated, truncated, info

    def _predict_and_replace(self, stacked_obs):
        """
        Extracts delayed leader state, predicts current state using DCNN, 
        and replaces it in the observation.
        """
        # 1. Parse Stacked Obs
        # Shape: (43 * FRAME_STACK,)
        frame_size = 43
        n_frames = cfg.ROBOT.FRAME_STACK
        
        frames = np.split(stacked_obs, n_frames)
        new_frames = []

        for frame in frames:
            # Frame: [Leader(15) | Follower(14) | Action(14)]
            # Leader: [Q(7), QD(7), Delay(1)]
            leader_part = frame[:15]
            follower_part = frame[15:29]
            action_part = frame[29:]
            
            # Extract Delay
            delay_sec = leader_part[14]
            # Convert delay to steps (approx)
            delay_steps = int(delay_sec * cfg.CONTROL_FREQ)
            
            if delay_steps > 0:
                # 2. Prepare for Prediction
                # Input to DCNN: State (14) + Action (Dummy 0 for Leader)
                current_state = torch.tensor(leader_part[:14], dtype=torch.float32, device=self.device).unsqueeze(0)
                dummy_action = torch.zeros((1, 1), dtype=torch.float32, device=self.device) # Action dim 1 for DCNN if we simplified it, or 0? 
                # Note: In train_sbsp we must ensure DCNN is init with correct dims.
                # Assuming DCNN takes 14 state + 0 action (since leader is auto).
                
                # 3. Recursive Rollout
                # Predict 'delay_steps' into the future
                with torch.no_grad():
                    pred_state = current_state
                    for _ in range(delay_steps):
                        # For Leader, action is irrelevant, pass zeros
                        pred_state = self.dcnn(pred_state, dummy_action)
                
                # 4. Reconstruct Frame
                pred_np = pred_state.cpu().numpy().squeeze()
                
                # Replace Q and QD with predicted
                new_leader_part = np.concatenate([pred_np, [0.0]]) # Set delay to 0 in obs since it's "corrected"
            else:
                new_leader_part = leader_part # No delay

            new_frame = np.concatenate([new_leader_part, follower_part, action_part])
            new_frames.append(new_frame)
            
        return np.concatenate(new_frames).astype(np.float32)