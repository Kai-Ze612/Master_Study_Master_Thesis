"""
Gymnasium Training environment with Autoregressive LSTM Integration.

LSTM: Output predicted q and qd
SAC: Learn to compensate torque based on predicted states

Goal: Min tracking error under delay with minimal torque compensation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from collections import deque
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import os

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import LocalRobotSimulator, TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.remote_robot_simulator import RemoteRobotSimulator
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

class TeleoperationEnvWithDelay(gym.Env):

    metadata = {'render_modes': ["human", "rgb_array"], 'render_fps': cfg.DEFAULT_CONTROL_FREQ}
    
    def __init__(
        self,
        delay_config: ExperimentConfig = ExperimentConfig.FULL_RANGE_COVER,
        trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
        randomize_trajectory: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        lstm_model_path: Optional[str] = cfg.LSTM_MODEL_PATH,
    ):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Rendering setup
        self.render_mode = render_mode
        self.viewer = None
        
        # RL/Env parameters
        self.max_episode_steps = cfg.MAX_EPISODE_STEPS
        self.control_freq = cfg.DEFAULT_CONTROL_FREQ
        self.current_step = 0
        
        # Robot parameters
        self.n_joints = cfg.N_JOINTS
        self.initial_qpos = cfg.INITIAL_JOINT_CONFIG.copy()
        self.max_joint_error = cfg.MAX_JOINT_ERROR_TERMINATION
        self.joint_limits_lower = cfg.JOINT_LIMITS_LOWER.copy()
        self.joint_limits_upper = cfg.JOINT_LIMITS_UPPER.copy()
        
        # Delay configuration
        self.delay_config = delay_config
        self.delay_simulator = DelaySimulator(control_freq=self.control_freq, config=delay_config, seed=seed)
        
        # Simulators
        self.leader = LocalRobotSimulator(trajectory_type=trajectory_type, randomize_params=randomize_trajectory)
        
        should_render_remote = (self.render_mode == "human")
        self.remote_robot = RemoteRobotSimulator(
            delay_config=delay_config,
            seed=seed,
            render=should_render_remote
        )
        
        # Buffers
        self.leader_q_history = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.remote_q_history = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        self.remote_qd_history = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        
        # Pre-computed Trajectory (for validation/ground truth)
        self._precomputed_trajectory_q: Optional[np.ndarray] = None
        self._precomputed_trajectory_qd: Optional[np.ndarray] = None
        
        # Warmup Logic
        self.warmup_steps = int(cfg.WARM_UP_DURATION * self.control_freq)
        self.steps_remaining_in_warmup = 0
        self.grace_period_steps = self.warmup_steps + int(cfg.NO_DELAY_DURATION * self.control_freq)
        
        ######################################################################
        # Load LSTM for state prediction
        self.lstm = None
        if lstm_model_path is not None and os.path.exists(lstm_model_path):
            self._load_lstm_model(lstm_model_path)
        else:
            pass # Training phase

        self._last_predicted_target: Optional[np.ndarray] = None
        self.max_ar_steps = cfg.MAX_AR_STEPS 

        # Normalized term for LSTM input
        self.prediction_ema = None
        self.ema_alpha = cfg.PREDICTION_EMA_ALPHA
        
        ######################################################################
        # Construct RL space
        self.action_space = spaces.Box(
            low=-cfg.MAX_TORQUE_COMPENSATION, high=cfg.MAX_TORQUE_COMPENSATION, 
            shape=(self.n_joints,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.OBS_DIM,), dtype=np.float32)
        
        ######################################################################
        self.last_target_q = None
        # Stats for info
        self._cached_prediction_error = 0.0
        self._cached_delay_steps = 0
        
    def _load_lstm_model(self, path):
        try:
            self.lstm = StateEstimator().to(self.device)
            # weights_only=False allows loading full checkpoints if needed
            checkpoint = torch.load(path, map_location=self.device)
            if 'state_estimator_state_dict' in checkpoint:
                self.lstm.load_state_dict(checkpoint['state_estimator_state_dict'])
            else:
                self.lstm.load_state_dict(checkpoint)
            self.lstm.eval()
            for p in self.lstm.parameters(): p.requires_grad = False
            print(f"[ENV] LSTM loaded: {path}")
        except Exception as e:
            print(f"[ENV] Warning: LSTM load failed: {e}")

    #######################################################################
    # Normalized and denormalized function 
    def _normalize_state(self, q, qd):
        """Normalize State (14D) to ~ N(0, 1)"""
        q_norm = (q - cfg.Q_MEAN) / cfg.Q_STD
        qd_norm = (qd - cfg.QD_MEAN) / cfg.QD_STD
        return np.concatenate([q_norm, qd_norm])

    def _normalize_input(self, q, qd, delay_scalar):
        """Normalize Input (15D = 14D State + 1D Delay)"""
        state_norm = self._normalize_state(q, qd)
        return np.concatenate([state_norm, [delay_scalar]])

    def _denormalize_state(self, pred_norm):
        """Denormalize State (14D) back to physical units"""
        q_norm = pred_norm[:7]
        qd_norm = pred_norm[7:]
        q = (q_norm * cfg.Q_STD) + cfg.Q_MEAN
        qd = (qd_norm * cfg.QD_STD) + cfg.QD_MEAN
        return np.concatenate([q, qd])
    #######################################################################
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset Signal Processing
        self.prediction_ema = None 
        self._last_predicted_target = None
        self._cached_prediction_error = 0.0
        self._cached_delay_steps = 0
        
        # Reset Leader & Precompute
        leader_start_q, _ = self.leader.reset(seed=seed)
        self._precompute_trajectory(leader_start_q)
        
        # Reset Remote
        self.remote_robot.reset(initial_qpos=self.initial_qpos)
        
        # Clear Buffers
        self.leader_q_history.clear()
        self.leader_qd_history.clear()
        self.remote_q_history.clear()
        self.remote_qd_history.clear()
        
        # Prefill Buffers (Warmup State)
        # We prefill with the starting position so the history isn't empty
        for _ in range(cfg.RNN_SEQUENCE_LENGTH + 20):
            self.leader_q_history.append(leader_start_q.copy())
            self.leader_qd_history.append(np.zeros(self.n_joints))
        
        start_q = leader_start_q.copy()
        for _ in range(cfg.REMOTE_HISTORY_LEN):
            self.remote_q_history.append(start_q.copy())
            self.remote_qd_history.append(np.zeros(self.n_joints))
            
        self.steps_remaining_in_warmup = self.warmup_steps
        self.last_target_q = start_q.copy()
        
        return self._get_observation(), {}

    def _precompute_trajectory(self, start_q):
        """Simulate the leader forward to get ground truth for the whole episode."""
        rollout_steps = self.max_episode_steps + cfg.ESTIMATOR_PREDICTION_HORIZON + 100
        
        # Save current leader state
        backup_state = (self.leader._q_current.copy(), self.leader._q_previous.copy(), 
                       self.leader._trajectory_time, self.leader._tick)
        
        temp_q = [start_q.copy()]
        temp_qd = [np.zeros(self.n_joints)]
        
        for _ in range(rollout_steps):
            q, qd, _, _, _, _ = self.leader.step()
            temp_q.append(q.copy())
            temp_qd.append(qd.copy())
            
        self._precomputed_trajectory_q = np.array(temp_q)
        self._precomputed_trajectory_qd = np.array(temp_qd)
        
        # Restore leader state
        self.leader._q_current = backup_state[0]
        self.leader._q_previous = backup_state[1]
        self.leader._trajectory_time = backup_state[2]
        self.leader._tick = backup_state[3]

    def step(self, action: np.ndarray):
        self.current_step += 1
        
        # 1. Step Leader
        new_leader_q, new_leader_qd, _, _, _, _ = self.leader.step()
        self.leader_q_history.append(new_leader_q.copy())
        self.leader_qd_history.append(new_leader_qd.copy())

        # 2. AR Prediction (Normalized + Smoothed)
        raw_prediction = self._perform_ar_prediction_step()
        
        # Apply EMA Filter to Denormalized Output
        if self.prediction_ema is None:
            self.prediction_ema = raw_prediction
        else:
            self.prediction_ema = self.ema_alpha * raw_prediction + (1.0 - self.ema_alpha) * self.prediction_ema
            
        self._last_predicted_target = self.prediction_ema

        # 3. Determine Safe Target for Remote
        if self.steps_remaining_in_warmup > 0:
            self.steps_remaining_in_warmup -= 1
            safe_target_q = self.initial_qpos.copy()
            torque_compensation = np.zeros(self.n_joints)
        else:
            raw_target_q = self._last_predicted_target[:self.n_joints]
            
            # Safety Clamp (prevent jumps > MAX_JOINT_CHANGE)
            if self.last_target_q is None: self.last_target_q = self.remote_robot.get_joint_state()[0]
            delta_q = raw_target_q - self.last_target_q
            clamped_delta_q = np.clip(delta_q, -cfg.MAX_JOINT_CHANGE_PER_STEP, cfg.MAX_JOINT_CHANGE_PER_STEP)
            
            safe_target_q = self.last_target_q + clamped_delta_q
            self.last_target_q = safe_target_q.copy()
            torque_compensation = action

        # 4. Step Remote Robot
        target_qd = self._last_predicted_target[self.n_joints:]
        self.remote_robot.step(
            safe_target_q, target_qd, torque_compensation, 
            true_local_q=self.get_true_current_target()[:self.n_joints]
        )
        
        # 5. Reward & Info
        reward, r_tracking = self._calculate_reward(action)
        remote_q, _ = self.remote_robot.get_joint_state()
        
        true_target_full = self.get_true_current_target()
        
        # Stats
        joint_error = np.linalg.norm(true_target_full[:self.n_joints] - remote_q)
        pred_error = np.linalg.norm(true_target_full[:self.n_joints] - self._last_predicted_target[:self.n_joints])
        self._cached_prediction_error = pred_error
        
        hist_len = len(self.leader_q_history)
        self._cached_delay_steps = 0 if self.current_step < self.grace_period_steps else \
                                   self.delay_simulator.get_observation_delay_steps(hist_len)

        terminated, term_penalty = self._check_termination(joint_error, pred_error, remote_q)
        reward += term_penalty
        # Modified: re-clip after termination penalty (safety net for NaN case)
        reward = float(np.clip(reward, cfg.REWARD_CLIP_MIN, cfg.REWARD_CLIP_MAX))
        
        truncated = self.current_step >= self.max_episode_steps
        if self.render_mode == "human": self.render()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _perform_ar_prediction_step(self) -> np.ndarray:
        """
        Performs Auto-Regressive Prediction.
        CRITICAL: Handles Normalization internally.
        """
        history_len = len(self.leader_q_history)
        
        # Grace Period: Perfect prediction
        if self.current_step < self.grace_period_steps:
            return np.concatenate([self.leader_q_history[-1], self.leader_qd_history[-1]])
        
        delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)
        
        # Fallback
        if delay_steps == 0 or self.lstm is None:
            idx = max(-1 - delay_steps, -len(self.leader_q_history))
            return np.concatenate([self.leader_q_history[idx], self.leader_qd_history[idx]])
            
        normalized_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        most_recent_idx = -1 - delay_steps
        
        # 1. Build Sequence of NORMALIZED inputs
        seq_buffer = []
        start_idx = most_recent_idx - cfg.RNN_SEQUENCE_LENGTH + 1
        for i in range(start_idx, most_recent_idx + 1):
            idx = max(-len(self.leader_q_history), i)
            # Normalize Q, Qd, Delay
            step_vec = self._normalize_input(
                self.leader_q_history[idx], 
                self.leader_qd_history[idx], 
                normalized_delay
            )
            seq_buffer.append(step_vec)
            
        input_tensor = torch.tensor(np.array(seq_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
        steps_to_predict = min(delay_steps, self.max_ar_steps)
        dt_norm_step = (1.0 / self.control_freq) / cfg.DELAY_INPUT_NORM_FACTOR
        
        with torch.no_grad():
            # Init Hidden State
            _, hidden = self.lstm.lstm(input_tensor)
            
            # Start AR loop from last frame of context
            curr_input = input_tensor[:, -1:, :]
            
            for _ in range(steps_to_predict):
                # Predict next state (Normalized)
                pred_state_norm, hidden = self.lstm.forward_step(curr_input, hidden)
                
                # Update Delay (Decrement)
                curr_delay_val = curr_input[0, 0, -1].item()
                next_delay_val = max(0.0, curr_delay_val - dt_norm_step)
                
                # Construct next input: [Pred State (Norm) + Next Delay]
                delay_t = torch.tensor([[[next_delay_val]]], device=self.device)
                curr_input = torch.cat([pred_state_norm, delay_t], dim=2)
        
        # 2. Denormalize Final Result
        final_pred_norm = curr_input.cpu().numpy()[0, 0, :14]
        return self._denormalize_state(final_pred_norm)

    def get_delayed_target_buffer(self, buffer_length: int) -> np.ndarray:
        """
        Returns NORMALIZED data buffer for LSTM training.
        This ensures the LSTM trainer receives data centered at 0 with std 1.
        """
        history_len = len(self.leader_q_history)
        delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)
        normalized_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        most_recent_idx = -1 - delay_steps
        
        buffer_seq = []
        for i in range(-buffer_length + 1, 1): 
            idx = np.clip(most_recent_idx + i, -history_len, -1)
            # NORMALIZE HERE
            step_vector = self._normalize_input(
                self.leader_q_history[idx],
                self.leader_qd_history[idx],
                normalized_delay
            )
            buffer_seq.append(step_vector)
            
        return np.array(buffer_seq).flatten().astype(np.float32)

    def get_future_target_sequence(self, horizon: int) -> np.ndarray:
        """
        Returns NORMALIZED future ground truth for LSTM loss calculation.
        """
        if self._precomputed_trajectory_q is None:
            # Fallback for initialization
            dummy = self._normalize_state(self.initial_qpos, np.zeros(self.n_joints))
            return np.tile(dummy, (horizon, 1)).flatten().astype(np.float32)

        history_len = len(self.leader_q_history)
        delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)
        start_idx = max(0, self.current_step - delay_steps + 1)
        
        targets = []
        for i in range(horizon):
            idx = min(start_idx + i, len(self._precomputed_trajectory_q) - 1)
            
            # Normalized target value
            state_norm = self._normalize_state(
                self._precomputed_trajectory_q[idx],
                self._precomputed_trajectory_qd[idx]
            )
            targets.append(state_norm)
            
        return np.array(targets).flatten().astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        """Constructs RL Observation vector."""
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        self.remote_q_history.append(remote_q.copy())
        self.remote_qd_history.append(remote_qd.copy())
        
        if self._last_predicted_target is None:
             self._last_predicted_target = self._perform_ar_prediction_step()
             
        pred_q = self._last_predicted_target[:self.n_joints]
        pred_qd = self._last_predicted_target[self.n_joints:]
        
        error_q = pred_q - remote_q
        error_qd = pred_qd - remote_qd
        
        rem_q_hist = np.concatenate(list(self.remote_q_history))
        rem_qd_hist = np.concatenate(list(self.remote_qd_history))
        
        hist_len = len(self.leader_q_history)
        d_steps = 0 if self.current_step < self.grace_period_steps else \
                    self.delay_simulator.get_observation_delay_steps(hist_len)
        norm_delay = float(d_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        
        obs = np.concatenate([
            remote_q, remote_qd, 
            rem_q_hist, rem_qd_hist,
            pred_q, pred_qd,
            error_q, error_qd,
            [norm_delay]
        ]).astype(np.float32)
        
        return obs

    def get_true_current_target(self) -> np.ndarray:
        if not self.leader_q_history:
            return np.concatenate([self.initial_qpos, np.zeros(self.n_joints)])
        return np.concatenate([self.leader_q_history[-1], self.leader_qd_history[-1]])

    def _calculate_reward(self, action: np.ndarray) -> Tuple[float, float]:
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        true_target = self.get_true_current_target()
        
        pos_err = true_target[:self.n_joints] - remote_q
        vel_err = true_target[self.n_joints:] - remote_qd
        
        # Position Tracking
        r_pos = -cfg.TRACKING_ERROR_SCALE * np.sum(pos_err**2)  # Tracking rewards
        # Velocity Tracking
        r_vel = -cfg.VELOCITY_ERROR_SCALE * np.sum(vel_err**2)  # Velocity rewards
        
        r_tracking = r_pos + r_vel
        
        # Penalty for large action
        r_action = -cfg.ACTION_PENALTY_WEIGHT * np.mean(action**2)
        
        # === Per-step safety penalty (Modified: implements paper Eq. 18) ===
        r_safe = 0.0
        
        # P_joint: joint limit violation
        at_limits = (np.any(remote_q <= self.joint_limits_lower + cfg.JOINT_LIMIT_MARGIN) or
                     np.any(remote_q >= self.joint_limits_upper - cfg.JOINT_LIMIT_MARGIN))
        if at_limits:
            r_safe += cfg.SAFETY_PENALTY_PER_STEP
        
        # P_track: tracking error exceeds threshold
        track_err_norm = np.linalg.norm(pos_err)
        if track_err_norm > cfg.SAFETY_TRACK_THRESHOLD:
            r_safe += cfg.SAFETY_PENALTY_PER_STEP
        
        # P_est: prediction divergence
        if self._last_predicted_target is not None:
            pred_q = self._last_predicted_target[:self.n_joints]
            true_q = true_target[:self.n_joints]
            pred_err_for_safe = np.linalg.norm(true_q - pred_q)
            if pred_err_for_safe > cfg.SAFETY_PRED_THRESHOLD:
                r_safe += cfg.SAFETY_PENALTY_PER_STEP
        # === End r_safe modification ===
        
        # === Reward clipping for training stability (Modified) ===
        total = r_tracking + r_action + r_safe
        total = float(np.clip(total, cfg.REWARD_CLIP_MIN, cfg.REWARD_CLIP_MAX))
        # === End clipping modification ===
        
        return total, float(r_tracking)

    def _check_termination(self, joint_error: float, prediction_error: float, remote_q: np.ndarray) -> Tuple[bool, float]:
        
        # NaN check (catastrophic case)
        if not np.all(np.isfinite(remote_q)): 
            return True, cfg.REWARD_CLIP_MIN  # Modified: was -100.0, now uses clip floor

        # Check joints at boundary
        at_limits = (np.any(remote_q <= self.joint_limits_lower + cfg.JOINT_LIMIT_MARGIN) or 
                    np.any(remote_q >= self.joint_limits_upper - cfg.JOINT_LIMIT_MARGIN))

        # Modified: use paper's threshold (0.5 rad) instead of self.max_joint_error (1.0)
        high_error = joint_error > cfg.SAFETY_TRACK_THRESHOLD
        
        # Modified: use cfg constant instead of magic number 0.3
        pred_divergence = prediction_error > cfg.SAFETY_PRED_THRESHOLD

        terminated = at_limits or high_error or pred_divergence
        
        # Modified: removed -10 termination penalty (avoid double-counting with per-step r_safe)
        penalty = 0.0
        return terminated, penalty

    def _get_info(self, phase="unknown"):
        is_warmup = self.current_step < self.grace_period_steps
        
        return {
            "prediction_error": self._cached_prediction_error,
            "current_delay_steps": self._cached_delay_steps,
            "is_in_warmup": is_warmup  # <--- CRITICAL FIX
        }

    def render(self):
        pass
        
    def close(self):
        if self.viewer is not None: plt.close(self.viewer)
