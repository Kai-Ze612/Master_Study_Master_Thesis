"""
Create RL Training Environment with Delays
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import mujoco
from collections import deque
from typing import Tuple, Dict, Any

from E2E_Teleoperation.E2E_RL.local_robot_simulator import LocalRobotSimulator, TrajectoryType
from E2E_Teleoperation.E2E_RL.remote_robot_simulator import RemoteRobotSimulator
from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
import E2E_Teleoperation.config.robot_config as cfg


class TeleoperationEnv(gym.Env):
    metadata = {'render_modes': ["human", "rgb_array"], 'render_fps': cfg.CONTROL_FREQ}
    
    def __init__(
        self,
        delay_config=ExperimentConfig.HIGH_VARIANCE,
        trajectory_type=TrajectoryType.FIGURE_8,
        randomize_trajectory=False,
        seed=None,
        render_mode=None,
        simulate_obs_timing: bool = True
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = cfg.MAX_EPISODE_STEPS
        
        # 1. Simulators
        self.delay_simulator = DelaySimulator(cfg.CONTROL_FREQ, config=delay_config, seed=seed)
        self.leader = LocalRobotSimulator(trajectory_type=trajectory_type, randomize_params=randomize_trajectory)
        self.remote = RemoteRobotSimulator(delay_config=delay_config, seed=seed, render=(render_mode=="human"), verbose=False)
        
        # 2. Teacher Setup
        self._teacher_model = self.remote.model
        self._teacher_data = mujoco.MjData(self._teacher_model)
        self._prev_total_torque = np.zeros(cfg.N_JOINTS)
        
        # 3. Buffers
        self.leader_hist = deque(maxlen=200)
        self.remote_hist_q = deque(maxlen=cfg.RNN_SEQUENCE_LENGTH)
        self.remote_hist_qd = deque(maxlen=cfg.RNN_SEQUENCE_LENGTH)
        
        # 4. Spaces
        self.action_space = spaces.Box(
            low=-cfg.MAX_ACTION_TORQUE, 
            high=cfg.MAX_ACTION_TORQUE, 
            shape=(cfg.N_JOINTS,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(cfg.OBS_DIM,), 
            dtype=np.float32
        )
        
        # State
        self.step_count = 0
        self.initial_qpos = cfg.INITIAL_JOINT_CONFIG.copy()
        
        # Previous action for smoothness penalty
        self._prev_action = np.zeros(cfg.N_JOINTS)
        
        # Cumulative error for early termination
        self._cumulative_error = 0.0
        self._error_window = deque(maxlen=50)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._prev_total_torque = np.zeros(cfg.N_JOINTS)
        self._prev_action = np.zeros(cfg.N_JOINTS)
        self._cumulative_error = 0.0
        self._error_window.clear()
        
        # 1. Reset Robots
        l_q, _ = self.leader.reset(seed=seed)
        l_qd = np.zeros(cfg.N_JOINTS) 
        
        self.remote.reset(initial_qpos=self.initial_qpos)
        r_q = self.initial_qpos.copy()
        r_qd = np.zeros(cfg.N_JOINTS)
        
        # 2. Clear & Fill history
        self.leader_hist.clear()
        self.remote_hist_q.clear()
        self.remote_hist_qd.clear()
        
        init_state = (l_q.copy(), np.zeros(cfg.N_JOINTS))
        for _ in range(cfg.RNN_SEQUENCE_LENGTH + 50):
            self.leader_hist.append(init_state)
            self.remote_hist_q.append(self.initial_qpos.copy())
            self.remote_hist_qd.append(np.zeros(cfg.N_JOINTS))
            
        # 3. Calculate Initial Teacher Action
        # FIX: Pass zero acceleration for the initial step
        l_qdd_init = np.zeros(cfg.N_JOINTS)
        teacher_torque = self._compute_teacher_torque(r_q, r_qd, l_q, l_qd, l_qdd_init)
        
        dist = np.linalg.norm(l_q - r_q)

        info = {
            'teacher_action': teacher_torque,
            'true_q': l_q.copy(),
            'true_qd': l_qd.copy(),
            'remote_q': r_q.copy(),
            'remote_qd': r_qd.copy(),
            'tracking_error': dist,
            'true_state_vector': np.concatenate([l_q, l_qd]),
            'has_new_obs': True
        }
            
        return self._get_obs(), info

    def step(self, action):
        self.step_count += 1
        
        # FIX: Retrieve Previous Velocity BEFORE stepping (to calc acceleration)
        # leader_hist always has data due to reset
        prev_l_qd = self.leader_hist[-1][1]

        # 1. Step Leader
        l_q, l_qd, _, _, _, _ = self.leader.step()
        self.leader_hist.append((l_q.copy(), l_qd.copy()))
        
        # FIX: Calculate Leader Acceleration (Feedforward) using Finite Difference
        dt = 1.0 / cfg.CONTROL_FREQ
        l_qdd = (l_qd - prev_l_qd) / dt

        # 2. Step Remote (Apply Action)
        self.remote.step(target_q=l_q, target_qd=l_qd, torque_input=action)
        r_q, r_qd = self.remote.get_joint_state()
        
        # Update History
        self.remote_hist_q.append(r_q)
        self.remote_hist_qd.append(r_qd)
        
        # 3. Calculate Teacher (for Loss/Reference)
        # target_q is l_q, target_qd is l_qd
        target_q, target_qd = self.leader_hist[-1]
        
        # FIX: Pass calculated l_qdd to the teacher
        teacher_torque = self._compute_teacher_torque(r_q, r_qd, target_q, target_qd, l_qdd)
        
        # 4. IMPROVED REWARD CALCULATION
        reward, reward_info = self._compute_reward(
            target_q, target_qd, r_q, r_qd, action, teacher_torque
        )
        
        # 5. Termination Conditions
        pos_error = np.linalg.norm(target_q - r_q)
        self._error_window.append(pos_error)
        
        # Terminate if: single large error OR sustained high error
        single_error_term = pos_error > cfg.MAX_JOINT_ERROR_TERMINATION
        sustained_error_term = (
            len(self._error_window) >= 50 and 
            np.mean(self._error_window) > cfg.MAX_JOINT_ERROR_TERMINATION * 0.7
        )
        
        terminated = single_error_term or sustained_error_term
        truncated = self.step_count >= self.max_episode_steps
        
        # 6. Update previous action
        self._prev_action = action.copy()
        
        # 7. INFO DICT
        info = {
            'teacher_action': teacher_torque,
            'true_q': target_q.copy(),
            'true_qd': target_qd.copy(),
            'remote_q': r_q.copy(),
            'remote_qd': r_qd.copy(),
            'tracking_error': pos_error,
            'true_state_vector': np.concatenate([target_q, target_qd]),
            'has_new_obs': True,
            'reward_info': reward_info
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _compute_reward(self, target_q, target_qd, r_q, r_qd, action, teacher_action):
        """
        Multi-component reward for stable learning.
        """
        # Normalize by joint limits for scale-invariance
        pos_error = np.abs(target_q - r_q)
        vel_error = np.abs(target_qd - r_qd)
        
        # 1. Position Reward (exponential, bounded)
        pos_error_norm = np.mean(pos_error)
        r_position = np.exp(-3.0 * pos_error_norm)
        
        # 2. Velocity Reward
        vel_error_norm = np.mean(vel_error) / 2.0
        r_velocity = np.exp(-1.0 * vel_error_norm) * 0.3
        
        # 3. Action Penalty
        action_norm = np.linalg.norm(action) / np.linalg.norm(cfg.TORQUE_LIMITS)
        r_action = -0.01 * action_norm
        
        # 4. Smoothness Penalty
        action_diff = np.linalg.norm(action - self._prev_action)
        action_diff_norm = action_diff / (2 * np.linalg.norm(cfg.TORQUE_LIMITS))
        r_smooth = -0.02 * action_diff_norm
        
        # 5. Teacher Imitation Bonus
        teacher_diff = np.linalg.norm(action - teacher_action)
        teacher_diff_norm = teacher_diff / (2 * np.linalg.norm(cfg.TORQUE_LIMITS))
        r_teacher = 0.1 * np.exp(-2.0 * teacher_diff_norm)
        
        # Total reward
        total_reward = r_position + r_velocity + r_action + r_smooth + r_teacher
        
        reward_info = {
            'r_position': r_position,
            'r_velocity': r_velocity,
            'r_action': r_action,
            'r_smooth': r_smooth,
            'r_teacher': r_teacher,
            'total': total_reward
        }
        
        return total_reward, reward_info

    def _compute_teacher_torque(self, curr_q, curr_qd, des_q, des_qd, des_qdd):
        """
        Calculates ID torque using Feedforward Acceleration + PD Feedback.
        """
        kp, kd = cfg.TEACHER_KP, cfg.TEACHER_KD
        
        # FIX: Include des_qdd (Feedforward) in the command
        # This prevents lag and saturation during continuous movement
        qdd_cmd = des_qdd + kp * (des_q - curr_q) + kd * (des_qd - curr_qd)
        
        self._teacher_data.qpos[:7] = curr_q
        self._teacher_data.qvel[:7] = curr_qd
        self._teacher_data.qacc[:7] = qdd_cmd
        
        mujoco.mj_inverse(self._teacher_model, self._teacher_data)
        raw_torque = self._teacher_data.qfrc_inverse[:7].copy()
        
        # Smoothing
        alpha = cfg.TEACHER_SMOOTHING
        smoothed_torque = (1 - alpha) * raw_torque + alpha * self._prev_total_torque
        self._prev_total_torque = smoothed_torque
        
        # Clip to physical limits
        return np.clip(smoothed_torque, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)

    def _get_obs_sequence(self) -> np.ndarray:
        # Simplified delay for training
        delay_steps = self.delay_simulator.get_state_delay_steps(len(self.leader_hist))
        norm_delay = delay_steps / cfg.DELAY_INPUT_NORM_FACTOR
        
        target_seq = []
        end_idx = len(self.leader_hist) - 1 - delay_steps
        start_idx = max(0, end_idx - cfg.RNN_SEQUENCE_LENGTH + 1)
        
        for i in range(cfg.RNN_SEQUENCE_LENGTH):
            curr_idx = start_idx + i
            if 0 <= curr_idx < len(self.leader_hist):
                q, qd = self.leader_hist[curr_idx]
            else:
                q, qd = self.leader_hist[0]
            
            q_norm = (q - cfg.Q_MEAN) / cfg.Q_STD
            qd_norm = (qd - cfg.QD_MEAN) / cfg.QD_STD
            target_seq.extend(np.concatenate([q_norm, qd_norm, [norm_delay]]))
            
        return np.array(target_seq, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        # 1. Remote State
        r_q, r_qd = self.remote_hist_q[-1], self.remote_hist_qd[-1]
        state_norm = np.concatenate([
            (r_q - cfg.Q_MEAN) / cfg.Q_STD, 
            (r_qd - cfg.QD_MEAN) / cfg.QD_STD
        ])
        
        # 2. Remote History
        hist_seq = []
        for i in range(cfg.RNN_SEQUENCE_LENGTH):
            q = (self.remote_hist_q[i] - cfg.Q_MEAN) / cfg.Q_STD
            qd = (self.remote_hist_qd[i] - cfg.QD_MEAN) / cfg.QD_STD
            hist_seq.extend(np.concatenate([q, qd]))
            
        # 3. Target History
        target_seq = self._get_obs_sequence()
        
        return np.concatenate([state_norm, hist_seq, target_seq], dtype=np.float32)

    def close(self):
        self.remote.close()