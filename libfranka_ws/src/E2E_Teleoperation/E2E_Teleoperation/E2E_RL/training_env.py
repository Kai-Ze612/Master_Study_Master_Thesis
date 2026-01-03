"""
Create RL Training Environment with Delays
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from E2E_Teleoperation.E2E_RL.leader_robot_simulator import LeaderRobotSimulator, TrajectoryType
from E2E_Teleoperation.E2E_RL.follower_robot_simulator import FollowerRobotSimulator
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
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = cfg.MAX_EPISODE_STEPS
        
        # 1. Simulators
        self.delay_simulator = DelaySimulator(cfg.CONTROL_FREQ, config=delay_config, seed=seed)
        self.leader = LeaderRobotSimulator(trajectory_type=trajectory_type, randomize_params=randomize_trajectory)
        self.follower = FollowerRobotSimulator(delay_config=delay_config, seed=seed, render=(render_mode=="human"), verbose=False)
        
        # 2. Buffers
        self.leader_hist = deque(maxlen=200)
        self.follower_hist_q = deque(maxlen=cfg.RNN_SEQUENCE_LENGTH)
        self.follower_hist_qd = deque(maxlen=cfg.RNN_SEQUENCE_LENGTH)
        
        # 3. Spaces
        self.action_space = spaces.Box(-cfg.MAX_ACTION_TORQUE, cfg.MAX_ACTION_TORQUE, shape=(cfg.N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(cfg.RL_OBS_DIM,), dtype=np.float32)
        
        self.step_count = 0
        self.initial_qpos = cfg.INITIAL_JOINT_CONFIG.copy()
        self._prev_action = np.zeros(cfg.N_JOINTS)
        
        # Current Leader State for Expert Calculation
        self._curr_leader_state = None 
        
        # [NEW] Curriculum Knob (1.0 = Full Difficulty, 0.0 = Real-time)
        # Default to 1.0 (Hard) so standard eval works, Trainer will lower it if needed.
        self.curriculum_scale = 1.0 

    # [NEW] Method to update difficulty from Trainer
    def set_curriculum_difficulty(self, scale: float):
        """
        Sets the difficulty of the delay.
        scale: 0.0 (No Delay) -> 1.0 (Full Configured Delay)
        """
        self.curriculum_scale = np.clip(scale, 0.0, 1.0)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._prev_action = np.zeros(cfg.N_JOINTS)
        
        # 1. Reset Robots
        l_q, _ = self.leader.reset(seed=seed)
        
        self.follower.reset(initial_qpos=self.initial_qpos)
        f_q = self.initial_qpos.copy()
        f_qd = np.zeros(cfg.N_JOINTS)
        
        # 2. Clear & Fill history
        self.leader_hist.clear()
        self.follower_hist_q.clear()
        self.follower_hist_qd.clear()
        
        init_state = (l_q.copy(), np.zeros(cfg.N_JOINTS))
        self._curr_leader_state = (l_q.copy(), np.zeros(cfg.N_JOINTS), np.zeros(cfg.N_JOINTS))
        
        for _ in range(cfg.RNN_SEQUENCE_LENGTH + 50):
            self.leader_hist.append(init_state)
            self.follower_hist_q.append(self.initial_qpos.copy())
            self.follower_hist_qd.append(np.zeros(cfg.N_JOINTS))
            
        info = {
            'leader_q': l_q.copy(),
            'follower_q': f_q.copy(),
            'true_state_vector': np.concatenate([l_q, np.zeros(cfg.N_JOINTS)]),
        }
        return self._get_obs(), info

    def get_expert_action(self):
        l_q, l_qd, l_qdd = self._curr_leader_state
        expert_torque = self.follower.compute_inverse_dynamics(l_q, l_qd, l_qdd)
        return expert_torque

    def set_action_delay_enabled(self, enabled: bool):
        if hasattr(self.follower, 'action_delay_enabled'):
            self.follower.action_delay_enabled = enabled

    def step(self, action):
        self.step_count += 1
        
        # 1. Step Leader
        l_q, l_qd, l_qdd, _, _, _, _ = self.leader.step()
        self.leader_hist.append((l_q.copy(), l_qd.copy()))
        self._curr_leader_state = (l_q, l_qd, l_qdd)
        
        # 2. Step Follower
        self.follower.step(torque_input=action)
        f_q, f_qd = self.follower.get_joint_state()
        
        self.follower_hist_q.append(f_q)
        self.follower_hist_qd.append(f_qd)
        
        # 3. Retrieve Targets
        target_q, target_qd = self.leader_hist[-1]
        
        # 4. Compute Base Reward
        reward, reward_info = self._compute_reward(target_q, target_qd, f_q, f_qd, action)
        
        # 5. Check Termination
        terminated, term_reason, term_penalty = self._check_termination(f_q, f_qd, target_q)
        
        if terminated:
            reward += term_penalty

        truncated = self.step_count >= self.max_episode_steps
        
        self._prev_action = action.copy()
        expert_action = self.get_expert_action()
        
        info = {
            'leader_q': target_q.copy(),
            'follower_q': f_q.copy(),
            'true_state_vector': np.concatenate([target_q, target_qd]),
            'expert_action': expert_action,
            'termination_reason': term_reason 
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _compute_reward(self, target_q, target_qd, r_q, r_qd, action):
        # 1. Position Error (Mean absolute error)
        pos_error = np.mean(np.abs(target_q - r_q))
        
        # 2. Velocity Error (Scaled down)
        vel_error = np.mean(np.abs(target_qd - r_qd)) / 2.0
        
        # 3. Component Rewards (Exponential decay)
        r_pos = np.exp(-3.0 * pos_error)
        r_vel = np.exp(-1.0 * vel_error) * 0.3
        
        # 4. Action Penalty
        action_norm = np.linalg.norm(action) / np.linalg.norm(cfg.TORQUE_LIMITS)
        r_act = -0.01 * action_norm
        
        # --- MODIFICATION START ---
        # 5. Global Scaling
        # We multiply by 0.05 so the max reward per step is ~0.065
        # Max Episode Reward becomes ~65 (Perfect stability range for SAC)
        reward_scale = 0.05
        total_reward = (r_pos + r_vel + r_act) * reward_scale
        # --- MODIFICATION END ---
        
        return total_reward, {
            "r_pos": r_pos, "r_vel": r_vel, "r_act": r_act, "err_pos": pos_error
        }

    def _check_termination(self, f_q, f_qd, target_q):
        """
        [MODIFIED] Stronger penalties (-200.0) to prevent lazy survival strategies.
        """
        # Condition 1: Non-Numerical (Simulation Divergence)
        if np.any(np.isnan(f_q)) or np.any(np.isinf(f_q)) or \
           np.any(np.isnan(f_qd)) or np.any(np.isinf(f_qd)):
            return True, "NaN_Simulation_Divergence", -200.0

        # Condition 2: Joint Limits
        if np.any(f_q < cfg.JOINT_LIMITS_LOWER) or np.any(f_q > cfg.JOINT_LIMITS_UPPER):
            return True, "Joint_Limit_Violation", -200.0

        # Condition 3: Max Joint Error (Safety Stop)
        if np.max(np.abs(target_q - f_q)) > cfg.MAX_JOINT_ERROR_TERMINATION:
            return True, "Max_Tracking_Error_Exceeded", -200.0

        return False, "None", 0.0
    
    def _get_obs_sequence(self) -> np.ndarray:
        # State Delay Simulation
        raw_delay_steps = self.delay_simulator.get_state_delay_steps(len(self.leader_hist))
        
        # [MODIFIED] Curriculum Scaling
        # If scale is 0.0 -> delay is 0. If scale is 1.0 -> full delay.
        delay_steps = int(raw_delay_steps * self.curriculum_scale)
        
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
        # 1. Follower State (14)
        f_q, f_qd = self.follower_hist_q[-1], self.follower_hist_qd[-1]
        state_norm = np.concatenate([(f_q - cfg.Q_MEAN)/cfg.Q_STD, (f_qd - cfg.QD_MEAN)/cfg.QD_STD])
        
        # --- DELETE THIS BLOCK START ---
        # hist_seq = []
        # for i in range(cfg.RNN_SEQ_LEN):
        #     idx = -1 - i
        #     q = self.follower_hist_q[idx]
        #     qd = self.follower_hist_qd[idx]
        #     norm_q = (q - cfg.Q_MEAN)/cfg.Q_STD
        #     norm_qd = (qd - cfg.QD_MEAN)/cfg.QD_STD
        #     hist_seq.append(np.concatenate([norm_q, norm_qd]))
        # hist_seq = np.concatenate(hist_seq)
        # --- DELETE THIS BLOCK END ---

        # 2. Leader History (1200)
        target_seq = self._get_obs_sequence()

        # 3. Prev Action (7)
        prev_action_norm = self._prev_action / cfg.MAX_ACTION_TORQUE
        
        # Concatenate ONLY: [State, LeaderHistory, Action]
        # This will result in shape (1221,)
        return np.concatenate([state_norm, target_seq, prev_action_norm], dtype=np.float32)

    def close(self):
        self.follower.close()