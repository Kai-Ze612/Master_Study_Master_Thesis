"""
Training Environment

Pipeline for End-to-End Teleoperation RL:
1. Leader Robot Simulator: True robot state (desired trajectory generation)
2. Delay Simulator: Simulated communication delay
3. E2E model output control torque
4. Follower Robot Simulator: Simulated robot with dynamics using the torque
5. Calculate reward
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
        self.max_episode_steps = cfg.ROBOT.MAX_EPISODE_STEPS
        
        # Enable Delay Simulator, Leader, and Follower robots
        self.delay_simulator = DelaySimulator(cfg.CONTROL_FREQ, config=delay_config, seed=seed)
        self.leader = LeaderRobotSimulator(trajectory_type=trajectory_type, randomize_params=randomize_trajectory)
        self.follower = FollowerRobotSimulator(delay_config=delay_config, seed=seed, render=(render_mode=="human"), verbose=False)
        
        # initial buffers
        self.leader_hist = deque(maxlen=200)
        self.follower_hist_q = deque(maxlen=cfg.ROBOT.RNN_SEQ_LEN)
        self.follower_hist_qd = deque(maxlen=cfg.ROBOT.RNN_SEQ_LEN)
        
        # Observation and Action Spaces
        self.action_space = spaces.Box(-cfg.ROBOT.MAX_ACTION_TORQUE, cfg.ROBOT.MAX_ACTION_TORQUE, shape=(cfg.ROBOT.N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(cfg.ROBOT.RL_OBS_DIM,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment and training parameters.
        """
        super().reset(seed=seed)
        
        # Internal step counter
        self.step_count = 0

        # Previous action history
        self._prev_action = np.zeros(cfg.N_JOINTS)
        # Initial parameters
        self.initial_qpos = cfg.INITIAL_JOINT_CONFIG.copy()
        # Current leader state
        self._curr_leader_state = None 
        
        # Reset follower Robot
        self.follower.reset(initial_qpos=self.initial_qpos)
        f_q = self.initial_qpos.copy()
        f_qd = np.zeros(cfg.N_JOINTS)

        # Clear history buffers
        self.leader_hist.clear()
        self.follower_hist_q.clear()
        self.follower_hist_qd.clear()
        
        # Reset leader robot
        l_q, _ = self.leader.reset(seed=seed) # get initial leader state
        init_state = (l_q.copy(), np.zeros(cfg.N_JOINTS))
        self._curr_leader_state = (l_q.copy(), np.zeros(cfg.N_JOINTS), np.zeros(cfg.N_JOINTS))

        # Prefill history buffers
        for _ in range(cfg.ROBOT.RNN_SEQ_LEN + 50):
            self.leader_hist.append(init_state)
            self.follower_hist_q.append(self.initial_qpos.copy())
            self.follower_hist_qd.append(np.zeros(cfg.N_JOINTS))
            
        info = {
            'leader_q': l_q.copy(),
            'follower_q': f_q.copy(),
            'true_state_vector': np.concatenate([l_q, np.zeros(cfg.N_JOINTS)]),
        }
        return self._get_obs(), info

    def step(self, action):
        self.step_count += 1
        
        # Step Physics
        l_q, l_qd, l_qdd, _, _, _, _ = self.leader.step()
        self.leader_hist.append((l_q.copy(), l_qd.copy()))
        self._curr_leader_state = (l_q, l_qd, l_qdd)
        
        # Feed action into Follower Robot (Mujoco Step)
        self.follower.step(torque_input=action)
        # Read follower state after action
        f_q, f_qd = self.follower.get_joint_state()
        
        self.follower_hist_q.append(f_q)
        self.follower_hist_qd.append(f_qd)
        
        # Calculate Reward
        target_q, target_qd = self.leader_hist[-1]
        reward, reward_info = self._compute_reward(target_q, target_qd, f_q, f_qd, action)
        
        # Check Termination
        terminated, term_reason, term_penalty = self._check_termination(f_q, f_qd, target_q)
        if terminated:
            reward += term_penalty

        truncated = self.step_count >= self.max_episode_steps
        self._prev_action = action.copy()
        
        info = {
            'leader_q': target_q.copy(),
            'follower_q': f_q.copy(),
            'true_state_vector': np.concatenate([target_q, target_qd]),
            'termination_reason': term_reason 
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _compute_reward(self, target_q, target_qd, f_q, f_qd, action):
        """
        Standard Dense Reward for Robotic Tracking.
        """
        
        # 1. Position Error (Euclidean distance in joint space)
        pos_error = np.linalg.norm(target_q - f_q)
        
        # 2. Velocity Error
        vel_error = np.linalg.norm(target_qd - f_qd)
        
        # 3. Action Penalty (Minimize energy)
        action_penalty = np.linalg.norm(action)
        
        # 4. Compute Gaussian Rewards
        r_pos = np.exp(-10.0 * pos_error) 
        r_vel = np.exp(-1.0 * vel_error)
        r_act = np.exp(-0.01 * action_penalty)
        
        # Weighted Sum
        # Prioritize Position (0.8), then Velocity (0.15), then Energy (0.05)
        reward = (0.8 * r_pos) + (0.15 * r_vel) + (0.05 * r_act)
        
        reward_info = {
            "r_pos": r_pos,
            "r_vel": r_vel,
            "r_act": r_act,
            "err_pos": pos_error
        }
        
        return reward, reward_info

    def _check_termination(self, f_q, f_qd, target_q):
        """
        FIXED: Soft Termination.
        Returns 0.0 penalty for functional failures to preserve reward normalization statistics.
        """
        # 1. NaN/Inf (True Simulation Crash) -> Keep Penalty
        if np.any(np.isnan(f_q)) or np.any(np.isinf(f_q)) or \
           np.any(np.isnan(f_qd)) or np.any(np.isinf(f_qd)):
            return True, "NaN_Simulation_Divergence", -10.0

        # 2. Joint Limits -> NO Penalty (Implicit loss of future reward is enough)
        if np.any(f_q < cfg.JOINT_LIMITS_LOWER) or np.any(f_q > cfg.JOINT_LIMITS_UPPER):
            return True, "Joint_Limit_Violation", 0.0

        # 3. Tracking Error -> NO Penalty
        if np.max(np.abs(target_q - f_q)) > cfg.ROBOT.MAX_JOINT_ERROR_TERMINATION:
            return True, "Max_Tracking_Error_Exceeded", 0.0

        return False, "None", 0.0
    
    def _get_obs_sequence(self) -> np.ndarray:
        """
        Constructs the leader history sequence with delay for LSTM input.
        """
        # Get raw delay steps
        raw_delay_steps = self.delay_simulator.get_state_delay_steps(len(self.leader_hist))
       
        # Normalized delay feature (Crucial for LSTM)
        norm_delay = raw_delay_steps / cfg.DELAY_INPUT_NORM_FACTOR
        
        target_seq = []
        end_idx = len(self.leader_hist) - 1 - raw_delay_steps
        start_idx = max(0, end_idx - cfg.ROBOT.RNN_SEQ_LEN + 1)
        
        for i in range(cfg.ROBOT.RNN_SEQ_LEN):
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
        """
        Contructs the observation vector for the RL agent.
        1. Remote Robot State (14)
        2. predict Leader State (after LSTM output)
        3. Previous Action (7)
        """
        
        # Remote Robot State (14)
        f_q, f_qd = self.follower_hist_q[-1], self.follower_hist_qd[-1]
        state_norm = np.concatenate([(f_q - cfg.Q_MEAN)/cfg.Q_STD, (f_qd - cfg.QD_MEAN)/cfg.QD_STD])
        
        # Leader History with Delay (1200), will feed to LSTM to get ture state prediction (14)
        target_seq = self._get_obs_sequence()

        # Previous Action (7)
        prev_action_norm = self._prev_action / cfg.MAX_ACTION_TORQUE
        
        # Total: 1221 dimensions
        return np.concatenate([state_norm, target_seq, prev_action_norm], dtype=np.float32)

    def close(self):
        self.follower.close()