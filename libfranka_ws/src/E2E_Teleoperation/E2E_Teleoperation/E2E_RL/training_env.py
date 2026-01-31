import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import mujoco

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
        
        self.delay_simulator = DelaySimulator(cfg.CONTROL_FREQ, config=delay_config, seed=seed)
        self.leader = LeaderRobotSimulator(trajectory_type=trajectory_type, randomize_params=randomize_trajectory)
        self.follower = FollowerRobotSimulator(delay_config=delay_config, seed=seed, render=(render_mode=="human"), verbose=False)
        
        self.leader_hist = deque(maxlen=cfg.ROBOT.LEADER_HISTORY_BUFFER_LEN)
        self.follower_hist_q = deque(maxlen=cfg.ROBOT.RNN_SEQ_LEN)
        self.follower_hist_qd = deque(maxlen=cfg.ROBOT.RNN_SEQ_LEN)
        self.action_hist = deque(maxlen=cfg.ROBOT.ACTION_HISTORY_LEN)
        
        self.action_space = spaces.Box(
            low=-cfg.ROBOT.MAX_ACTION_TORQUE,
            high=cfg.ROBOT.MAX_ACTION_TORQUE,
            shape=(cfg.ROBOT.N_JOINTS,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(cfg.ROBOT.RL_OBS_DIM,),
            dtype=np.float32
        )
        
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        l_q, _ = self.leader.reset(seed=seed)
        f_q, _ = self.follower.reset(seed=seed)
        self.delay_simulator.reset()
        
        self.leader_hist.clear()
        self.follower_hist_q.clear()
        self.follower_hist_qd.clear()
        self.action_hist.clear()
        
        for _ in range(cfg.ROBOT.LEADER_HISTORY_BUFFER_LEN):
            self.leader_hist.append((l_q, np.zeros(7)))
            
        for _ in range(cfg.ROBOT.RNN_SEQ_LEN):
            self.follower_hist_q.append(f_q)
            self.follower_hist_qd.append(np.zeros(7))
            
        for _ in range(cfg.ROBOT.ACTION_HISTORY_LEN):
            self.action_hist.append(np.zeros(7))
            
        self.step_count = 0
        
        obs = self._get_obs()
        
        true_state = np.concatenate([
            (l_q - cfg.ROBOT.Q_MEAN) / cfg.ROBOT.Q_STD,
            (np.zeros(7) - cfg.ROBOT.QD_MEAN) / cfg.ROBOT.QD_STD
        ])
        
        info = {
            'true_state_vector': true_state.astype(np.float32),
            'leader_q': l_q.copy(),
            'leader_qd': np.zeros(7),
            'follower_q': f_q.copy(),
            'follower_qd': np.zeros(7),
            'delay': 0.0
        }
        return obs, info

    def step(self, action):
        # 1. Step Leader
        l_q, l_qd, _, _, _, _, _ = self.leader.step()
        self.leader_hist.append((l_q, l_qd))
        
        # 2. Get Delayed Leader
        delayed_l_q, delayed_l_qd, current_delay_sec = self.delay_simulator.get_delayed_state(self.leader_hist)
        
        # 3. Apply Action Directly (No Gravity Comp)
        # [MODIFIED] Removed manual gravity compensation block to rely on simulator/network
        final_torque = action 
        
        # 4. Step Follower
        follower_info = self.follower.step(final_torque)
        f_q = follower_info['q_follower']
        
        # --- CRITICAL SAFETY CHECK ---
        if not np.all(np.isfinite(f_q)):
            f_q = cfg.INITIAL_JOINT_CONFIG.copy()
            f_qd = np.zeros(7)
            # Penalize crashing
            reward = -1.0 # [RECOMMENDED FIX] Lowered from -500 to prevent Q-collapse
            terminated = True
            truncated = False
            
            true_state = np.zeros(14) 
            info = {
                'leader_q': l_q.copy(),
                'leader_qd': l_qd.copy(),
                'follower_q': f_q.copy(),
                'follower_qd': f_qd.copy(),
                'true_state_vector': true_state.astype(np.float32),
                'delay': current_delay_sec,
                'crash': True
            }
            self.follower_hist_q.append(f_q)
            self.follower_hist_qd.append(f_qd)
            self.action_hist.append(action) 
            
            return self._get_obs(), reward, terminated, truncated, info

        # Compute Follower Velocity
        if len(self.follower_hist_q) > 0:
            f_qd = (f_q - self.follower_hist_q[-1]) / cfg.DT
        else:
            f_qd = np.zeros(7)
            
        self.follower_hist_q.append(f_q)
        self.follower_hist_qd.append(f_qd)
        self.action_hist.append(action)
        
        target_q, target_qd = self.leader_hist[-1]
        reward = self._compute_reward(f_q, f_qd, target_q, target_qd, action)
        
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_episode_steps
        
        pos_error = np.linalg.norm(target_q - f_q)
        if pos_error > 2.0: 
            reward += cfg.REWARD.PENALTY_DIVERGENCE
            terminated = True
            
        true_state = np.concatenate([
            (target_q - cfg.ROBOT.Q_MEAN) / cfg.ROBOT.Q_STD,
            (target_qd - cfg.ROBOT.QD_MEAN) / cfg.ROBOT.QD_STD
        ])
        
        info = {
            'leader_q': target_q.copy(),
            'leader_qd': target_qd.copy(),
            'follower_q': f_q.copy(),
            'follower_qd': f_qd.copy(),
            'true_state_vector': true_state.astype(np.float32), 
            'delay': current_delay_sec
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs_sequence(self):
        combined_seq = []
        start_idx = len(self.follower_hist_q) - cfg.ROBOT.RNN_SEQ_LEN
        
        for i in range(cfg.ROBOT.RNN_SEQ_LEN):
            curr_idx = start_idx + i
            if curr_idx < 0: continue
            
            # Retrieve past states
            l_q_delayed, l_qd_delayed, delay_sec = self.delay_simulator.get_delayed_state(self.leader_hist, offset_indices=cfg.ROBOT.RNN_SEQ_LEN - 1 - i)
            
            l_q_norm = (l_q_delayed - cfg.ROBOT.Q_MEAN) / cfg.ROBOT.Q_STD
            l_qd_norm = (l_qd_delayed - cfg.ROBOT.QD_MEAN) / cfg.ROBOT.QD_STD
            norm_delay = delay_sec / 1.0 
            
            f_q = self.follower_hist_q[i]
            f_qd = self.follower_hist_qd[i]
            f_q_norm = (f_q - cfg.ROBOT.Q_MEAN) / cfg.ROBOT.Q_STD
            f_qd_norm = (f_qd - cfg.ROBOT.QD_MEAN) / cfg.ROBOT.QD_STD
            
            if i < len(self.action_hist):
                act = self.action_hist[i]
            else:
                act = np.zeros(7)
                
            act_norm = act / cfg.ROBOT.MAX_ACTION_TORQUE
            
            step_data = np.concatenate([l_q_norm, l_qd_norm, [norm_delay], f_q_norm, f_qd_norm, act_norm])
            combined_seq.extend(step_data)
            
        return np.array(combined_seq, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        f_q, f_qd = self.follower_hist_q[-1], self.follower_hist_qd[-1]
        state_norm = np.concatenate([(f_q - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, (f_qd - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD])
        
        target_seq = self._get_obs_sequence()
        
        act_hist_array = np.array(self.action_hist, dtype=np.float32)
        act_hist_norm = act_hist_array / cfg.ROBOT.MAX_ACTION_TORQUE
        
        act_hist_flat = act_hist_norm.flatten()
        
        # [CRITICAL FIX] Matching logic with evaluation script
        # We use the second to last action (t-1) because 't' is not yet observed
        if len(self.action_hist) > 1:
            prev_act = self.action_hist[-2] / cfg.ROBOT.MAX_ACTION_TORQUE
        else:
            prev_act = np.zeros(7)
            
        return np.concatenate([state_norm, target_seq, act_hist_flat, prev_act], dtype=np.float32)

    def _compute_reward(self, f_q, f_qd, t_q, t_qd, action):
        pos_err = np.linalg.norm(t_q - f_q)
        vel_err = np.linalg.norm(t_qd - f_qd)
        
        r_pos = np.exp(-cfg.REWARD.SCALE_POS * pos_err)
        r_vel = np.exp(-cfg.REWARD.SCALE_VEL * vel_err)
        
        energy = np.mean(np.square(action))
        
        if len(self.action_hist) > 1:
            smoothness = np.linalg.norm(action - self.action_hist[-2])
        else:
            smoothness = 0.0
            
        reward = (cfg.REWARD.W_POS * r_pos) + \
                 (cfg.REWARD.W_VEL * r_vel) - \
                 (cfg.REWARD.W_ENERGY * energy) - \
                 (cfg.REWARD.W_SMOOTH * smoothness)
                 
        return np.clip(reward, cfg.REWARD.MIN_CLIP, cfg.REWARD.MAX_CLIP)