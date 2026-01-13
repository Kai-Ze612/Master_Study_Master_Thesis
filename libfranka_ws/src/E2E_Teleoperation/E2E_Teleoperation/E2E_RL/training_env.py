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
        
        # Simulators
        self.delay_simulator = DelaySimulator(cfg.CONTROL_FREQ, config=delay_config, seed=seed)
        self.leader = LeaderRobotSimulator(trajectory_type=trajectory_type, randomize_params=randomize_trajectory)
        self.follower = FollowerRobotSimulator(delay_config=delay_config, seed=seed, render=(render_mode=="human"), verbose=False)
        
        # Buffers
        self.leader_hist = deque(maxlen=cfg.ROBOT.LEADER_HISTORY_BUFFER_LEN)
        self.follower_hist_q = deque(maxlen=cfg.ROBOT.RNN_SEQ_LEN)
        self.follower_hist_qd = deque(maxlen=cfg.ROBOT.RNN_SEQ_LEN)
        self.action_hist = deque(maxlen=cfg.ROBOT.ACTION_HISTORY_LEN)
        
        # Spaces
        self.action_space = spaces.Box(-cfg.ROBOT.MAX_ACTION_TORQUE, cfg.ROBOT.MAX_ACTION_TORQUE, shape=(cfg.ROBOT.N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(cfg.ROBOT.RL_OBS_DIM,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._prev_action = np.zeros(cfg.N_JOINTS)
                
        # Reset Robots
        l_q, _ = self.leader.reset(seed=seed) 
        self.initial_qpos = cfg.INITIAL_JOINT_CONFIG.copy()
        self.follower.reset(initial_qpos=self.initial_qpos)
        
        f_q = self.initial_qpos.copy()
        f_qd = np.zeros(cfg.ROBOT.N_JOINTS)
        
        # Clear Buffers
        self.leader_hist.clear()
        self.follower_hist_q.clear()
        self.follower_hist_qd.clear()
        self.action_hist.clear()
        for _ in range(cfg.ROBOT.ACTION_HISTORY_LEN):
            self.action_hist.append(np.zeros(cfg.ROBOT.N_JOINTS))
        
        # Set initial state
        init_state = (l_q.copy(), np.zeros(cfg.ROBOT.N_JOINTS))
        for _ in range(cfg.ROBOT.RNN_SEQ_LEN + 50):
            self.leader_hist.append(init_state)
            self.follower_hist_q.append(self.initial_qpos.copy())
            self.follower_hist_qd.append(np.zeros(cfg.ROBOT.N_JOINTS))
        
        # --- FIX: Truth is Leader Only (14) ---
        l_q_norm = (l_q - cfg.Q_MEAN) / cfg.Q_STD
        l_qd_norm = (np.zeros(7) - cfg.QD_MEAN) / cfg.QD_STD
        
        true_state = np.concatenate([l_q_norm, l_qd_norm])
        # --------------------------------------

        info = {
            'leader_q': l_q.copy(),
            'follower_q': f_q.copy(),
            'true_state_vector': true_state.astype(np.float32), 
        }
        return self._get_obs(), info

    def step(self, action):
        self.step_count += 1
        
        # 1. Step Leader
        l_q, l_qd, l_qdd, _, _, _, _ = self.leader.step()
        self.leader_hist.append((l_q.copy(), l_qd.copy()))
        
        # 2. Step Follower
        self.follower.step(torque_input=action)
        f_q, f_qd = self.follower.get_joint_state()
        
        self.action_hist.append(action.copy())
        self.follower_hist_q.append(f_q)
        self.follower_hist_qd.append(f_qd)
        
        # 3. Reward (FIXED: Now passing 'action')
        target_q, target_qd = self.leader_hist[-1]
        reward, _ = self._compute_reward(target_q, target_qd, f_q, f_qd, action)
        
        # 4. Termination Check
        terminated, term_reason, term_penalty = self._check_early_stop(f_q, f_qd, target_q)
        reward += term_penalty

        truncated = self.step_count >= self.max_episode_steps
        
        # Update previous action for next step's smoothness calc
        self._prev_action = action.copy()
        
        # 5. Construct State
        # --- FIX: Truth is Leader Only (14) ---
        t_q_norm = (target_q - cfg.Q_MEAN) / cfg.Q_STD
        t_qd_norm = (target_qd - cfg.QD_MEAN) / cfg.QD_STD
        
        true_state = np.concatenate([t_q_norm, t_qd_norm])
        
        info = {
            'leader_q': target_q.copy(),
            'follower_q': f_q.copy(),
            'true_state_vector': true_state.astype(np.float32), 
            'termination_reason': term_reason 
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _check_early_stop(self, f_q, f_qd, target_q):
        if np.any(np.isnan(f_q)) or np.any(np.isinf(f_q)):
            return True, "Simulation_Divergence", -10.0
        if np.any(f_q < cfg.JOINT_LIMITS_LOWER) or np.any(f_q > cfg.JOINT_LIMITS_UPPER):
            return False, "Joint_Limit_Violation", -0.1 
        max_error = np.max(np.abs(target_q - f_q))
        if max_error > cfg.ROBOT.MAX_JOINT_ERROR_TERMINATION:
            return False, "Max_Tracking_Error_Exceeded", -0.1 
        return False, "None", 0.0
    
    def _compute_reward(self, target_q, target_qd, f_q, f_qd, action):
        # 1. Tracking Components
        pos_error = np.linalg.norm(target_q - f_q)
        vel_error = np.linalg.norm(target_qd - f_qd)
        
        r_pos = np.exp(-2.0 * pos_error) 
        r_vel = np.exp(-1.0 * vel_error)
        
        # 2. Action Penalties (NORMALIZED)
        max_torque = cfg.ROBOT.MAX_ACTION_TORQUE
        
        # Normalize action [-1, 1]
        act_norm = action / max_torque
        
        # Energy Penalty
        r_energy = -np.mean(np.square(act_norm)) 
        
        # Smoothness Penalty
        prev_act_norm = self._prev_action / max_torque
        change_norm = act_norm - prev_act_norm
        r_smoothness = -np.mean(np.square(change_norm))
        
        # 3. Weights (Your tuned values)
        w_pos = 2.0
        w_vel = 0.5
        w_energy = 0.1   # Strengthened as per your plan
        w_smooth = 0.2   # Strengthened
        
        raw_reward = (w_pos * r_pos) + (w_vel * r_vel) + (w_energy * r_energy) + (w_smooth * r_smoothness)
        
        # 4. Clipping (Safety Valve)
        reward = np.clip(raw_reward, -5.0, 3.0)
        
        return reward, {}
    
    def _get_obs_sequence(self) -> np.ndarray:
        raw_delay_steps = self.delay_simulator.get_state_delay_steps(len(self.leader_hist))
        norm_delay = raw_delay_steps / cfg.DELAY_INPUT_NORM_FACTOR
        
        end_idx = len(self.leader_hist) - 1 - raw_delay_steps
        start_idx = max(0, end_idx - cfg.ROBOT.RNN_SEQ_LEN + 1)
        
        combined_seq = [] 

        for i in range(cfg.ROBOT.RNN_SEQ_LEN):
            curr_idx = start_idx + i
            
            # 1. Leader [15]
            if 0 <= curr_idx < len(self.leader_hist):
                l_q, l_qd = self.leader_hist[curr_idx]
            else:
                l_q, l_qd = self.leader_hist[0]
            
            l_q_norm = (l_q - cfg.Q_MEAN) / cfg.Q_STD
            l_qd_norm = (l_qd - cfg.QD_MEAN) / cfg.QD_STD
            
            # 2. Follower [14]
            if curr_idx < len(self.follower_hist_q):
                f_q = self.follower_hist_q[curr_idx]
                f_qd = self.follower_hist_qd[curr_idx]
            else:
                if len(self.follower_hist_q) > 0:
                    f_q, f_qd = self.follower_hist_q[-1], self.follower_hist_qd[-1]
                else:
                    f_q = cfg.INITIAL_JOINT_CONFIG
                    f_qd = np.zeros(7)

            f_q_norm = (f_q - cfg.Q_MEAN) / cfg.Q_STD
            f_qd_norm = (f_qd - cfg.QD_MEAN) / cfg.QD_STD
            
            # 3. Action [7]
            if curr_idx < len(self.action_hist):
                act = self.action_hist[curr_idx]
            else:
                if len(self.action_hist) > 0:
                    act = self.action_hist[-1]
                else:
                    act = np.zeros(7)
                
            act_norm = act / cfg.MAX_ACTION_TORQUE
            
            # Interleave
            step_data = np.concatenate([l_q_norm, l_qd_norm, [norm_delay], f_q_norm, f_qd_norm, act_norm])
            combined_seq.extend(step_data)
            
        return np.array(combined_seq, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        f_q, f_qd = self.follower_hist_q[-1], self.follower_hist_qd[-1]
        state_norm = np.concatenate([(f_q - cfg.Q_MEAN)/cfg.Q_STD, (f_qd - cfg.QD_MEAN)/cfg.QD_STD])
        
        target_seq = self._get_obs_sequence()
        
        act_hist_array = np.array(self.action_hist, dtype=np.float32)
        act_hist_norm = act_hist_array / cfg.MAX_ACTION_TORQUE
        act_hist_flat = act_hist_norm.flatten() 
        
        prev_action_norm = self._prev_action / cfg.MAX_ACTION_TORQUE
        
        return np.concatenate([
            state_norm,      
            target_seq,      
            act_hist_flat,   
            prev_action_norm 
        ], dtype=np.float32)
    
    def close(self):
        self.follower.close()