"""
PD Gain Tuning Training Environment
------------------------------------
RL outputs state-dependent PD gains (Kp, Kd) instead of torque residuals.

Architecture:
    τ = Kp(s) * (q_delayed - q_current) + Kd(s) * (qd_delayed - qd_current)
    
Where:
    - Kp(s), Kd(s) are 7-dim vectors output by RL policy
    - q_delayed, qd_delayed are the delayed leader states (what operator sees)
    - Gravity compensation is added instantly in FollowerRobotSimulator

Action Space: 14-dim (Kp[7] + Kd[7]), normalized to [-1, 1]
    action[0:7]  -> Kp gains (scaled to [KP_MIN, KP_MAX])
    action[7:14] -> Kd gains (scaled to [KD_MIN, KD_MAX])
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from ASAC.leader_robot_simulator import LeaderRobotSimulator
from ASAC.follower_robot_simulator import FollowerRobotSimulator
from ASAC.utils.delay_simulator import DelaySimulator, ExperimentConfig
import ASAC.config.robot_config as cfg


class PDGainTuningEnv(gym.Env):
    """
    Environment where RL learns optimal PD gains for teleoperation.
    
    Key differences from Residual RL:
    1. Action space: 14-dim (Kp + Kd) instead of 7-dim torque residual
    2. RL directly outputs the controller gains
    3. No separate PD baseline - RL IS the controller
    """
    
    metadata = {'render_modes': ["human", "rgb_array"], 'render_fps': 60}
    
    def __init__(
        self,
        delay_config=ExperimentConfig.HIGH_VARIANCE,
        trajectory_type="figure_8",
        randomize_trajectory=False,
        seed=None,
        render_mode=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = cfg.ROBOT.MAX_EPISODE_STEPS
        
        # Delay simulator (for observation delay)
        self.delay_simulator = DelaySimulator(cfg.CONTROL_FREQ, config=delay_config, seed=seed)
        
        # Handle trajectory type
        if not isinstance(trajectory_type, str):
            trajectory_type = trajectory_type.value
            
        # Leader robot (generates target trajectory)
        self.leader = LeaderRobotSimulator(
            trajectory_type=trajectory_type, 
            randomize_params=randomize_trajectory
        )
        
        # Follower robot (with visualization if requested)
        self.follower = FollowerRobotSimulator(
            delay_config=delay_config, 
            seed=seed, 
            render=(render_mode == "human"),
            render_fps=60,
            verbose=True
        )
        
        # History buffers
        self.leader_hist = deque(maxlen=cfg.ROBOT.LEADER_HISTORY_BUFFER_LEN)
        self.follower_hist_q = deque(maxlen=cfg.ROBOT.RNN_SEQ_LEN)
        self.follower_hist_qd = deque(maxlen=cfg.ROBOT.RNN_SEQ_LEN)
        self.gains_hist = deque(maxlen=cfg.ROBOT.ACTION_HISTORY_LEN)  # Store Kp, Kd history
        
        # Action space: 14-dim normalized [-1, 1] for Kp (7) + Kd (7)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(cfg.ROBOT.N_JOINTS * 2,),  # 14 dims
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(cfg.ROBOT.RL_OBS_DIM,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.step_count = 0
        
        # Current gains (for observation and logging)
        self._current_kp = cfg.PD_GAINS.KP_BASE.copy()
        self._current_kd = cfg.PD_GAINS.KD_BASE.copy()
        self._current_delayed_target_q = np.zeros(cfg.ROBOT.N_JOINTS, dtype=np.float32)
        self._current_delayed_target_qd = np.zeros(cfg.ROBOT.N_JOINTS, dtype=np.float32)

    def _action_to_gains(self, action: np.ndarray) -> tuple:
        """
        Convert normalized action [-1, 1] to actual PD gains.
        
        Args:
            action: 14-dim array, action[0:7] for Kp, action[7:14] for Kd
            
        Returns:
            (Kp, Kd): Tuple of 7-dim gain arrays
        """
        action = np.clip(action, -1.0, 1.0)
        
        # Scale from [-1, 1] to [MIN, MAX]
        # gain = MIN + (action + 1) / 2 * RANGE
        kp_normalized = (action[:7] + 1.0) / 2.0  # [0, 1]
        kd_normalized = (action[7:14] + 1.0) / 2.0  # [0, 1]
        
        kp = cfg.PD_GAINS.KP_MIN + kp_normalized * cfg.PD_GAINS.KP_RANGE
        kd = cfg.PD_GAINS.KD_MIN + kd_normalized * cfg.PD_GAINS.KD_RANGE
        
        return kp.astype(np.float32), kd.astype(np.float32)

    def _gains_to_action(self, kp: np.ndarray, kd: np.ndarray) -> np.ndarray:
        """
        Convert actual PD gains to normalized action [-1, 1].
        Inverse of _action_to_gains.
        """
        kp_normalized = (kp - cfg.PD_GAINS.KP_MIN) / (cfg.PD_GAINS.KP_RANGE + 1e-6)
        kd_normalized = (kd - cfg.PD_GAINS.KD_MIN) / (cfg.PD_GAINS.KD_RANGE + 1e-6)
        
        action_kp = kp_normalized * 2.0 - 1.0  # [0,1] -> [-1,1]
        action_kd = kd_normalized * 2.0 - 1.0
        
        return np.concatenate([action_kp, action_kd]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset components
        l_q, _ = self.leader.reset(seed=seed)
        f_q, _ = self.follower.reset(seed=seed)
        self.delay_simulator.reset()
        
        # Clear histories
        self.leader_hist.clear()
        self.follower_hist_q.clear()
        self.follower_hist_qd.clear()
        self.gains_hist.clear()
        
        # Initialize histories
        zero_qd = np.zeros(cfg.ROBOT.N_JOINTS, dtype=np.float32)
        init_gains = np.concatenate([cfg.PD_GAINS.KP_BASE, cfg.PD_GAINS.KD_BASE])
        
        for _ in range(cfg.ROBOT.LEADER_HISTORY_BUFFER_LEN):
            self.leader_hist.append((l_q.copy(), zero_qd.copy()))
        
        for _ in range(cfg.ROBOT.RNN_SEQ_LEN):
            self.follower_hist_q.append(f_q.copy())
            self.follower_hist_qd.append(zero_qd.copy())
        
        for _ in range(cfg.ROBOT.ACTION_HISTORY_LEN):
            self.gains_hist.append(init_gains.copy())
        
        # Reset state
        self.step_count = 0
        self._current_kp = cfg.PD_GAINS.KP_BASE.copy()
        self._current_kd = cfg.PD_GAINS.KD_BASE.copy()
        self._current_delayed_target_q = l_q.copy()
        self._current_delayed_target_qd = zero_qd.copy()
        
        return self._get_obs(), {
            'kp': self._current_kp.copy(),
            'kd': self._current_kd.copy()
        }

    def step(self, action: np.ndarray):
        """
        Execute one step with RL-output PD gains.
        
        Args:
            action: 14-dim normalized action (Kp[7] + Kd[7])
        """
        self.step_count += 1
        
        # 1. Convert action to PD gains
        kp, kd = self._action_to_gains(action)
        self._current_kp = kp.copy()
        self._current_kd = kd.copy()
        
        # 2. Step leader to get new target
        l_q, l_qd, _, _, _, _, _ = self.leader.step()
        self.leader_hist.append((l_q.copy(), l_qd.copy()))
        
        # 3. Get delayed leader state (what operator "sees")
        delayed_l_q, delayed_l_qd, current_delay_sec = self.delay_simulator.get_delayed_state(
            self.leader_hist
        )
        self._current_delayed_target_q = delayed_l_q.copy()
        self._current_delayed_target_qd = delayed_l_qd.copy()
        
        # 4. Get current follower state
        current_f_q = self.follower_hist_q[-1]
        current_f_qd = self.follower_hist_qd[-1]
        
        # 5. Compute PD control with RL gains
        q_error = delayed_l_q - current_f_q
        qd_error = delayed_l_qd - current_f_qd
        
        pd_torque = kp * q_error + kd * qd_error
        pd_torque = np.clip(pd_torque, -cfg.ROBOT.MAX_ACTION_TORQUE, cfg.ROBOT.MAX_ACTION_TORQUE)
        
        # 6. Step follower (gravity compensation handled internally)
        follower_info = self.follower.step(pd_torque)
        f_q = follower_info['q_follower']
        
        # Safety check
        if not np.all(np.isfinite(f_q)):
            return (
                self._get_obs(),
                cfg.REWARD.PENALTY_DIVERGENCE,
                True, False,
                {'crash': True, 'reason': 'NaN in joint positions'}
            )
        
        # 7. Compute follower velocity
        f_qd = (f_q - self.follower_hist_q[-1]) / cfg.DT
        
        # 8. Update histories
        self.follower_hist_q.append(f_q.copy())
        self.follower_hist_qd.append(f_qd.copy())
        gains_concat = np.concatenate([kp, kd])
        self.gains_hist.append(gains_concat.copy())
        
        # 9. Compute reward (against TRUE target, not delayed)
        target_q, target_qd = self.leader_hist[-1]
        reward = self._compute_reward(f_q, f_qd, target_q, target_qd, kp, kd)
        
        # 10. Termination conditions
        pos_error = np.linalg.norm(target_q - f_q)
        terminated = pos_error > cfg.ROBOT.MAX_JOINT_ERROR_TERMINATION
        truncated = self.step_count >= self.max_episode_steps
        
        # Info dict
        info = {
            'position_error': pos_error,
            'velocity_error': np.linalg.norm(target_qd - f_qd),
            'current_delay': current_delay_sec,
            'kp': kp.copy(),
            'kd': kd.copy(),
            'kp_mean': np.mean(kp),
            'kd_mean': np.mean(kd),
            'pd_torque_norm': np.linalg.norm(pd_torque),
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs_sequence(self) -> np.ndarray:
        """Build history sequence for observation."""
        combined_seq = []
        
        for i in range(cfg.ROBOT.RNN_SEQ_LEN):
            offset = cfg.ROBOT.RNN_SEQ_LEN - 1 - i
            l_q_delayed, l_qd_delayed, delay_sec = self.delay_simulator.get_delayed_state(
                self.leader_hist, offset_indices=offset
            )
            
            # Normalize states
            l_q_norm = (l_q_delayed - cfg.ROBOT.Q_MEAN) / cfg.ROBOT.Q_STD
            l_qd_norm = (l_qd_delayed - cfg.ROBOT.QD_MEAN) / cfg.ROBOT.QD_STD
            
            f_q = self.follower_hist_q[i]
            f_qd = self.follower_hist_qd[i]
            f_q_norm = (f_q - cfg.ROBOT.Q_MEAN) / cfg.ROBOT.Q_STD
            f_qd_norm = (f_qd - cfg.ROBOT.QD_MEAN) / cfg.ROBOT.QD_STD
            
            # Normalize gains history
            gains = self.gains_hist[i] if i < len(self.gains_hist) else np.concatenate([cfg.PD_GAINS.KP_BASE, cfg.PD_GAINS.KD_BASE])
            gains_norm = self._gains_to_action(gains[:7], gains[7:14])  # Already [-1, 1]
            
            step_data = np.concatenate([
                l_q_norm, l_qd_norm, [delay_sec],
                f_q_norm, f_qd_norm, gains_norm
            ])
            combined_seq.extend(step_data)
        
        return np.array(combined_seq, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Construct observation."""
        f_q = self.follower_hist_q[-1]
        f_qd = self.follower_hist_qd[-1]
        
        # Current state (normalized)
        state_norm = np.concatenate([
            (f_q - cfg.ROBOT.Q_MEAN) / cfg.ROBOT.Q_STD,
            (f_qd - cfg.ROBOT.QD_MEAN) / cfg.ROBOT.QD_STD
        ])
        
        # History sequence
        target_seq = self._get_obs_sequence()
        
        # Previous gains (normalized)
        if len(self.gains_hist) > 1:
            prev_gains = self.gains_hist[-2]
        else:
            prev_gains = np.concatenate([cfg.PD_GAINS.KP_BASE, cfg.PD_GAINS.KD_BASE])
        prev_gains_norm = self._gains_to_action(prev_gains[:7], prev_gains[7:14])
        
        return np.concatenate([state_norm, target_seq, prev_gains_norm], dtype=np.float32)

    def _compute_reward(
        self, 
        f_q: np.ndarray, 
        f_qd: np.ndarray, 
        target_q: np.ndarray, 
        target_qd: np.ndarray, 
        kp: np.ndarray,
        kd: np.ndarray
    ) -> float:
        """
        Compute reward for PD gain tuning.
        
        Components:
        1. Position tracking (primary)
        2. Velocity tracking
        3. Gain smoothness (penalize rapid gain changes)
        4. Gain regularization (penalize extreme gains)
        """
        # Tracking errors
        pos_err = np.linalg.norm(target_q - f_q)
        vel_err = np.linalg.norm(target_qd - f_qd)
        
        # Exponential reward shaping
        r_pos = np.exp(-cfg.REWARD.SCALE_POS * pos_err)
        r_vel = np.exp(-cfg.REWARD.SCALE_VEL * vel_err)
        
        # Gain smoothness penalty (penalize rapid changes)
        if len(self.gains_hist) > 1:
            prev_gains = self.gains_hist[-2]
            prev_kp, prev_kd = prev_gains[:7], prev_gains[7:14]
            kp_change = np.linalg.norm(kp - prev_kp) / np.linalg.norm(cfg.PD_GAINS.KP_RANGE)
            kd_change = np.linalg.norm(kd - prev_kd) / np.linalg.norm(cfg.PD_GAINS.KD_RANGE)
            gain_smoothness = kp_change + kd_change
        else:
            gain_smoothness = 0.0
        r_smooth = -cfg.REWARD.W_SMOOTH * gain_smoothness
        
        # Gain regularization (penalize deviation from base gains)
        kp_dev = np.linalg.norm(kp - cfg.PD_GAINS.KP_BASE) / np.linalg.norm(cfg.PD_GAINS.KP_RANGE)
        kd_dev = np.linalg.norm(kd - cfg.PD_GAINS.KD_BASE) / np.linalg.norm(cfg.PD_GAINS.KD_RANGE)
        r_gain_reg = -cfg.REWARD.W_GAIN_REG * (kp_dev + kd_dev)
        
        # Combine
        reward = (
            cfg.REWARD.W_POS * r_pos +
            cfg.REWARD.W_VEL * r_vel +
            r_smooth +
            r_gain_reg
        )
        
        return float(np.clip(reward, cfg.REWARD.MIN_CLIP, cfg.REWARD.MAX_CLIP))

    def render(self):
        """Render is handled automatically by FollowerRobotSimulator."""
        pass

    def close(self):
        """Clean up."""
        self.follower.close()


# =============================================================================
# Factory Functions
# =============================================================================

def make_gain_tuning_env(rank: int, args, seed_offset: int = 0):
    """Factory for creating PD gain tuning environments."""
    def _init():
        render_mode = "human" if getattr(args, "render", False) and rank == 0 else None
        
        return PDGainTuningEnv(
            delay_config=ExperimentConfig(args.delay_config),
            trajectory_type="figure_8",
            randomize_trajectory=True,
            seed=args.seed + rank + seed_offset,
            render_mode=render_mode,
        )
    return _init


def make_gain_tuning_eval_env(args):
    """Create evaluation environment for PD gain tuning."""
    return PDGainTuningEnv(
        delay_config=ExperimentConfig(args.delay_config),
        trajectory_type="figure_8",
        randomize_trajectory=False,
        seed=args.seed + 99999,
        render_mode=None,  # Never render eval env (avoids multiple viewer crash)
    )