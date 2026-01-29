"""
SBSP Environment: RL Tunes Gains, Controller Tracks Delayed State
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import mujoco

from SBSP.leader_robot_simulator import LeaderRobotSimulator, TrajectoryType
from SBSP.follower_robot_simulator import FollowerRobotSimulator
from SBSP.utils.delay_simulator import DelaySimulator, ExperimentConfig
import SBSP.config.robot_config as cfg

class SBSPEnv(gym.Env):
    metadata = {'render_modes': ["human"], 'render_fps': cfg.CONTROL_FREQ}
    
    def __init__(
        self,
        delay_config=ExperimentConfig.HIGH_VARIANCE,
        trajectory_type=TrajectoryType.FIGURE_8,
        randomize_trajectory=False,
        seed=None,
        render_mode=None,
    ):
        super().__init__()
        self.leader = LeaderRobotSimulator(trajectory_type=trajectory_type, randomize_params=randomize_trajectory)
        self.follower = FollowerRobotSimulator(delay_config=delay_config, seed=seed, render=(render_mode=="human"), verbose=False)
        self.delay_sim = DelaySimulator(cfg.CONTROL_FREQ, config=delay_config, seed=seed)
        
        self.obs_history = deque(maxlen=cfg.ROBOT.FRAME_STACK)
        self.leader_hist = deque(maxlen=cfg.ROBOT.LEADER_HISTORY_BUFFER_LEN)
        
        # Action: 14 (7 Kp + 7 Kd)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
        # Observation: Stacked Frames
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.ROBOT.RL_OBS_DIM,), dtype=np.float32)
        
        self.last_action = np.zeros(14)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        l_q, _ = self.leader.reset(seed=seed)
        f_q, _ = self.follower.reset(seed=seed)
        self.delay_sim.reset()
        
        self.leader_hist.clear()
        self.obs_history.clear()
        
        for _ in range(cfg.ROBOT.LEADER_HISTORY_BUFFER_LEN):
            self.leader_hist.append((l_q, np.zeros(7)))
            
        l_init = np.concatenate([(l_q - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, np.zeros(7), [0.0]])
        f_init = np.concatenate([(f_q - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, np.zeros(7)])
        act_init = np.zeros(14)
        
        init_frame = np.concatenate([l_init, f_init, act_init])
        for _ in range(cfg.ROBOT.FRAME_STACK):
            self.obs_history.append(init_frame)
            
        self.last_action = np.zeros(14)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action_norm):
        # 1. Update Simulators
        l_q_new, l_qd_new, _, _, _, _, _ = self.leader.step()
        self.leader_hist.append((l_q_new, l_qd_new))
        
        # Get Delayed Reference (This is what the controller tracks)
        ref_q, ref_qd, ref_delay = self.delay_sim.get_delayed_state(self.leader_hist)
        
        # 2. Decode PD Gains from RL
        action_norm = np.clip(action_norm, -1.0, 1.0)
        kp = (action_norm[:7] + 1)/2 * (cfg.SBSP.KP_MAX - cfg.SBSP.KP_MIN) + cfg.SBSP.KP_MIN
        kd = (action_norm[7:] + 1)/2 * (cfg.SBSP.KD_MAX - cfg.SBSP.KD_MIN) + cfg.SBSP.KD_MIN
        
        # 3. Control Law: Track Delayed State
        f_q, f_qd = self.follower.get_joint_state()
        
        self.follower.data.qpos[:7] = f_q
        self.follower.data.qvel[:7] = f_qd
        mujoco.mj_forward(self.follower.model, self.follower.data)
        gravity = self.follower.data.qfrc_bias[:7].copy()
        
        tau = (kp * (ref_q - f_q)) + (kd * (ref_qd - f_qd)) + gravity
        
        # 4. Step
        info = self.follower.step(tau)
        f_q_new = info['q_follower']
        f_qd_new = info['qd_follower']
        
        # 5. Obs
        l_vec = np.concatenate([
            (ref_q - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, 
            (ref_qd - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD, 
            [ref_delay]
        ])
        f_vec = np.concatenate([
            (f_q_new - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, 
            (f_qd_new - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD
        ])
        frame = np.concatenate([l_vec, f_vec, self.last_action])
        self.obs_history.append(frame)
        self.last_action = action_norm.copy()
        
        # 6. Reward & Info
        reward = self._compute_reward(f_q_new, f_qd_new, l_q_new, l_qd_new, action_norm)
        
        self.step_count += 1
        truncated = self.step_count >= cfg.ROBOT.MAX_EPISODE_STEPS
        
        if np.linalg.norm(l_q_new - f_q_new) > 2.5:
            reward = cfg.REWARD.PENALTY_DIVERGENCE
            truncated = True
            
        # Export 'true_leader_q' for the Algorithm's prediction loss
        # Also cast metrics to float for safe multiprocessing
        info = {
            'true_leader_q': l_q_new.astype(np.float32), 
            'kp_mean': float(np.mean(kp)),
            'kd_mean': float(np.mean(kd)),
            'position_error': float(np.linalg.norm(l_q_new - f_q_new))
        }
        
        return self._get_obs(), reward, False, truncated, info

    def _get_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def _compute_reward(self, f_q, f_qd, t_q, t_qd, action):
        pos_err = np.linalg.norm(t_q - f_q)
        vel_err = np.linalg.norm(t_qd - f_qd)
        r_pos = cfg.REWARD.W_POS * np.exp(-cfg.REWARD.SCALE_POS * pos_err)
        r_vel = cfg.REWARD.W_VEL * np.exp(-cfg.REWARD.SCALE_VEL * vel_err)
        energy = np.mean(np.square(action)) * cfg.REWARD.W_ENERGY
        smooth = 0.0
        if self.step_count > 0:
            smooth = np.mean(np.square(action - self.last_action)) * cfg.REWARD.W_SMOOTH
        return r_pos + r_vel - energy - smooth

# Helper Factories
def make_sbsp_env(rank, args):
    def _init():
        delay_cfg = ExperimentConfig.HIGH_VARIANCE
        if args.delay_config == 0: delay_cfg = ExperimentConfig.NO_DELAY
        elif args.delay_config == 1: delay_cfg = ExperimentConfig.LOW_DELAY
        elif args.delay_config == 2: delay_cfg = ExperimentConfig.HIGH_DELAY
        
        render_mode = "human" if (args.render and rank == 0) else None
        return SBSPEnv(delay_config=delay_cfg, trajectory_type=TrajectoryType.FIGURE_8, randomize_trajectory=True, seed=args.seed + rank, render_mode=render_mode)
    return _init

def make_sbsp_eval_env(args):
    return SBSPEnv(delay_config=ExperimentConfig.HIGH_VARIANCE, trajectory_type=TrajectoryType.FIGURE_8, randomize_trajectory=True, seed=args.seed + 1000, render_mode=None)