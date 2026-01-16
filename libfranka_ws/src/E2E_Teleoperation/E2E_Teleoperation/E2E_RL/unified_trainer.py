"""
Unified Trainer: NaN Guard Edition
"""

"""
Unified Trainer: NaN Guard Edition
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from E2E_Teleoperation.E2E_RL.e2e_network import JointActor, JointCritic
from E2E_Teleoperation.E2E_RL.e2e_algorithm import ResidualSAC
import E2E_Teleoperation.config.robot_config as cfg


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, 
                 state_dim: int, device: torch.device):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device
        
        self.obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.next_obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.act_buf = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.done_buf = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.state_buf = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
    
    def add(self, obs, action, reward, next_obs, done, state):
        self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.next_obs_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.act_buf[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rew_buf[self.ptr] = float(reward)
        self.done_buf[self.ptr] = float(done)
        self.state_buf[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def add_batch(self, obs, action, reward, next_obs, done, state):
        N = obs.shape[0]
        indices = np.arange(self.ptr, self.ptr + N) % self.capacity
        self.obs_buf[indices] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.next_obs_buf[indices] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.act_buf[indices] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rew_buf[indices] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.done_buf[indices] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        state_stacked = np.array(state)
        self.state_buf[indices] = torch.as_tensor(state_stacked, dtype=torch.float32, device=self.device)
        self.ptr = (self.ptr + N) % self.capacity
        self.size = min(self.size + N, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            'obs': self.obs_buf[idxs],
            'actions': self.act_buf[idxs],
            'rewards': self.rew_buf[idxs].unsqueeze(1),
            'next_obs': self.next_obs_buf[idxs],
            'dones': self.done_buf[idxs].unsqueeze(1),
            'true_state_vector': self.state_buf[idxs]
        }


class UnifiedTrainer:
    def __init__(self, env, output_dir: Path, eval_env=None):
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = JointActor().to(self.device)
        self.critic = JointCritic().to(self.device)
        self.critic_target = JointCritic().to(self.device)
        
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
        self.logger = logging.getLogger(__name__)
        log_dir = self.output_dir / "files"
        log_dir.mkdir(parents=True, exist_ok=True)
        if not self.logger.handlers:
            self.logger.addHandler(logging.FileHandler(log_dir / "training.log"))
            self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.writer = SummaryWriter(log_dir=self.output_dir)
        
        if cfg.ROBOT.PRETRAINED_ACTOR_PATH.exists():
            try:
                self.actor.load_state_dict(torch.load(cfg.ROBOT.PRETRAINED_ACTOR_PATH, map_location=self.device))
                self.logger.info(">>> Loaded Pre-trained Actor.")
            except Exception as e:
                self.logger.warning(f">>> Failed to load pre-trained actor: {e}")

        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.TRAIN.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)
        
        self.sac = ResidualSAC(
            self.actor, self.critic, self.critic_target,
            self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer,
            self.log_alpha, gamma=cfg.TRAIN.GAMMA, tau=cfg.SAC.TARGET_TAU
        )
        
        self.buffer = ReplayBuffer(
            cfg.TRAIN.BUFFER_SIZE, cfg.ROBOT.RL_OBS_DIM, cfg.ROBOT.N_JOINTS,
            cfg.ROBOT.ESTIMATOR_OUTPUT_DIM, self.device
        )
        
        self.num_envs = self.env.num_envs if hasattr(self.env, "num_envs") else 1
        self.warmup_steps = cfg.TRAIN.WARMUP_STEPS // self.num_envs
        
    def log(self, msg):
        self.logger.info(msg)
        
    def train_e2e(self):
        self.log(f">>> E2E RL STARTED | Mode: {'JOINT' if cfg.TRAIN.JOINT_OPTIMIZATION else 'DECOUPLED'}")
        
        global_step = 0
        grad_updates_pending = 0
        best_eval_reward = -np.inf
        
        obs = self.env.reset()
        if isinstance(obs, tuple): obs = obs[0]
        
        # WARMUP
        self.log(">>> Starting Warmup...")
        for _ in range(self.warmup_steps):
            actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
            if self.num_envs == 1:
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions[0])
                dones = terminated or truncated
                self.buffer.add(obs, actions, rewards, next_obs, dones, infos['true_state_vector'])
            else:
                next_obs, rewards, dones, infos = self.env.step(actions)
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, [i['true_state_vector'] for i in infos])
            obs = next_obs
            global_step += self.num_envs
        
        # MAIN LOOP
        pbar = tqdm(initial=global_step, total=cfg.TRAIN.TOTAL_TIMESTEPS, desc="Training")
        
        while global_step < cfg.TRAIN.TOTAL_TIMESTEPS:
            # --- 1. SAMPLE ACTION & LOGGING PREP ---
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device)
                if self.num_envs == 1 and obs_t.ndim == 1:
                    obs_t = obs_t.unsqueeze(0)
                
                # Sample action AND prediction for logging
                actions_tensor, _, pred_leader_tensor, _ = self.actor.sample(obs_t)
                actions = actions_tensor.detach().cpu().numpy()
                pred_leader_np = pred_leader_tensor.detach().cpu().numpy()

            # --- 2. STEP ENV ---
            if self.num_envs == 1:
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions[0])
                dones = terminated or truncated
                
                # --- [DEBUG] Collect Metrics for Logging ---
                true_leader_q = infos['leader_q']
                # Denormalize Prediction
                pred_leader_q = (pred_leader_np[0][:7] * cfg.ROBOT.Q_STD) + cfg.ROBOT.Q_MEAN
                follower_q = infos['follower_q']
            else:
                next_obs, rewards, dones, infos = self.env.step(actions)
            
            if self.num_envs == 1:
                self.buffer.add(obs, actions, rewards, next_obs, dones, infos['true_state_vector'])
            else:
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, [i['true_state_vector'] for i in infos])
            
            obs = next_obs
            global_step += self.num_envs
            grad_updates_pending += self.num_envs
            
            # --- 3. RL UPDATE ---
            if grad_updates_pending >= cfg.TRAIN.TRAIN_FREQUENCY:
                updates = np.clip(int(grad_updates_pending * 0.5), 1, 64)
                grad_updates_pending = 0
                
                for _ in range(updates):
                    batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
                    metrics = self.sac.update(batch, update_actor=True)
                    
                    # --- [CRITICAL] NaN GUARD ---
                    if np.isnan(metrics['actor_loss']) or np.isnan(metrics['pred_loss']):
                        self.log("\n[FATAL] NaN DETECTED IN LOSS! IMMEDIATE DUMP:")
                        self.log(f"Step: {global_step} | Metrics: {metrics}")
                        torch.save(self.actor.state_dict(), self.output_dir / "CRASH_actor.pth")
                        sys.exit(1)
                    # -----------------------------

                    if global_step % cfg.TRAIN.LOG_FREQ == 0:
                        # Compute Debug Metrics
                        if self.num_envs == 1:
                            pred_err = np.mean(np.abs(true_leader_q - pred_leader_q))
                            track_err = np.mean(np.abs(true_leader_q - follower_q))
                        else:
                            pred_err = 0.0 # Placeholder for vec env
                            track_err = 0.0

                        self.log(f"Step {global_step} | R: {np.mean(rewards):.2f} | RL_L: {metrics['actor_loss']:.3f} | Pred_L: {metrics['pred_loss']:.3f}")
                        self.log(f"   >>> [Debug] Pred_Err: {pred_err:.4f} | Track_Err: {track_err:.4f}")
                        
                        self.writer.add_scalar("Train/Reward", np.mean(rewards), global_step)
                        self.writer.add_scalar("Train/Pred_Error_Rad", pred_err, global_step)
                        self.writer.add_scalar("Loss/RL", metrics['actor_loss'], global_step)
            
            if global_step % cfg.TRAIN.EVAL_INTERVAL == 0:
                current_eval_reward = self._run_evaluation_episodes(global_step)
                if current_eval_reward > best_eval_reward:
                    best_eval_reward = current_eval_reward
                    self._save_checkpoint(global_step, is_best=True)
                self._save_checkpoint(global_step, is_best=False)
            
            pbar.update(self.num_envs)
        
        pbar.close()
        self.log(">>> Training Finished.")

    def _run_evaluation_episodes(self, step):
        eval_rewards = []
        for episode_idx in range(cfg.EVAL.NUM_EPISODES):
            reset_ret = self.eval_env.reset()
            obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
            total_rew = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                    mu, _, _, _ = self.actor(obs_t)
                    action = torch.tanh(mu) * self.actor.action_scale
                    action_np = action.cpu().numpy()[0]
                
                next_obs, reward, terminated, truncated, _ = self.eval_env.step(action_np)
                done = terminated or truncated
                total_rew += reward
                obs = next_obs
            
            eval_rewards.append(total_rew)
        
        avg_eval = np.mean(eval_rewards)
        self.log(f">>> Eval Reward: {avg_eval:.2f}")
        return avg_eval

    def _save_checkpoint(self, step, is_best=False):
        filename = "best_model.pth" if is_best else "latest_model.pth"
        torch.save(self.actor.state_dict(), self.output_dir / filename)