"""
Unified Trainer: Pure E2E RL (SAC)
----------------------------------
Standard Reinforcement Learning loop without expert intervention or DAgger.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import sys
import gc
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
        
        self.obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, pin_memory=True)
        self.next_obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, pin_memory=True)
        self.act_buf = torch.zeros((capacity, action_dim), dtype=torch.float32, pin_memory=True)
        self.rew_buf = torch.zeros(capacity, dtype=torch.float32, pin_memory=True)
        self.done_buf = torch.zeros(capacity, dtype=torch.float32, pin_memory=True)
        self.state_buf = torch.zeros((capacity, state_dim), dtype=torch.float32, pin_memory=True)
    
    def add(self, obs, action, reward, next_obs, done, state):
        self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32)
        self.next_obs_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32)
        self.act_buf[self.ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.rew_buf[self.ptr] = float(reward)
        self.done_buf[self.ptr] = float(done)
        self.state_buf[self.ptr] = torch.as_tensor(state, dtype=torch.float32)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def add_batch(self, obs, action, reward, next_obs, done, state):
        N = obs.shape[0]
        indices = np.arange(self.ptr, self.ptr + N) % self.capacity
        
        self.obs_buf[indices] = torch.as_tensor(obs, dtype=torch.float32)
        self.next_obs_buf[indices] = torch.as_tensor(next_obs, dtype=torch.float32)
        self.act_buf[indices] = torch.as_tensor(action, dtype=torch.float32)
        self.rew_buf[indices] = torch.as_tensor(reward, dtype=torch.float32)
        self.done_buf[indices] = torch.as_tensor(done, dtype=torch.float32)
        self.state_buf[indices] = torch.as_tensor(state, dtype=torch.float32)
        
        self.ptr = (self.ptr + N) % self.capacity
        self.size = min(self.size + N, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': self.obs_buf[idxs].to(self.device, non_blocking=True),
            'actions': self.act_buf[idxs].to(self.device, non_blocking=True),
            'rewards': self.rew_buf[idxs].to(self.device, non_blocking=True).unsqueeze(1),
            'next_obs': self.next_obs_buf[idxs].to(self.device, non_blocking=True),
            'dones': self.done_buf[idxs].to(self.device, non_blocking=True).unsqueeze(1),
            'true_state_vector': self.state_buf[idxs].to(self.device, non_blocking=True)
        }


class UnifiedTrainer:
    def __init__(self, env, output_dir: Path, eval_env=None):
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env
        
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = JointActor().to(self.device)
        self.critic = JointCritic().to(self.device)
        self.critic_target = JointCritic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.TRAIN.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)
        
        # SAC Algorithm
        self.sac = ResidualSAC(
            self.actor,
            self.critic,
            self.critic_target,
            self.actor_optimizer,
            self.critic_optimizer,
            self.alpha_optimizer,
            self.log_alpha,
            gamma=cfg.TRAIN.GAMMA,
            tau=cfg.SAC.TARGET_TAU
        )
        
        # Replay Buffer
        self.buffer = ReplayBuffer(
            cfg.TRAIN.BUFFER_SIZE,
            cfg.ROBOT.RL_OBS_DIM,
            cfg.ROBOT.N_JOINTS,
            cfg.ROBOT.ESTIMATOR_OUTPUT_DIM,
            self.device
        )
        
        # Logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(log_dir=cfg.ROBOT.LOG_DIR / self.output_dir.name)
        
        self.num_envs = self.env.num_envs if hasattr(self.env, "num_envs") else 1
        self.warmup_steps = cfg.TRAIN.WARMUP_STEPS // self.num_envs
        
        # Load Pre-trained if available
        if cfg.ROBOT.PRETRAINED_ACTOR_PATH.exists():
            self.actor.load_state_dict(torch.load(cfg.ROBOT.PRETRAINED_ACTOR_PATH, map_location=self.device))
            self.logger.info(">>> Loaded Pre-trained Actor.")
        
    def log(self, msg):
        self.logger.info(msg)
        
    def train_e2e(self):
        self.log("Trainer Initialized. Device: {}".format(self.device))
        self.log("\n============================================================\n>>> PURE E2E RL TRAINING STARTED\n>>> Config: {} Warmup | Patience: {} evals\n============================================================".format(cfg.TRAIN.WARMUP_STEPS, cfg.TRAIN.EARLY_STOP_PATIENCE))
        
        global_step = 0
        grad_updates_pending = 0
        best_eval_reward = -np.inf
        no_improvement_count = 0
        
        # Warmup: Random Data Collection
        self.log(">>> Starting Warmup (Random Actions)...")
        obs = self.env.reset()
        for _ in range(cfg.TRAIN.WARMUP_STEPS):
            actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
            next_obs, rewards, dones, _, infos = self.env.step(actions)
            
            if self.num_envs == 1:
                self.buffer.add(obs, actions, rewards, next_obs, dones, infos['true_state_vector'])
            else:
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, [info['true_state_vector'] for info in infos])
            
            obs = next_obs
            global_step += self.num_envs
        
        # Main Training Loop
        pbar = tqdm(initial=global_step, total=cfg.TRAIN.TOTAL_TIMESTEPS, desc="Training Steps")
        
        while global_step < cfg.TRAIN.TOTAL_TIMESTEPS:
            actions = self.actor.sample(torch.as_tensor(obs, device=self.device))[0].cpu().numpy()
            next_obs, rewards, dones, _, infos = self.env.step(actions)
            
            if self.num_envs == 1:
                self.buffer.add(obs, actions, rewards, next_obs, dones, infos['true_state_vector'])
            else:
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, [info['true_state_vector'] for info in infos])
            
            obs = next_obs
            global_step += self.num_envs
            grad_updates_pending += self.num_envs
            
            if grad_updates_pending >= cfg.TRAIN.TRAIN_FREQUENCY:
                updates_to_run = np.clip(int(grad_updates_pending * 0.5), 1, 64)
                grad_updates_pending = 0
                
                for _ in range(updates_to_run):
                    batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
                    metrics = self.sac.update(batch)
                    
                    if global_step % cfg.TRAIN.LOG_FREQ == 0:
                        self.log(f"Step {global_step} | R: {np.mean(rewards):.2f} | Q: {metrics['q1_loss']:.1f} | AuxLoss: {metrics['pred_loss']:.4f}")
                        self.writer.add_scalar("Train/Reward", np.mean(rewards), global_step)
                        self.writer.add_scalar("Train/Q_Loss", metrics['q1_loss'], global_step)
                        self.writer.add_scalar("Train/Aux_Loss", metrics['pred_loss'], global_step)
            
            if global_step % cfg.TRAIN.EVAL_INTERVAL == 0 and global_step >= cfg.TRAIN.WARMUP_STEPS:
                current_eval_reward = self._run_evaluation_episodes(global_step)
                
                improvement = current_eval_reward - best_eval_reward
                if improvement > cfg.TRAIN.EARLY_STOP_MIN_DELTA:
                    best_eval_reward = current_eval_reward
                    no_improvement_count = 0
                    self.log(f">>> New Best Model! Reward: {best_eval_reward:.2f}")
                    self._save_checkpoint(global_step, is_best=True)
                else:
                    no_improvement_count += 1
                    self.log(f">>> No improvement. Patience: {no_improvement_count}/{cfg.TRAIN.EARLY_STOP_PATIENCE}")
                
                self._save_checkpoint(global_step, is_best=False)

                if cfg.TRAIN.ENABLE_EARLY_STOP and no_improvement_count >= cfg.TRAIN.EARLY_STOP_PATIENCE:
                    self.log(f"\n[STOP] Early Stopping Triggered! No improvement for {no_improvement_count} evals.")
                    self.log(f"Best Reward Achieved: {best_eval_reward:.2f}")
                    break
            
            pbar.update(self.num_envs)
        
        pbar.close()
        self.log(">>> Training Finished.")

    def _run_evaluation_episodes(self, step):
        eval_rewards = []
        
        for episode_idx in range(5):
            reset_ret = self.eval_env.reset()
            if isinstance(reset_ret, tuple): obs = reset_ret[0]
            else: obs = reset_ret

            total_rew = 0
            done = False
            step_count = 0
            
            if episode_idx == 0:
                self.log(f"\n=== FULL EPISODE DEBUG (Step {step}) ===")
                self.log(f"Step | True q (7 joints) | Pred q (7 joints) | Actions (7 joints) | Follower q (7 joints) | Tracking Error | Pred Error")
                self.log("-" * 300)

            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                    mu, _, pred_state_t, _ = self.actor(obs_t)
                    action = (torch.tanh(mu) * self.actor.action_scale).cpu().numpy()[0]
                    pred_state = pred_state_t.cpu().numpy()[0]

                next_obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                total_rew += reward
                
                if episode_idx == 0 and (step_count % 50 == 0 or step_count == 0):
                    # Extract full vectors
                    true_q = info['true_state_vector'][:7]  # Normalized leader q
                    pred_q = pred_state[:7]                 # Normalized pred q
                    follower_q = info['follower_q']         # Raw follower q (assuming denormalized; adjust if needed)

                    # Compute errors
                    tracking_error = np.linalg.norm(true_q - follower_q)
                    pred_error = np.linalg.norm(true_q - pred_q)

                    # Round to 3 decimals for readability
                    true_q_rounded = np.round(true_q, 3)
                    pred_q_rounded = np.round(pred_q, 3)
                    action_rounded = np.round(action, 3)
                    follower_q_rounded = np.round(follower_q, 3)
                    tracking_error_rounded = round(tracking_error, 3)
                    pred_error_rounded = round(pred_error, 3)

                    # One-line progress summary (averages/norms for quick view)
                    true_q_norm = np.linalg.norm(true_q)
                    pred_q_norm = np.linalg.norm(pred_q)
                    action_norm = np.linalg.norm(action)
                    follower_q_norm = np.linalg.norm(follower_q)
                    self.log(f"Progress Step {step_count}: True q Norm {true_q_norm:.3f} | Pred q Norm {pred_q_norm:.3f} | Action Norm {action_norm:.3f} | Follower q Norm {follower_q_norm:.3f} | Tracking Err {tracking_error_rounded} | Pred Err {pred_error_rounded}")

                    # Multi-line details
                    self.log(f"Step: {step_count}")
                    self.log(f"True q: {true_q_rounded}")
                    self.log(f"Pred q: {pred_q_rounded}")
                    self.log(f"Actions: {action_rounded}")
                    self.log(f"Follower q: {follower_q_rounded}")
                    self.log(f"Tracking Error: {tracking_error_rounded}")
                    self.log(f"Pred Error: {pred_error_rounded}\n")

                step_count += 1
                obs = next_obs
            
            # Log per-episode reward
            self.log(f"Episode {episode_idx} Reward: {total_rew:.2f}")
        
        avg_eval = np.mean(eval_rewards)
        std_eval = np.std(eval_rewards)
        self.log(f"--- Eval @ {step}: {avg_eval:.2f} +/- {std_eval:.2f} ---")
        self.writer.add_scalar("Eval/Reward", avg_eval, step)
        return avg_eval

    def _save_checkpoint(self, step, is_best=False):
        filename = "best_model.pth" if is_best else "latest_model.pth"
        save_path = self.output_dir / filename
        torch.save({
            'step': step,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
        }, save_path)