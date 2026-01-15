"""
Unified Trainer: Pure E2E RL (SAC) - Optimized & Fixed
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
        
        # Buffers on Device
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
        
        # 1. INITIALIZE NETWORKS
        self.actor = JointActor().to(self.device)
        self.critic = JointCritic().to(self.device)
        self.critic_target = JointCritic().to(self.device)
        
        # Logging Setup
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
        self.logger = logging.getLogger(__name__)
        log_dir = self.output_dir / "files"
        log_dir.mkdir(parents=True, exist_ok=True)
        if not self.logger.handlers:
            self.logger.addHandler(logging.FileHandler(log_dir / "training.log"))
            self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.writer = SummaryWriter(log_dir=self.output_dir)
        
        # 2. LOAD & SYNC
        if cfg.ROBOT.PRETRAINED_ACTOR_PATH.exists():
            try:
                self.actor.load_state_dict(torch.load(cfg.ROBOT.PRETRAINED_ACTOR_PATH, map_location=self.device))
                self.logger.info(">>> Loaded Pre-trained Actor.")
            except Exception as e:
                self.logger.warning(f">>> Failed to load pre-trained actor: {e}")

        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 3. SETUP OPTIMIZERS
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.TRAIN.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)
        
        # 4. COMPILATION
        if hasattr(torch, "compile"):
            try:
                self.actor = torch.compile(self.actor)
                self.critic = torch.compile(self.critic)
                self.critic_target = torch.compile(self.critic_target)
            except Exception as e:
                self.logger.warning(f"[WARNING] torch.compile failed: {e}")

        # 5. INITIALIZE ALGORITHM & BUFFER
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
        no_improvement_count = 0
        
        obs = self.env.reset()
        if isinstance(obs, tuple): obs = obs[0]
        
        # ==========================================
        # WARMUP LOOP
        # ==========================================
        self.log(">>> Starting Warmup...")
        for _ in range(self.warmup_steps):
            # Creates (N, 7) array. For num_envs=1, this is (1, 7).
            actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
            
            # [FIX] Handle 5-tuple return (Gymnasium) vs 4-tuple return (VecEnv)
            if self.num_envs == 1:
                # Gymnasium: obs, reward, terminated, truncated, info
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions[0])
                dones = terminated or truncated
            else:
                # VecEnv: obs, rewards, dones, infos
                next_obs, rewards, dones, infos = self.env.step(actions)
            
            if self.num_envs == 1:
                self.buffer.add(obs, actions, rewards, next_obs, dones, infos['true_state_vector'])
            else:
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, [i['true_state_vector'] for i in infos])
            
            obs = next_obs
            global_step += self.num_envs
        
        # ==========================================
        # MAIN TRAINING LOOP
        # ==========================================
        pbar = tqdm(initial=global_step, total=cfg.TRAIN.TOTAL_TIMESTEPS, desc="Training")
        
        while global_step < cfg.TRAIN.TOTAL_TIMESTEPS:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device)
                
                # [FIX]: Unsqueeze for inference if single env (Dim,) -> (1, Dim)
                if self.num_envs == 1 and obs_t.ndim == 1:
                    obs_t = obs_t.unsqueeze(0)
                    
                actions = self.actor.sample(obs_t)[0].detach().cpu().numpy()
                
            # [FIX] Handle 5-tuple vs 4-tuple
            if self.num_envs == 1:
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions[0])
                dones = terminated or truncated
            else:
                next_obs, rewards, dones, infos = self.env.step(actions)
            
            if self.num_envs == 1:
                self.buffer.add(obs, actions, rewards, next_obs, dones, infos['true_state_vector'])
            else:
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, [i['true_state_vector'] for i in infos])
            
            obs = next_obs
            global_step += self.num_envs
            grad_updates_pending += self.num_envs
            
            if grad_updates_pending >= cfg.TRAIN.TRAIN_FREQUENCY:
                updates = np.clip(int(grad_updates_pending * 0.5), 1, 64)
                grad_updates_pending = 0
                
                for _ in range(updates):
                    batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
                    metrics = self.sac.update(batch, update_actor=True)
                    
                    if global_step % cfg.TRAIN.LOG_FREQ == 0:
                        self.log(f"Step {global_step} | R: {np.mean(rewards):.2f} | RL_L: {metrics['actor_loss']:.2f} | Pred_L: {metrics['pred_loss']:.3f}")
                        self.writer.add_scalar("Train/Reward", np.mean(rewards), global_step)
                        self.writer.add_scalar("Loss/RL", metrics['actor_loss'], global_step)
                        self.writer.add_scalar("Loss/Pred", metrics['pred_loss'], global_step)
            
            if global_step % cfg.TRAIN.EVAL_INTERVAL == 0 and global_step >= cfg.TRAIN.WARMUP_STEPS:
                current_eval_reward = self._run_evaluation_episodes(global_step)
                
                if current_eval_reward > best_eval_reward:
                    best_eval_reward = current_eval_reward
                    no_improvement_count = 0
                    self._save_checkpoint(global_step, is_best=True)
                else:
                    no_improvement_count += 1
                
                self._save_checkpoint(global_step, is_best=False)

                if cfg.TRAIN.ENABLE_EARLY_STOP and no_improvement_count >= cfg.TRAIN.EARLY_STOP_PATIENCE:
                    self.log("[STOP] Early Stopping Triggered.")
                    break
            
            pbar.update(self.num_envs)
        
        pbar.close()
        self.log(">>> Training Finished.")

    def _run_evaluation_episodes(self, step):
        eval_rewards = []
        
        self.log(f"\n>>> EVALUATION AT STEP {step}")
        self.log("=" * 100)
        
        for episode_idx in range(cfg.EVAL.NUM_EPISODES):
            reset_ret = self.eval_env.reset()
            obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
            total_rew = 0
            done = False
            ep_step = 0
            
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                    if obs_t.ndim == 1:
                        obs_t = obs_t.unsqueeze(0)

                    mu, _, pred_state_t, _ = self.actor(obs_t)
                    # Use deterministic action for eval
                    action = torch.tanh(mu) * self.actor.action_scale
                    
                    action_np = action.cpu().numpy()[0]
                    # Get NORMALIZED prediction
                    pred_state_norm = pred_state_t.cpu().numpy()[0]

                next_obs, reward, terminated, truncated, info = self.eval_env.step(action_np)
                done = terminated or truncated
                total_rew += reward
                
                # --- DETAILED VERTICAL LOGGING ---
                # Only log for the first episode, every DEBUG_PRINT_INTERVAL steps
                if episode_idx == 0 and ep_step % cfg.EVAL.DEBUG_PRINT_INTERVAL == 0:
                    # 1. Retrieve Raw Values
                    true_q = info['leader_q']      # Target (Leader)
                    follower_q = info['follower_q'] # Actual (Follower)
                    
                    # 2. Denormalize Prediction
                    pred_q = (pred_state_norm[:7] * cfg.ROBOT.Q_STD) + cfg.ROBOT.Q_MEAN
                    
                    # 3. Calculate Error
                    err_q = true_q - follower_q
                    
                    # 4. Helper for formatting array
                    def fmt_arr(arr):
                        return "[" + ", ".join([f"{x:6.3f}" for x in arr]) + "]"

                    self.log(f"--- Step {ep_step} ---")
                    self.log(f"Leader (True): {fmt_arr(true_q)}")
                    self.log(f"Leader (Pred): {fmt_arr(pred_q)}")
                    self.log(f"Follower     : {fmt_arr(follower_q)}")
                    self.log(f"Action (Tau) : {fmt_arr(action_np)}")
                    self.log(f"Error (T-F)  : {fmt_arr(err_q)}")
                    self.log("-" * 60)
                # -----------------------------

                obs = next_obs
                ep_step += 1
            
            eval_rewards.append(total_rew)
        
        avg_eval = np.mean(eval_rewards)
        self.log(f">>> Avg Reward: {avg_eval:.2f}")
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