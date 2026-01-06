"""
Unified Trainer for End-to-End Teleoperation
- Phase 1: Pre-train LSTM on random data (Supervised Learning)
- Phase 2: Fine tuning (SAC) with Pre-trained LSTM (Reinforcement Learning)
"""

"""
Unified Trainer: Pretrain -> Warmup -> Residual RL -> E2E Fine-Tune
MODIFIED: Expert Data Initialization + Multi-Step Updates for Parallel Envs
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Dict
from torch.utils.tensorboard import SummaryWriter

from E2E_Teleoperation.E2E_RL.e2e_network import JointActor, JointCritic
from E2E_Teleoperation.E2E_RL.e2e_algorithm import ResidualSAC
import E2E_Teleoperation.config.robot_config as cfg

# --- REPLAY BUFFER ---
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
        # Support adding batches from Vector Envs
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

# --- UNIFIED TRAINER ---
class UnifiedTrainer:
    def __init__(self, env, output_dir: Path):
        self.env = env
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Detect Environment Type
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vec_env = self.num_envs > 1
        
        self._setup_logging()
        self.writer = SummaryWriter(log_dir=str(output_dir / "logs"))
        
        self.REWARD_SCALE = 0.1  
        self.WARMUP_STEPS = 50_000 
        self.TOTAL_STEPS = cfg.TRAIN.TOTAL_TIMESTEPS
        self.BATCH_SIZE = 256
        
        self.actor = JointActor().to(self.device)
        self.critic = JointCritic().to(self.device)
        self.critic_target = JointCritic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.residual_net.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.log_alpha = torch.tensor([np.log(0.1)], dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        self.algo = ResidualSAC(
            self.actor, self.critic, self.critic_target,
            self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer,
            self.log_alpha
        )
        
        self.buffer = ReplayBuffer(
            cfg.TRAIN.BUFFER_SIZE, cfg.ROBOT.RL_OBS_DIM, 
            cfg.ROBOT.N_JOINTS, cfg.ROBOT.ROBOT_STATE_DIM, self.device
        )
        
        self.log(f"Trainer Initialized. Parallel Envs: {self.num_envs}")

    def _setup_logging(self):
        self.logger = logging.getLogger("ResidualTrainer")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        log_file = self.output_dir / "training_log.txt"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)

    def log(self, msg):
        self.logger.info(msg)

    def _reset_env(self):
        res = self.env.reset()
        if self.is_vec_env:
            return res 
        else:
            return res[0] 

    def _step_env(self, action):
        if self.is_vec_env:
            next_obs, reward, done, info = self.env.step(action)
            return next_obs, reward, done, info
        else:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            return next_obs, reward, terminated or truncated, info

    def train_e2e(self):
        self.log("\n=== PHASE 1: SUPERVISED LSTM PRE-TRAINING ===")
        
        # MODIFIED: Collect EXPERT Data (Actor) instead of Random
        self._collect_initial_data(steps=10_000)
        
        self._pretrain_lstm(steps=5_000)
        
        self.log("\n=== PHASE 2 & 3: RL TRAINING ===")
        obs = self._reset_env()
        fine_tune_mode = False
        steps_per_iter = self.num_envs
        
        for global_step in range(0, self.TOTAL_STEPS, steps_per_iter):
            
            # --- ACTION SELECTION (ALWAYS POLICY) ---
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                
                if not self.is_vec_env:
                     obs_t = obs_t.unsqueeze(0) 
                
                action_t, _, _, _ = self.actor.sample(obs_t)
                action = action_t.cpu().numpy()
                
                if not self.is_vec_env:
                    action = action[0]

            # --- STEP ---
            next_obs, reward, done, info = self._step_env(action)
            scaled_reward = reward * self.REWARD_SCALE
            
            # --- STORE ---
            if self.is_vec_env:
                true_states = np.stack([i['true_state_vector'] for i in info])
                self.buffer.add_batch(obs, action, scaled_reward, next_obs, done, true_states)
            else:
                true_state = info.get('true_state_vector', np.zeros(14))
                self.buffer.add(obs, action, scaled_reward, next_obs, float(done), true_state)
            
            obs = next_obs
            
            if not self.is_vec_env and done:
                obs = self._reset_env()

            # --- UPDATE ---
            if self.buffer.size > self.BATCH_SIZE:
                is_warmup = global_step < self.WARMUP_STEPS
                
                if global_step >= 200_000 and not fine_tune_mode:
                    self.log("!!! UNFREEZING LSTM FOR E2E FINE-TUNING !!!")
                    self.actor_optimizer.add_param_group({'params': self.actor.base_encoder.parameters(), 'lr': 1e-5})
                    fine_tune_mode = True
                    for p in self.actor.base_encoder.parameters(): p.requires_grad = True

                # MODIFIED: Run multiple updates to match parallel collection rate
                # Ratio: 1 Update per 1 Environment Step
                updates_to_run = self.num_envs
                
                for _ in range(updates_to_run):
                    metrics = self.algo.update(
                        self.buffer.sample(self.BATCH_SIZE), 
                        update_actor=(not is_warmup),
                        fine_tune_encoder=fine_tune_mode
                    )
                
                if global_step % 1000 == 0:
                     avg_rew = np.mean(reward)
                     self.log(f"Step {global_step} | C: {metrics['critic_loss']:.2f} | A: {metrics['actor_loss']:.2f} | "
                              f"P: {metrics['pred_loss']:.3f} | R_avg: {avg_rew:.2f}")
                     
                     self.writer.add_scalar("Loss/Critic", metrics['critic_loss'], global_step)
                     self.writer.add_scalar("Loss/Actor", metrics['actor_loss'], global_step)
                     self.writer.add_scalar("Reward/Avg", avg_rew, global_step)

    def _collect_initial_data(self, steps):
        # MODIFIED: Collects data using the ACTOR (Base Policy), not Random
        self.log(f"Collecting {steps} steps of EXPERT (BASE POLICY) data...")
        
        obs = self._reset_env()
        iters = steps // self.num_envs if self.is_vec_env else steps
        
        for _ in range(iters):
            # Use Actor to sample actions (No Grads)
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                if not self.is_vec_env: obs_t = obs_t.unsqueeze(0)
                
                action_t, _, _, _ = self.actor.sample(obs_t)
                action = action_t.cpu().numpy()
                if not self.is_vec_env: action = action[0]
            
            next_obs, reward, done, info = self._step_env(action)
            
            if self.is_vec_env:
                true_states = np.stack([i['true_state_vector'] for i in info])
                self.buffer.add_batch(obs, action, reward, next_obs, done, true_states)
            else:
                true_state = info.get('true_state_vector', np.zeros(14))
                self.buffer.add(obs, action, reward, next_obs, float(done), true_state)
            
            obs = next_obs
            if not self.is_vec_env and done: 
                obs = self._reset_env()

    def _pretrain_lstm(self, steps):
        self.log(f"Pre-training LSTM for {steps} steps...")
        opt = optim.Adam(self.actor.base_encoder.parameters(), lr=1e-3)
        for p in self.actor.base_encoder.parameters(): p.requires_grad = True
            
        for i in range(steps):
            batch = self.buffer.sample(self.BATCH_SIZE)
            _, pred_state, _ = self.actor.base_encoder(batch['obs'][:, -1207:-7].view(-1, 80, 15)) 
            loss = nn.MSELoss()(pred_state, batch['true_state_vector'])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 1000 == 0: self.log(f"Pretrain Loss: {loss.item():.4f}")
        
        self.log("Re-freezing LSTM.")
        for p in self.actor.base_encoder.parameters(): p.requires_grad = False