"""
Unified Trainer for End-to-End Teleoperation
- Phase 1: Pre-train LSTM on random data (Supervised Learning)
- Phase 2: Fine tuning (SAC) with Pre-trained LSTM (Reinforcement Learning)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter  # Added TensorBoard

from E2E_Teleoperation.E2E_RL.e2e_network import JointActor, JointCritic
from E2E_Teleoperation.E2E_RL.e2e_algorithm import SACAlgorithm # Now utilizing your algorithm file
import E2E_Teleoperation.config.robot_config as cfg

class ReplayBuffer:
    """
    Replay Buffer for storing experience tuples.
    Optimized to return dict for SACAlgorithm.
    """
    def __init__(self, capacity, obs_dim, action_dim, state_dim, device):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device
        
        # Pre-allocate memory
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.state_buf = np.zeros((capacity, state_dim), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, state):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.state_buf[self.ptr] = state
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def add_batch(self, obs, action, reward, next_obs, done, state):
        N = obs.shape[0]
        indices = np.arange(self.ptr, self.ptr + N) % self.capacity
        self.obs_buf[indices] = obs
        self.next_obs_buf[indices] = next_obs
        self.act_buf[indices] = action
        self.rew_buf[indices] = reward
        self.done_buf[indices] = done
        self.state_buf[indices] = state
        self.ptr = (self.ptr + N) % self.capacity
        self.size = min(self.size + N, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': torch.as_tensor(self.obs_buf[idxs], device=self.device),
            'actions': torch.as_tensor(self.act_buf[idxs], device=self.device),
            'rewards': torch.as_tensor(self.rew_buf[idxs], device=self.device),
            'next_obs': torch.as_tensor(self.next_obs_buf[idxs], device=self.device),
            'dones': torch.as_tensor(self.done_buf[idxs], device=self.device),
            'true_state_vector': torch.as_tensor(self.state_buf[idxs], device=self.device)
        }

class UnifiedTrainer:
    def __init__(self, env, output_dir):
        self.env = env
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = getattr(env, "num_envs", 1)

        # 1. Logging Setup
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        print(f"Training on: {self.device} | Logs: {self.output_dir}")

        # 2. Networks
        self.actor = JointActor().to(self.device)
        self.critic = JointCritic().to(self.device)
        self.target_critic = JointCritic().to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 3. Optimizers (Using Config LRs)
        # Split actor params to handle Encoder freezing/diff LR if needed
        self.actor_optimizer = optim.Adam([
            {'params': self.actor.encoder.parameters(), 'lr': cfg.TRAIN.ENCODER_LR}, 
            {'params': self.actor.net.parameters(), 'lr': cfg.TRAIN.ACTOR_LR},
            {'params': self.actor.mu.parameters(), 'lr': cfg.TRAIN.ACTOR_LR},
            {'params': self.actor.log_std.parameters(), 'lr': cfg.TRAIN.ACTOR_LR},
        ])
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)

        # Alpha (Entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)

        # 4. Initialize Algorithm
        self.algorithm = SACAlgorithm(
            actor=self.actor,
            critic=self.critic,
            critic_target=self.target_critic,
            actor_opt=self.actor_optimizer,
            critic_opt=self.critic_optimizer,
            alpha_opt=self.alpha_optimizer,
            log_alpha=self.log_alpha
        )

        # 5. Replay Buffer
        self.buffer = ReplayBuffer(
            cfg.TRAIN.BUFFER_SIZE, 
            cfg.ROBOT.RL_OBS_DIM, 
            cfg.ROBOT.N_JOINTS, 
            cfg.ROBOT.ROBOT_STATE_DIM, 
            self.device
        )
        
        self.best_eval_reward = -np.inf
        
        # Pre-training Loss function
        self.sl_loss = nn.MSELoss()

    def train_e2e(self):
        """
        Execute the full Phase 1 (SL) -> Phase 2 (RL) pipeline
        """
        # ==================================================
        # PHASE 1: Data Collection & Supervised Learning
        # ==================================================
        print(f"\n[Phase 1] Collecting Random Data for Pre-training...")
        self._collect_random_data(steps=5000)

        print(f"[Phase 1] Pre-training LSTM Encoder (Supervised Learning)...")
        self._train_encoder_supervised(steps=cfg.TRAIN.STAGE1_STEPS)
        
        # Save Pre-trained Encoder
        self.save_checkpoint("phase1_pretrained")

        # ==================================================
        # PHASE 2: End-to-End Reinforcement Learning
        # ==================================================
        print(f"\n[Phase 2] Starting End-to-End SAC Training...")
        
        obs = self._reset_env()
        
        # Total interactions divided by num_envs because we step all envs at once
        total_steps = cfg.TRAIN.STAGE2_STEPS
        
        for step in range(1, (total_steps // self.num_envs) + 1):
            global_step = step * self.num_envs
            
            # 1. Action Selection
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device).float()
                # Handle single env case
                if self.num_envs == 1 and obs_t.ndim == 1: 
                    obs_t = obs_t.unsqueeze(0)
                
                # Sample action
                action_t, _, _, _, _ = self.actor.sample(obs_t)
                action = action_t.cpu().numpy()
                
                if self.num_envs == 1: 
                    action = action[0]

            # 2. Environment Step
            next_obs, reward, done, state = self._step_env(action)

            # 3. Add to Buffer
            if self.num_envs > 1:
                self.buffer.add_batch(obs, action, reward, next_obs, done, state)
            else:
                self.buffer.add(obs, action, reward, next_obs, done, state)

            obs = next_obs

            # 4. Update (Gradient Descent)
            # Use Config Batch Size (4096) for GPU utilization
            if self.buffer.size > cfg.SAC.WARMUP_STEPS:
                batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
                update_info = self.algorithm.update(batch)
                
                if step % cfg.TRAIN.LOG_FREQ == 0:
                    self._log_metrics(update_info, global_step)

            # 5. Evaluation
            if global_step % cfg.TRAIN.VAL_FREQ == 0:
                self.evaluate(global_step)

        self.env.close()
        self.writer.close()

    def _collect_random_data(self, steps):
        """Run random policy to fill buffer for pre-training"""
        obs = self._reset_env()
        for _ in range(steps // self.num_envs):
            if self.num_envs > 1:
                action = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
            else:
                action = self.env.action_space.sample()
            
            next_obs, reward, done, state = self._step_env(action)
            
            if self.num_envs > 1:
                self.buffer.add_batch(obs, action, reward, next_obs, done, state)
            else:
                self.buffer.add(obs, action, reward, next_obs, done, state)
            obs = next_obs

    def _train_encoder_supervised(self, steps):
        """
        Phase 1 Loop: Train ONLY the encoder/predictor to predict next state.
        This stabilizes the LSTM before RL starts.
        """
        self.actor.train()
        for i in range(steps):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            
            # Forward pass just to get prediction
            # We use .sample() but we only care about pred_state
            _, _, pred_state, _, _ = self.actor.sample(batch['obs'])
            
            true_state = batch['true_state_vector']
            loss = self.sl_loss(pred_state, true_state)
            
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            
            if i % cfg.TRAIN.LOG_FREQ == 0:
                print(f"Phase 1 Step {i} | Encoder Loss: {loss.item():.5f}")
                self.writer.add_scalar("Phase1/Encoder_Loss", loss.item(), i)

    def _step_env(self, action):
        """Helper to handle single/vec env API differences"""
        step_res = self.env.step(action)
        
        if self.num_envs > 1:
            next_obs, reward, done, infos = step_res
            state = np.stack([i['true_state_vector'] for i in infos])
            # For vec env, 'done' is usually a boolean array. 
            # SubprocVecEnv resets automatically, so we don't need manual reset.
            return next_obs, reward, done, state
        else:
            next_obs, reward, term, trunc, info = step_res
            done = float(term or trunc)
            state = info['true_state_vector']
            if term or trunc:
                next_obs, _ = self.env.reset()
            return next_obs, reward, done, state

    def _reset_env(self):
        res = self.env.reset()
        if isinstance(res, tuple):
            return res[0]
        return res

    def _log_metrics(self, info, step):
        print(f"Step {step} | Critic: {info['critic_loss']:.3f} | Actor: {info['actor_loss']:.3f}")
        for k, v in info.items():
            self.writer.add_scalar(f"Phase2/{k}", v, step)

    def evaluate(self, step):
        avg_reward = 0
        eval_episodes = 5
        
        for _ in range(eval_episodes):
            obs = self._reset_env()
            if self.num_envs > 1: obs = obs[0] # Just evaluate on one instance
            
            done = False
            ep_rew = 0
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
                    mu, _, _, _, _ = self.actor(obs_t)
                    # Deterministic action for eval
                    act = torch.tanh(mu).cpu().numpy()[0] * cfg.ROBOT.TORQUE_LIMITS
                
                # Step (handle vec env vs single env for eval)
                if self.num_envs > 1:
                     # Broadcast action to all envs to keep them stepping, but only track 0
                     step_act = np.tile(act, (self.num_envs, 1))
                     o2_b, r_b, d_b, _ = self.env.step(step_act)
                     obs, r, done = o2_b[0], r_b[0], d_b[0]
                else:
                    o2, r, term, trunc, _ = self.env.step(act)
                    obs, r, done = o2, r, term or trunc
                
                ep_rew += r
            avg_reward += ep_rew
        
        avg_reward /= eval_episodes
        print(f"Step {step} | Eval Reward: {avg_reward:.2f}")
        self.writer.add_scalar("Eval/Reward", avg_reward, step)
        
        if avg_reward > self.best_eval_reward:
            self.best_eval_reward = avg_reward
            self.save_checkpoint("best")
        self.save_checkpoint("latest")

    def save_checkpoint(self, label):
        path = self.output_dir / f"ckpt_{label}.pth"
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'algorithm': self.algorithm.update_step
        }, path)