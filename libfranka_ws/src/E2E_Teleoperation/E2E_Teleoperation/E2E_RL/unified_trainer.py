"""
Unified Trainer for End-to-End Teleoperation
- Phase 1: Pre-train LSTM on random data (Supervised Learning)
- Phase 2: Fine tuning (SAC) with Pre-trained LSTM (Reinforcement Learning)
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import json
import time
import logging

from torch.utils.tensorboard import SummaryWriter
from E2E_Teleoperation.E2E_RL.e2e_network import JointActor, JointCritic
import E2E_Teleoperation.config.robot_config as cfg

class ReplayBuffer:
    """
    Optimized Replay Buffer with GPU-friendly data transfer.
    """
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, 
                 state_dim: int, device: torch.device):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device
        
        # Use pinned memory for faster CPU->GPU transfer
        self.obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, pin_memory=True)
        self.next_obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, pin_memory=True)
        self.act_buf = torch.zeros((capacity, action_dim), dtype=torch.float32, pin_memory=True)
        self.rew_buf = torch.zeros(capacity, dtype=torch.float32, pin_memory=True)
        self.done_buf = torch.zeros(capacity, dtype=torch.float32, pin_memory=True)
        self.state_buf = torch.zeros((capacity, state_dim), dtype=torch.float32, pin_memory=True)
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            next_obs: np.ndarray, done: float, state: np.ndarray):
        """Add single transition"""
        self.obs_buf[self.ptr] = torch.from_numpy(obs)
        self.next_obs_buf[self.ptr] = torch.from_numpy(next_obs)
        self.act_buf[self.ptr] = torch.from_numpy(action)
        self.rew_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.state_buf[self.ptr] = torch.from_numpy(state)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def add_batch(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, 
                  next_obs: np.ndarray, done: np.ndarray, state: np.ndarray):
        """Add batch of transitions (for vectorized environments)"""
        N = obs.shape[0]
        indices = np.arange(self.ptr, self.ptr + N) % self.capacity
        
        self.obs_buf[indices] = torch.from_numpy(obs)
        self.next_obs_buf[indices] = torch.from_numpy(next_obs)
        self.act_buf[indices] = torch.from_numpy(action)
        self.rew_buf[indices] = torch.from_numpy(reward)
        self.done_buf[indices] = torch.from_numpy(done)
        self.state_buf[indices] = torch.from_numpy(state)
        
        self.ptr = (self.ptr + N) % self.capacity
        self.size = min(self.size + N, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch with efficient GPU transfer"""
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
    """
    Unified trainer with comprehensive logging, optimization, and 2-Phase Training.
    """
    def __init__(self, env, output_dir: Path, load_checkpoint: Optional[str] = None):
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self._setup_file_logging()
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = getattr(env, "num_envs", 1)
        
        self.log_info(f"Device: {self.device}")
        self.log_info(f"Number of parallel environments: {self.num_envs}")
        
        # Initialize networks
        self.actor = JointActor().to(self.device)
        self.critic = JointCritic().to(self.device)
        self.target_critic = JointCritic().to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Log network architecture
        self._log_network_info()
        
        # Initialize optimizers (Note: Using cfg.TRAIN.* paths)
        self._setup_optimizers()
        
        # Entropy temperature
        self.target_entropy = -float(cfg.N_JOINTS) * cfg.SAC.TARGET_ENTROPY_RATIO
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.SAC.ALPHA_LR)
        
        # Replay buffer
        self.buffer = ReplayBuffer(
            cfg.TRAIN.BUFFER_SIZE, 
            cfg.ROBOT.RL_OBS_DIM, 
            cfg.ROBOT.N_JOINTS, 
            cfg.ROBOT.ROBOT_STATE_DIM, 
            self.device
        )
        
        # Loss function for Phase 1 (Supervised Learning)
        self.sl_loss = nn.MSELoss()
        
        # Training hyperparameters
        self.gamma = cfg.TRAIN.GAMMA
        self.tau = cfg.SAC.TARGET_TAU
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        
        # Training statistics
        self.global_step = 0
        self.update_count = 0
        self.episode_count = 0
        self.best_eval_reward = -np.inf
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.update_times = []
        
        # Load checkpoint if provided
        if load_checkpoint:
            self._load_checkpoint(load_checkpoint)
        
        self.log_info("Trainer initialized successfully")

    def _setup_file_logging(self):
        """Setup file logging for training progress"""
        self.logger = logging.getLogger("E2E_Training")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates on re-init
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        log_file = self.output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_info(self, message: str):
        self.logger.info(message)
    
    def _log_network_info(self):
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        encoder_params = sum(p.numel() for p in self.actor.encoder.parameters())
        
        self.log_info(f"Actor parameters: {actor_params:,}")
        self.log_info(f"LSTM Encoder parameters: {encoder_params:,}")
        self.log_info(f"Critic parameters: {critic_params:,}")
    
    def _setup_optimizers(self):
        """Setup optimizers with differential learning rates"""
        # Actor optimizer with differential LR
        actor_params = [
            {'params': self.actor.encoder.parameters(), 'lr': cfg.TRAIN.ENCODER_LR},
            {'params': self.actor.net.parameters(), 'lr': cfg.TRAIN.ACTOR_LR},
            {'params': self.actor.mu.parameters(), 'lr': cfg.TRAIN.ACTOR_LR},
            {'params': self.actor.log_std.parameters(), 'lr': cfg.TRAIN.ACTOR_LR},
        ]
        self.actor_optimizer = optim.Adam(actor_params)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        
        self.log_info(f"Optimizer configuration:")
        self.log_info(f"  Encoder LR: {cfg.TRAIN.ENCODER_LR}")
        self.log_info(f"  Actor LR: {cfg.TRAIN.ACTOR_LR}")
        self.log_info(f"  Critic LR: {cfg.TRAIN.CRITIC_LR}")

    def train_e2e(self):
        """Main training loop"""
        
        # ==========================================
        # PHASE 1: PRE-TRAINING
        # ==========================================
        self.log_info("\n" + "="*70)
        self.log_info("PHASE 1: LSTM PRE-TRAINING (SUPERVISED LEARNING)")
        self.log_info("="*70)
        
        # 1. Collect Random Data
        # We need a decent amount of data to train the LSTM physics predictor
        self._random_exploration_phase(num_steps=5000)
        
        # 2. Train Encoder
        # This aligns the LSTM state predictions before RL starts
        self._train_encoder_supervised(steps=cfg.TRAIN.STAGE1_STEPS)
        
        self.save_checkpoint("phase1_pretrained", eval_reward=0.0)
        
        # ==========================================
        # PHASE 2: RL FINE-TUNING
        # ==========================================
        self.log_info("\n" + "="*70)
        self.log_info("PHASE 2: END-TO-END SAC TRAINING")
        self.log_info("="*70)
        self.log_info(f"Total timesteps: {cfg.TOTAL_TIMESTEPS}")
        self.log_info(f"Batch size: {self.batch_size}")
        
        obs = self._reset_env()
        episode_reward = 0.0
        episode_length = 0
        start_time = time.time()
        
        # Calculate iteration count based on parallel envs
        num_iterations = cfg.TOTAL_TIMESTEPS // self.num_envs
        
        for step in range(1, num_iterations + 1):
            
            # 1. Select Action
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                if self.num_envs == 1:
                    obs_t = obs_t.unsqueeze(0)
                
                action_t, _, _, _, _ = self.actor.sample(obs_t)
                action = action_t.cpu().numpy()
                
                if self.num_envs == 1:
                    action = action[0]
            
            # 2. Step Environment
            next_obs, reward, done, info = self._env_step(action)
            
            # 3. Store Transition
            self._store_transition(obs, action, reward, next_obs, done, info)
            
            # 4. Update Stats
            if self.num_envs == 1:
                episode_reward += reward
                episode_length += 1
            else:
                episode_reward += reward[0] # Just track first env for rough progress
                episode_length += 1
            
            obs = next_obs
            self.global_step += self.num_envs
            
            # 5. Update Networks
            if self.buffer.size > self.batch_size:
                update_info = self.update_sac()
                
                if self.update_count % cfg.TRAIN.LOG_FREQ == 0:
                    self._log_training_metrics(update_info, start_time)
            
            # 6. Handle Episode End
            # Check if any env is done (for vec envs, they auto-reset, so 'done' is transient)
            is_done = done if self.num_envs == 1 else done[0]
            
            if is_done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_count += 1
                
                self.writer.add_scalar('Episode/Reward', episode_reward, self.global_step)
                self.writer.add_scalar('Episode/Length', episode_length, self.global_step)
                
                if self.episode_count % 10 == 0:
                    avg_rew = np.mean(self.episode_rewards[-10:])
                    self.log_info(f"Ep {self.episode_count} | Rew: {episode_reward:.1f} | Avg(10): {avg_rew:.1f}")
                
                # Reset stats
                episode_reward = 0.0
                episode_length = 0
                if self.num_envs == 1: obs = self._reset_env()
            
            # 7. Evaluate
            if self.global_step % cfg.EVAL_INTERVAL == 0:
                self.evaluate()
        
        self.log_info("\nTRAINING COMPLETED")
        self._log_final_statistics()
        self.writer.close()
        self.env.close()

    def _random_exploration_phase(self, num_steps: int):
        """Collect random transitions to fill buffer"""
        self.log_info(f"Collecting {num_steps} random transitions...")
        obs = self._reset_env()
        collected = 0
        
        while collected < num_steps:
            if self.num_envs > 1:
                action = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
            else:
                action = self.env.action_space.sample()
            
            next_obs, reward, done, info = self._env_step(action)
            self._store_transition(obs, action, reward, next_obs, done, info)
            
            obs = next_obs
            collected += self.num_envs
            
            # Handle reset for single env
            if self.num_envs == 1 and done:
                obs = self._reset_env()
        
        self.log_info(f"Buffer size: {self.buffer.size}")

    def _train_encoder_supervised(self, steps: int):
        """
        Phase 1: Train ONLY the LSTM encoder using Supervised Learning.
        Objective: Minimize MSE(Predicted_State, True_Physics_State)
        """
        self.log_info(f"Starting Supervised Pre-training for {steps} steps...")
        self.actor.train()
        
        for i in range(1, steps + 1):
            batch = self.buffer.sample(self.batch_size)
            
            # Forward pass to get state prediction
            # We ignore action/log_prob here
            _, _, pred_state, _, _ = self.actor.sample(batch['obs'])
            
            # Calculate Loss
            loss = self.sl_loss(pred_state, batch['true_state_vector'])
            
            # Optimize
            self.actor_optimizer.zero_grad()
            loss.backward()
            # Clip for stability
            nn.utils.clip_grad_norm_(self.actor.encoder.parameters(), 1.0)
            self.actor_optimizer.step()
            
            if i % cfg.TRAIN.LOG_FREQ == 0:
                self.writer.add_scalar('Phase1/Encoder_Loss', loss.item(), i)
                if i % 2000 == 0:
                    self.log_info(f"Phase 1 Step {i} | Encoder Loss: {loss.item():.6f}")

    def _env_step(self, action) -> Tuple:
        """Unified environment step"""
        step_res = self.env.step(action)
        if self.num_envs > 1:
            next_obs, reward, done, infos = step_res
            return next_obs, reward, done, infos
        else:
            next_obs, reward, term, trunc, info = step_res
            done = term or trunc
            return next_obs, reward, done, info
    
    def _reset_env(self):
        res = self.env.reset()
        if isinstance(res, tuple):
            obs, _ = res
        else:
            obs = res
        return obs
    
    def _store_transition(self, obs, action, reward, next_obs, done, info):
        if self.num_envs > 1:
            state = np.stack([i['true_state_vector'] for i in info])
            self.buffer.add_batch(obs, action, reward, next_obs, done, state)
        else:
            self.buffer.add(obs, action, reward, next_obs, float(done), 
                          info['true_state_vector'])

    def update_sac(self) -> Dict[str, float]:
        """Single SAC update step"""
        self.update_count += 1
        update_start = time.time()
        
        batch = self.buffer.sample(self.batch_size)
        
        # --- 1. Critic Update ---
        with torch.no_grad():
            next_action, next_log_prob, next_pred_state, _, _ = self.actor.sample(batch['next_obs'])
            target_Q1, target_Q2 = self.target_critic(next_pred_state, next_action)
            alpha = self.log_alpha.exp()
            target_V = torch.min(target_Q1, target_Q2) - alpha * next_log_prob
            target_Q = batch['rewards'] + (1.0 - batch['dones']) * self.gamma * target_V
        
        _, _, curr_pred_state, _, _ = self.actor.sample(batch['obs'])
        curr_Q1, curr_Q2 = self.critic(curr_pred_state.detach(), batch['actions'])
        
        critic_loss = nn.MSELoss()(curr_Q1, target_Q) + nn.MSELoss()(curr_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.SAC.GRAD_CLIP_CRITIC)
        self.critic_optimizer.step()
        
        # --- 2. Actor Update ---
        new_action, log_prob, pred_state, _, _ = self.actor.sample(batch['obs'])
        
        # SAC Loss
        q1_new, q2_new = self.critic(pred_state, new_action)
        q_new = torch.min(q1_new, q2_new)
        sac_loss = (alpha.detach() * log_prob - q_new).mean()
        
        # Auxiliary Prediction Loss (keep physics consistent)
        pred_loss = self.sl_loss(pred_state, batch['true_state_vector'])
        total_actor_loss = sac_loss + (5.0 * pred_loss) # Weight 5.0
        
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.SAC.GRAD_CLIP_ACTOR)
        self.actor_optimizer.step()
        
        # --- 3. Alpha Update ---
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # --- 4. Target Soft Update ---
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        update_time = time.time() - update_start
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': sac_loss.item(),
            'pred_loss': pred_loss.item(),
            'alpha': alpha.item(),
            'critic_grad_norm': critic_grad_norm.item(),
            'actor_grad_norm': actor_grad_norm.item(),
            'update_time': update_time
        }

    def _log_training_metrics(self, metrics: Dict[str, float], start_time: float):
        """Log metrics to TensorBoard and Console"""
        # TensorBoard
        for k, v in metrics.items():
            if 'loss' in k or 'norm' in k or 'alpha' in k:
                self.writer.add_scalar(f"Train/{k}", v, self.global_step)
        
        # Performance
        elapsed = time.time() - start_time
        sps = self.global_step / (elapsed + 1e-6)
        self.writer.add_scalar('Performance/StepsPerSec', sps, self.global_step)
        
        # Console (Sparse)
        self.log_info(
            f"Step {self.global_step:7d} | "
            f"CritLoss: {metrics['critic_loss']:6.3f} | "
            f"ActLoss: {metrics['actor_loss']:6.3f} | "
            f"PredLoss: {metrics['pred_loss']:6.3f} | "
            f"Alpha: {metrics['alpha']:5.3f} | "
            f"SPS: {sps:4.0f}"
        )

    def evaluate(self):
        """Evaluation Loop"""
        self.log_info(f"--- Evaluation at Step {self.global_step} ---")
        eval_rewards = []
        
        # Run 3 eval episodes
        for _ in range(3):
            obs = self._reset_env()
            if self.num_envs > 1: obs = obs[0]
            
            done = False
            ep_rew = 0
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32).view(1, -1)
                    mu, _, _, _, _ = self.actor(obs_t)
                    # Deterministic action
                    act = torch.tanh(mu).cpu().numpy()[0] * cfg.ROBOT.TORQUE_LIMITS
                
                # Step (handle single/vec differences)
                if self.num_envs > 1:
                     step_res = self.env.step(np.tile(act, (self.num_envs, 1)))
                     obs, r, done = step_res[0][0], step_res[1][0], step_res[2][0]
                else:
                    step_res = self.env.step(act)
                    obs, r, done = step_res[0], step_res[1], step_res[2] or step_res[3]
                
                ep_rew += r
            eval_rewards.append(ep_rew)
        
        avg_reward = np.mean(eval_rewards)
        self.writer.add_scalar('Eval/Reward', avg_reward, self.global_step)
        self.log_info(f"Eval Average Reward: {avg_reward:.2f}")
        
        if avg_reward > self.best_eval_reward:
            self.best_eval_reward = avg_reward
            self.save_checkpoint("best", avg_reward)
        self.save_checkpoint("latest", avg_reward)

    def save_checkpoint(self, label: str, eval_reward: float):
        path = self.output_dir / f"checkpoint_{label}.pth"
        torch.save({
            'global_step': self.global_step,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'best_eval_reward': self.best_eval_reward,
        }, path)

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.global_step = checkpoint['global_step']
        self.best_eval_reward = checkpoint.get('best_eval_reward', -np.inf)
        self.log_info(f"Loaded checkpoint from {path} (Step {self.global_step})")

    def _log_final_statistics(self):
        if len(self.episode_rewards) > 0:
            self.log_info(f"Final Best Eval Reward: {self.best_eval_reward:.2f}")
            self.log_info(f"Total Episodes: {self.episode_count}")