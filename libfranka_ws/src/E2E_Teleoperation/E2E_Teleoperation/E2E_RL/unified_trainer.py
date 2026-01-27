"""
Unified Trainer (Fixed Version)
-------------------------------
Key Fixes:
1. Proper alpha initialization for fine-tuning from BC
2. Separate evaluation environment to prevent shape mismatch
3. Better warmup strategy using BC policy instead of random
4. Gradient monitoring and early divergence detection
5. Learning rate scheduling for stability
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- Project Imports ---
from E2E_Teleoperation.E2E_RL.e2e_network import JointActor, JointCritic
from E2E_Teleoperation.E2E_RL.e2e_algorithm import ResidualSAC
import E2E_Teleoperation.config.robot_config as cfg


class ReplayBuffer:
    """
    Experience Replay Buffer for Off-Policy RL.
    Storage on CPU to prevent GPU OOM.
    """

    def __init__(
        self, 
        capacity: int, 
        obs_dim: int, 
        action_dim: int, 
        state_dim: int, 
        training_device: torch.device
    ):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.training_device = training_device
        self.storage_device = torch.device("cpu")
        
        print(f"[Memory] Initializing ReplayBuffer on {self.storage_device} (Capacity: {capacity})...")
        self.obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.storage_device)
        self.next_obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.storage_device)
        self.act_buf = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.storage_device)
        self.rew_buf = torch.zeros(capacity, dtype=torch.float32, device=self.storage_device)
        self.done_buf = torch.zeros(capacity, dtype=torch.float32, device=self.storage_device)
        self.state_buf = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.storage_device)
    
    def add(
        self, 
        obs: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_obs: np.ndarray, 
        done: bool, 
        state: np.ndarray
    ) -> None:
        self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.storage_device)
        self.next_obs_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.storage_device)
        self.act_buf[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.storage_device)
        self.rew_buf[self.ptr] = float(reward)
        self.done_buf[self.ptr] = float(done)
        self.state_buf[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.storage_device)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def add_batch(
        self, 
        obs: np.ndarray, 
        action: np.ndarray, 
        reward: np.ndarray, 
        next_obs: np.ndarray, 
        done: np.ndarray, 
        state: np.ndarray
    ) -> None:
        N = obs.shape[0]
        indices = np.arange(self.ptr, self.ptr + N) % self.capacity
        
        self.obs_buf[indices] = torch.as_tensor(obs, dtype=torch.float32, device=self.storage_device)
        self.next_obs_buf[indices] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.storage_device)
        self.act_buf[indices] = torch.as_tensor(action, dtype=torch.float32, device=self.storage_device)
        self.rew_buf[indices] = torch.as_tensor(reward, dtype=torch.float32, device=self.storage_device)
        self.done_buf[indices] = torch.as_tensor(done, dtype=torch.float32, device=self.storage_device)
        
        state_stacked = np.array(state)
        self.state_buf[indices] = torch.as_tensor(state_stacked, dtype=torch.float32, device=self.storage_device)
        
        self.ptr = (self.ptr + N) % self.capacity
        self.size = min(self.size + N, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = torch.randint(0, self.size, (batch_size,))
        
        return {
            'obs': self.obs_buf[idxs].to(self.training_device),
            'actions': self.act_buf[idxs].to(self.training_device),
            'rewards': self.rew_buf[idxs].unsqueeze(1).to(self.training_device),
            'next_obs': self.next_obs_buf[idxs].to(self.training_device),
            'dones': self.done_buf[idxs].unsqueeze(1).to(self.training_device),
            'true_state_vector': self.state_buf[idxs].to(self.training_device)
        }


class UnifiedTrainer:
    """
    Main Trainer Class for E2E Residual SAC (Fixed Version).
    """

    def __init__(self, env: Any, output_dir: Path, eval_env: Optional[Any] = None):
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 1. Setup Logging ---
        self._setup_logging()
        
        # --- 2. Initialize Models ---
        self.actor = JointActor().to(self.device)
        self.critic = JointCritic().to(self.device)
        self.critic_target = JointCritic().to(self.device)
        
        self._load_pretrained_checkpoint()
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # --- 3. Optimizers with different LR for encoder vs policy ---
        # [FIX] Separate parameter groups for better fine-tuning control
        encoder_param_ids = {p.data_ptr() for p in self.actor.base_encoder.parameters()}
        policy_params = [p for p in self.actor.parameters() if p.data_ptr() not in encoder_param_ids]
        encoder_params = list(self.actor.base_encoder.parameters())

        self.actor_optimizer = optim.Adam([
            {'params': encoder_params, 'lr': cfg.TRAIN.ENCODER_LR},
            {'params': policy_params, 'lr': cfg.TRAIN.ACTOR_LR}
        ])
                
        self.actor_optimizer = optim.Adam([
            {'params': encoder_params, 'lr': cfg.TRAIN.ENCODER_LR},  # Lower LR for pretrained LSTM
            {'params': policy_params, 'lr': cfg.TRAIN.ACTOR_LR}
        ])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        
        # [FIX] Initialize alpha lower for fine-tuning (less exploration needed)
        # alpha = exp(log_alpha), so log_alpha = -2 gives alpha ≈ 0.135
        initial_log_alpha = math.log(cfg.SAC.INITIAL_ALPHA)
        self.log_alpha = torch.tensor([initial_log_alpha], requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)
        
        # --- 4. Algorithm & Buffer ---
        self.sac = ResidualSAC(
            self.actor, self.critic, self.critic_target,
            self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer,
            self.log_alpha, 
            gamma=cfg.TRAIN.GAMMA, 
            tau=cfg.SAC.TARGET_TAU,
            reward_scale=cfg.SAC.REWARD_SCALE
        )
        
        self.buffer = ReplayBuffer(
            cfg.TRAIN.BUFFER_SIZE, cfg.ROBOT.RL_OBS_DIM, cfg.ROBOT.N_JOINTS,
            cfg.ROBOT.ESTIMATOR_OUTPUT_DIM, self.device
        )
        
        # Environment Handling
        self.num_envs = getattr(self.env, "num_envs", 1)
        self.warmup_steps = cfg.TRAIN.WARMUP_STEPS // self.num_envs

        # Tracking
        self.global_step = 0
        self.best_eval_reward = -float('inf')
        
        # [NEW] Gradient monitoring
        self._grad_norms = {'actor': [], 'critic': []}
        self._loss_history = {'actor': [], 'critic': [], 'pred': []}
        
        # [NEW] Learning rate scheduler for stability
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.actor_optimizer, T_0=50000, T_mult=2, eta_min=1e-6
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.critic_optimizer, T_0=50000, T_mult=2, eta_min=1e-5
        )

    def _setup_logging(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
        self.logger = logging.getLogger(__name__)
        log_dir = self.output_dir / "files"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.logger.handlers:
            self.logger.addHandler(logging.FileHandler(log_dir / "training.log"))
            self.logger.addHandler(logging.StreamHandler(sys.stdout))
        
        self.writer = SummaryWriter(log_dir=str(self.output_dir))

    def _load_pretrained_checkpoint(self) -> None:
        """Loads pre-trained BC weights safely."""
        ckpt_path = cfg.ROBOT.PRETRAINED_ACTOR_PATH
        if ckpt_path.exists():
            try:
                self.logger.info(f">>> Loading Checkpoint: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                
                if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                    self.actor.load_state_dict(checkpoint['actor'])
                    self.logger.info(">>> Loaded 'actor' state_dict successfully.")
                    
                    # [NEW] Log checkpoint info
                    if 'epoch' in checkpoint:
                        self.logger.info(f"    BC Epoch: {checkpoint['epoch']}, Loss: {checkpoint.get('loss', 'N/A')}")
                else:
                    self.actor.load_state_dict(checkpoint)
                    self.logger.info(">>> Loaded raw state_dict successfully.")
                    
            except Exception as e:
                self.logger.error(f">>> FAILED to load checkpoint: {e}")
                self.logger.warning(">>> Starting from SCRATCH (Random Weights). Risky!")
        else:
            self.logger.warning(f">>> No checkpoint found at {ckpt_path}. Starting from scratch.")

    def _warmup(self, use_policy: bool = True) -> np.ndarray:
        """
        Fills the replay buffer before training.
        
        [FIX] Option to use BC policy for warmup instead of random actions.
        This provides better initial Q-value estimates since the data comes
        from a reasonable policy rather than random noise.
        """
        self.logger.info(f">>> Starting Warmup Phase (use_policy={use_policy})...")
        obs, _ = self.env.reset()
        
        self.actor.eval()  # Disable dropout etc.
        
        for step in range(self.warmup_steps):
            if use_policy:
                # Use the BC-pretrained policy (deterministic for stability)
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device)
                    if self.num_envs == 1 and obs_t.ndim == 1:
                        obs_t = obs_t.unsqueeze(0)
                    mu, log_std, _, _, _ = self.actor(obs_t)
                    # Add small noise for exploration during warmup
                    noise_scale = 0.1 * (1 - step / self.warmup_steps)  # Decay noise
                    noise = torch.randn_like(mu) * noise_scale
                    actions = torch.tanh(mu + noise) * self.actor.action_scale
                    actions = actions.cpu().numpy()
                    if self.num_envs == 1:
                        actions = actions[0]
            else:
                # Random actions (original behavior)
                if self.num_envs == 1:
                    actions = self.env.action_space.sample()
                else:
                    actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
            
            if self.num_envs == 1:
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
                dones = terminated or truncated
                self.buffer.add(obs, actions, rewards, next_obs, dones, infos['true_state_vector'])
                
                if dones:
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs
            else:
                next_obs, rewards, dones, infos = self.env.step(actions)
                true_states = [i['true_state_vector'] for i in infos]
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, true_states)
                obs = next_obs
            
            self.global_step += self.num_envs
            
            # Log warmup progress
            if step % 1000 == 0:
                self.logger.info(f"    Warmup: {step}/{self.warmup_steps} steps")
        
        self.actor.train()
        self.logger.info(f">>> Warmup Complete. Buffer size: {self.buffer.size}")
        return obs

    def train_e2e(self) -> None:
        """Main Training Loop."""
        self.logger.info(f">>> E2E RL STARTED | Mode: {'JOINT' if cfg.TRAIN.JOINT_OPTIMIZATION else 'DECOUPLED'}")
        self.logger.info(f"    Initial Alpha: {self.log_alpha.exp().item():.4f}")
        self.logger.info(f"    Target Entropy: {self.sac.target_entropy:.2f}")
        
        # Use policy-based warmup for better initial Q-estimates
        obs = self._warmup(use_policy=True)
        grad_updates_pending = 0
        
        # Early stopping tracking
        no_improvement_steps = 0
        best_tracking_error = float('inf')
        
        pbar = tqdm(initial=self.global_step, total=cfg.TRAIN.TOTAL_TIMESTEPS, desc="Training")
        
        while self.global_step < cfg.TRAIN.TOTAL_TIMESTEPS:
            # --- 1. Select Action ---
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device)
                if self.num_envs == 1 and obs_t.ndim == 1:
                    obs_t = obs_t.unsqueeze(0)
                
                actions_tensor, _, pred_leader_tensor, _, _ = self.actor.sample(obs_t)
                actions = actions_tensor.cpu().numpy()
                pred_leader_np = pred_leader_tensor.cpu().numpy()

            # --- 2. Environment Step ---
            if self.num_envs == 1:
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions[0])
                dones = terminated or truncated
                self.buffer.add(obs, actions[0], rewards, next_obs, dones, infos['true_state_vector'])
                
                # Debug Info
                true_leader_q = infos['leader_q']
                pred_leader_q = (pred_leader_np[0][:7] * cfg.ROBOT.Q_STD) + cfg.ROBOT.Q_MEAN
                follower_q = infos['follower_q']
                
                if dones:
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs
            else:
                next_obs, rewards, dones, infos = self.env.step(actions)
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, [i['true_state_vector'] for i in infos])
                true_leader_q, pred_leader_q, follower_q = None, None, None
                obs = next_obs

            self.global_step += self.num_envs
            grad_updates_pending += self.num_envs
            
            # --- 3. Gradient Update ---
            if grad_updates_pending >= cfg.TRAIN.TRAIN_FREQUENCY:
                metrics = self._update_policy(grad_updates_pending)
                grad_updates_pending = 0
                
                # Update LR schedulers
                self.actor_scheduler.step()
                self.critic_scheduler.step()
                
                # Log Metrics
                if self.global_step % cfg.TRAIN.LOG_FREQ == 0:
                    self._log_metrics(metrics, rewards, true_leader_q, pred_leader_q, follower_q)
                    
                    # Check for training divergence
                    if self._check_divergence(metrics):
                        self.logger.warning(">>> Training divergence detected! Saving checkpoint and reducing LR...")
                        self._save_checkpoint(is_best=False, suffix="_divergence")
                        # Reduce learning rates
                        for param_group in self.actor_optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        for param_group in self.critic_optimizer.param_groups:
                            param_group['lr'] *= 0.5

            # --- 4. Evaluation & Checkpointing ---
            if self.global_step % cfg.TRAIN.EVAL_INTERVAL == 0:
                eval_metrics = self._evaluate_and_save()
                
                # Early stopping check
                if cfg.TRAIN.ENABLE_EARLY_STOP:
                    if eval_metrics['tracking_error'] < best_tracking_error - cfg.TRAIN.EARLY_STOP_MIN_DELTA:
                        best_tracking_error = eval_metrics['tracking_error']
                        no_improvement_steps = 0
                    else:
                        no_improvement_steps += cfg.TRAIN.EVAL_INTERVAL
                        
                    if no_improvement_steps >= cfg.TRAIN.EARLY_STOP_PATIENCE * cfg.TRAIN.EVAL_INTERVAL:
                        self.logger.info(f">>> Early stopping triggered. No improvement for {no_improvement_steps} steps.")
                        break
            
            pbar.update(self.num_envs)
        
        pbar.close()
        self._save_checkpoint(is_best=False, suffix="_final")
        self.logger.info(">>> Training Finished.")

    def _update_policy(self, pending_steps: int) -> Dict[str, float]:
        """Performs SAC gradient updates."""
        updates = np.clip(int(pending_steps * 0.5), 1, 64)
        last_metrics = {}
        
        for _ in range(updates):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            metrics = self.sac.update(batch, update_actor=True)
            last_metrics = metrics
            
            # Record losses for divergence detection
            self._loss_history['actor'].append(metrics.get('actor_loss', 0))
            self._loss_history['critic'].append(metrics.get('critic_loss', 0))
            self._loss_history['pred'].append(metrics.get('pred_loss', 0))
            
            # Keep only recent history
            for key in self._loss_history:
                if len(self._loss_history[key]) > 1000:
                    self._loss_history[key] = self._loss_history[key][-500:]
            
            # NaN Guard
            if np.isnan(metrics.get('actor_loss', 0)) or np.isnan(metrics.get('pred_loss', 0)):
                self._handle_nan_crash(metrics)
        
        return last_metrics

    def _check_divergence(self, metrics: Dict[str, float]) -> bool:
        """Check for signs of training divergence."""
        # Check Q-value explosion
        if abs(metrics.get('q1_mean', 0)) > 1000:
            self.logger.warning(f"Q-value explosion: {metrics.get('q1_mean', 0):.2f}")
            return True
        
        # Check loss spike
        if len(self._loss_history['critic']) > 100:
            recent_mean = np.mean(self._loss_history['critic'][-100:])
            overall_mean = np.mean(self._loss_history['critic'])
            if recent_mean > 10 * overall_mean and overall_mean > 0:
                self.logger.warning(f"Critic loss spike: recent={recent_mean:.2f}, overall={overall_mean:.2f}")
                return True
        
        return False

    def _handle_nan_crash(self, metrics: Dict[str, float]) -> None:
        """Dumps state and exits upon NaN detection."""
        self.logger.error("\n[FATAL] NaN DETECTED IN LOSS! IMMEDIATE DUMP:")
        self.logger.error(f"Step: {self.global_step} | Metrics: {metrics}")
        self._save_checkpoint(is_best=False, suffix="_CRASH")
        sys.exit(1)

    def _log_metrics(self, metrics, rewards, true_q, pred_q, follow_q) -> None:
        """Logs training stats to console and TensorBoard."""
        pred_err = 0.0
        track_err = 0.0
        
        if self.num_envs == 1 and true_q is not None:
            pred_err = np.mean(np.abs(true_q - pred_q))
            track_err = np.mean(np.abs(true_q - follow_q))

        # Console Log
        self.logger.info(
            f"Step {self.global_step} | R: {np.mean(rewards):.2f} | "
            f"Actor_L: {metrics.get('actor_loss', 0):.3f} | Pred_L: {metrics.get('pred_loss', 0):.3f} | "
            f"Alpha: {metrics.get('alpha', 0):.3f}"
        )
        self.logger.info(f"   >>> Q: mean={metrics.get('q1_mean', 0):.2f}, min={metrics.get('q1_min', 0):.2f}, max={metrics.get('q1_max', 0):.2f}")
        self.logger.info(f"   >>> [Debug] Pred_Err: {pred_err:.4f} | Track_Err: {track_err:.4f}")
        
        # TensorBoard Log
        self.writer.add_scalar("Train/Reward", np.mean(rewards), self.global_step)
        self.writer.add_scalar("Train/Pred_Error_Rad", pred_err, self.global_step)
        self.writer.add_scalar("Train/Tracking_Error_Rad", track_err, self.global_step)
        self.writer.add_scalar("Loss/Actor", metrics.get('actor_loss', 0), self.global_step)
        self.writer.add_scalar("Loss/Critic", metrics.get('critic_loss', 0), self.global_step)
        self.writer.add_scalar("Loss/Prediction", metrics.get('pred_loss', 0), self.global_step)
        self.writer.add_scalar("SAC/Alpha", metrics.get('alpha', 0), self.global_step)
        self.writer.add_scalar("SAC/Q_mean", metrics.get('q1_mean', 0), self.global_step)
        self.writer.add_scalar("SAC/LogProb", metrics.get('log_prob_mean', 0), self.global_step)
        
        # Learning rates
        self.writer.add_scalar("LR/Actor", self.actor_optimizer.param_groups[0]['lr'], self.global_step)
        self.writer.add_scalar("LR/Critic", self.critic_optimizer.param_groups[0]['lr'], self.global_step)

    def _evaluate_and_save(self) -> Dict[str, float]:
        """Runs evaluation episodes and saves checkpoints."""
        eval_metrics = self._run_evaluation_episodes()
        
        if eval_metrics['reward'] > self.best_eval_reward:
            self.best_eval_reward = eval_metrics['reward']
            self._save_checkpoint(is_best=True)
            self.logger.info(f">>> New Best Model Saved (Reward: {eval_metrics['reward']:.2f})")
        
        self._save_checkpoint(is_best=False)
        return eval_metrics

    def _run_evaluation_episodes(self) -> Dict[str, float]:
        """Executes evaluation loop without noise/exploration."""
        eval_rewards = []
        eval_tracking_errors = []
        
        self.actor.eval()
        
        for _ in range(cfg.EVAL.NUM_EPISODES):
            obs, _ = self.eval_env.reset()
            total_rew = 0
            total_track_err = 0
            steps = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                    mu, _, _, _, _ = self.actor(obs_t)
                    # Deterministic Action for Eval
                    action = torch.tanh(mu) * self.actor.action_scale
                    action_np = action.cpu().numpy()[0]
                
                next_obs, reward, terminated, truncated, info = self.eval_env.step(action_np)
                done = terminated or truncated
                total_rew += reward
                
                # Track position error
                if 'leader_q' in info and 'follower_q' in info:
                    track_err = np.linalg.norm(info['leader_q'] - info['follower_q'])
                    total_track_err += track_err
                    steps += 1
                
                obs = next_obs
            
            eval_rewards.append(total_rew)
            if steps > 0:
                eval_tracking_errors.append(total_track_err / steps)
        
        self.actor.train()
        
        avg_eval = np.mean(eval_rewards)
        avg_track_err = np.mean(eval_tracking_errors) if eval_tracking_errors else 0
        
        self.writer.add_scalar("Eval/Reward", avg_eval, self.global_step)
        self.writer.add_scalar("Eval/Tracking_Error", avg_track_err, self.global_step)
        
        self.logger.info(f">>> Eval: Reward={avg_eval:.2f}, TrackErr={avg_track_err:.4f}")
        
        return {'reward': float(avg_eval), 'tracking_error': float(avg_track_err)}

    def _save_checkpoint(self, is_best: bool = False, suffix: str = "") -> None:
        if is_best:
            filename = "best_model.pth"
        else:
            filename = f"latest_model{suffix}.pth"
        
        # Save comprehensive checkpoint
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'global_step': self.global_step,
            'best_eval_reward': self.best_eval_reward,
        }
        torch.save(checkpoint, self.output_dir / filename)