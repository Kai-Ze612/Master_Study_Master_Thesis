"""
Unified Trainer (Professional Edition)
--------------------------------------
Orchestrates the End-to-End Reinforcement Learning training loop using Residual SAC.

Features:
- Modular Architecture: Separates warmup, training, logging, and evaluation.
- CPU Replay Buffer: Mitigates CUDA OOM errors by storing large history on RAM.
- NaN Guard: Automatically detects loss divergence and saves crash checkpoints.
- Robust Checkpointing: Handles strict PyTorch security settings for safe loading.
- [FIX] NumPy Compatibility: Replaced deprecated np.float with python float.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union

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
    
    Storage Strategy: 
    - Stores transition data on CPU RAM to prevent GPU OOM errors given large sequence history.
    - Moves only sampled batches to GPU during training.
    """

    def __init__(
        self, 
        capacity: int, 
        obs_dim: int, 
        action_dim: int, 
        state_dim: int, 
        training_device: torch.device
    ):
        """
        Args:
            capacity: Max number of transitions.
            obs_dim: Dimension of observation vector.
            action_dim: Dimension of action vector.
            state_dim: Dimension of true state vector (for auxiliary loss).
            training_device: Device where training happens (e.g., 'cuda:0').
        """
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.training_device = training_device
        self.storage_device = torch.device("cpu")
        
        # Pre-allocate memory on CPU
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
        """Adds a single transition to the buffer."""
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
        """Adds a batch of transitions (from vectorized envs) to the buffer."""
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
        """Samples a batch and moves it to the training device (GPU)."""
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
    Main Trainer Class for E2E Residual SAC.
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
        
        # --- 3. Optimizers ---
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.TRAIN.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)
        
        # --- 4. Algorithm & Buffer ---
        self.sac = ResidualSAC(
            self.actor, self.critic, self.critic_target,
            self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer,
            self.log_alpha, gamma=cfg.TRAIN.GAMMA, tau=cfg.SAC.TARGET_TAU
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
        # [FIX] Use float('inf') instead of np.float('inf')
        self.best_eval_reward = -float('inf')

    def _setup_logging(self) -> None:
        """Configures Python logging and TensorBoard."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
        self.logger = logging.getLogger(__name__)
        log_dir = self.output_dir / "files"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.logger.handlers:
            self.logger.addHandler(logging.FileHandler(log_dir / "training.log"))
            self.logger.addHandler(logging.StreamHandler(sys.stdout))
        
        self.writer = SummaryWriter(log_dir=str(self.output_dir))

    def _load_pretrained_checkpoint(self) -> None:
        """Loads pre-trained BC weights safely, handling PyTorch 2.6 security."""
        ckpt_path = cfg.ROBOT.PRETRAINED_ACTOR_PATH
        if ckpt_path.exists():
            try:
                self.logger.info(f">>> Loading Checkpoint: {ckpt_path}")
                # weights_only=False allows loading Numpy arrays/Dicts
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                
                if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                    self.actor.load_state_dict(checkpoint['actor'])
                    self.logger.info(">>> Loaded 'actor' state_dict successfully.")
                else:
                    self.actor.load_state_dict(checkpoint)
                    self.logger.info(">>> Loaded raw state_dict successfully.")
                    
            except Exception as e:
                self.logger.error(f">>> FAILED to load checkpoint: {e}")
                self.logger.warning(">>> Starting from SCRATCH (Random Weights). Risky!")
        else:
            self.logger.warning(f">>> No checkpoint found at {ckpt_path}. Starting from scratch.")

    def _warmup(self) -> np.ndarray:
        """Fills the replay buffer with random actions before training begins."""
        self.logger.info(">>> Starting Warmup Phase...")
        obs, _ = self.env.reset()
        
        for _ in range(self.warmup_steps):
            if self.num_envs == 1:
                actions = np.array([self.env.action_space.sample()])
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions[0])
                dones = terminated or truncated
                self.buffer.add(obs, actions, rewards, next_obs, dones, infos['true_state_vector'])
            else:
                actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
                next_obs, rewards, dones, infos = self.env.step(actions)
                true_states = [i['true_state_vector'] for i in infos]
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, true_states)
            
            obs = next_obs
            self.global_step += self.num_envs
            
        return obs

    def train_e2e(self) -> None:
        """Main Training Loop."""
        self.logger.info(f">>> E2E RL STARTED | Mode: {'JOINT' if cfg.TRAIN.JOINT_OPTIMIZATION else 'DECOUPLED'}")
        
        obs = self._warmup()
        grad_updates_pending = 0
        
        pbar = tqdm(initial=self.global_step, total=cfg.TRAIN.TOTAL_TIMESTEPS, desc="Training")
        
        while self.global_step < cfg.TRAIN.TOTAL_TIMESTEPS:
            # --- 1. Select Action ---
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device)
                if self.num_envs == 1 and obs_t.ndim == 1:
                    obs_t = obs_t.unsqueeze(0)
                
                actions_tensor, _, pred_leader_tensor, _ = self.actor.sample(obs_t)
                actions = actions_tensor.detach().cpu().numpy()
                pred_leader_np = pred_leader_tensor.detach().cpu().numpy()

            # --- 2. Environment Step ---
            if self.num_envs == 1:
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions[0])
                dones = terminated or truncated
                self.buffer.add(obs, actions, rewards, next_obs, dones, infos['true_state_vector'])
                
                # Debug Info
                true_leader_q = infos['leader_q']
                pred_leader_q = (pred_leader_np[0][:7] * cfg.ROBOT.Q_STD) + cfg.ROBOT.Q_MEAN
                follower_q = infos['follower_q']
            else:
                next_obs, rewards, dones, infos = self.env.step(actions)
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, [i['true_state_vector'] for i in infos])
                true_leader_q, pred_leader_q, follower_q = None, None, None # Skip debug for vec env

            obs = next_obs
            self.global_step += self.num_envs
            grad_updates_pending += self.num_envs
            
            # --- 3. Gradient Update ---
            if grad_updates_pending >= cfg.TRAIN.TRAIN_FREQUENCY:
                metrics = self._update_policy(grad_updates_pending)
                grad_updates_pending = 0
                
                # Log Metrics
                if self.global_step % cfg.TRAIN.LOG_FREQ == 0:
                    self._log_metrics(metrics, rewards, true_leader_q, pred_leader_q, follower_q)

            # --- 4. Evaluation & Checkpointing ---
            if self.global_step % cfg.TRAIN.EVAL_INTERVAL == 0:
                self._evaluate_and_save()
            
            pbar.update(self.num_envs)
        
        pbar.close()
        self.logger.info(">>> Training Finished.")

    def _update_policy(self, pending_steps: int) -> Dict[str, float]:
        """Performs SAC gradient updates."""
        updates = np.clip(int(pending_steps * 0.5), 1, 64)
        last_metrics = {}
        
        for _ in range(updates):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            metrics = self.sac.update(batch, update_actor=True)
            last_metrics = metrics
            
            # NaN Guard
            if np.isnan(metrics['actor_loss']) or np.isnan(metrics['pred_loss']):
                self._handle_nan_crash(metrics)
        
        return last_metrics

    def _handle_nan_crash(self, metrics: Dict[str, float]) -> None:
        """Dumps state and exits upon NaN detection."""
        self.logger.error("\n[FATAL] NaN DETECTED IN LOSS! IMMEDIATE DUMP:")
        self.logger.error(f"Step: {self.global_step} | Metrics: {metrics}")
        torch.save(self.actor.state_dict(), self.output_dir / "CRASH_actor.pth")
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
            f"Actor_L: {metrics['actor_loss']:.3f} | Pred_L: {metrics['pred_loss']:.3f}"
        )
        self.logger.info(f"   >>> [Debug] Pred_Err: {pred_err:.4f} | Track_Err: {track_err:.4f}")
        
        # TensorBoard Log
        self.writer.add_scalar("Train/Reward", np.mean(rewards), self.global_step)
        self.writer.add_scalar("Train/Pred_Error_Rad", pred_err, self.global_step)
        self.writer.add_scalar("Train/Tracking_Error_Rad", track_err, self.global_step)
        self.writer.add_scalar("Loss/Actor", metrics['actor_loss'], self.global_step)
        self.writer.add_scalar("Loss/Critic", metrics['critic_loss'], self.global_step)
        self.writer.add_scalar("Loss/Prediction", metrics['pred_loss'], self.global_step)

    def _evaluate_and_save(self) -> None:
        """Runs evaluation episodes and saves checkpoints."""
        avg_reward = self._run_evaluation_episodes()
        
        if avg_reward > self.best_eval_reward:
            self.best_eval_reward = avg_reward
            self._save_checkpoint(is_best=True)
            self.logger.info(f">>> New Best Model Saved (Reward: {avg_reward:.2f})")
        
        self._save_checkpoint(is_best=False)

    def _run_evaluation_episodes(self) -> float:
        """Executes evaluation loop without noise/exploration."""
        eval_rewards = []
        for _ in range(cfg.EVAL.NUM_EPISODES):
            obs, _ = self.eval_env.reset()
            total_rew = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                    mu, _, _, _ = self.actor(obs_t)
                    # Deterministic Action for Eval
                    action = torch.tanh(mu) * self.actor.action_scale
                    action_np = action.cpu().numpy()[0]
                
                next_obs, reward, terminated, truncated, _ = self.eval_env.step(action_np)
                done = terminated or truncated
                total_rew += reward
                obs = next_obs
            
            eval_rewards.append(total_rew)
        
        avg_eval = np.mean(eval_rewards)
        self.writer.add_scalar("Eval/Reward", avg_eval, self.global_step)
        return float(avg_eval)

    def _save_checkpoint(self, is_best: bool = False) -> None:
        filename = "best_model.pth" if is_best else "latest_model.pth"
        torch.save(self.actor.state_dict(), self.output_dir / filename)