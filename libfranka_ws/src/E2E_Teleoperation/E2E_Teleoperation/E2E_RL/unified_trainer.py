"""
Unified Trainer (Fixed: Removed Critic Warmup to prevent Q-Collapse)
-----------------------------------------------------------------------
Optimized for stable transition from BC to RL.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import math

import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- Project Imports ---
from E2E_Teleoperation.E2E_RL.e2e_network import JointActor, JointCritic
from E2E_Teleoperation.E2E_RL.e2e_algorithm import ResidualSACWithFrozenLSTM
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
        encoder_param_ids = {p.data_ptr() for p in self.actor.base_encoder.parameters()}
        policy_params = [p for p in self.actor.parameters() if p.data_ptr() not in encoder_param_ids]
        encoder_params = list(self.actor.base_encoder.parameters())

        self.actor_optimizer = optim.Adam([
            {'params': encoder_params, 'lr': cfg.TRAIN.ENCODER_LR},
            {'params': policy_params, 'lr': cfg.TRAIN.ACTOR_LR}
        ])
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        
        # [MODIFIED] Initialize alpha significantly lower to prevent negative Q-drift during fine-tuning
        initial_log_alpha = math.log(cfg.SAC.INITIAL_ALPHA)
        self.log_alpha = torch.tensor([initial_log_alpha], requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)
        
        # --- 4. Algorithm & Buffer ---
        # [MODIFIED] Extended freeze period to 100k steps to ensure Critic stability first
        self.sac = ResidualSACWithFrozenLSTM(
            self.actor, self.critic, self.critic_target,
            self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer,
            self.log_alpha, 
            gamma=cfg.TRAIN.GAMMA, 
            tau=cfg.SAC.TARGET_TAU,
            reward_scale=cfg.SAC.REWARD_SCALE,
            freeze_lstm_steps=100000 
        )
        
        self.buffer = ReplayBuffer(
            cfg.TRAIN.BUFFER_SIZE, cfg.ROBOT.RL_OBS_DIM, cfg.ROBOT.N_JOINTS,
            cfg.ROBOT.ESTIMATOR_OUTPUT_DIM, self.device
        )
        
        self.num_envs = getattr(self.env, "num_envs", 1)
        self.warmup_steps = cfg.TRAIN.WARMUP_STEPS // self.num_envs

        self.global_step = 0
        self.best_eval_reward = -float('inf')
        
        self._loss_history = {'actor': [], 'critic': [], 'pred': []}
        
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
        ckpt_path = cfg.ROBOT.PRETRAINED_ACTOR_PATH
        if ckpt_path.exists():
            try:
                self.logger.info(f">>> Loading Checkpoint: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                
                if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                    self.actor.load_state_dict(checkpoint['actor'])
                    self.logger.info(">>> Loaded 'actor' state_dict successfully.")
                else:
                    self.actor.load_state_dict(checkpoint)
                    self.logger.info(">>> Loaded raw state_dict successfully.")
            except Exception as e:
                self.logger.error(f">>> FAILED to load checkpoint: {e}")
        else:
            self.logger.warning(f">>> No checkpoint found at {ckpt_path}.")

    def _warmup(self, use_policy: bool = True) -> np.ndarray:
        self.logger.info(f">>> Starting Warmup Phase (use_policy={use_policy})...")
        
        if self.num_envs > 1:
            obs = self.env.reset()
        else:
            obs, _ = self.env.reset()
        
        self.actor.eval()
        
        for step in range(self.warmup_steps):
            if use_policy:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device)
                    if self.num_envs == 1 and obs_t.ndim == 1:
                        obs_t = obs_t.unsqueeze(0)
                    mu, _, _, _, _ = self.actor(obs_t)
                    actions = torch.tanh(mu).cpu().numpy()
                    if self.num_envs == 1:
                        actions = actions[0]
            else:
                actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)]) if self.num_envs > 1 else self.env.action_space.sample()
            
            if self.num_envs == 1:
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
                dones = terminated or truncated
                self.buffer.add(obs, actions, rewards, next_obs, dones, infos['true_state_vector'])
                obs = self.env.reset()[0] if dones else next_obs
            else:
                next_obs, rewards, dones, infos = self.env.step(actions)
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, [i['true_state_vector'] for i in infos])
                obs = next_obs
            
            self.global_step += self.num_envs
        
        self.actor.train()
        self.logger.info(f">>> Warmup Complete. Buffer size: {self.buffer.size}")
        return obs

    def _warmup_critic(self, steps: int = 15000) -> None:
        """
        [MODIFIED] Trains the critic on warmup data with increased steps and actor in eval mode.
        """
        self.logger.info(f">>> Starting Critic Warmup ({steps} steps)...")
        self.actor.eval() 
        
        for i in range(steps):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            metrics = self.sac.update(batch, update_actor=False)
            
            if i % 1000 == 0:
                self.logger.info(f"    Warmup {i}/{steps} | Loss: {metrics['critic_loss']:.4f} | Q_mean: {metrics['q1_mean']:.2f}")
                
        self.actor.train()
        self.logger.info(">>> Critic Warmup Complete.")

    def train_e2e(self) -> None:
        self.logger.info(f">>> E2E RL STARTED | Mode: {'JOINT' if cfg.TRAIN.JOINT_OPTIMIZATION else 'DECOUPLED'}")
        
        obs = self._warmup(use_policy=True)
        
        # [CRITICAL FIX] SKIPPED CRITIC WARMUP
        # Rationale: The BC policy performs poorly in dynamics initially. 
        # Forcing the critic to learn these negative values (-137 Q) without allowing 
        # the actor to adapt causes a collapse. We start joint training immediately.
        self.logger.info(">>> Skipping Critic Warmup to allow immediate Actor adaptation.")
        # self._warmup_critic(steps=15000) 
        
        grad_updates_pending = 0
        pbar = tqdm(initial=self.global_step, total=cfg.TRAIN.TOTAL_TIMESTEPS, desc="Training")
        
        while self.global_step < cfg.TRAIN.TOTAL_TIMESTEPS:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device)
                if self.num_envs == 1 and obs_t.ndim == 1:
                    obs_t = obs_t.unsqueeze(0)
                actions_tensor, _, _, _, _ = self.actor.sample(obs_t)
                actions = actions_tensor.cpu().numpy()

            if self.num_envs == 1:
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions[0])
                dones = terminated or truncated
                self.buffer.add(obs, actions[0], rewards, next_obs, dones, infos['true_state_vector'])
                obs = self.env.reset()[0] if dones else next_obs
            else:
                next_obs, rewards, dones, infos = self.env.step(actions)
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, [i['true_state_vector'] for i in infos])
                obs = next_obs

            self.global_step += self.num_envs
            grad_updates_pending += self.num_envs
            
            if grad_updates_pending >= cfg.TRAIN.TRAIN_FREQUENCY:
                metrics = self._update_policy(grad_updates_pending)
                grad_updates_pending = 0
                self.actor_scheduler.step()
                self.critic_scheduler.step()
                
                if self.global_step % cfg.TRAIN.LOG_FREQ == 0:
                    self._log_metrics(metrics, rewards)
            
            if self.global_step % cfg.TRAIN.EVAL_INTERVAL == 0:
                self._evaluate_and_save()
            
            pbar.update(self.num_envs)
        
        pbar.close()
        self._save_checkpoint(is_best=False, suffix="_final")

    def _update_policy(self, pending_steps: int) -> Dict[str, float]:
        updates = np.clip(int(pending_steps * 0.5), 1, 64)
        last_metrics = {}
        for _ in range(updates):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            # update_actor=True allows the policy head to adapt immediately
            # Note: The LSTM is still frozen by the ResidualSACWithFrozenLSTM class logic
            metrics = self.sac.update(batch, update_actor=True)
            last_metrics = metrics
        return last_metrics

    def _log_metrics(self, metrics, rewards) -> None:
        self.logger.info(f"Step {self.global_step} | R: {np.mean(rewards):.2f} | Alpha: {metrics.get('alpha', 0):.3f}")
        self.writer.add_scalar("Train/Reward", np.mean(rewards), self.global_step)
        self.writer.add_scalar("Loss/Actor", metrics.get('actor_loss', 0), self.global_step)
        self.writer.add_scalar("Loss/Critic", metrics.get('critic_loss', 0), self.global_step)

    def _evaluate_and_save(self) -> Dict[str, float]:
        eval_metrics = self._run_evaluation_episodes()
        if eval_metrics['reward'] > self.best_eval_reward:
            self.best_eval_reward = eval_metrics['reward']
            self._save_checkpoint(is_best=True)
        self._save_checkpoint(is_best=False)
        return eval_metrics

    def _run_evaluation_episodes(self) -> Dict[str, float]:
        eval_rewards = []
        self.actor.eval()
        for _ in range(cfg.EVAL.NUM_EPISODES):
            obs, _ = self.eval_env.reset()
            total_rew, done = 0, False
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                    mu, _, _, _, _ = self.actor(obs_t)
                    action_np = torch.tanh(mu).cpu().numpy()[0]
                next_obs, reward, terminated, truncated, _ = self.eval_env.step(action_np)
                done, total_rew, obs = terminated or truncated, total_rew + reward, next_obs
            eval_rewards.append(total_rew)
        self.actor.train()
        return {'reward': float(np.mean(eval_rewards))}

    def _save_checkpoint(self, is_best: bool = False, suffix: str = "") -> None:
        filename = "best_model.pth" if is_best else f"latest_model{suffix}.pth"
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'global_step': self.global_step,
            'best_eval_reward': self.best_eval_reward,
        }
        torch.save(checkpoint, self.output_dir / filename)