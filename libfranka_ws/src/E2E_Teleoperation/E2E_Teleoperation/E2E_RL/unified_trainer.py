"""
Unified Trainer: Pretrain -> Critic Warmup -> Residual RL -> E2E Fine-Tune
MODIFIED: Expert Data Init + Offline Critic Warmup + Zero Env Warmup + Config Fixes
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
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig

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
        
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vec_env = self.num_envs > 1
        
        self._setup_logging()
        self.writer = SummaryWriter(log_dir=str(output_dir / "logs"))
        
        # --- CONFIG HYPERPARAMETERS ---
        self.REWARD_SCALE = cfg.SAC.REWARD_SCALE
        self.WARMUP_STEPS = 0  # Warmup done offline via Critic Pre-training
        
        self.TOTAL_STEPS = cfg.TRAIN.TOTAL_TIMESTEPS
        self.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
        self.EVAL_INTERVAL = cfg.TRAIN.EVAL_INTERVAL
        
        self.actor = JointActor().to(self.device)
        self.critic = JointCritic().to(self.device)
        self.critic_target = JointCritic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.residual_net.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        
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
        
        self.log(f"Trainer Initialized. Envs: {self.num_envs}, Batch: {self.BATCH_SIZE}")

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
        # 1. Collect Data using Inverse Dynamics
        self.log("\n=== PHASE 1: EXPERT DATA COLLECTION ===")
        self._collect_initial_data(steps=10_000)
        
        # 2. Train LSTM on the collected expert data
        self.log("\n=== PHASE 2: SUPERVISED LSTM PRE-TRAINING ===")
        self._pretrain_lstm(steps=5_000)
        
        # 3. Train Critic on the collected expert data
        self.log("\n=== PHASE 2.5: CRITIC WARMUP (Q-NETWORK) ===")
        self._pretrain_critic(steps=5_000)
        
        self.log("\n=== PHASE 3: RL TRAINING ===")
        self.log("Freezing LSTM for initial RL phase...")
        self.actor.base_encoder.requires_grad_(False)
        
        obs = self._reset_env()
        fine_tune_mode = False
        steps_per_iter = self.num_envs
        
        for global_step in range(0, self.TOTAL_STEPS, steps_per_iter):
            
            # --- EVALUATION LOOP ---
            if global_step > 0 and global_step % self.EVAL_INTERVAL == 0:
                self._evaluate(global_step)

            # --- ACTION SELECTION (Policy) ---
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                if not self.is_vec_env: obs_t = obs_t.unsqueeze(0) 
                
                # Sample action (RL Policy)
                action_t, _, _, _ = self.actor.sample(obs_t)
                action = action_t.cpu().numpy()
                if not self.is_vec_env: action = action[0]

            # --- STEP ENV ---
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
            if not self.is_vec_env and done: obs = self._reset_env()

            # --- UPDATE ---
            if self.buffer.size > self.BATCH_SIZE:
                is_warmup = global_step < self.WARMUP_STEPS
                
                if global_step >= 200_000 and not fine_tune_mode:
                    self.log("!!! UNFREEZING LSTM FOR E2E FINE-TUNING !!!")
                    self.actor.base_encoder.requires_grad_(True)
                    self.actor_optimizer.add_param_group({'params': self.actor.base_encoder.parameters(), 'lr': 1e-5})
                    fine_tune_mode = True

                updates_to_run = self.num_envs
                for _ in range(updates_to_run):
                    metrics = self.algo.update(
                        self.buffer.sample(self.BATCH_SIZE), 
                        update_actor=(not is_warmup),
                        fine_tune_encoder=fine_tune_mode
                    )
                                
                if global_step % 1000 == 0:
                    avg_rew = np.mean(reward)
                    
                    # LOGGING: Print to console with new diagnostics
                    self.log(f"Step {global_step} | C: {metrics['critic_loss']:.2f} | A: {metrics['actor_loss']:.2f} | "
                             f"Q: {metrics['q_mean']:.2f} | Ent: {metrics['entropy']:.2f} | "
                             f"ActNorm: {metrics['action_norm']:.3f} | R: {avg_rew:.2f}")
                    
                    # LOGGING: Write to TensorBoard
                    self.writer.add_scalar("Loss/Critic", metrics['critic_loss'], global_step)
                    self.writer.add_scalar("Loss/Actor", metrics['actor_loss'], global_step)
                    self.writer.add_scalar("Loss/Pred", metrics['pred_loss'], global_step)
                    self.writer.add_scalar("Reward/Avg", avg_rew, global_step)
                    self.writer.add_scalar("Param/Alpha", metrics['alpha'], global_step)
                    
                    # Debug Scalars (NEW)
                    self.writer.add_scalar("Debug/Q_Mean", metrics['q_mean'], global_step)
                    self.writer.add_scalar("Debug/Q_Max", metrics['q_max'], global_step)
                    self.writer.add_scalar("Debug/Entropy", metrics['entropy'], global_step)
                    self.writer.add_scalar("Debug/Action_Norm", metrics['action_norm'], global_step)
    
    def _collect_initial_data(self, steps):
        """Phase 1: Collect 'Golden' Data using Inverse Dynamics"""
        self.log(f"Collecting {steps} steps of TRUE EXPERT data (Delay Disabled for Action ONLY)...")
        
        # --- FIX: DISABLE ACTION DELAY ONLY ---
        # Do NOT change self.env.delay_simulator.config
        
        if self.is_vec_env:
            self.env.env_method("set_action_delay_enabled", False)
        else:
            # We assume single env for debugging
            # Save original state just in case
            original_state = self.env.follower.action_delay_enabled
            
            # Disable ONLY the action delay queue
            self.env.follower.action_delay_enabled = False
            # self.env.delay_simulator.config = ExperimentConfig.NO_DELAY  <-- DELETE THIS LINE

        obs = self._reset_env()
        iters = steps // self.num_envs if self.is_vec_env else steps
        
        for _ in range(iters):
            # ... (Collection loop remains the same) ...
            if self.is_vec_env:
                expert_actions = self.env.env_method("get_expert_action")
                action = np.array(expert_actions)
            else:
                action = self.env.get_expert_action()
            
            next_obs, reward, done, info = self._step_env(action)
            
            # Add to buffer...
            if self.is_vec_env:
                 true_states = np.stack([i['true_state_vector'] for i in info])
                 self.buffer.add_batch(obs, action, reward, next_obs, done, true_states)
            else:
                 true_state = info.get('true_state_vector', np.zeros(14))
                 self.buffer.add(obs, action, float(reward), next_obs, float(done), true_state)
                 
            obs = next_obs
            if not self.is_vec_env and done: obs = self._reset_env()
        
        # --- RESTORE ACTION DELAY ---
        self.log("Expert Data Collection Complete. Restoring Action Delay...")
        if self.is_vec_env:
            self.env.env_method("set_action_delay_enabled", True)
        else:
            self.env.follower.action_delay_enabled = True
        
        self.log(f"Buffer Size: {self.buffer.size}")

    def _pretrain_lstm(self, steps):
        """Phase 2: Train LSTM on buffer data"""
        self.log(f"Pre-training LSTM for {steps} steps...")
        self.actor.base_encoder.requires_grad_(True)
        opt = optim.Adam(self.actor.base_encoder.parameters(), lr=1e-3)
            
        for i in range(steps):
            batch = self.buffer.sample(self.BATCH_SIZE)
            _, pred_state, _ = self.actor.base_encoder(batch['obs'][:, -1207:-7].view(-1, 80, 15)) 
            loss = nn.MSELoss()(pred_state, batch['true_state_vector'])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 1000 == 0: self.log(f"LSTM Pretrain Loss: {loss.item():.4f}")
        
        self.log("Re-freezing LSTM.")
        self.actor.base_encoder.requires_grad_(False)
    
    def _pretrain_critic(self, steps):
        """Phase 2.5: Train Critic on buffer data (Offline Q-Warmup)"""
        self.log(f"Pre-training Critic for {steps} steps...")
        # We use the ResidualSAC.update method but force update_actor=False
        
        for i in range(steps):
            batch = self.buffer.sample(self.BATCH_SIZE)
            
            # This updates ONLY the Critic (Q-networks)
            metrics = self.algo.update(batch, update_actor=False, fine_tune_encoder=False)
            
            if i % 1000 == 0:
                self.log(f"Critic Warmup Loss: {metrics['critic_loss']:.4f}")

    def _evaluate(self, step):
        """
        Evaluation Loop with Detailed Metrics Logging.
        Tracks: True Q, Pred Q, Remote Q, Action, Pred Error, Tracking Error
        """
        self.log("\n--- Starting Evaluation (Detailed) ---")
        
        eval_rewards = []
        eval_pred_errors = []
        eval_track_errors = []
        
        eval_episodes = 5
        
        for ep_idx in range(eval_episodes):
            obs = self._reset_env()
            
            # Initialization for Vector Env
            if self.is_vec_env:
                current_rewards = np.zeros(self.num_envs)
                final_rewards = np.zeros(self.num_envs)
                finished_mask = np.zeros(self.num_envs, dtype=bool)
            else:
                total_rew = 0
                done = False

            # --- EPISODE LOOP ---
            while True:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                    if not self.is_vec_env: obs_t = obs_t.unsqueeze(0)
                    
                    # 1. Get Actor Output AND Predicted State
                    # forward() returns: mu_res, log_std, pred_state, next_hidden
                    mu_res, _, pred_state_t, _ = self.actor(obs_t)
                    
                    action = mu_res.cpu().numpy()
                    pred_state = pred_state_t.cpu().numpy()
                    
                    if not self.is_vec_env: 
                        action = action[0]
                        pred_state = pred_state[0]

                # 2. Step Environment
                next_obs, reward, done, info = self._step_env(action)
                
                # --- EXTRACT METRICS (From Env 0 only) ---
                if self.is_vec_env:
                    # Info is a list of dicts. We peek at the first environment (Env 0)
                    true_state = info[0]['true_state_vector'] # [q(7), qd(7)]
                    remote_q = obs[0, :7]                     # From Observation
                    curr_pred_q = pred_state[0, :7]           # From LSTM
                    curr_action = action[0]                   # Torque
                else:
                    true_state = info['true_state_vector']
                    remote_q = obs[:7]
                    curr_pred_q = pred_state[:7]
                    curr_action = action

                true_q = true_state[:7]

                # Calculate Errors (Euclidean Norm)
                pred_err = np.linalg.norm(true_q - curr_pred_q)
                track_err = np.linalg.norm(true_q - remote_q)
                
                # Store errors for averaging
                eval_pred_errors.append(pred_err)
                eval_track_errors.append(track_err)

                # --- HANDLE REWARDS & TERMINATION (The Fix) ---
                if self.is_vec_env:
                    current_rewards += reward * (1 - finished_mask)
                    new_done = done & (~finished_mask)
                    
                    if np.any(new_done):
                        final_rewards[new_done] = current_rewards[new_done]
                        finished_mask = finished_mask | new_done
                    
                    # Stop evaluation if ANY environment finishes (to save time)
                    if np.any(done):
                        # Calculate mean of finished episodes
                        avg_ep_rew = np.mean(final_rewards[new_done])
                        eval_rewards.append(avg_ep_rew)
                        break
                else:
                    total_rew += reward
                    if done:
                        eval_rewards.append(total_rew)
                        break
                
                obs = next_obs
            
            # --- PRINT SNAPSHOT (At end of Episode 0) ---
            if ep_idx == 0:
                self.log(f"\n[Eval Snapshot | Ep {ep_idx} End]")
                self.log(f"  True Q:     {np.array2string(true_q, precision=3, suppress_small=True)}")
                self.log(f"  Predict Q:  {np.array2string(curr_pred_q, precision=3, suppress_small=True)}")
                self.log(f"  Remote Q:   {np.array2string(remote_q, precision=3, suppress_small=True)}")
                self.log(f"  Action:     {np.array2string(curr_action, precision=3, suppress_small=True)}")
                self.log(f"  -> Pred Error: {pred_err:.4f} | Track Error: {track_err:.4f}")

        # --- SUMMARY ---
        mean_rew = np.mean(eval_rewards)
        mean_pred_err = np.mean(eval_pred_errors)
        mean_track_err = np.mean(eval_track_errors)
        
        self.log(f"\nEvaluation Result at Step {step}:")
        self.log(f"  Avg Reward: {mean_rew:.2f}")
        self.log(f"  Avg Pred Error:  {mean_pred_err:.4f} (Is LSTM failing?)")
        self.log(f"  Avg Track Error: {mean_track_err:.4f} (Is Actor failing?)")
        
        # Tensorboard
        self.writer.add_scalar("Eval/MeanReward", mean_rew, step)
        self.writer.add_scalar("Eval/PredError", mean_pred_err, step)
        self.writer.add_scalar("Eval/TrackError", mean_track_err, step)
        
        self._reset_env()