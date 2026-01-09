"""
Unified Trainer: Expert Data (Fixed) -> BC (Unfrozen) -> DAgger RL (Smart Intervention)
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
        
        self.REWARD_SCALE = cfg.SAC.REWARD_SCALE
        self.TOTAL_STEPS = cfg.TRAIN.TOTAL_TIMESTEPS
        self.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
        self.EVAL_INTERVAL = cfg.TRAIN.EVAL_INTERVAL
        
        self.actor = JointActor().to(self.device)
        self.critic = JointCritic().to(self.device)
        self.critic_target = JointCritic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam([
            {'params': self.actor.base_encoder.lstm_cell.parameters(), 'lr': cfg.TRAIN.ENCODER_LR},
            {'params': self.actor.residual_net.parameters(), 'lr': cfg.TRAIN.ACTOR_LR},
            {'params': self.actor.res_mu.parameters(), 'lr': cfg.TRAIN.ACTOR_LR},
            {'params': self.actor.res_log_std.parameters(), 'lr': cfg.TRAIN.ACTOR_LR},
            {'params': self.actor.aux_head.parameters(), 'lr': cfg.TRAIN.ACTOR_LR} 
        ])
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        
        self.log_alpha = torch.tensor([np.log(0.1)], dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)
        
        self.algo = ResidualSAC(
            self.actor, self.critic, self.critic_target,
            self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer,
            self.log_alpha
        )
        
        self.buffer = ReplayBuffer(
            cfg.TRAIN.BUFFER_SIZE, cfg.ROBOT.RL_OBS_DIM, 
            cfg.ROBOT.N_JOINTS, cfg.ROBOT.ROBOT_STATE_DIM, self.device
        )
        
        self.log(f"Trainer Initialized. Device: {self.device}")

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
        if self.is_vec_env: return res 
        else: return res[0] 

    def _step_env(self, action):
        if self.is_vec_env:
            return self.env.step(action)
        else:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            return next_obs, reward, terminated or truncated, info

    def train_e2e(self):
        # Phase 1
        self.log("\n>>> PHASE 1: GENERATING EXPERT DATA")
        expert_buffer = self.generate_expert_data(steps=cfg.TRAIN.EXPERT_DATA_STEPS)
        
        # Phase 2
        self.log(f"\n>>> PHASE 2: BEHAVIORAL CLONING (NORMALIZED)")
        self._train_bc(expert_buffer)
        
        del expert_buffer
        gc.collect()
        
        # Phase 3
        self.log("\n>>> PHASE 3: DAgger RL FINE-TUNING")
        self._run_rl_loop()

    def generate_expert_data(self, steps):
        temp_buffer = ReplayBuffer(steps, cfg.ROBOT.RL_OBS_DIM, cfg.ROBOT.N_JOINTS, cfg.ROBOT.ROBOT_STATE_DIM, self.device)
        
        if self.is_vec_env: self.env.env_method("set_action_delay_enabled", False)
        else: self.env.follower.action_delay_enabled = False
            
        obs = self._reset_env()
        collected = 0
        episodes = 0
        
        pbar = tqdm(total=steps, desc="Expert Gen")
        
        while collected < steps:
            if self.is_vec_env:
                expert_actions = self.env.env_method("get_expert_action")
                action = np.array(expert_actions)
                incr = self.num_envs
            else:
                action = self.env.get_expert_action()
                incr = 1
            
            # Step with Real Reward
            next_obs, reward, done, info = self._step_env(action)
            
            if self.is_vec_env:
                true_states = np.stack([i['true_state_vector'] for i in info])
                temp_buffer.add_batch(obs, action, reward, next_obs, done, true_states)
            else:
                true_state = info.get('true_state_vector', np.zeros(14))
                temp_buffer.add(obs, action, reward, next_obs, float(done), true_state)
            
            obs = next_obs
            collected += incr
            pbar.update(incr)
            
            # CRITICAL: Reset LSTM on episode boundaries
            if not self.is_vec_env and done: 
                obs = self._reset_env()
                episodes += 1
                
        pbar.close()
        if self.is_vec_env: self.env.env_method("set_action_delay_enabled", True)
        else: self.env.follower.action_delay_enabled = True
            
        return temp_buffer

    def _train_bc(self, expert_buffer):
        """Phase 2 with Normalization and Clipping"""
        optimizer = optim.Adam(self.actor.parameters(), lr=cfg.TRAIN.BC_LR)
        aux_weight = cfg.TRAIN.WEIGHT_PRE_LOSS_START
        
        self.actor.train()
        self.actor.base_encoder.requires_grad_(True)
        
        for epoch in range(cfg.TRAIN.BC_EPOCHS):
            num_batches = expert_buffer.size // cfg.TRAIN.BATCH_SIZE
            avg_bc_loss = 0
            avg_aux_loss = 0
            
            for _ in range(num_batches):
                batch = expert_buffer.sample(cfg.TRAIN.BATCH_SIZE)
                
                mu, _, pred_state, _ = self.actor(batch['obs'])
                pred_action = torch.tanh(mu) * self.actor.action_scale
                
                # 
                # FIX: Normalize loss so 87Nm and 12Nm joints have equal weight
                norm_pred = pred_action / self.actor.action_scale
                norm_target = batch['actions'] / self.actor.action_scale
                bc_loss = F.mse_loss(norm_pred, norm_target)
                
                aux_loss = F.mse_loss(pred_state, batch['true_state_vector'])
                total_loss = bc_loss + (aux_weight * aux_loss)
                
                optimizer.zero_grad()
                total_loss.backward()
                
                # 
                # FIX: Clip Gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                avg_bc_loss += bc_loss.item()
                avg_aux_loss += aux_loss.item()
            
            aux_weight = max(cfg.TRAIN.WEIGHT_PRE_LOSS_END, aux_weight * cfg.TRAIN.BC_AUX_LOSS_DECAY)
            
            if epoch % 5 == 0:
                self.log(f"BC Epoch {epoch:02d} | NormLoss: {avg_bc_loss/num_batches:.4f} | Aux: {avg_aux_loss/num_batches:.4f}")

    def _run_rl_loop(self):
        obs = self._reset_env()
        
        self.actor.base_encoder.requires_grad_(False)
        encoder_frozen = True
        self.actor.residual_net.requires_grad_(True)
        
        ENCODER_FREEZE_STEPS = 25_000 
        INTERVENTION_STEPS = 50_000
        
        # BC Warmup
        self.log(">>> RL WARMUP: Collecting 5,000 steps with BC Policy...")
        warmup_steps = 0
        with torch.no_grad():
            while warmup_steps < 5000:
                obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                if not self.is_vec_env: obs_t = obs_t.unsqueeze(0)
                action_t, _, _, _ = self.actor.sample(obs_t)
                action = action_t.cpu().numpy()
                if not self.is_vec_env: action = action[0]
                
                next_obs, reward, done, info = self._step_env(action)
                if self.is_vec_env:
                    true_states = np.stack([i['true_state_vector'] for i in info])
                    self.buffer.add_batch(obs, action, reward * self.REWARD_SCALE, next_obs, done, true_states)
                    warmup_steps += self.num_envs
                else:
                    true_state = info.get('true_state_vector', np.zeros(14))
                    self.buffer.add(obs, action, reward * self.REWARD_SCALE, next_obs, float(done), true_state)
                    warmup_steps += 1
                obs = next_obs
                if not self.is_vec_env and done: obs = self._reset_env()
        
        self.log(">>> WARMUP COMPLETE. Starting RL Updates.")

        for global_step in range(0, self.TOTAL_STEPS, self.num_envs):
            
            if encoder_frozen and global_step >= ENCODER_FREEZE_STEPS:
                self.log(f"Step {global_step}: Unfreezing ENCODER.")
                self.actor.base_encoder.requires_grad_(True)
                encoder_frozen = False
            
            if global_step > 0 and global_step % self.EVAL_INTERVAL == 0:
                self._evaluate(global_step)

            # --- DAgger LOGIC ---
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                if not self.is_vec_env: obs_t = obs_t.unsqueeze(0)
                action_t, _, _, _ = self.actor.sample(obs_t)
                rl_action_raw = action_t.cpu().numpy()
                if not self.is_vec_env: rl_action_raw = rl_action_raw[0]

            if self.is_vec_env:
                id_action = np.array(self.env.env_method("get_expert_action"))[0]
                student_action = rl_action_raw[0]
            else:
                id_action = self.env.get_expert_action()
                student_action = rl_action_raw

            progress = min(1.0, global_step / INTERVENTION_STEPS)
            alpha = 0.8 * (1.0 - progress)
            
            # 
            # FIX: Normalized Divergence Threshold
            # Normalize actions to [0, 1] relative to torque limits for fair comparison
            norm_student = student_action / cfg.TORQUE_LIMITS
            norm_id = id_action / cfg.TORQUE_LIMITS
            diff_norm = np.linalg.norm(norm_student - norm_id)
            
            # Threshold: 0.1 (10% total error) -> 0.5 (50% error allowed later)
            threshold = 0.1 + (0.4 * progress)
            
            final_action = student_action
            is_intervention = False
            
            if diff_norm > threshold or np.random.random() < alpha:
                is_intervention = True
                corrected = (alpha * id_action) + ((1.0 - alpha) * student_action)
                if self.is_vec_env:
                    rl_action_raw[0] = corrected
                    final_action = rl_action_raw
                else:
                    final_action = corrected

            next_obs, reward, done, info = self._step_env(final_action)
            
            if self.is_vec_env:
                true_states = np.stack([i['true_state_vector'] for i in info])
                self.buffer.add_batch(obs, final_action, reward * self.REWARD_SCALE, next_obs, done, true_states)
            else:
                true_state = info.get('true_state_vector', np.zeros(14))
                self.buffer.add(obs, final_action, reward * self.REWARD_SCALE, next_obs, float(done), true_state)
            
            obs = next_obs
            if not self.is_vec_env and done: obs = self._reset_env()

            if self.buffer.size > self.BATCH_SIZE:
                updates = self.num_envs
                for _ in range(updates):
                    metrics = self.algo.update(
                        self.buffer.sample(self.BATCH_SIZE), 
                        update_actor=True, 
                        fine_tune_encoder=(not encoder_frozen)
                    )
                
                if global_step % 1000 == 0:
                    avg_rew = np.mean(reward)
                    status = "FROZEN" if encoder_frozen else "E2E"
                    self.log(f"Step {global_step} [{status}] | R: {avg_rew:.2f} | NormDiv: {diff_norm:.2f} | Thr: {threshold:.2f} | Int: {is_intervention}")
                    
                    self.writer.add_scalar("Reward/Avg", avg_rew, global_step)
                    self.writer.add_scalar("Loss/Actor", metrics['actor_loss'], global_step)
                    self.writer.add_scalar("Debug/NormActionDiv", diff_norm, global_step)

    def _evaluate(self, step):
        self.log(f"\n--- Evaluation at Step {step} ---")
        eval_rewards = []
        for _ in range(3):
            obs = self._reset_env()
            total_rew = 0
            done = False
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                    mu, _, _, _ = self.actor(obs_t)
                    action = (torch.tanh(mu) * self.actor.action_scale).cpu().numpy()[0]
                obs, reward, done, _ = self.env.step(action)
                total_rew += reward
            eval_rewards.append(total_rew)
        self.log(f"  Avg Reward: {np.mean(eval_rewards):.2f}")