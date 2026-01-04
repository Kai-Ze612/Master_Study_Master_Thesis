"""
E2E_Teleoperation/E2E_RL/unified_trainer.py
"""


import torch
import torch.nn.functional as F
import numpy as np
import copy
from pathlib import Path

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.e2e_network import LSTM, JointActor, JointCritic
from E2E_Teleoperation.E2E_RL.e2e_algorithm import SACAlgorithm

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # We store this for the Prediction Loss
        self.leader_states = np.zeros((capacity, 14), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, leader_state, expert_action=None):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.leader_states[self.ptr] = leader_state
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': torch.FloatTensor(self.obs[ind]).to(self.device),
            'actions': torch.FloatTensor(self.actions[ind]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[ind]).to(self.device),
            'next_obs': torch.FloatTensor(self.next_obs[ind]).to(self.device),
            'dones': torch.FloatTensor(self.dones[ind]).to(self.device),
            'true_state_vector': torch.FloatTensor(self.leader_states[ind]).to(self.device)
        }

class UnifiedTrainer:
    def __init__(self, env, output_dir, device="cuda"):
        self.env = env
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dimensions
        self.is_vector_env = hasattr(env, "num_envs")
        if self.is_vector_env:
            obs_dim = env.single_observation_space.shape[0]
            action_dim = env.single_action_space.shape[0]
        else:
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]

        # Networks
        self.actor = JointActor().to(self.device)
        self.critic = JointCritic().to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Shared Encoder Pointer
        self.encoder = self.actor.encoder

        # Automatic Entropy Tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.TRAIN.ACTOR_LR)

        # Replay Buffer
        self.buffer = ReplayBuffer(cfg.TRAIN.BUFFER_SIZE, obs_dim, action_dim, self.device)

    def _log(self, msg):
        print(msg)
        with open(self.output_dir / "training_log.txt", "a") as f:
            f.write(msg + "\n")

    def _add_to_buffer(self, obs, action, reward, next_obs, term, trunc, info):
        # Handle Vector Env vs Single Env
        if self.is_vector_env:
            for i in range(len(obs)):
                done = float(term[i] or trunc[i])
                # Extract True Leader State from info for Prediction Loss
                true_state = info[i]['true_state_vector']
                self.buffer.add(obs[i], action[i], reward[i], next_obs[i], done, true_state)
        else:
            done = float(term or trunc)
            true_state = info['true_state_vector']
            self.buffer.add(obs, action, reward, next_obs, done, true_state)

    def train_stage2_e2e(self):
        self._log("\n>>> STARTING FINAL E2E TRAINING")
        
        # 1. OPTIMIZER STRATEGY: DIFFERENTIAL LEARNING RATES
        # Actor = Fast (3e-4), Encoder = Slow (3e-5)
        actor_lr = cfg.TRAIN.ACTOR_LR
        encoder_lr = actor_lr * 0.1  # FORCE 10x SMALLER LR
        
        self._log(f"Strategy: Differential LR | Actor: {actor_lr} | Encoder: {encoder_lr}")
        
        opt_actor = torch.optim.Adam([
            {'params': self.actor.net.parameters(), 'lr': actor_lr},     # Policy MLP
            {'params': self.encoder.parameters(), 'lr': encoder_lr}      # LSTM Eyes
        ])
        
        opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        
        sac = SACAlgorithm(self.actor, self.critic, self.critic_target, 
                           opt_actor, opt_critic, self.opt_alpha, self.log_alpha)
        
        # Config Override
        sac.ENCODER_WARMUP_STEPS = 5000  # Keep safety lock
        
        # Force Full Difficulty
        if self.is_vector_env: self.env.env_method("set_curriculum_difficulty", 1.0)
        else: self.env.set_curriculum_difficulty(1.0)

        # =========================================================================
        # PHASE 1: SUPERVISED PRE-TRAINING (100% LSTM, 0% SAC)
        # =========================================================================
        PRETRAIN_STEPS = 10000 
        COLLECTION_STEPS = 5000 
        self._log(f"\n[PHASE 1] Collecting {COLLECTION_STEPS} steps of random data...")
        
        if self.is_vector_env: obs = self.env.reset()
        else: obs, info = self.env.reset()
        
        for _ in range(COLLECTION_STEPS):
            if self.is_vector_env: action = np.array([self.env.action_space.sample() for _ in range(len(obs))])
            else: action = self.env.action_space.sample()
            
            step_res = self.env.step(action)
            
            # Unpack
            if self.is_vector_env:
                next_obs, reward, dones, next_infos = step_res
                terminated = dones; truncated = [False]*len(dones); next_info_arg = next_infos
            else:
                next_obs, reward, terminated, truncated, next_info_arg = step_res

            self._add_to_buffer(obs, action, reward, next_obs, terminated, truncated, next_info_arg)
            obs = next_obs
            
            if not self.is_vector_env and (terminated or truncated): 
                obs, info = self.env.reset()
                
        # Train Loop
        self._log(f"[PHASE 1] Training LSTM...")
        self.encoder.train()
        for i in range(PRETRAIN_STEPS):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            _, _, pred_state, _, _ = self.actor.sample(batch['obs'])
            true_state = batch['true_state_vector']
            
            pred_loss = F.mse_loss(pred_state, true_state)
            
            opt_actor.zero_grad()
            pred_loss.backward()
            opt_actor.step()
            
            if (i+1) % 2000 == 0: self._log(f"   Pre-train Step {i+1} | MSE: {pred_loss.item():.5f}")

        # Clear Buffer
        self.buffer.ptr = 0
        self.buffer.size = 0
        self._log("[PHASE 1] Complete. Buffer Cleared.")

        # =========================================================================
        # PHASE 2: E2E RL (99% SAC, 1% LSTM via LR)
        # =========================================================================
        self._log("\n[PHASE 2] Starting E2E RL Training...")
        
        best_eval_reward = -float('inf')
        acc_track_err = 0.0
        log_steps_count = 0
        
        if self.is_vector_env: obs = self.env.reset()
        else: obs, info = self.env.reset()
        
        # Fast Start Config
        TRAIN_BATCH_SIZE = 4096
        MIN_STEPS_TO_START = 256

        for step in range(cfg.TRAIN.STAGE2_STEPS):
            
            # Action Selection
            if self.buffer.size < MIN_STEPS_TO_START:
                if self.is_vector_env: 
                    action = np.array([self.env.action_space.sample() for _ in range(len(obs))])
                    pred_state = np.zeros((len(obs), 14))
                else: 
                    action = self.env.action_space.sample()
                    pred_state = np.zeros(14)
            else:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).to(self.device)
                    if not self.is_vector_env: obs_t = obs_t.unsqueeze(0)
                    action_tensor, _, pred_state_tensor, _, _ = self.actor.sample(obs_t)
                    
                    action = action_tensor.cpu().numpy(); pred_state = pred_state_tensor.cpu().numpy()
                    if not self.is_vector_env: action = action[0]; pred_state = pred_state[0]

            # Step
            step_res = self.env.step(action)
            
            if self.is_vector_env:
                next_obs, reward, dones, next_infos = step_res
                terminated = dones; truncated = [False]*len(dones); next_info_arg = next_infos
                true_q = np.array([inf['true_state_vector'][:7] for inf in next_infos])
                remote_q = np.array([inf['follower_q'] for inf in next_infos])
            else:
                next_obs, reward, terminated, truncated, next_info = step_res
                next_info_arg = next_info
                true_q = next_info['true_state_vector'][:7]; remote_q = next_info['follower_q']

            # Metrics
            acc_track_err += np.mean(np.abs(true_q - remote_q))
            log_steps_count += 1
            
            self._add_to_buffer(obs, action, reward, next_obs, terminated, truncated, next_info_arg)
            obs = next_obs
            
            # Update
            if self.buffer.size >= TRAIN_BATCH_SIZE:
                updates = self.env.num_envs if self.is_vector_env else 1
                for _ in range(updates):
                    sac.update(self.buffer.sample(TRAIN_BATCH_SIZE))
                
            if not self.is_vector_env and (terminated or truncated): obs, info = self.env.reset()

            # Logs
            if step % 1000 == 0 and step > 0:
                self._log(f"Step {step} | TrkErr: {acc_track_err/log_steps_count:.4f}")
                acc_track_err = 0.0; log_steps_count = 0

            # Eval
            if step % 5000 == 0 and step > 0:
                curr_rew = self.evaluate(num_episodes=3)
                if curr_rew > best_eval_reward:
                    best_eval_reward = curr_rew
                    torch.save(self.actor.state_dict(), self.output_dir / "stage2_best.pth")
                self._log(f">>> EVAL | Reward: {curr_rew:.1f} | Best: {best_eval_reward:.1f}")

    def evaluate(self, num_episodes=3):
        # Evaluation logic (same as before)
        total_reward = 0.0
        for _ in range(num_episodes):
            if self.is_vector_env: obs = self.env.reset() # Simplified for vector
            else: obs, _ = self.env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).to(self.device)
                    if not self.is_vector_env: obs_t = obs_t.unsqueeze(0)
                    action, _, _, _, _ = self.actor.sample(obs_t)
                    action = action.cpu().numpy()
                    if not self.is_vector_env: action = action[0]
                
                if self.is_vector_env:
                    obs, rew, dones, _ = self.env.step(action)
                    total_reward += np.mean(rew)
                    done = np.any(dones)
                else:
                    obs, rew, term, trunc, _ = self.env.step(action)
                    total_reward += rew
                    done = term or trunc
        return total_reward / num_episodes