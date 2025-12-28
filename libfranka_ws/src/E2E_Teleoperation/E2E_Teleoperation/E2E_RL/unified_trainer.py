"""
E2E_Teleoperation/E2E_RL/unified_trainer.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
from pathlib import Path

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.sac_policy_network import LSTM, JointActor, JointCritic
from E2E_Teleoperation.E2E_RL.sac_training_algorithm import SACAlgorithm

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
        
        # Aux data for BC and Encoder
        self.leader_states = np.zeros((capacity, 14), dtype=np.float32)
        self.expert_actions = np.zeros((capacity, action_dim), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, leader_state, expert_action):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        
        self.leader_states[self.ptr] = leader_state
        if expert_action is not None:
            self.expert_actions[self.ptr] = expert_action
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': torch.FloatTensor(self.obs[idxs]).to(self.device),
            'next_obs': torch.FloatTensor(self.next_obs[idxs]).to(self.device),
            'actions': torch.FloatTensor(self.actions[idxs]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[idxs]).to(self.device),
            'dones': torch.FloatTensor(self.dones[idxs]).to(self.device),
            'true_state_vector': torch.FloatTensor(self.leader_states[idxs]).to(self.device),
            'expert_action': torch.FloatTensor(self.expert_actions[idxs]).to(self.device)
        }

class UnifiedTrainer:
    def __init__(self, env, output_dir, is_vector_env=False):
        self.env = env
        self.output_dir = Path(output_dir)
        self.log_file = self.output_dir / "training_log.txt"
        self.is_vector_env = is_vector_env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize Networks
        self.encoder = LSTM().to(self.device)
        self.actor = JointActor(self.encoder).to(self.device)
        self.critic = JointCritic(self.encoder).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # 2. Buffer
        self.buffer = ReplayBuffer(cfg.TRAIN.BUFFER_SIZE, cfg.ROBOT.OBS_DIM, 7, self.device)
        
        # 3. Alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)

    def _log(self, msg):
        print(msg)
        with open(self.log_file, "a") as f: f.write(msg + "\n")

    def _add_to_buffer(self, obs, action, reward, next_obs, terminated, truncated, info):
        # Handle Vector Env vs Single Env extraction
        if self.is_vector_env:
            for i in range(len(obs)):
                done = terminated[i] or truncated[i]
                self.buffer.add(
                    obs[i], action[i], reward[i], next_obs[i], done,
                    info['true_state_vector'][i], info['expert_action'][i]
                )
        else:
            done = terminated or truncated
            self.buffer.add(
                obs, action, reward, next_obs, done,
                info['true_state_vector'], info['expert_action']
            )

    def _collect_data(self, steps, use_teacher=False):
        """
        Collects data for the replay buffer.
        If use_teacher=True, the agent ignores the policy and executes 
        the 'Expert Action' (Inverse Dynamics) from the environment.
        """
        mode_str = "TEACHER" if use_teacher else "RANDOM/POLICY"
        self._log(f">> Collecting {steps} steps ({mode_str})...")
        
        obs, info = self.env.reset()
        for _ in range(steps):
            if use_teacher:
                # Ask environment for the perfect torque to track Leader
                if self.is_vector_env:
                    action = info['expert_action'] # Vector env needs support, simplified here for Single
                else:
                    action = self.env.get_expert_action()
            else: 
                action = self.env.action_space.sample()
            
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            self._add_to_buffer(obs, action, reward, next_obs, terminated, truncated, next_info)
            
            obs = next_obs
            info = next_info
            
            if not self.is_vector_env and (terminated or truncated):
                obs, info = self.env.reset()

    def train_stage1_bc(self):
        """
        STAGE 1: Behavioral Cloning (BC) + Encoder Pre-training
        We use the 'Teacher' (Inverse Dynamics) to generate perfect trajectories.
        The Encoder learns to predict the Leader State.
        The Actor learns to clone the Expert Action (Inverse Dynamics).
        """
        self._log("\n>>> STAGE 1: BC & Encoder Pre-training")
        
        # 1. Collect Expert Data (Teacher Mode)
        # This ensures the buffer is full of "Perfect Tracking" examples.
        self._collect_data(steps=20000, use_teacher=True)
        
        opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=cfg.TRAIN.ENCODER_LR)
        opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.TRAIN.ACTOR_LR)
        
        best_loss = float('inf')

        for step in range(cfg.TRAIN.STAGE1_STEPS):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            
            # --- A. Encoder Training (Supervised: Pred State vs Leader State) ---
            _, _, pred_state, _, _ = self.actor.forward(batch['obs'])
            loss_enc = F.mse_loss(pred_state, batch['true_state_vector'])
            
            opt_encoder.zero_grad()
            loss_enc.backward()
            opt_encoder.step()
            
            # --- B. Actor BC Training (Supervised: Actor Action vs Expert Action) ---
            # We detach the encoder features to not destabilize the LSTM with Actor gradients yet
            pred_action, _, _, _, _ = self.actor.sample(batch['obs'])
            loss_bc = F.mse_loss(pred_action, batch['expert_action'])
            
            opt_actor.zero_grad()
            loss_bc.backward()
            opt_actor.step()
            
            if step % 1000 == 0:
                total_loss = loss_enc.item() + loss_bc.item()
                self._log(f"Stage 1 | Step {step} | Enc Loss: {loss_enc.item():.5f} | BC Loss: {loss_bc.item():.5f}")
                
                if total_loss < best_loss:
                    best_loss = total_loss
                    torch.save(self.encoder.state_dict(), self.output_dir / "stage1_encoder.pth")
                    torch.save(self.actor.state_dict(), self.output_dir / "stage1_actor.pth")

        self._log(">>> Stage 1 Complete.")

    def train_stage2_e2e(self):
        """
        STAGE 2: End-to-End SAC Fine-tuning
        """
        self._log("\n>>> STAGE 2: E2E SAC Fine-tuning")
        
        # Optimizers (Encoder LR is lower for stability)
        opt_actor = torch.optim.Adam([
            {'params': self.actor.net.parameters(), 'lr': cfg.TRAIN.ACTOR_LR}, 
            {'params': self.encoder.parameters(), 'lr': cfg.TRAIN.ENCODER_LR * 0.1}
        ])
        opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        
        sac = SACAlgorithm(self.actor, self.critic, self.critic_target, 
                           opt_actor, opt_critic, self.opt_alpha, self.log_alpha)
        
        obs, info = self.env.reset()
        ep_reward = 0
        best_avg_reward = -float('inf')
        recent_rewards = []
        
        for step in range(cfg.TRAIN.STAGE2_STEPS):
            # Sample Action (Policy)
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device) if not self.is_vector_env \
                   else torch.FloatTensor(obs).to(self.device)
                action, _, _, _, _ = self.actor.sample(obs_t)
                action = action.cpu().numpy()
                if not self.is_vector_env: action = action[0]

            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            ep_reward += np.mean(reward) if self.is_vector_env else reward

            self._add_to_buffer(obs, action, reward, next_obs, terminated, truncated, next_info)
            obs = next_obs
            info = next_info
            
            # Update
            if self.buffer.size > cfg.TRAIN.BATCH_SIZE:
                sac.update(self.buffer.sample(cfg.TRAIN.BATCH_SIZE))

            # Logging & Reset
            if not self.is_vector_env and (terminated or truncated):
                self._log(f"Step {step} | Reward: {ep_reward:.1f}")
                recent_rewards.append(ep_reward)
                if len(recent_rewards) > 10: recent_rewards.pop(0)
                
                if np.mean(recent_rewards) > best_avg_reward:
                    best_avg_reward = np.mean(recent_rewards)
                    torch.save(self.actor.state_dict(), self.output_dir / "stage2_best.pth")

                obs, info = self.env.reset()
                ep_reward = 0
            
            if step % 50000 == 0:
                 torch.save(self.actor.state_dict(), self.output_dir / f"stage2_ckpt_{step}.pth")