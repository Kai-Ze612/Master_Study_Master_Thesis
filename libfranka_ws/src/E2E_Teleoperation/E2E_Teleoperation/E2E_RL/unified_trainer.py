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

from E2E_Teleoperation.E2E_RL.e2e_network import JointActor, JointCritic
import E2E_Teleoperation.config.robot_config as cfg

class ReplayBuffer:
    """
    Replay Buffer for storing experience tuples.
    Stores:
    - obs: RL observation
    - actions: Actions taken
    - rewards: Rewards received
    - next_obs: Next observations
    - dones: Done flags
    - state: Physics ground truth state
    """
    def __init__(self, capacity, obs_dim, action_dim, state_dim, device):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.state_buf = np.zeros((capacity, state_dim), dtype=np.float32)  # Store physics ground truth
    
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
        return (
            torch.as_tensor(self.obs_buf[idxs], device=self.device),
            torch.as_tensor(self.act_buf[idxs], device=self.device),
            torch.as_tensor(self.rew_buf[idxs], device=self.device),
            torch.as_tensor(self.next_obs_buf[idxs], device=self.device),
            torch.as_tensor(self.done_buf[idxs], device=self.device),
            torch.as_tensor(self.state_buf[idxs], device=self.device)  # Add to sample
        )

class UnifiedTrainer:
    def __init__(self, env, output_dir):
        self.env = env
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = getattr(env, "num_envs", 1)  # for log

        # Networks
        self.actor = JointActor().to(self.device)
        self.critic = JointCritic().to(self.device)
        self.target_critic = JointCritic().to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers (Differential LR)
        actor_params = [
            {'params': self.actor.encoder.parameters(), 'lr': 1e-5},
            {'params': self.actor.net.parameters(), 'lr': 1e-4},
            {'params': self.actor.mu.parameters(), 'lr': 1e-4},
            {'params': self.actor.log_std.parameters(), 'lr': 1e-4},
        ]
        self.actor_optimizer = optim.Adam(actor_params)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Entropy
        self.target_entropy = -float(cfg.N_JOINTS)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        self.alpha = self.log_alpha.exp()

        self.buffer = ReplayBuffer(cfg.BUFFER_SIZE, cfg.RL_OBS_DIM, cfg.N_JOINTS, cfg.ROBOT.ROBOT_STATE_DIM, self.device)
        
        # Best Reward Tracking
        self.best_eval_reward = -np.inf
        self.gamma = cfg.GAMMA
        self.tau = cfg.TAU

    def train_e2e(self):
        print(f"\n Phase 1: LSTM Pre-training on Random Data")
        res = self.env.reset()
        
        if isinstance(res, tuple):
            obs, info = res
        else:
            obs = res
            info = {} 

        for _ in range(5000 // self.num_envs):
            if self.num_envs > 1:
                action = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
                next_obs, reward, done, infos = self.env.step(action)
                state = np.stack([i['true_state_vector'] for i in infos])
                self.buffer.add_batch(obs, action, reward, next_obs, done, state)
            else:
                action = self.env.action_space.sample()
                next_obs, reward, term, trunc, info = self.env.step(action)
                done = term or trunc
                self.buffer.add(obs, action, reward, next_obs, float(done), info['true_state_vector'])
                if done:
                    next_obs, info = self.env.reset()
            obs = next_obs

        print("\n")
        print(f"Setup   : {self.num_envs} Parallel Env(s)")
        
        res = self.env.reset()
        if isinstance(res, tuple):
            obs, info = res
        else:
            obs = res
            info = {}  # Placeholder

        for step in range(1, (cfg.TOTAL_TIMESTEPS // self.num_envs) + 1):
            # 1. Action Selection
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device).float()
                if self.num_envs == 1: obs_t = obs_t.unsqueeze(0)
                action_t, _, _, _, _ = self.actor.sample(obs_t)
                action = action_t.cpu().numpy()
                if self.num_envs == 1: action = action[0]

            # 2. Env Step
            step_res = self.env.step(action)
            if self.num_envs > 1:
                next_obs, reward, done, infos = step_res
                state = np.stack([i['true_state_vector'] for i in infos])
                self.buffer.add_batch(obs, action, reward, next_obs, done, state)
            else:
                next_obs, reward, term, trunc, info = step_res
                done = term or trunc
                self.buffer.add(obs, action, reward, next_obs, float(done), info['true_state_vector'])
                if done:
                    next_obs, info = self.env.reset()
            obs = next_obs

            # 3. Update
            if self.buffer.size > 1000:
                self.update_sac(256)

            # 4. Eval & Save
            global_step = step * self.num_envs
            if global_step % cfg.EVAL_INTERVAL == 0:
                self.evaluate(global_step)
                    
    def update_sac(self, batch_size):
        if not hasattr(self, 'update_count'):
            self.update_count = 0
        
        if self.update_count % 100 == 0:
            print(f"Update Step: {self.update_count} | Processing LSTM Rollout...")
        
        o, a, r, o2, d, true_state = self.buffer.sample(batch_size)
        
        # Critic Update
        with torch.no_grad():
            a2, lp2, p2, _, _ = self.actor.sample(o2)
            q1_t, q2_t = self.target_critic(p2, a2)
            q_target = r.unsqueeze(1) + self.gamma * (1 - d.unsqueeze(1)) * (torch.min(q1_t, q2_t) - self.alpha * lp2)

        _, _, p, _, _ = self.actor.sample(o)
        q1, q2 = self.critic(p.detach(), a)
        loss_q = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        # Actor Update
        a_pi, lp, p_pi, _, _ = self.actor.sample(o)
        q1_pi, q2_pi = self.critic(p_pi, a_pi)
        sac_loss = (self.alpha * lp - torch.min(q1_pi, q2_pi)).mean()
        
        pred_loss = nn.MSELoss()(p_pi, true_state)
        pred_weight = 5.0
        total_actor_loss = sac_loss + (pred_weight * pred_loss)

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha Update
        loss_alpha = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Target Soft Update
        for p, p_t in zip(self.critic.parameters(), self.target_critic.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)
        
        self.update_count += 1

    def evaluate(self, step):
        avg_reward = 0
        for _ in range(3):
            res = self.env.reset()
            obs = res[0] if (isinstance(res, tuple) or self.num_envs > 1) else res
            if self.num_envs > 1: obs = obs[0]
            
            done = False
            ep_rew = 0
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device).float().view(1, -1)
                    mu, _, _, _, _ = self.actor(obs_t)
                    act = torch.tanh(mu).cpu().numpy()[0] * cfg.ROBOT.TORQUE_LIMITS
                
                step_res = self.env.step(np.tile(act, (self.num_envs, 1)) if self.num_envs > 1 else act)
                if self.num_envs > 1:
                    o2_b, r_b, d_b, _ = step_res
                    obs, ep_rew, done = o2_b[0], ep_rew + r_b[0], d_b[0]
                else:
                    o2, r, term, trunc, _ = step_res
                    obs, ep_rew, done = o2, ep_rew + r, term or trunc
            avg_reward += ep_rew
        
        avg_reward /= 3
        print(f"Step {step} | Eval Reward: {avg_reward:.2f} | Best: {self.best_eval_reward:.2f}")
        
        if avg_reward > self.best_eval_reward:
            self.best_eval_reward = avg_reward
            self.save_checkpoint("best")
        self.save_checkpoint("latest")

    def save_checkpoint(self, label):
        path = self.output_dir / f"stage2_{label}.pth"
        torch.save(self.actor.state_dict(), path)