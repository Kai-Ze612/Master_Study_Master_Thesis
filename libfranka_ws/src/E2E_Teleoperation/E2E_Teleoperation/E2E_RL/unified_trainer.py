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
        if self.is_vector_env:
            for i in range(len(obs)):
                done = terminated[i] or truncated[i]
                if isinstance(info, (list, tuple)):
                    true_state = info[i]['true_state_vector']
                    expert_act = info[i].get('expert_action', np.zeros(7))
                else:
                    true_state = info['true_state_vector'][i]
                    expert_act = info['expert_action'][i]
                self.buffer.add(obs[i], action[i], reward[i], next_obs[i], done, true_state, expert_act)
        else:
            done = terminated or truncated
            expert_act = info.get('expert_action', np.zeros(7))
            self.buffer.add(obs, action, reward, next_obs, done, info['true_state_vector'], expert_act)

    def evaluate(self, num_episodes=10):
        self.actor.eval(); self.encoder.eval()
        eval_rewards = []
        finished_episodes = 0
        if self.is_vector_env:
            obs = self.env.reset()
            current_rewards = np.zeros(len(obs))
            while finished_episodes < num_episodes:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).to(self.device)
                    mu, _, _, _, _ = self.actor.forward(obs_t)
                    action = (torch.tanh(mu) * self.actor.scale).cpu().numpy()
                next_obs, reward, dones, _ = self.env.step(action)
                current_rewards += reward
                for i, done in enumerate(dones):
                    if done:
                        if finished_episodes < num_episodes:
                            eval_rewards.append(current_rewards[i])
                            finished_episodes += 1
                        current_rewards[i] = 0.0
                obs = next_obs
        else:
            for _ in range(num_episodes):
                obs, _ = self.env.reset()
                done = False; ep_reward = 0.0
                while not done:
                    with torch.no_grad():
                        obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                        mu, _, _, _, _ = self.actor.forward(obs_t)
                        action = (torch.tanh(mu) * self.actor.scale).cpu().numpy()[0]
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                    obs = next_obs
                eval_rewards.append(ep_reward)
        self.actor.train(); self.encoder.train()
        return np.mean(eval_rewards)

    def train_stage2_e2e(self):
        """
        PURE E2E: Simultaneous Learning (No Phases)
        """
        self._log("\n>>> STARTING PURE SIMULTANEOUS TRAINING")
        self._log("Method: Joint Optimization (RL + Aux Prediction Loss)")
        
        opt_actor = torch.optim.Adam([
            {'params': self.actor.net.parameters(), 'lr': cfg.TRAIN.ACTOR_LR}, 
            {'params': self.encoder.parameters(), 'lr': cfg.TRAIN.ENCODER_LR} 
        ])
        opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        
        sac = SACAlgorithm(self.actor, self.critic, self.critic_target, 
                           opt_actor, opt_critic, self.opt_alpha, self.log_alpha)
        
        # [THE KEY]: Let LSTM learn immediately. 
        sac.ENCODER_WARMUP_STEPS = 0 
        
        # --- CRITICAL FIX: Slower Curriculum ---
        CURRICULUM_DURATION = 200_000 
        
        best_eval_reward = -float('inf')
        no_improvement_count = 0
        
        # --- SAFETY FIX: Infinite Patience for Thesis Run ---
        PATIENCE = 500
        DEBUG_STEP_TRIGGER = 114000
        
        if self.is_vector_env: obs = self.env.reset(); info = {} 
        else: obs, info = self.env.reset()
        
        acc_track_err = 0.0
        acc_max_err = 0.0
        acc_pred_err = 0.0
        log_steps_count = 0

        # --- THE ONLY LOOP ---
        for step in range(cfg.TRAIN.STAGE2_STEPS):
            
            # Curriculum
            if step < CURRICULUM_DURATION: difficulty = step / CURRICULUM_DURATION
            else: difficulty = 1.0
            
            if self.is_vector_env: self.env.env_method("set_curriculum_difficulty", difficulty)
            else: self.env.set_curriculum_difficulty(difficulty)
            
            # Action
            # If buffer is empty, Random. If not, Policy.
            if self.buffer.size < cfg.TRAIN.BATCH_SIZE:
                if self.is_vector_env: 
                    action = np.array([self.env.action_space.sample() for _ in range(len(obs))])
                    # Dummy pred state: (Batch, 14)
                    pred_state = np.zeros((len(obs), 14))
                else: 
                    action = self.env.action_space.sample()
                    # Dummy pred state: (14,) -> 1D Array for Single Env
                    pred_state = np.zeros(14) 
            else:
                with torch.no_grad():
                    if step > DEBUG_STEP_TRIGGER: print(f"[DEBUG] Step {step}: Sampling...", flush=True)
                    obs_t = torch.FloatTensor(obs).to(self.device)
                    if not self.is_vector_env: obs_t = obs_t.unsqueeze(0)
                    action_tensor, _, pred_state_tensor, _, _ = self.actor.sample(obs_t)
                    action = action_tensor.cpu().numpy(); pred_state = pred_state_tensor.cpu().numpy()
                    if not self.is_vector_env: action = action[0]; pred_state = pred_state[0]

            # Step
            step_result = self.env.step(action)
            
            if self.is_vector_env:
                next_obs, reward, dones, next_infos = step_result
                terminated = dones; truncated = [False] * len(dones); next_info_arg = next_infos
                true_q = np.array([inf['true_state_vector'][:7] for inf in next_infos])
                remote_q = np.array([inf['follower_q'] for inf in next_infos])
                pred_q = pred_state[:, :7]
                
                # --- Metrics ---
                acc_track_err += np.mean(np.abs(true_q - remote_q))
                acc_max_err += np.mean(np.max(np.abs(true_q - remote_q), axis=1)) 
                acc_pred_err += np.mean(np.abs(true_q - pred_q))
            else:
                next_obs, reward, terminated, truncated, next_info = step_result
                next_info_arg = next_info
                true_q = next_info['true_state_vector'][:7]; remote_q = next_info['follower_q']; pred_q = pred_state[:7]
                
                # --- Metrics ---
                acc_track_err += np.mean(np.abs(true_q - remote_q))
                acc_max_err += np.max(np.abs(true_q - remote_q))
                acc_pred_err += np.mean(np.abs(true_q - pred_q))

            log_steps_count += 1
            self._add_to_buffer(obs, action, reward, next_obs, terminated, truncated, next_info_arg)
            obs = next_obs; info = next_info_arg
            
            # Update (Only starts after buffer has batch_size samples)
            if self.buffer.size > cfg.TRAIN.BATCH_SIZE:
                if step > DEBUG_STEP_TRIGGER: print(f"[DEBUG] Step {step}: Updating...", flush=True)
                sac.update(self.buffer.sample(cfg.TRAIN.BATCH_SIZE))
                
            if not self.is_vector_env and (terminated or truncated): obs, info = self.env.reset()

            # --- Heartbeat ---
            if step % 1000 == 0 and step > 0:
                avg_track = acc_track_err / log_steps_count
                avg_max = acc_max_err / log_steps_count
                avg_pred = acc_pred_err / log_steps_count
                
                if self.is_vector_env:
                    snap_true = true_q[0]; snap_pred = pred_q[0]; snap_rem = remote_q[0]
                else:
                    snap_true = true_q; snap_pred = pred_q; snap_rem = remote_q
                fmt = lambda x: np.array2string(x[:4], precision=2, suppress_small=True, separator=', ')
                
                self._log(
                    f"Step {step}/{cfg.TRAIN.STAGE2_STEPS} | Diff: {difficulty:.2f} | "
                    f"TrkErr: {avg_track:.4f} | MaxJointErr: {avg_max:.4f} | PredErr: {avg_pred:.4f}\n"
                    f"   True Q: {fmt(snap_true)}\n"
                    f"   Pred Q: {fmt(snap_pred)}\n"
                    f"   Remo Q: {fmt(snap_rem)}"
                )
                acc_track_err = 0.0; acc_max_err = 0.0; acc_pred_err = 0.0; log_steps_count = 0

            # --- Eval ---
            if step % 5000 == 0 and step > 0:
                self._log(f"--- Step {step}: Running Full Evaluation (10 eps) ---")
                current_eval_reward = self.evaluate(num_episodes=10)
                is_best = current_eval_reward > best_eval_reward
                save_msg = " [SAVED NEW BEST]" if is_best else ""
                if is_best:
                    best_eval_reward = current_eval_reward
                    no_improvement_count = 0
                    torch.save(self.actor.state_dict(), self.output_dir / "stage2_best.pth")
                else:
                    no_improvement_count += 1
                self._log(f"--- Eval Result: Reward {current_eval_reward:.1f} | Best {best_eval_reward:.1f}{save_msg} ---\n")
                if self.is_vector_env: obs = self.env.reset()
                else: obs, info = self.env.reset()
                
                # Extended Patience check
                if no_improvement_count >= PATIENCE:
                    self._log(f"\n!!! EARLY STOPPING: No improvement for {PATIENCE * 5000} steps !!!")
                    break
            
            if step % 50000 == 0:
                 torch.save(self.actor.state_dict(), self.output_dir / f"stage2_ckpt_{step}.pth")