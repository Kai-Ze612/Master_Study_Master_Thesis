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
        
        # --- OPTIMIZATION: Compile models ---
        if int(torch.__version__.split(".")[0]) >= 2:
            print(">>> Compiling models with torch.compile() for speed...")
            self.actor.net = torch.compile(self.actor.net)
            self.critic.q1 = torch.compile(self.critic.q1)
            self.critic.q2 = torch.compile(self.critic.q2)

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
                    expert_act = info[i]['expert_action']
                else:
                    true_state = info['true_state_vector'][i]
                    expert_act = info['expert_action'][i]
                
                self.buffer.add(obs[i], action[i], reward[i], next_obs[i], done, true_state, expert_act)
        else:
            done = terminated or truncated
            self.buffer.add(obs, action, reward, next_obs, done, info['true_state_vector'], info['expert_action'])

    def _collect_data(self, steps, use_teacher=False):
        mode_str = "TEACHER" if use_teacher else "RANDOM/POLICY"
        self._log(f">> Collecting {steps} steps ({mode_str})...")
        
        if use_teacher:
            if self.is_vector_env: self.env.env_method("set_action_delay_enabled", False)
            else: self.env.follower.action_delay_enabled = False
        
        if self.is_vector_env:
            obs = self.env.reset()
            if use_teacher:
                expert_actions = self.env.env_method("get_expert_action")
                info = {'expert_action': expert_actions}
            else: info = {} 
        else:
            obs, info = self.env.reset()
            if use_teacher: info['expert_action'] = self.env.get_expert_action()
        
        for _ in range(steps):
            if use_teacher:
                if self.is_vector_env:
                    if isinstance(info, (list, tuple)): action = np.array([d['expert_action'] for d in info])
                    else: action = np.array(info['expert_action']) 
                else: action = info['expert_action']
            else: action = self.env.action_space.sample()
            
            step_result = self.env.step(action)
            
            if self.is_vector_env:
                next_obs, reward, dones, next_infos = step_result
                terminated = dones; truncated = [False] * len(dones); next_info_arg = next_infos 
            else:
                next_obs, reward, terminated, truncated, next_info = step_result
                next_info_arg = next_info

            self._add_to_buffer(obs, action, reward, next_obs, terminated, truncated, next_info_arg)
            obs = next_obs; info = next_info_arg 
            
            if not self.is_vector_env and (terminated or truncated):
                obs, info = self.env.reset()
                if use_teacher: info['expert_action'] = self.env.get_expert_action()

        if use_teacher:
            if self.is_vector_env: self.env.env_method("set_action_delay_enabled", True)
            else: self.env.follower.action_delay_enabled = True

    def train_stage1_bc(self):
        self._log("\n>>> STAGE 1: BC & Encoder Pre-training")
        self._collect_data(steps=20000, use_teacher=True)
        
        opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=cfg.TRAIN.ENCODER_LR)
        opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.TRAIN.ACTOR_LR)
        best_loss = float('inf')

        for step in range(cfg.TRAIN.STAGE1_STEPS):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            _, _, pred_state, _, _ = self.actor.forward(batch['obs'])
            loss_enc = F.mse_loss(pred_state, batch['true_state_vector'])
            opt_encoder.zero_grad(); loss_enc.backward(); opt_encoder.step()
            
            pred_action, _, _, _, _ = self.actor.sample(batch['obs'])
            loss_bc = F.mse_loss(pred_action, batch['expert_action'])
            opt_actor.zero_grad(); loss_bc.backward(); opt_actor.step()
            
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
        STAGE 2: E2E SAC Fine-tuning with Curriculum Learning
        """
        self._log("\n>>> STAGE 2: E2E SAC Fine-tuning (CURRICULUM MODE)")
        
        opt_actor = torch.optim.Adam([
            {'params': self.actor.net.parameters(), 'lr': cfg.TRAIN.ACTOR_LR}, 
            # Note: Kept default LR from config, but strongly recommend setting this lower (0.001 * LR) 
            # if using Curriculum to avoid forgetting features.
            {'params': self.encoder.parameters(), 'lr': cfg.TRAIN.ENCODER_LR * 0.1}
        ])
        opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        
        sac = SACAlgorithm(self.actor, self.critic, self.critic_target, 
                           opt_actor, opt_critic, self.opt_alpha, self.log_alpha)
        
        # --- Curriculum Settings ---
        CURRICULUM_DURATION = 200_000 # Ramps difficulty 0.0 -> 1.0 over 200k steps
        
        if self.is_vector_env: obs = self.env.reset(); info = {} 
        else: obs, info = self.env.reset()
        
        # --- Trackers ---
        num_envs = len(obs) if self.is_vector_env else 1
        env_running_rewards = np.zeros(num_envs)
        finished_ep_rewards = []
        acc_track_err = 0.0; acc_pred_err = 0.0; log_steps_count = 0
        best_mean_total_reward = -float('inf'); no_improvement_count = 0; PATIENCE = 50; recent_total_rewards = []

        for step in range(cfg.TRAIN.STAGE2_STEPS):
            
            # --- [CURRICULUM UPDATE] ---
            if step < CURRICULUM_DURATION:
                difficulty = step / CURRICULUM_DURATION
            else:
                difficulty = 1.0
            
            if self.is_vector_env: self.env.env_method("set_curriculum_difficulty", difficulty)
            else: self.env.set_curriculum_difficulty(difficulty)
            # ---------------------------

            # 1. Action & Prediction
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).to(self.device)
                if not self.is_vector_env: obs_t = obs_t.unsqueeze(0)
                action_tensor, _, pred_state_tensor, _, _ = self.actor.sample(obs_t)
                action = action_tensor.cpu().numpy(); pred_state = pred_state_tensor.cpu().numpy()
                if not self.is_vector_env: action = action[0]; pred_state = pred_state[0]

            # 2. Step Env
            step_result = self.env.step(action)
            
            if self.is_vector_env:
                next_obs, reward, dones, next_infos = step_result
                terminated = dones; truncated = [False] * len(dones); next_info_arg = next_infos
                env_running_rewards += reward
                for i, done in enumerate(dones):
                    if done:
                        finished_ep_rewards.append(env_running_rewards[i]); env_running_rewards[i] = 0.0
                true_q_batch = np.array([inf['true_state_vector'][:7] for inf in next_infos])
                remote_q_batch = np.array([inf['follower_q'] for inf in next_infos])
                pred_q_batch = pred_state[:, :7]
                acc_track_err += np.mean(np.abs(true_q_batch - remote_q_batch))
                acc_pred_err += np.mean(np.abs(true_q_batch - pred_q_batch))
            else:
                next_obs, reward, terminated, truncated, next_info = step_result
                next_info_arg = next_info
                env_running_rewards[0] += reward
                if terminated or truncated:
                    finished_ep_rewards.append(env_running_rewards[0]); env_running_rewards[0] = 0.0
                true_q = next_info['true_state_vector'][:7]; remote_q = next_info['follower_q']; pred_q = pred_state[:7]
                acc_track_err += np.mean(np.abs(true_q - remote_q))
                acc_pred_err += np.mean(np.abs(true_q - pred_q))

            log_steps_count += 1

            # 3. Buffer & Update
            self._add_to_buffer(obs, action, reward, next_obs, terminated, truncated, next_info_arg)
            obs = next_obs; info = next_info_arg
            
            if self.buffer.size > cfg.TRAIN.BATCH_SIZE:
                sac.update(self.buffer.sample(cfg.TRAIN.BATCH_SIZE))
                
            if not self.is_vector_env and (terminated or truncated): obs, info = self.env.reset()

            # --- LOGGING ---
            if step % 1000 == 0 and step > 0:
                avg_track = acc_track_err / log_steps_count
                avg_pred = acc_pred_err / log_steps_count
                
                if len(finished_ep_rewards) > 0:
                    avg_total_reward = np.mean(finished_ep_rewards); finished_ep_rewards = [] 
                else:
                    avg_total_reward = best_mean_total_reward if best_mean_total_reward > -float('inf') else 0.0
                    
                recent_total_rewards.append(avg_total_reward)
                if len(recent_total_rewards) > 10: recent_total_rewards.pop(0)
                smooth_total_rew = np.mean(recent_total_rewards)
                
                is_best = smooth_total_rew > best_mean_total_reward
                save_msg = " [SAVED]" if is_best else ""
                
                if is_best and smooth_total_rew != 0.0:
                    best_mean_total_reward = smooth_total_rew; no_improvement_count = 0
                    torch.save(self.actor.state_dict(), self.output_dir / "stage2_best.pth")
                else: no_improvement_count += 1

                if self.is_vector_env:
                    snap_true = true_q_batch[0]; snap_pred = pred_q_batch[0]; snap_rem = remote_q_batch[0]; snap_torq = action[0]
                else:
                    snap_true = true_q; snap_pred = pred_q; snap_rem = remote_q; snap_torq = action

                fmt = lambda x: np.array2string(x[:4], precision=2, suppress_small=True, separator=', ')
                
                self._log(
                    f"Step {step}/{cfg.TRAIN.STAGE2_STEPS} | Diff: {difficulty:.2f} | "
                    f"EpRet: {avg_total_reward:.1f} | Best: {best_mean_total_reward:.1f}{save_msg} | "
                    f"TrkErr: {avg_track:.4f}\n"
                    f"   True Q: {fmt(snap_true)}\n"
                    f"   Pred Q: {fmt(snap_pred)}\n"
                    f"   Remo Q: {fmt(snap_rem)}"
                )
                
                if no_improvement_count >= PATIENCE:
                    self._log(f"\n!!! EARLY STOPPING: No improvement for {PATIENCE * 1000} steps !!!")
                    break
                
                acc_track_err = 0.0; acc_pred_err = 0.0; log_steps_count = 0
                
            if step % 50000 == 0:
                 torch.save(self.actor.state_dict(), self.output_dir / f"stage2_ckpt_{step}.pth")