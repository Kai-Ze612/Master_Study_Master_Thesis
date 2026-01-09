"""
Unified Trainer: Expert Data Init -> Behavioral Cloning (Dynamic) -> E2E RL Fine-Tune
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
        
        # --- CONFIG HYPERPARAMETERS ---
        self.REWARD_SCALE = cfg.SAC.REWARD_SCALE
        self.TOTAL_STEPS = cfg.TRAIN.TOTAL_TIMESTEPS
        self.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
        self.EVAL_INTERVAL = cfg.TRAIN.EVAL_INTERVAL
        
        # Initialize Networks
        self.actor = JointActor().to(self.device)
        self.critic = JointCritic().to(self.device)
        self.critic_target = JointCritic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # --- FIX: OPTIMIZER GROUPS ---
        # We must NOT use 'self.actor.base_encoder.parameters()' because it includes the aux_head.
        # Instead, we explicitly target 'lstm_cell' for the Encoder LR,
        # and 'aux_head' (predictor) for the Actor LR.
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
        
        # Main Replay Buffer (For RL Phase)
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

    # =========================================================================
    # CORE PIPELINE: EXPERT GEN -> BEHAVIORAL CLONING -> RL
    # =========================================================================
    def train_e2e(self):
        """
        Executes the Real E2E Training Pipeline:
        1. Generate 'God Mode' Expert Data (Inverse Dynamics, No Action Delay)
        2. Train Behavioral Cloning (Latent + Action) with Dynamic Weighting
        3. Run RL Fine-Tuning (Residual SAC) from scratch
        """
        
        # --- PHASE 1: GENERATE EXPERT DATA (OFFLINE) ---
        self.log("\n" + "="*50)
        self.log(">>> PHASE 1: GENERATING EXPERT DATA (INVERSE DYNAMICS)")
        self.log("="*50)
        
        expert_buffer = self.generate_expert_data(steps=cfg.TRAIN.EXPERT_DATA_STEPS)
        self.log(f"Expert Buffer Generated. Size: {expert_buffer.size}")
        
        # --- PHASE 2: BEHAVIORAL CLONING (IMITATION) ---
        self.log("\n" + "="*50)
        self.log(f">>> PHASE 2: IMITATION LEARNING (BC) - {cfg.TRAIN.BC_EPOCHS} Epochs")
        self.log("="*50)
        
        self._train_bc(expert_buffer)
        
        # --- CLEANUP ---
        self.log("Phase 2 Complete. Freeing Expert Buffer memory...")
        del expert_buffer
        gc.collect()
        
        # --- PHASE 3: RL FINE-TUNING ---
        self.log("\n" + "="*50)
        self.log(">>> PHASE 3: RL FINE-TUNING (RESIDUAL SAC)")
        self.log("="*50)
        
        # 3.1 Random Warmup
        self.log("Collecting Random Warmup Data (Valid Delay)...")
        self._collect_random_warmup(steps=cfg.TRAIN.WARMUP_STEPS)
        
        # 3.2 Main RL Loop
        self.log("Starting Main RL Loop...")
        self._run_rl_loop()

    # =========================================================================
    # PHASE 1 HELPER: EXPERT DATA GENERATION
    # =========================================================================
    def generate_expert_data(self, steps):
        """
        Generates expert demonstrations by disabling action delay for the teacher (ID).
        This provides perfect (obs, expert_action) pairs where expert_action is the 
        ideal torque to track the leader given the *current* state.
        """
        temp_buffer = ReplayBuffer(steps, cfg.ROBOT.RL_OBS_DIM, cfg.ROBOT.N_JOINTS, cfg.ROBOT.ROBOT_STATE_DIM, self.device)
        
        # 1. Disable Action Delay (God Mode for Label Generation)
        if self.is_vec_env:
            self.env.env_method("set_action_delay_enabled", False)
        else:
            self.env.follower.action_delay_enabled = False
            
        obs = self._reset_env()
        collected = 0
        
        while collected < steps:
            # Get Inverse Dynamics Action (Teacher)
            if self.is_vec_env:
                expert_actions = self.env.env_method("get_expert_action")
                action = np.array(expert_actions)
                step_increment = self.num_envs
            else:
                action = self.env.get_expert_action()
                step_increment = 1
                
            next_obs, _, done, info = self._step_env(action)
            
            # Store in Temp Buffer
            if self.is_vec_env:
                true_states = np.stack([i['true_state_vector'] for i in info])
                # Reward is 0.0 because BC doesn't use reward
                temp_buffer.add_batch(obs, action, np.zeros(self.num_envs), next_obs, done, true_states)
            else:
                true_state = info.get('true_state_vector', np.zeros(14))
                temp_buffer.add(obs, action, 0.0, next_obs, float(done), true_state)
            
            obs = next_obs
            if not self.is_vec_env and done: 
                obs = self._reset_env()
                
            collected += step_increment
            if collected % 5000 == 0:
                self.log(f"Generated {collected}/{steps} expert samples...")

        # 2. Re-enable Action Delay (Restore Reality for Student)
        if self.is_vec_env:
            self.env.env_method("set_action_delay_enabled", True)
        else:
            self.env.follower.action_delay_enabled = True
            
        return temp_buffer

    # =========================================================================
    # PHASE 2 HELPER: DYNAMIC BEHAVIORAL CLONING
    # =========================================================================
    def _train_bc(self, expert_buffer):
        """
        Supervised Training loop.
        Loss = Action_MSE + (Aux_Weight * State_MSE)
        Aux_Weight decays over time to shift focus from 'Physics' to 'Control'.
        """
        # Use a separate optimizer for BC if desired, or re-use actor_optimizer
        # We use a simple Adam here to ensure clean slate for params not in actor_optimizer (if any)
        # But using self.actor_optimizer is better to keep momentum states if we wanted continuous training.
        # Here we re-init a fresh optimizer to strictly follow config BC_LR.
        optimizer = optim.Adam(self.actor.parameters(), lr=cfg.TRAIN.BC_LR)
        
        aux_weight = cfg.TRAIN.WEIGHT_PRE_LOSS_START
        self.actor.train()
        self.actor.base_encoder.requires_grad_(True) # Ensure LSTM is training
        
        for epoch in range(cfg.TRAIN.BC_EPOCHS):
            num_batches = expert_buffer.size // cfg.TRAIN.BATCH_SIZE
            avg_bc_loss = 0
            avg_aux_loss = 0
            
            for _ in range(num_batches):
                batch = expert_buffer.sample(cfg.TRAIN.BATCH_SIZE)
                
                # Forward Pass (Deterministic=True for BC usually, but here we train distribution mean)
                # We want the MEAN of the policy to match the expert
                mu, _, pred_state, _ = self.actor(batch['obs'])
                pred_action = torch.tanh(mu) * self.actor.action_scale
                
                # 1. Imitation Loss (Match Expert Torque)
                bc_loss = F.mse_loss(pred_action, batch['actions'])
                
                # 2. Aux Loss (Physics Grounding - Match True Leader State)
                # 'true_state_vector' in buffer is [LeaderQ, LeaderQd]
                aux_loss = F.mse_loss(pred_state, batch['true_state_vector'])
                
                # Weighted Sum
                total_loss = bc_loss + (aux_weight * aux_loss)
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                avg_bc_loss += bc_loss.item()
                avg_aux_loss += aux_loss.item()
            
            # Decay the Aux Weight
            aux_weight = max(cfg.TRAIN.WEIGHT_PRE_LOSS_END, aux_weight * cfg.TRAIN.BC_AUX_LOSS_DECAY)
            
            if epoch % 5 == 0 or epoch == cfg.TRAIN.BC_EPOCHS - 1:
                self.log(f"BC Epoch {epoch:03d} | BC_Loss: {avg_bc_loss/num_batches:.4f} | "
                         f"Aux_Loss: {avg_aux_loss/num_batches:.4f} | W_Aux: {aux_weight:.3f}")

    # =========================================================================
    # PHASE 3 HELPER: RL LOOPS
    # =========================================================================
    def _collect_random_warmup(self, steps):
        obs = self._reset_env()
        collected = 0
        while collected < steps:
            if self.is_vec_env:
                action = np.random.uniform(-1, 1, size=(self.num_envs, cfg.ROBOT.N_JOINTS)) * cfg.ROBOT.MAX_ACTION_TORQUE
                next_obs, reward, done, info = self._step_env(action)
                true_states = np.stack([i['true_state_vector'] for i in info])
                self.buffer.add_batch(obs, action, reward * self.REWARD_SCALE, next_obs, done, true_states)
                collected += self.num_envs
            else:
                action = np.random.uniform(-1, 1, size=cfg.ROBOT.N_JOINTS) * cfg.ROBOT.MAX_ACTION_TORQUE
                next_obs, reward, done, info = self._step_env(action)
                true_state = info.get('true_state_vector', np.zeros(14))
                self.buffer.add(obs, action, reward * self.REWARD_SCALE, next_obs, float(done), true_state)
                if done: obs = self._reset_env()
                collected += 1
            obs = next_obs

    def _run_rl_loop(self):
        """
        Phase 3: RL Fine-Tuning Loop.
        CORRECTED STRATEGY:
        1. Freeze Encoder ONLY (Protect Vision).
        2. UNFREEZE Actor IMMEDIATELY (Allow Policy to fix BC errors).
        3. Safety Monitor: Reset if RL deviates wildly from Physics.
        """
        obs = self._reset_env()
        
        self.log("Phase 3 Start: Freezing Encoder ONLY. Actor is ACTIVE.")
        
        # 1. Freeze Encoder (Protect the Brain)
        self.actor.base_encoder.requires_grad_(False)
        encoder_frozen = True
        
        # 2. Ensure Actor is Active (Allow the Body to learn)
        self.actor.residual_net.requires_grad_(True)
        self.actor.res_mu.requires_grad_(True)
        self.actor.res_log_std.requires_grad_(True)
        
        ENCODER_FREEZE_STEPS = 25_000 
        
        for global_step in range(0, self.TOTAL_STEPS, self.num_envs):
            
            # --- SCHEDULED UNFREEZING ---
            if encoder_frozen and global_step >= ENCODER_FREEZE_STEPS:
                self.log(f"Step {global_step}: Unfreezing ENCODER for Full E2E Fine-Tuning!")
                self.actor.base_encoder.requires_grad_(True)
                encoder_frozen = False
            
            # --- EVALUATION ---
            if global_step > 0 and global_step % self.EVAL_INTERVAL == 0:
                self._evaluate(global_step)

            # --- ACTION SELECTION ---
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                if not self.is_vec_env: obs_t = obs_t.unsqueeze(0)
                action_t, _, _, _ = self.actor.sample(obs_t)
                action = action_t.cpu().numpy()
                if not self.is_vec_env: action = action[0]

            # ==================================================================
            # >>> SAFETY MONITOR & DEBUG <<<
            # ==================================================================
            # 1. Get Teacher Action
            if self.is_vec_env:
                expert_actions = self.env.env_method("get_expert_action")
                id_action = np.array(expert_actions)[0]
                rl_action_debug = action[0]
            else:
                id_action = self.env.get_expert_action()
                rl_action_debug = action

            # 2. Check Divergence
            diff_norm = np.linalg.norm(rl_action_debug - id_action)
            
            # 3. SAFETY RESET
            # If RL is doing something insane (Action Divergence > 35.0), 
            # it means the robot is crashing or fighting physics. 
            # RESET immediately to prevent buffer pollution.
            force_reset = False
            if diff_norm > 35.0:
                if global_step % 10 == 0: # Avoid spamming log
                    self.log(f"[SAFETY] Step {global_step} Divergence {diff_norm:.2f} -> FORCING RESET")
                force_reset = True
                
            # Log periodically
            if global_step % 500 == 0:
                self.log(f"[DEBUG Step {global_step}] Action Div: {diff_norm:.2f}")
                self.log(f"  RL: {np.round(rl_action_debug[:4], 1)}")
                self.log(f"  ID: {np.round(id_action[:4], 1)}")
            # ==================================================================

            # --- ENV STEP ---
            next_obs, reward, done, info = self._step_env(action)
            
            # If Safety Reset triggered, override Done
            if force_reset:
                done = True 
                # Penalize reward slightly to discourage crashing
                reward = -5.0 

            # --- STORAGE ---
            if self.is_vec_env:
                true_states = np.stack([i['true_state_vector'] for i in info])
                self.buffer.add_batch(obs, action, reward * self.REWARD_SCALE, next_obs, done, true_states)
            else:
                true_state = info.get('true_state_vector', np.zeros(14))
                self.buffer.add(obs, action, reward * self.REWARD_SCALE, next_obs, float(done), true_state)
            
            obs = next_obs
            if (not self.is_vec_env and done) or force_reset: 
                obs = self._reset_env()

            # --- UPDATE ---
            if self.buffer.size > self.BATCH_SIZE:
                updates = self.num_envs
                for _ in range(updates):
                    # Actor is ALWAYS updated now
                    metrics = self.algo.update(
                        self.buffer.sample(self.BATCH_SIZE), 
                        update_actor=True, 
                        fine_tune_encoder=(not encoder_frozen)
                    )
                
                if global_step % 1000 == 0:
                    avg_rew = np.mean(reward)
                    status = "ENCODER_FROZEN" if encoder_frozen else "FULL_E2E"
                    self.log(f"Step {global_step} [{status}] | R: {avg_rew:.2f} | "
                             f"A_Loss: {metrics['actor_loss']:.2f} | Div: {diff_norm:.1f}")
                    
                    self.writer.add_scalar("Reward/Avg", avg_rew, global_step)
                    self.writer.add_scalar("Loss/Actor", metrics['actor_loss'], global_step)
                    self.writer.add_scalar("Debug/ActionDiv", diff_norm, global_step)

    def _evaluate(self, step):
        """
        Evaluation Loop. Prints True vs. Predicted vs. Remote state for debugging.
        """
        self.log(f"\n--- Evaluation at Step {step} ---")
        
        eval_rewards = []
        eval_pred_errors = []
        eval_track_errors = []
        eval_episodes = 5
        
        for ep_idx in range(eval_episodes):
            obs = self._reset_env()
            ep_step_count = 0 
            
            if self.is_vec_env:
                current_rewards = np.zeros(self.num_envs)
                final_rewards = np.zeros(self.num_envs)
                finished_mask = np.zeros(self.num_envs, dtype=bool)
            else:
                total_rew = 0
                done = False

            while True:
                ep_step_count += 1
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                    if not self.is_vec_env: obs_t = obs_t.unsqueeze(0)
                    
                    # Deterministic Action for Eval
                    mu_res, _, pred_state_t, _ = self.actor(obs_t)
                    action = (torch.tanh(mu_res) * self.actor.action_scale).cpu().numpy()
                    pred_state = pred_state_t.cpu().numpy()
                    
                    if not self.is_vec_env: 
                        action = action[0]
                        pred_state = pred_state[0]

                next_obs, reward, done, info = self._step_env(action)
                
                # Metrics Extraction
                if self.is_vec_env:
                    true_state_vec = info[0]['true_state_vector']
                    # Assuming true_state_vector is [LeaderQ(7), LeaderQd(7)]
                    true_q = true_state_vec[:7]
                    curr_f_q = info[0]['follower_q']
                    curr_pred_vec = pred_state[0]
                else:
                    true_state_vec = info['true_state_vector']
                    true_q = true_state_vec[:7]
                    curr_f_q = info['follower_q']
                    curr_pred_vec = pred_state

                # Errors
                pred_err = np.linalg.norm(true_state_vec - curr_pred_vec)
                track_err = np.linalg.norm(true_q - curr_f_q)
                
                eval_pred_errors.append(pred_err)
                eval_track_errors.append(track_err)

                # Termination Handling
                if self.is_vec_env:
                    current_rewards += reward * (1 - finished_mask)
                    new_done = done & (~finished_mask)
                    if np.any(new_done):
                        final_rewards[new_done] = current_rewards[new_done]
                        finished_mask = finished_mask | new_done
                    if np.any(done):
                        avg_ep_rew = np.mean(final_rewards[new_done])
                        eval_rewards.append(avg_ep_rew)
                        break
                else:
                    total_rew += reward
                    if done:
                        # Print Termination Reason
                        reason = info.get('termination_reason', 'None')
                        if reason != 'None':
                            self.log(f"  Eval Ep {ep_idx} Terminated: {reason}")
                        eval_rewards.append(total_rew)
                        break
                
                obs = next_obs
            
        mean_rew = np.mean(eval_rewards)
        mean_pred_err = np.mean(eval_pred_errors)
        mean_track_err = np.mean(eval_track_errors)
        
        self.log(f"  Result: Avg Reward: {mean_rew:.2f} | Pred Error: {mean_pred_err:.4f} | Track Error: {mean_track_err:.4f}")
        
        self.writer.add_scalar("Eval/MeanReward", mean_rew, step)
        self.writer.add_scalar("Eval/PredError", mean_pred_err, step)
        self.writer.add_scalar("Eval/TrackError", mean_track_err, step)