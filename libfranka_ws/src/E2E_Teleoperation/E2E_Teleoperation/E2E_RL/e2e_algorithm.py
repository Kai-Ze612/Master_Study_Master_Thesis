"""
LSTM + SAC algorithm
This script defines the backpropagation and update steps for the End-to-End

Components:
1. Critic (Q-Function):
   - Learns to estimate the value of state-action pairs Q(s, a).
   - Loss: MSE between Current Q and the Bellman Target (r + gamma * V_next).
   - Inputs: Includes 'Remote State' (Ground Truth) for Asymmetric Learning.

2. Actor (Policy):
   - Learns the optimal action distribution pi(a|s).
   - Loss: Standard SAC Max-Entropy Objective + Auxiliary State Prediction Loss.
   - The Auxiliary Loss enforces consistency in the latent state encoder (LSTM).

3. Alpha (Entropy Coefficient):
   - Controls the trade-off between exploration and exploitation.
   - Optimized automatically to maintain a fixed Target Entropy.
"""

import torch
import torch.nn.functional as F
import E2E_Teleoperation.config.robot_config as cfg

class ResidualSAC:
    def __init__(
        self,
        actor,
        critic,
        critic_target,
        actor_optimizer,
        critic_optimizer,
        alpha_optimizer,
        log_alpha,
    ):
       
        # Initialize networks and optimizers
        self.actor = actor # Current Policy
        self.critic = critic # Current Q networks
        self.critic_target = critic_target # This is bellman equation
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.log_alpha = log_alpha
        self.gamma = cfg.TRAIN.GAMMA
        self.tau = cfg.SAC.TARGET_TAU
        
        # Automatic Entropy Tuning using Ratio from Config
        self.target_entropy = -float(cfg.ROBOT.N_JOINTS) * cfg.SAC.TARGET_ENTROPY_RATIO

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update(self, batch, update_actor=True, fine_tune_encoder=False):
        # --- FIX: HANDLE DICTIONARY BATCH WITH CORRECT KEYS ---
        if isinstance(batch, dict):
            # Extract tensors using the keys defined in unified_trainer.py ReplayBuffer
            obs = batch['obs']
            action = batch['actions']          # <--- Fixed Key
            reward = batch['rewards']          # <--- Fixed Key
            next_obs = batch['next_obs']
            done = batch['dones']              # <--- Fixed Key
            true_state = batch['true_state_vector'] # <--- Fixed Key
        else:
            # Fallback for tuple unpacking (if used elsewhere)
            obs, action, reward, next_obs, done, true_state = batch
        
        # Move to device
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device) 
        true_state = true_state.to(self.device)

        # --------------------------
        # 1. CRITIC UPDATE
        # --------------------------
        with torch.no_grad():
            # Get next action from actor
            next_mu, next_log_std, next_pred_state, _ = self.actor(next_obs)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mu, next_std)
            next_action = next_dist.rsample()
            next_action_tanh = torch.tanh(next_action)
            
            # Compute Log Prob
            log_prob_next = next_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            log_prob_next -= torch.log(self.actor.action_scale * (1 - next_action_tanh.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
            
            # Target Q
            target_q1, target_q2 = self.critic_target(next_obs, next_action_tanh, next_pred_state)
            target_q = torch.min(target_q1, target_q2) - (self.log_alpha.exp() * log_prob_next)
            q_target = reward + (1 - done) * self.gamma * target_q

        # Current Q
        if fine_tune_encoder:
             _, _, pred_state_curr, _ = self.actor(obs)
        else:
             with torch.no_grad():
                 _, _, pred_state_curr, _ = self.actor(obs)

        q1, q2 = self.critic(obs, action, pred_state_curr)
        
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        # [DEBUG] Capture Q-statistics
        with torch.no_grad():
            q_mean = (q1.mean() + q2.mean()) / 2
            q_max = torch.max(q1.max(), q2.max())
            q_min = torch.min(q1.min(), q2.min())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --------------------------
        # 2. ACTOR UPDATE
        # --------------------------
        actor_loss_val = 0.0
        pred_loss_val = 0.0
        entropy_val = 0.0
        action_norm_val = 0.0
        
        if update_actor:
            mu, log_std, pred_state, _ = self.actor(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action_sample = dist.rsample()
            action_tanh = torch.tanh(action_sample)
            
            log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(self.actor.action_scale * (1 - action_tanh.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
            
            # [DEBUG] Capture Policy Statistics
            with torch.no_grad():
                entropy_val = -log_prob.mean().item()
                action_norm_val = action_tanh.abs().mean().item()

            # Actor Loss
            q1_pi, q2_pi = self.critic(obs, action_tanh, pred_state)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            alpha = self.log_alpha.exp()
            actor_loss = ((alpha * log_prob) - min_q_pi).mean()
            
            # Auxiliary Prediction Loss
            pred_loss = F.mse_loss(pred_state, true_state)
            
            w_pre = cfg.TRAIN.WEIGHT_PRE_LOSS
            total_actor_loss = actor_loss + (w_pre * pred_loss)

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor_optimizer.step()
            
            actor_loss_val = actor_loss.item()
            pred_loss_val = pred_loss.item()
            
            # --- FIX 3: STABLE ALPHA UPDATE (Full Recomputation) ---
            # We must re-sample the action from the *updated* policy to get the correct entropy gradient.
            with torch.no_grad():
                 # Re-run forward pass with the updated actor weights
                 mu_t, log_std_t, _, _ = self.actor(obs)
                 std_t = log_std_t.exp()
                 dist_t = torch.distributions.Normal(mu_t, std_t)
                 
                 # Resample fresh action
                 act_sample_t = dist_t.rsample() 
                 act_tanh_t = torch.tanh(act_sample_t)
                 
                 # Compute log prob
                 log_prob_t = dist_t.log_prob(act_sample_t).sum(dim=-1, keepdim=True)
                 log_prob_t -= torch.log(self.actor.action_scale * (1 - act_tanh_t.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)

            alpha_loss = -(self.log_alpha * (log_prob_t + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        if update_actor: 
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "actor_loss": actor_loss_val,
            "critic_loss": critic_loss.item(),
            "pred_loss": pred_loss_val,
            "alpha": self.log_alpha.exp().item(),
            "q_mean": q_mean.item(),
            "q_max": q_max.item(),
            "q_min": q_min.item(),
            "entropy": entropy_val,
            "action_norm": action_norm_val
        }