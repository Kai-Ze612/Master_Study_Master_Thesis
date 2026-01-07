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
        """
        1. Update Critic Networks
            - Compute Bellman Target using next state predictions from Actor
            - MSE Loss between Current Q and Target Q
        2. Update Actor Network
            - Sample actions from current policy
            - Compute Actor Loss using Min Q values from Critic
            - Add Auxiliary State Prediction Loss
        3. Update Alpha (Entropy Coefficient)
            - Optimize alpha to maintain a fixed target entropy
        4. Soft Update Target Critic Networks
            - Update target networks using tau parameter
        5. Return Losses and Alpha Value
        """
        
        obs = batch['obs']
        actions_scaled = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        true_states = batch['true_state_vector']

        scale = self.actor.action_scale
        actions_normalized = actions_scaled / scale

        # 1. CRITIC UPDATE
        with torch.no_grad():
            next_mu, next_log_std, next_pred_state, _ = self.actor(next_obs)
            next_std = next_log_std.exp()
            dist = torch.distributions.Normal(next_mu, next_std)
            next_action_sample = dist.rsample() 
            
            # This value is in [-1, 1]
            next_action_tanh = torch.tanh(next_action_sample) 
            
            # Log Prob correction uses scale
            log_prob = dist.log_prob(next_action_sample).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(scale * (1 - next_action_tanh.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
            
            alpha = self.log_alpha.exp()

            # Normalized action for target Q calculation
            target_q1, target_q2 = self.critic_target(next_obs, next_action_tanh, next_pred_state)
            target_q = torch.min(target_q1, target_q2) - alpha * log_prob
            
            q_target = rewards + (1 - dones) * self.gamma * target_q

        # Current Q Calculation
        if not fine_tune_encoder:
            with torch.no_grad():
                _, _, current_pred_state, _ = self.actor(obs)
        else:
             _, _, current_pred_state, _ = self.actor(obs)

        # Use Normalized Actions from Buffer
        q1, q2 = self.critic(obs, actions_normalized, current_pred_state)

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        # [DEBUG] Capture Q-statistics
        with torch.no_grad():
            q_mean = (q1.mean() + q2.mean()) / 2
            q_max = torch.max(q1.max(), q2.max())
            q_min = torch.min(q1.min(), q2.min())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ACTOR UPDATE
        actor_loss_val = 0.0
        pred_loss_val = 0.0
        entropy_val = 0.0      # <--- ADD THIS
        action_norm_val = 0.0  # <--- ADD THIS
        
        if update_actor:
            mu, log_std, pred_state, _ = self.actor(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action_sample = dist.rsample()
            
            # This is [-1, 1]
            action_tanh = torch.tanh(action_sample)
            
            log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(scale * (1 - action_tanh.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
            
            # Use Tanh (Normalized) Action for Actor Loss
            q1_pi, q2_pi = self.critic(obs, action_tanh, pred_state)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            alpha = self.log_alpha.exp().detach()
            actor_loss = (alpha * log_prob - min_q_pi).mean()
            
            pred_loss = F.mse_loss(pred_state, true_states)
            
            # Use weight from config if it exists, else 1.0
            w_pre = getattr(cfg.TRAIN, 'WEIGHT_PRE_LOSS', 1.0)
            total_actor_loss = actor_loss + (w_pre * pred_loss)

            with torch.no_grad():
                entropy_val = -log_prob.mean().item()
                action_norm_val = action_tanh.abs().mean().item() # Approx magnitude
            
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor_optimizer.step()
            
            actor_loss_val = actor_loss.item()
            pred_loss_val = pred_loss.item()
            
            # Alpha Update
            with torch.no_grad():
                 _, log_std_t, _, _ = self.actor(obs)
                 std_t = log_std_t.exp()
                 dist_t = torch.distributions.Normal(mu, std_t)
                 act_sample_t = dist_t.sample()
                 act_tanh_t = torch.tanh(act_sample_t)
                 log_prob_t = dist_t.log_prob(act_sample_t).sum(dim=-1, keepdim=True)
                 log_prob_t -= torch.log(scale * (1 - act_tanh_t.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)

            alpha_loss = -(self.log_alpha * (log_prob_t + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft Update Target Critic Networks
        if update_actor: 
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "actor_loss": actor_loss_val,
            "critic_loss": critic_loss.item(),
            "pred_loss": pred_loss_val,
            "alpha": self.log_alpha.exp().item(),
            # [DEBUG] New Diagnostics
            "q_mean": q_mean.item(),
            "q_max": q_max.item(),
            "q_min": q_min.item(),
            "entropy": entropy_val,
            "action_norm": action_norm_val
        }