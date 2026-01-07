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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        Optimization process
        
        3 Phases training:
        1. Critic Optimization:
            Goal: Minimize Bellman Error
        2. Actor Optimization:
        
        """
        
        # Batch Data
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        true_states = batch['true_state_vector']

        # 1. CRITIC UPDATE
        with torch.no_grad():
            # Get next action from Actor (Current Policy)
            next_mu, next_log_std, next_pred_state, _ = self.actor(next_obs)

            # Sample next action
            next_std = next_log_std.exp()
            dist = torch.distributions.Normal(next_mu, next_std)
            next_action_sample = dist.rsample() 
            
            # Tanh Squash (SAC requirement)
            next_action_tanh = torch.tanh(next_action_sample)
            
            # Scale to torque limits (Crucial for Q-val accuracy)
            scale = self.actor.action_scale
            next_action_scaled = next_action_tanh * scale

            # Log Prob correction
            log_prob = dist.log_prob(next_action_sample).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(scale * (1 - next_action_tanh.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
            
            alpha = self.log_alpha.exp()

            # Target Q Calculation
            # Input: (next_obs, next_action, next_pred_state)
            target_q1, target_q2 = self.critic_target(next_obs, next_action_scaled, next_pred_state)
            target_q = torch.min(target_q1, target_q2) - alpha * log_prob
            
            # Bellman Target
            q_target = rewards + (1 - dones) * self.gamma * target_q

        # Current Q Calculation
        # We need current prediction for the critic input
        if not fine_tune_encoder:
            with torch.no_grad():
                _, _, current_pred_state, _ = self.actor(obs)
        else:
             _, _, current_pred_state, _ = self.actor(obs)

        # Input: (obs, actions, current_pred_state)
        q1, q2 = self.critic(obs, actions, current_pred_state)
        
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. ACTOR UPDATE
        actor_loss_val = 0.0
        pred_loss_val = 0.0
        
        if update_actor:
            # Re-run actor to get current graph for backprop
            mu, log_std, pred_state, _ = self.actor(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action_sample = dist.rsample()
            action_tanh = torch.tanh(action_sample)
            
            scale = self.actor.action_scale
            action_scaled = action_tanh * scale
            
            # Log Prob
            log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(scale * (1 - action_tanh.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
            
            # Q-values for Actor Loss
            q1_pi, q2_pi = self.critic(obs, action_scaled, pred_state)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            # SAC Objective
            alpha = self.log_alpha.exp().detach()
            actor_loss = (alpha * log_prob - min_q_pi).mean()
            
            # State prediction loss
            pred_loss = F.mse_loss(pred_state, true_states)
            
            # Total Loss
            total_actor_loss = actor_loss + (cfg.TRAIN.WEIGHT_PRE_LOSS * pred_loss)

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            
            self.actor_optimizer.step()
            
            actor_loss_val = actor_loss.item()
            pred_loss_val = pred_loss.item()

            # ALPHA UPDATE ---
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

        # SOFT UPDATE TARGET NETS
        if update_actor: 
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "actor_loss": actor_loss_val,
            "critic_loss": critic_loss.item(),
            "pred_loss": pred_loss_val,
            "alpha": self.log_alpha.exp().item()
        }