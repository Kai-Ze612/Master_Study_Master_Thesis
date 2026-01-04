"""
E2E Update Algorithm:


"""


import numpy as np
import torch
import torch.nn.functional as F
import E2E_Teleoperation.config.robot_config as cfg

class SACAlgorithm:
    def __init__(self, actor, critic, critic_target, actor_opt, critic_opt, alpha_opt, log_alpha):
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.actor_optimizer = actor_opt
        self.critic_optimizer = critic_opt
        self.alpha_optimizer = alpha_opt
        self.log_alpha = log_alpha
        
        self.target_entropy = -float(cfg.ROBOT.N_JOINTS) * cfg.SAC.TARGET_ENTROPY_RATIO
        self.gamma = cfg.TRAIN.GAMMA
        self.tau = cfg.SAC.TARGET_TAU
        
        # -----------------------------------------------------------
        # SAFETY LOCK: FREEZE ENCODER AT START OF PHASE 2
        # -----------------------------------------------------------
        self.update_step = 0
        self.ENCODER_WARMUP_STEPS = 5000 
        
        # Determine device reliably from network parameters
        self.device = next(actor.parameters()).device

    def update(self, batch):
        self.update_step += 1
        
        obs = batch['obs']
        action = batch['actions']
        reward = batch['rewards']
        next_obs = batch['next_obs']
        not_done = 1. - batch['dones']
        
        # 1. Fix Alpha Crash (Convert log to exp)
        alpha = self.log_alpha.exp()
        
        # -------------------------
        # 1. Critic Update
        # -------------------------
        with torch.no_grad():
            next_action, next_log_prob, next_pred_state, _, _ = self.actor.sample(next_obs)
            
            # Target Q-values
            target_Q1, target_Q2 = self.critic_target(next_pred_state, next_action)
            target_V = torch.min(target_Q1, target_Q2) - alpha.detach() * next_log_prob
            target_Q = reward + (not_done * self.gamma * target_V)
            
        # Current Q-values
        _, _, curr_pred_state, _, _ = self.actor.sample(obs)
        curr_Q1, curr_Q2 = self.critic(curr_pred_state.detach(), action)
        
        critic_loss = F.mse_loss(curr_Q1, target_Q) + F.mse_loss(curr_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------------------
        # 2. Joint Actor + Encoder Update
        # -------------------------
        new_action, log_prob, pred_state, _, _ = self.actor.sample(obs)
        
        # A. SAC Loss (Maximize Reward)
        q1_new, q2_new = self.critic(pred_state, new_action)
        q_new = torch.min(q1_new, q2_new)
        sac_loss = (alpha.detach() * log_prob - q_new).mean()
        
        # B. Prediction Loss (Maintain Physics)
        true_state = batch['true_state_vector']
        pred_loss = F.mse_loss(pred_state, true_state)
        
        # C. Total Loss (Weighted)
        # Weight = 5.0 (Enough to guide, not enough to dominate)
        pred_weight = 5.0 
        total_actor_loss = sac_loss + (pred_weight * pred_loss)
        
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        
        # --- SAFETY LOCK: WARMUP ---
        # Prevent initial random gradients from breaking the pre-trained LSTM
        if self.update_step < self.ENCODER_WARMUP_STEPS:
            # Zero out gradients for the LSTM (Encoder) only
            for param in self.actor.encoder.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.SAC.GRAD_CLIP_ACTOR)
        self.actor_optimizer.step()

        # -------------------------
        # 3. Alpha Update
        # -------------------------
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # -------------------------
        # 4. Soft Update
        # -------------------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": sac_loss.item(),
            "pred_loss": pred_loss.item(),
            "alpha": alpha.item()
        }