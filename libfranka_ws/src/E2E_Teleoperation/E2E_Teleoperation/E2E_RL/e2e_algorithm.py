"""
LSTM + SAC algorithm (Optimized with Mixed Precision)
"""

"""
LSTM + SAC algorithm (Optimized with Mixed Precision and Joint/Decoupled support)
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler 
import E2E_Teleoperation.config.robot_config as cfg

class ResidualSAC:
    def __init__(self, actor, critic, critic_target, 
                 actor_optimizer, critic_optimizer, alpha_optimizer,
                 log_alpha,
                 target_entropy=None, gamma=0.99, tau=0.005):
        
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        
        self.log_alpha = log_alpha
        self.scaler = GradScaler('cuda')

        if target_entropy is None:
            self.target_entropy = -torch.prod(torch.tensor(cfg.ROBOT.MAX_ACTION_TORQUE.shape)).item()
        else:
            self.target_entropy = target_entropy
            
        self.gamma = gamma
        self.tau = tau
        self.aux_loss_weight = cfg.TRAIN.AUX_LOSS_GRADIENT_SCALE

    def update(self, batch, update_actor=True):
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        true_state = batch['true_state_vector']

        # ----------------------------
        # 1. CRITIC UPDATE (Standard)
        # ----------------------------
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                next_mu, next_log_std, next_pred_state, _ = self.actor(next_obs)
                next_std = next_log_std.exp()
                next_action_dist = torch.distributions.Normal(next_mu, next_std)
                next_action = torch.tanh(next_action_dist.rsample()) * self.actor.action_scale
                
                log_prob = next_action_dist.log_prob(next_action_dist.rsample()).sum(-1, keepdim=True)
                log_prob -= torch.log(self.actor.action_scale * (1 - (next_action/self.actor.action_scale).pow(2)) + 1e-6).sum(-1, keepdim=True)
                
                target_q1, target_q2 = self.critic_target(next_obs, next_action, next_pred_state)
                target_q = torch.min(target_q1, target_q2) - (self.log_alpha.exp() * log_prob)
                q_target = rewards + (1 - dones) * self.gamma * target_q

        with torch.amp.autocast('cuda'):
            # Detach prediction for critic update so critic loss doesn't affect LSTM
            with torch.no_grad():
                _, _, current_pred_state_no_grad, _ = self.actor(obs)
            
            q1, q2 = self.critic(obs, actions, current_pred_state_no_grad)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.TRAIN.GRAD_CLIP)
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()

        metrics = {"critic_loss": critic_loss.item(), "q1": q1.mean().item()}

        if update_actor:
            # ----------------------------
            # 2. ACTOR UPDATE
            # ----------------------------
            if cfg.TRAIN.JOINT_OPTIMIZATION:
                # === PATH A: JOINT (Summed Loss, Single Backward) ===
                with torch.amp.autocast('cuda'):
                    mu, log_std, pred_state, _ = self.actor(obs)
                    
                    # Pred Loss (Supervised)
                    pred_loss = F.mse_loss(pred_state, true_state)
                    
                    # RL Loss
                    std = log_std.exp()
                    dist = torch.distributions.Normal(mu, std)
                    x_t = dist.rsample()
                    y_t = torch.tanh(x_t)
                    pred_action = y_t * self.actor.action_scale
                    
                    log_prob = dist.log_prob(x_t)
                    log_prob -= torch.log(self.actor.action_scale * (1 - y_t.pow(2)) + 1e-6)
                    log_prob = log_prob.sum(1, keepdim=True)
                    
                    # Use attached pred_state so RL can shape embedding
                    q1_pi, q2_pi = self.critic(obs, pred_action, pred_state)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    
                    rl_loss = ((self.log_alpha.exp() * log_prob) - min_q_pi).mean()
                    
                    # Sum Losses
                    total_actor_loss = rl_loss + (self.aux_loss_weight * pred_loss)
                
                self.actor_optimizer.zero_grad()
                self.scaler.scale(total_actor_loss).backward()
                self.scaler.unscale_(self.actor_optimizer)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.TRAIN.GRAD_CLIP)
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
                
            else:
                # === PATH B: DECOUPLED (Separate Backward Passes) ===
                # Step 1: Prediction Update
                with torch.amp.autocast('cuda'):
                    _, _, pred_state, _ = self.actor(obs)
                    pred_loss = F.mse_loss(pred_state, true_state)
                
                self.actor_optimizer.zero_grad()
                # Manually scale gradients here since we are not summing losses
                self.scaler.scale(pred_loss * self.aux_loss_weight).backward() 
                self.scaler.unscale_(self.actor_optimizer)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.TRAIN.GRAD_CLIP)
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
                
                # Step 2: RL Update
                with torch.amp.autocast('cuda'):
                    mu, log_std, pred_state_detached, _ = self.actor(obs)
                    # Use detached prediction so RL doesn't fight Pred gradients immediately
                    
                    std = log_std.exp()
                    dist = torch.distributions.Normal(mu, std)
                    x_t = dist.rsample()
                    y_t = torch.tanh(x_t)
                    pred_action = y_t * self.actor.action_scale
                    
                    log_prob = dist.log_prob(x_t)
                    log_prob -= torch.log(self.actor.action_scale * (1 - y_t.pow(2)) + 1e-6)
                    log_prob = log_prob.sum(1, keepdim=True)
                    
                    q1_pi, q2_pi = self.critic(obs, pred_action, pred_state_detached.detach())
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    
                    rl_loss = ((self.log_alpha.exp() * log_prob) - min_q_pi).mean()

                self.actor_optimizer.zero_grad()
                self.scaler.scale(rl_loss).backward()
                self.scaler.unscale_(self.actor_optimizer)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.TRAIN.GRAD_CLIP)
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
                
                total_actor_loss = rl_loss + pred_loss # For logging

            # --- Alpha Update (Shared) ---
            with torch.amp.autocast('cuda'):
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.alpha_optimizer)
            self.scaler.update()
            
            # Soft Update Targets
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            metrics.update({
                "actor_loss": rl_loss.item(),
                "pred_loss": pred_loss.item(),
                "total_loss": total_actor_loss.item(),
                "alpha": self.log_alpha.exp().item()
            })
            
        return metrics