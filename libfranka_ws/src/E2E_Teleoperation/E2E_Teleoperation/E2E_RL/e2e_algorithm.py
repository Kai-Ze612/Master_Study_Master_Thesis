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
import torch.optim as optim
import E2E_Teleoperation.config.robot_config as cfg

class ResidualSAC:
    def __init__(self, actor, critic, critic_target, 
                 actor_optimizer, critic_optimizer, alpha_optimizer,
                 log_alpha,
                 target_entropy=None, gamma=0.99, tau=0.005, alpha_lr=3e-4):
        
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        
        self.log_alpha = log_alpha
        
        # Automatic Entropy Tuning
        if target_entropy is None:
            self.target_entropy = -torch.prod(torch.tensor(cfg.ROBOT.MAX_ACTION_TORQUE.shape)).item()
        else:
            self.target_entropy = target_entropy
            
        self.gamma = gamma
        self.tau = tau
        
        self.update_counter = 0  # Counter for logging diagnostics

    def update(self, batch, update_actor=True, fine_tune_encoder=True):
        self.update_counter += 1
        
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        true_state = batch['true_state_vector'] # Ground Truth [Leader(14)]

        # ----------------------------
        # 1. CRITIC UPDATE
        # ----------------------------
        with torch.no_grad():
            next_mu, next_log_std, next_pred_state, _ = self.actor(next_obs)
            next_std = next_log_std.exp()
            next_action_dist = torch.distributions.Normal(next_mu, next_std)
            next_action = torch.tanh(next_action_dist.rsample()) * self.actor.action_scale
            
            # Target Q-Value
            target_q1, target_q2 = self.critic_target(next_obs, next_action, next_pred_state)
            target_q = torch.min(target_q1, target_q2)
            
            # Entropy term
            log_prob = next_action_dist.log_prob(next_action_dist.rsample()).sum(-1, keepdim=True)
            log_prob -= torch.log(self.actor.action_scale * (1 - (next_action/self.actor.action_scale).pow(2)) + 1e-6).sum(-1, keepdim=True)
            
            target_q = target_q - (self.log_alpha.exp() * log_prob)
            q_target = rewards + (1 - dones) * self.gamma * target_q
            
            # REMOVED CLAMPING (As discussed previously)
            # q_target = torch.clamp(q_target, -100.0, 100.0) 

        # Current Q
        # Detach latent here to stop Critic from modifying the LSTM
        with torch.no_grad():
            _, _, current_pred_state, _ = self.actor(obs)
            
        q1, q2 = self.critic(obs, actions, current_pred_state.detach())
        
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.TRAIN.GRAD_CLIP_CRITIC)
        self.critic_optimizer.step()

        metrics = {
            "critic_loss": critic_loss.item(), 
            "q1": q1.mean().item(),
            "q_target": q_target.mean().item() # Useful debug
        }

        # ----------------------------
        # 2. ACTOR & ALPHA UPDATE
        # ----------------------------
        if update_actor:
            # Re-run actor to get computation graph
            mu, log_std, pred_state, _ = self.actor(obs)
            
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            x_t = dist.rsample()
            y_t = torch.tanh(x_t)
            pred_action = y_t * self.actor.action_scale
            
            log_prob = dist.log_prob(x_t)
            log_prob -= torch.log(self.actor.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            
            # Policy Loss
            # Detach pred_state for critic input so policy gradient doesn't go through Q->Latent->LSTM
            # But DO allow gradients through pred_state->LSTM from the Aux Loss below
            q1_pi, q2_pi = self.critic(obs, pred_action, pred_state.detach())
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            actor_loss = ((self.log_alpha.exp() * log_prob) - min_q_pi).mean()
            
            # Auxiliary Loss (Physics Consistency)
            # This is what trains the LSTM to track the leader
            pred_loss = F.mse_loss(pred_state, true_state)
            
            # Total Loss
            # Weighted sum allows gradients to flow to LSTM from pred_loss
            total_actor_loss = actor_loss + (cfg.TRAIN.AUX_LOSS_WEIGHT * pred_loss)

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            
            # --- DIAGNOSTIC START ---
            # Check if gradients are reaching the LSTM weights
            if self.update_counter % 100 == 0:
                if hasattr(self.actor.base_encoder, 'lstm_cell'):
                    lstm_grad = self.actor.base_encoder.lstm_cell.weight_ih.grad
                    if lstm_grad is not None:
                        grad_norm = lstm_grad.norm().item()
                        print(f"[DEBUG Step {self.update_counter}] LSTM Grad Norm: {grad_norm:.6f} | Pred Loss: {pred_loss.item():.4f}")
                    else:
                        print(f"[DEBUG Step {self.update_counter}] LSTM Grad IS NONE! (Check fine_tune_encoder flag)")
            # --- DIAGNOSTIC END ---

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.TRAIN.GRAD_CLIP_ACTOR)
            self.actor_optimizer.step()
            
            # --- 3. ALPHA UPDATE ---
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # --- 4. TARGET UPDATE ---
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            metrics.update({
                "actor_loss": actor_loss.item(),
                "pred_loss": pred_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "alpha": self.log_alpha.exp().item()
            })
            
        return metrics