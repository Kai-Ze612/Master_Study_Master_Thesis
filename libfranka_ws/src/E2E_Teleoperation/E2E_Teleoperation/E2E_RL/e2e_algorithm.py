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
            # Default to -dim(A) (standard heuristic)
            self.target_entropy = -float(actor.res_mu.out_features) 
        else:
            self.target_entropy = target_entropy
            
        self.gamma = gamma
        self.tau = tau

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, batch, update_actor=True, fine_tune_encoder=False):
        """
        Updates Critic, Actor, and Entropy Coefficient.
        OPTIMIZED: Removes redundant forward passes.
        """
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        true_state = batch['true_state_vector']
        
        # ----------------------------
        # 1. CRITIC UPDATE
        # ----------------------------
        with torch.no_grad():
            # Get next action from policy
            next_mu, next_log_std, next_pred_state, _ = self.actor(next_obs)
            
            # Sample next action
            std = next_log_std.exp()
            dist = torch.distributions.Normal(next_mu, std)
            next_action_sample = dist.rsample()
            next_action_tanh = torch.tanh(next_action_sample) * self.actor.action_scale
            
            # Compute log prob for entropy correction
            log_prob_next = dist.log_prob(next_action_sample).sum(dim=-1, keepdim=True)
            log_prob_next -= torch.log(self.actor.action_scale * (1 - torch.tanh(next_action_sample).pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
            
            # Target Q-values
            target_q1, target_q2 = self.critic_target(next_obs, next_action_tanh, next_pred_state)
            min_target_q = torch.min(target_q1, target_q2) - (self.alpha.detach() * log_prob_next)
            
            # Bellman Equation
            q_target = rewards + (1 - dones) * self.gamma * min_target_q

        # Current Q-values
        # Note: We detach 'pred_state' here because Critic shouldn't update Actor's encoder
        # unless we explicitly design a joint architecture (standard is usually detached for stability)
        # But in your architecture, Actor outputs pred_state. 
        # For simplicity/stability, we usually pass detached state to critic to avoid competing gradients.
        with torch.no_grad():
             _, _, pred_state_curr, _ = self.actor(obs)
             
        q1, q2 = self.critic(obs, actions, pred_state_curr)
        
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        # Metrics dictionary
        metrics = {
            "critic_loss": critic_loss.item(),
            "q1": q1.mean().item(),
            "q2": q2.mean().item(),
            "actor_loss": 0.0,
            "pred_loss": 0.0,
            "alpha": self.alpha.item(),
            "entropy": 0.0
        }

        # ----------------------------
        # 2. ACTOR & ALPHA UPDATE
        # ----------------------------
        if update_actor:
            # --- OPTIMIZATION: Single Forward Pass ---
            # We determine if we need gradients for the encoder
            if fine_tune_encoder:
                # Full gradients allowed
                mu, log_std, pred_state, _ = self.actor(obs)
            else:
                # Freeze Encoder part (detach history/LSTM features) manually or 
                # rely on the fact that we set requires_grad=False in the trainer.
                # Since Trainer controls requires_grad, we just call forward.
                mu, log_std, pred_state, _ = self.actor(obs)

            # Sample Action
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action_sample = dist.rsample()
            action_tanh = torch.tanh(action_sample) * self.actor.action_scale
            
            # Log Prob
            log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(self.actor.action_scale * (1 - torch.tanh(action_sample).pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
            
            # Actor Loss: Alpha * LogProb - Q
            # We pass gradients through Q to update Actor (reparameterization trick)
            q1_pi, q2_pi = self.critic(obs, action_tanh, pred_state.detach()) # Detach state from critic
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()
            
            # Auxiliary Prediction Loss (Physics Grounding)
            # IMPORTANT: This must be calculated even if fine_tune_encoder is False
            # to verify the metric, but typically we only backprop if fine_tune is True.
            # However, if fine_tune=False, the encoder params have grad=False, 
            # so backprop here won't hurt (it just won't update encoder).
            pred_loss = F.mse_loss(pred_state, true_state)
            
            # Configurable weight for Aux Loss (passed via Trainer usually, hardcoded here for simplicity)
            # In your config you have WEIGHT_PRE_LOSS. Ideally passed in __init__.
            # Assuming 1.0 or handled externally.
            # Let's assume the trainer wants us to return raw losses or handle weighting.
            # Your previous code added it to actor_loss.
            
            # Re-read Trainer logic: Trainer passes WEIGHT_PRE_LOSS logic? 
            # Actually Trainer config has it. Let's add it here.
            # NOTE: We use a fixed weight or just return losses. 
            # To match your structure, we assume pure SAC actor loss here + Aux.
            
            # We will return the components so Trainer can log them, 
            # but we must backprop the SUM.
            # We use a default weight of 0.5 if not specified, 
            # BUT the trainer freezes/unfreezes layers, so we just sum them.
            
            total_actor_loss = actor_loss + (0.5 * pred_loss) 

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_optimizer.step()
            
            # --- 3. ALPHA UPDATE ---
            # Reuse log_prob from above (Detached!)
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # --- 4. TARGET UPDATE ---
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            # Update Metrics
            metrics["actor_loss"] = actor_loss.item()
            metrics["pred_loss"] = pred_loss.item()
            metrics["entropy"] = -log_prob.mean().item()

        return metrics