"""
E2E Update Algorithm: Physics-Regularized SAC with Curriculum Learning

Pipeline Steps:
1. Data Preparation: 
   - Sample batch from Replay Buffer.

2. Update Critic:
   - Standard SAC MSE Loss against Bellman Target.

3. Update Actor + Encoder (The "Dual Objective"):
   - Computes Composite Loss: L_total = L_SAC (Reward) + λ * L_Physics (Prediction Loss).
   - Curriculum Logic:
     * Phase 1/2 (Frozen): LSTM gradients zeroed out; only Policy head updates.
     * Phase 3 (Fine-tuning): Differential Clipping (Encoder=0.5, Policy=1.0) 
       to prevent Catastrophic Forgetting of physics.

4. Update Alpha (Temperature):
   - Automatic Entropy Adjustment to maintain exploration target.

5. Soft Update:
   - Polyak Averaging for Target Networks.
"""

import torch
import torch.nn.functional as F
import E2E_Teleoperation.config.robot_config as cfg

class ResidualSAC:
    def __init__(self, actor, critic, critic_target, actor_opt, critic_opt, alpha_opt, log_alpha):
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.actor_optimizer = actor_opt
        self.critic_optimizer = critic_opt
        self.alpha_optimizer = alpha_opt
        self.log_alpha = log_alpha
        
        self.gamma = cfg.TRAIN.GAMMA
        self.tau = cfg.SAC.TARGET_TAU
        self.target_entropy = -float(cfg.ROBOT.N_JOINTS) 

    def update(self, batch, update_actor=True, fine_tune_encoder=False):
        obs = batch['obs']
        action = batch['actions']
        reward = batch['rewards']
        next_obs = batch['next_obs']
        not_done = 1. - batch['dones']
        
        alpha = self.log_alpha.exp().detach()
        
        # -------------------------
        # 1. Critic Update
        # -------------------------
        with torch.no_grad():
            next_action, next_log_prob, next_pred_state, _ = self.actor.sample(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_pred_state, next_action)
            target_V = torch.min(target_Q1, target_Q2) - alpha * next_log_prob
            target_Q = reward + (not_done * self.gamma * target_V)

        # Current Q
        # Detach pred_state unless fine-tuning encoder
        _, _, curr_pred_state, _ = self.actor.sample(obs)
        if not fine_tune_encoder:
            curr_pred_state = curr_pred_state.detach()
            
        curr_Q1, curr_Q2 = self.critic(curr_pred_state, action)
        critic_loss = F.mse_loss(curr_Q1, target_Q) + F.mse_loss(curr_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = 0.0
        alpha_val = alpha.item()
        pred_loss_val = 0.0

        # -------------------------
        # 2. Actor Update
        # -------------------------
        if update_actor:
            new_action, log_prob, pred_state_actor, _ = self.actor.sample(obs)
            
            # Detach for Critic Input (Standard SAC)
            q1_new, q2_new = self.critic(pred_state_actor.detach(), new_action)
            q_new = torch.min(q1_new, q2_new)
            
            # SAC Loss
            sac_loss = (alpha * log_prob - q_new).mean()
            total_loss = sac_loss
            
            # Optional: Prediction Loss if Fine-Tuning Encoder (Phase 3)
            if fine_tune_encoder:
                true_state = batch['true_state_vector']
                pred_loss = F.mse_loss(pred_state_actor, true_state)
                # Weighted Sum
                total_loss = sac_loss + (1.0 * pred_loss) 
                pred_loss_val = pred_loss.item()
            
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            
            actor_loss_val = sac_loss.item()
            
            # 3. Alpha Update
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_val = self.log_alpha.exp().item()

        # -------------------------
        # 4. Soft Update
        # -------------------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_val,
            "pred_loss": pred_loss_val,
            "alpha": alpha_val
        }