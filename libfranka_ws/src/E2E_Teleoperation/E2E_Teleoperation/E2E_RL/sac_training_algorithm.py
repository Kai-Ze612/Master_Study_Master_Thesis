"""
E2E_Teleoperation/E2E_RL/sac_training_algorithm.py
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
        # STRATEGY: ENCODER WARMUP (Effective LR=0 for Encoder)
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
        not_done = 1.0 - batch['dones']
        true_state = batch['true_state_vector']
        
        # Optional: Expert Action for Inverse Dynamics (ID) Regularization
        # (Assuming you add this to your buffer later, otherwise it stays None)
        expert_action = batch.get('expert_action', None) 

        # -------------------------
        # 1. Critic Update
        # -------------------------
        with torch.no_grad():
            # Get next state from Actor/Encoder
            # Note: next_pred_state is used (14-dim), not the full latent feat
            next_action, next_log_prob, next_pred_state, _, _ = self.actor.sample(next_obs)
            
            target_q1, target_q2 = self.critic_target(next_pred_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            alpha = self.log_alpha.exp()
            target_value = reward + not_done * self.gamma * (target_q - alpha * next_log_prob)

        # Current state estimates
        _, _, curr_pred_state, _, _ = self.actor.sample(obs)
        
        # We detach here strictly for the CRITIC update. 
        # The Critic should update its own weights to match the Target Value.
        # It should NOT update the Encoder to make the state "easier to guess".
        current_q1, current_q2 = self.critic(curr_pred_state.detach(), action)
        
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.SAC.GRAD_CLIP_CRITIC)
        self.critic_optimizer.step()

        # -------------------------
        # 2. Actor & Encoder Update (The E2E Part)
        # -------------------------
        new_action, log_prob, pred_state, _, _ = self.actor.sample(obs)
        
        # [TRUE E2E] We pass pred_state WITH GRADIENTS.
        # This allows SAC loss to backpropagate all the way to the LSTM Encoder.
        q1_new, q2_new = self.critic(pred_state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        alpha = self.log_alpha.exp().detach()
        sac_loss = (alpha * log_prob - q_new).mean()
        
        # [STABILITY FIX] Strong Supervised Loss
        # We assume pred_state (14) vs true_state (14)
        pred_loss = F.mse_loss(pred_state, true_state)
        
        # Weight increased to 500.0 to balance the ~30.0 SAC loss
        # This acts as an "Anchor" for the encoder.
        pred_weight = 500.0
        
        # [ID REGULARIZATION]
        # If expert data is available, we add a Behavioral Cloning term.
        # Even without it, the 'pred_weight' protects the encoder.
        bc_loss = 0.0
        if expert_action is not None:
            bc_loss = F.mse_loss(new_action, expert_action)
            bc_weight = 10.0
            total_actor_loss = sac_loss + (pred_weight * pred_loss) + (bc_weight * bc_loss)
        else:
            total_actor_loss = sac_loss + (pred_weight * pred_loss)
        
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        
        # -----------------------------------------------------------
        # [THESIS STRATEGY] LR = 0 FOR ENCODER (WARMUP)
        # -----------------------------------------------------------
        # If we are in the "Chaos Phase" (Steps < 5000), we manually
        # zero out the gradients on the encoder parameters.
        # This effectively sets LR=0 for the Encoder, while the Actor Head keeps learning.
        if self.update_step < self.ENCODER_WARMUP_STEPS:
            # Access the encoder submodule inside the actor
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
            "bc_loss": bc_loss.item() if isinstance(bc_loss, torch.Tensor) else 0.0,
            "alpha": alpha.item()
        }