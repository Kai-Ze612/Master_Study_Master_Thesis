"""
LSTM + SAC algorithm (Fixed: Q-value explosion and log_prob bug)

Key Fixes:
1. Fixed log_prob calculation using the SAME sampled action
2. Added reward normalization option
3. Added Q-value clipping to prevent explosion
4. Added logging for Q-value monitoring
"""

import torch
import torch.nn.functional as F
import E2E_Teleoperation.config.robot_config as cfg


class ResidualSAC:
    def __init__(self, actor, critic, critic_target, 
                 actor_optimizer, critic_optimizer, alpha_optimizer,
                 log_alpha,
                 target_entropy=None, 
                 gamma=0.99, 
                 tau=0.005,
                 reward_scale=0.1,      # Scale down rewards
                 q_clip=(-200, 200)):   # Clip Q-values for stability
        
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.log_alpha = log_alpha
        
        # Target entropy: -dim(action_space) is standard
        if target_entropy is None:
            self.target_entropy = -cfg.ROBOT.N_JOINTS  # -7 for 7-DOF
        else:
            self.target_entropy = target_entropy
            
        self.gamma = gamma
        self.tau = tau
        self.aux_loss_weight = cfg.TRAIN.AUX_LOSS_GRADIENT_SCALE
        self.reward_scale = reward_scale
        self.q_clip = q_clip

    def _sample_action_and_log_prob(self, mu, log_std, action_scale):
        """Correct SAC sampling with tanh squashing."""
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * action_scale
        
        # Log prob with tanh correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

    def update(self, batch, update_actor=True):
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards'] * self.reward_scale  # Scale rewards!
        next_obs = batch['next_obs']
        dones = batch['dones']
        true_state = batch['true_state_vector']
        
        # Extract follower states for critic
        follower_state = obs[:, :14]
        next_follower_state = next_obs[:, :14]

        # ----------------------------
        # 1. CRITIC UPDATE
        # ----------------------------
        with torch.no_grad():
            # Get next action and latent from actor
            next_mu, next_log_std, next_pred_state, _, next_latent = self.actor(next_obs)
            next_action, next_log_prob = self._sample_action_and_log_prob(
                next_mu, next_log_std, self.actor.action_scale
            )
            
            # Target Q-values using compact inputs
            target_q1, target_q2 = self.critic_target(
                next_follower_state, next_latent, next_pred_state, next_action
            )
            target_q = torch.min(target_q1, target_q2) - (self.log_alpha.exp() * next_log_prob)
            
            # Clip target Q for stability
            if self.q_clip is not None:
                target_q = torch.clamp(target_q, self.q_clip[0], self.q_clip[1])
            
            q_target = rewards + (1 - dones) * self.gamma * target_q

        # Current Q-values (need latent from actor, but detached)
        with torch.no_grad():
            _, _, current_pred_state, _, current_latent = self.actor(obs)
        
        q1, q2 = self.critic(follower_state, current_latent, current_pred_state, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.TRAIN.GRAD_CLIP)
        self.critic_optimizer.step()

        metrics = {
            "critic_loss": critic_loss.item(), 
            "q1_mean": q1.mean().item(),
            "q1_max": q1.max().item(),
            "q1_min": q1.min().item(),
            "q_target_mean": q_target.mean().item(),
        }

        if update_actor:
            if cfg.TRAIN.JOINT_OPTIMIZATION:
                raise NotImplementedError("Joint optimization not implemented in this version")
            else:
                # === DECOUPLED MODE ===
                
                # Step 1: LSTM Prediction Update
                _, _, pred_state, _, _ = self.actor(obs)
                pred_loss = F.mse_loss(pred_state, true_state)
                
                self.actor_optimizer.zero_grad()
                (pred_loss * self.aux_loss_weight).backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.TRAIN.GRAD_CLIP)
                self.actor_optimizer.step()
                
                # Step 2: RL Policy Update
                mu, log_std, pred_state_for_critic, _, latent_for_critic = self.actor(obs)
                pred_action, log_prob = self._sample_action_and_log_prob(
                    mu, log_std, self.actor.action_scale
                )
                
                # Critic evaluation with compact inputs
                # Detach latent and pred_state so policy gradient doesn't flow through LSTM
                q1_pi, q2_pi = self.critic(
                    follower_state, 
                    latent_for_critic.detach(), 
                    pred_state_for_critic.detach(), 
                    pred_action
                )
                min_q_pi = torch.min(q1_pi, q2_pi)
                
                # SAC actor loss
                alpha = self.log_alpha.exp().detach()
                rl_loss = (alpha * log_prob - min_q_pi).mean()

                self.actor_optimizer.zero_grad()
                rl_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.TRAIN.GRAD_CLIP)
                self.actor_optimizer.step()
                
                total_actor_loss = rl_loss + pred_loss

            # Alpha Update
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Soft Target Update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            metrics.update({
                "actor_loss": rl_loss.item(),
                "pred_loss": pred_loss.item(),
                "total_loss": total_actor_loss.item(),
                "alpha": self.log_alpha.exp().item(),
                "log_prob_mean": log_prob.mean().item(),
                "min_q_pi": min_q_pi.mean().item(),
            })
            
        return metrics