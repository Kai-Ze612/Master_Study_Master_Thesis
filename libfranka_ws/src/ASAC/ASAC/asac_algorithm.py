"""
A-SAC Algorithm Implementation
------------------------------
Soft Actor-Critic with stability measures.
All hyperparameters read from robot_config.py
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import ASAC.config.robot_config as cfg


class AugmentedSAC:
    """
    Stabilized Soft Actor-Critic for PD Gain Tuning.
    
    Stability mechanisms:
    - Policy delay: Update actor every N critic updates
    - Target smoothing: Add noise to target actions (TD3-style)
    - Q-clipping: Bound Q-values to reasonable range
    - Alpha bounds: Prevent entropy collapse or explosion
    """
    
    def __init__(
        self, 
        actor, 
        critic, 
        critic_target, 
        actor_optimizer: optim.Optimizer, 
        critic_optimizer: optim.Optimizer, 
        alpha_optimizer: optim.Optimizer,
        log_alpha: torch.Tensor,
        target_entropy: float = None, 
    ):
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.log_alpha = log_alpha
        
        # Read hyperparameters from config
        self.gamma = cfg.TRAIN.GAMMA
        self.tau = cfg.SAC.TARGET_TAU
        self.policy_delay = cfg.SAC.POLICY_DELAY
        self.target_noise = cfg.SAC.TARGET_NOISE
        self.noise_clip = cfg.SAC.NOISE_CLIP
        self.alpha_min = cfg.SAC.ALPHA_MIN
        self.alpha_max = cfg.SAC.ALPHA_MAX
        self.q_clip = cfg.SAC.Q_CLIP
        self.grad_clip = cfg.TRAIN.GRAD_CLIP
        
        # Target entropy (default: -dim(action))
        if target_entropy is None:
            self.target_entropy = -float(cfg.ROBOT.N_JOINTS * 2)  # 14 for gain tuning
        else:
            self.target_entropy = target_entropy
            
        # Update counters
        self.critic_updates = 0
        self.actor_updates = 0
        
        # Monitoring
        self._last_q_mean = 0.0
        self._last_q_std = 0.0

    def update(self, batch: dict) -> dict:
        """Perform one gradient update step."""
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        
        # Get alpha with clamping
        alpha = torch.clamp(self.log_alpha.exp(), self.alpha_min, self.alpha_max)

        # ====================================================================
        # 1. Critic Update (always performed)
        # ====================================================================
        with torch.no_grad():
            # Sample next actions from current policy
            next_mu, next_log_std = self.actor(next_obs)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mu, next_std)
            
            next_action_sample = next_dist.rsample()
            next_action_raw = torch.tanh(next_action_sample) * self.actor.action_scale
            
            # TD3-style target policy smoothing
            target_noise = torch.randn_like(next_action_raw) * self.target_noise
            target_noise = target_noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_action = next_action_raw + target_noise
            next_action = torch.clamp(next_action, -self.actor.action_scale, self.actor.action_scale)
            
            # Log probability
            log_prob = next_dist.log_prob(next_action_sample).sum(-1, keepdim=True)
            log_prob -= torch.log(
                self.actor.action_scale * (1 - (next_action_raw / self.actor.action_scale).pow(2)) + 1e-6
            ).sum(-1, keepdim=True)
            
            # Target Q with entropy
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * log_prob
            target_q = torch.clamp(target_q, -self.q_clip, self.q_clip)
            
            # Bellman target
            q_target = rewards + (1 - dones) * self.gamma * target_q

        # Current Q estimates
        q1, q2 = self.critic(obs, actions)
        
        # Q-clipping for loss
        q1_clipped = torch.clamp(q1, -self.q_clip, self.q_clip)
        q2_clipped = torch.clamp(q2, -self.q_clip, self.q_clip)
        
        # Store Q stats for monitoring
        self._last_q_mean = ((q1 + q2) / 2).mean().item()
        self._last_q_std = ((q1 + q2) / 2).std().item()
        
        # Critic loss
        critic_loss = F.mse_loss(q1_clipped, q_target) + F.mse_loss(q2_clipped, q_target)

        # Critic optimization
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.grad_clip
        )
        self.critic_optimizer.step()
        
        self.critic_updates += 1

        # ====================================================================
        # 2. Actor Update (delayed - every N critic updates)
        # ====================================================================
        actor_loss = torch.tensor(0.0, device=obs.device)
        alpha_loss = torch.tensor(0.0, device=obs.device)
        actor_grad_norm = 0.0
        
        if self.critic_updates % self.policy_delay == 0:
            # Freeze critic
            for param in self.critic.parameters():
                param.requires_grad = False
            
            # Sample actions from current policy
            mu, log_std = self.actor(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            
            x_t = dist.rsample()
            y_t = torch.tanh(x_t)
            pred_action = y_t * self.actor.action_scale
            
            # Log probability
            log_prob_actor = dist.log_prob(x_t)
            log_prob_actor -= torch.log(self.actor.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob_actor = log_prob_actor.sum(1, keepdim=True)
            
            # Q values for actor actions
            q1_pi, q2_pi = self.critic(obs, pred_action)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            # Actor loss
            actor_loss = (alpha.detach() * log_prob_actor - min_q_pi).mean()

            # Actor optimization
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_clip
            )
            self.actor_optimizer.step()
            
            # Unfreeze critic
            for param in self.critic.parameters():
                param.requires_grad = True
            
            self.actor_updates += 1

            # ================================================================
            # 3. Alpha Update
            # ================================================================
            alpha_loss = -(self.log_alpha * (log_prob_actor + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Clamp log_alpha
            with torch.no_grad():
                self.log_alpha.data.clamp_(np.log(self.alpha_min), np.log(self.alpha_max))

            # ================================================================
            # 4. Soft Target Update
            # ================================================================
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item(),
            "q_mean": self._last_q_mean,
            "q_std": self._last_q_std,
            "critic_grad_norm": float(critic_grad_norm) if torch.is_tensor(critic_grad_norm) else critic_grad_norm,
            "actor_grad_norm": float(actor_grad_norm) if torch.is_tensor(actor_grad_norm) else actor_grad_norm,
        }