"""
SBSP Algorithm: Standard SAC (Aux loss moved to Pre-training)
"""
import torch
import torch.nn.functional as F
import SBSP.config.robot_config as cfg

class SBSPAlgorithm:
    def __init__(self, actor, critic, critic_target, 
                 actor_optimizer, critic_optimizer, alpha_optimizer,
                 log_alpha,
                 gamma=0.99, 
                 tau=0.005,
                 reward_scale=1.0):
        
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.log_alpha = log_alpha
        
        self.target_entropy = -float(cfg.ROBOT.ACTION_DIM) * cfg.SAC.TARGET_ENTROPY_RATIO
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale

    def update(self, batch):
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards'] * self.reward_scale
        next_obs = batch['next_obs']
        dones = batch['dones']
        # true_state is no longer used here, it's used in DCNN training
        
        # --- 1. CRITIC UPDATE ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - (self.log_alpha.exp() * next_log_prob)
            q_target = rewards + (1 - dones) * self.gamma * target_q

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 2. ACTOR UPDATE (Standard SAC) ---
        new_action, log_prob = self.actor.sample(obs)
        
        q1_pi, q2_pi = self.critic(obs, new_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp()
        sac_loss = (alpha.detach() * log_prob - min_q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        sac_loss.backward()
        self.actor_optimizer.step()

        # --- 3. ALPHA UPDATE ---
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.log_alpha.data.clamp_(-5.0, 2.0)

        # --- 4. TARGET UPDATE ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": sac_loss.item(),
            "alpha": alpha.item(),
            "q_mean": q1.mean().item()
        }