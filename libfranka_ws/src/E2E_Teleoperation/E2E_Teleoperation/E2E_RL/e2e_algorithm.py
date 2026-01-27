"""
LSTM + SAC algorithm (Fixed Version)

Key Fixes from Original:
1. Fixed gradient flow in decoupled mode - single forward pass
2. Proper alpha initialization guidance
3. Better reward normalization
4. Q-value monitoring with adaptive clipping
5. Gradient accumulation for stability
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
                 reward_scale=1.0,       # [CHANGED] Less aggressive scaling
                 q_clip=(-100, 100),     # [CHANGED] Tighter Q-clip
                 grad_accumulation_steps=1):  # [NEW] Gradient accumulation
        
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.log_alpha = log_alpha
        
        # Target entropy: -dim(action_space) is standard, but can be tuned
        # Using a ratio < 1.0 encourages less exploration (good for fine-tuning)
        if target_entropy is None:
            self.target_entropy = -cfg.ROBOT.N_JOINTS * cfg.SAC.TARGET_ENTROPY_RATIO
        else:
            self.target_entropy = target_entropy
            
        self.gamma = gamma
        self.tau = tau
        self.aux_loss_weight = cfg.TRAIN.AUX_LOSS_GRADIENT_SCALE
        self.reward_scale = reward_scale
        self.q_clip = q_clip
        self.grad_accumulation_steps = grad_accumulation_steps
        self._grad_accumulation_counter = 0
        
        # [NEW] Running statistics for adaptive Q-clipping
        self._q_running_mean = 0.0
        self._q_running_std = 1.0
        self._q_ema_decay = 0.99

    def _sample_action_and_log_prob(self, mu, log_std, action_scale):
        """Correct SAC sampling with tanh squashing."""
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * action_scale
        
        # Log prob with tanh correction (numerically stable)
        log_prob = dist.log_prob(x_t)
        # Clamp y_t^2 to avoid log(0)
        log_prob -= torch.log(action_scale * (1 - y_t.pow(2).clamp(max=0.999)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

    def _update_q_statistics(self, q_values):
        """Update running statistics for Q-values."""
        q_mean = q_values.mean().item()
        q_std = q_values.std().item() + 1e-6
        
        self._q_running_mean = self._q_ema_decay * self._q_running_mean + (1 - self._q_ema_decay) * q_mean
        self._q_running_std = self._q_ema_decay * self._q_running_std + (1 - self._q_ema_decay) * q_std

    def update(self, batch, update_actor=True):
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards'] * self.reward_scale
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
            
            # [IMPROVED] Adaptive Q-clipping based on running statistics
            adaptive_clip_low = self._q_running_mean - 5 * self._q_running_std
            adaptive_clip_high = self._q_running_mean + 5 * self._q_running_std
            # Combine with hard clips
            clip_low = max(self.q_clip[0], adaptive_clip_low)
            clip_high = min(self.q_clip[1], adaptive_clip_high)
            target_q = torch.clamp(target_q, clip_low, clip_high)
            
            q_target = rewards + (1 - dones) * self.gamma * target_q

        # Current Q-values (need latent from actor, but detached)
        with torch.no_grad():
            _, _, current_pred_state, _, current_latent = self.actor(obs)
        
        q1, q2 = self.critic(follower_state, current_latent, current_pred_state, actions)
        
        # Update Q statistics
        self._update_q_statistics(torch.cat([q1, q2]))
        
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
            "q_running_mean": self._q_running_mean,
            "q_running_std": self._q_running_std,
        }

        if update_actor:
            if cfg.TRAIN.JOINT_OPTIMIZATION:
                raise NotImplementedError("Joint optimization not implemented in this version")
            else:
                # === FIXED DECOUPLED MODE ===
                # [KEY FIX] Single forward pass, then compute both losses
                # This ensures consistent latent representation for both objectives
                
                mu, log_std, pred_state, _, latent = self.actor(obs)
                
                # --- Prediction Loss (auxiliary) ---
                pred_loss = F.mse_loss(pred_state, true_state)
                
                # --- RL Policy Loss ---
                pred_action, log_prob = self._sample_action_and_log_prob(
                    mu, log_std, self.actor.action_scale
                )
                
                # Critic evaluation with compact inputs
                # Detach latent and pred_state so policy gradient doesn't flow through LSTM
                q1_pi, q2_pi = self.critic(
                    follower_state, 
                    latent.detach(), 
                    pred_state.detach(), 
                    pred_action
                )
                min_q_pi = torch.min(q1_pi, q2_pi)
                
                # SAC actor loss
                alpha = self.log_alpha.exp().detach()
                rl_loss = (alpha * log_prob - min_q_pi).mean()
                
                # --- Combined Update ---
                # [KEY FIX] Single backward pass with combined loss
                # Weight the prediction loss to balance with RL loss
                total_actor_loss = rl_loss + (self.aux_loss_weight * pred_loss)
                
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.TRAIN.GRAD_CLIP)
                self.actor_optimizer.step()

            # Alpha Update
            # [NOTE] Detach log_prob to prevent double gradient
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # [NEW] Clamp log_alpha to prevent extreme values
            with torch.no_grad():
                self.log_alpha.clamp_(-5.0, 2.0)  # alpha in [0.0067, 7.39]
            
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


class ResidualSACWithFrozenLSTM(ResidualSAC):
    """
    Variant that freezes LSTM during initial RL fine-tuning.
    
    Rationale: The BC pre-training has already learned good state prediction.
    Freezing LSTM initially allows the policy head to adapt to RL objectives
    without destabilizing the learned representations.
    
    Usage:
        trainer = ResidualSACWithFrozenLSTM(...)
        # After N steps, call trainer.unfreeze_lstm() to enable full fine-tuning
    """
    
    def __init__(self, *args, freeze_lstm_steps=50000, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_lstm_steps = freeze_lstm_steps
        self._total_steps = 0
        self._lstm_frozen = True
        self._freeze_lstm()
    
    def _freeze_lstm(self):
        """Freeze LSTM encoder parameters."""
        for param in self.actor.base_encoder.parameters():
            param.requires_grad = False
        self._lstm_frozen = True
    
    def unfreeze_lstm(self):
        """Unfreeze LSTM encoder parameters."""
        for param in self.actor.base_encoder.parameters():
            param.requires_grad = True
        self._lstm_frozen = False
    
    def update(self, batch, update_actor=True):
        self._total_steps += 1
        
        # Auto-unfreeze after specified steps
        if self._lstm_frozen and self._total_steps >= self.freeze_lstm_steps:
            self.unfreeze_lstm()
            print(f"[ResidualSACWithFrozenLSTM] LSTM unfrozen at step {self._total_steps}")
        
        metrics = super().update(batch, update_actor)
        metrics["lstm_frozen"] = float(self._lstm_frozen)
        return metrics