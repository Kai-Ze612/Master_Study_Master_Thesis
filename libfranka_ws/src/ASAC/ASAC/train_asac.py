"""
PD Gain Tuning A-SAC Training Script
-------------------------------------
Train RL to output optimal state-dependent PD gains (Kp, Kd).

Architecture:
    τ = Kp(s) * (q_target - q_current) + Kd(s) * (qd_target - qd_current) + gravity

Usage:
    # Train with visualization (single env required)
    python3 -m ASAC.train_asac --num-envs 1 --render --config 3
    
    # Train without visualization (faster, multi-env)
    python3 -m ASAC.train_asac --num-envs 8 --config 3
"""

import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv

import ASAC.config.robot_config as cfg
from ASAC.training_env import make_gain_tuning_env, make_gain_tuning_eval_env
from ASAC.asac_network import GainTuningActor, GainTuningCritic
from ASAC.asac_algorithm import AugmentedSAC
from ASAC.utils.replay_buffer import ReplayBuffer


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    num_eval_episodes: int = 10
    eval_interval: int = cfg.TRAIN.EVAL_INTERVAL
    early_stop_patience: int = 20
    min_improvement: float = 0.005


class EpisodeTracker:
    """Track rolling episode statistics."""
    
    def __init__(self, num_envs: int, window_size: int = 100):
        self.num_envs = num_envs
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        
        from collections import deque
        self.reward_history = deque(maxlen=window_size)
        self.length_history = deque(maxlen=window_size)
        self.kp_history = deque(maxlen=window_size)
        self.kd_history = deque(maxlen=window_size)
        self.total_episodes = 0
        
        self._episode_kp_means = [[] for _ in range(num_envs)]
        self._episode_kd_means = [[] for _ in range(num_envs)]
    
    def step(self, rewards, dones, infos=None):
        rewards = np.atleast_1d(rewards)
        dones = np.atleast_1d(dones)
        
        for i in range(self.num_envs):
            self.episode_rewards[i] += rewards[i]
            self.episode_lengths[i] += 1
            
            # Track gains
            if infos and isinstance(infos, list) and len(infos) > i:
                if 'kp_mean' in infos[i]:
                    self._episode_kp_means[i].append(infos[i]['kp_mean'])
                if 'kd_mean' in infos[i]:
                    self._episode_kd_means[i].append(infos[i]['kd_mean'])
            
            if dones[i]:
                self.reward_history.append(self.episode_rewards[i])
                self.length_history.append(self.episode_lengths[i])
                
                if self._episode_kp_means[i]:
                    self.kp_history.append(np.mean(self._episode_kp_means[i]))
                if self._episode_kd_means[i]:
                    self.kd_history.append(np.mean(self._episode_kd_means[i]))
                
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0
                self._episode_kp_means[i] = []
                self._episode_kd_means[i] = []
                self.total_episodes += 1
    
    def get_stats(self):
        if not self.reward_history:
            return {}
        return {
            'mean_reward': float(np.mean(self.reward_history)),
            'std_reward': float(np.std(self.reward_history)),
            'mean_length': float(np.mean(self.length_history)),
            'mean_kp': float(np.mean(self.kp_history)) if self.kp_history else 0,
            'mean_kd': float(np.mean(self.kd_history)) if self.kd_history else 0,
            'total_episodes': self.total_episodes
        }


class GainTuningSACTrainer:
    """Trainer for PD Gain Tuning SAC."""
    
    def __init__(self, env, eval_env, output_dir, eval_config):
        self.env = env
        self.eval_env = eval_env
        self.output_dir = Path(output_dir)
        self.eval_config = eval_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_logging()
        self._setup_networks()
        self._setup_buffer()
        
        self.global_step = 0
        self.num_envs = getattr(env, "num_envs", 1)
        self.warmup_steps = cfg.TRAIN.WARMUP_STEPS // self.num_envs
        
        self.best_eval_reward = -np.inf
        self.evals_without_improvement = 0
        self.eval_history = []
        self.episode_tracker = EpisodeTracker(self.num_envs)
    
    def _setup_logging(self):
        log_file = self.output_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("GainTuningSAC")
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
        
        # Save config
        config_dict = {
            'pd_gains': {
                'kp_min': cfg.PD_GAINS.KP_MIN.tolist(),
                'kp_max': cfg.PD_GAINS.KP_MAX.tolist(),
                'kd_min': cfg.PD_GAINS.KD_MIN.tolist(),
                'kd_max': cfg.PD_GAINS.KD_MAX.tolist(),
            },
            'train': {
                'batch_size': cfg.TRAIN.BATCH_SIZE,
                'buffer_size': cfg.TRAIN.BUFFER_SIZE,
                'actor_lr': cfg.TRAIN.ACTOR_LR,
                'critic_lr': cfg.TRAIN.CRITIC_LR,
            },
            'eval': asdict(self.eval_config)
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _setup_networks(self):
        obs_dim = cfg.ROBOT.RL_OBS_DIM
        action_dim = cfg.ROBOT.N_JOINTS * 2  # 14 (Kp + Kd)
        
        self.actor = GainTuningActor(input_dim=obs_dim).to(self.device)
        self.critic = GainTuningCritic(obs_dim=obs_dim).to(self.device)
        self.critic_target = GainTuningCritic(obs_dim=obs_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.TRAIN.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        
        self.log_alpha = torch.tensor(
            [np.log(cfg.SAC.INITIAL_ALPHA)],
            requires_grad=True,
            device=self.device,
            dtype=torch.float32
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)
        
        self.sac = AugmentedSAC(
            self.actor, self.critic, self.critic_target,
            self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer,
            self.log_alpha
        )
        
        self.logger.info(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters()):,}")
        self.logger.info(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters()):,}")
        self.logger.info(f"Action dim: {action_dim} (Kp[7] + Kd[7])")
    
    def _setup_buffer(self):
        obs_dim = cfg.ROBOT.RL_OBS_DIM
        action_dim = cfg.ROBOT.N_JOINTS * 2  # 14
        self.logger.info(f"Buffer: obs_dim={obs_dim}, action_dim={action_dim}")
        self.buffer = ReplayBuffer(cfg.TRAIN.BUFFER_SIZE, obs_dim, action_dim, self.device)
    
    def evaluate(self):
        """Run evaluation episodes."""
        self.actor.eval()
        eval_rewards = []
        eval_pos_errors = []
        eval_kp_means = []
        eval_kd_means = []
        
        for _ in range(self.eval_config.num_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            pos_errors = []
            kp_means = []
            kd_means = []
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    mu, _ = self.actor(obs_t)
                    action = torch.tanh(mu) * self.actor.action_scale
                    action = action.cpu().numpy().squeeze()
                
                obs, reward, term, trunc, info = self.eval_env.step(action)
                done = term or trunc
                episode_reward += reward
                
                if 'position_error' in info:
                    pos_errors.append(info['position_error'])
                if 'kp_mean' in info:
                    kp_means.append(info['kp_mean'])
                if 'kd_mean' in info:
                    kd_means.append(info['kd_mean'])
            
            eval_rewards.append(episode_reward)
            if pos_errors:
                eval_pos_errors.append(np.mean(pos_errors))
            if kp_means:
                eval_kp_means.append(np.mean(kp_means))
            if kd_means:
                eval_kd_means.append(np.mean(kd_means))
        
        self.actor.train()
        
        return {
            'eval_mean_reward': float(np.mean(eval_rewards)),
            'eval_std_reward': float(np.std(eval_rewards)),
            'eval_mean_pos_error': float(np.mean(eval_pos_errors)) if eval_pos_errors else 0,
            'eval_mean_kp': float(np.mean(eval_kp_means)) if eval_kp_means else 0,
            'eval_mean_kd': float(np.mean(eval_kd_means)) if eval_kd_means else 0,
        }
    
    def save_checkpoint(self, filename, is_best=False):
        checkpoint = {
            'global_step': self.global_step,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'best_eval_reward': self.best_eval_reward,
        }
        torch.save(checkpoint, self.output_dir / filename)
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            torch.save(self.actor.state_dict(), self.output_dir / 'best_actor.pth')
    
    def train(self):
        # Reset environment
        if self.num_envs > 1:
            obs = self.env.reset()
        else:
            obs, _ = self.env.reset()
        
        # Warmup
        self.logger.info(f"Warmup: {self.warmup_steps * self.num_envs} steps...")
        for _ in tqdm(range(self.warmup_steps), desc="Warmup", leave=False):
            if self.num_envs > 1:
                actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
                next_obs, rewards, dones, infos = self.env.step(actions)
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones)
            else:
                action = self.env.action_space.sample()
                next_obs, reward, term, trunc, info = self.env.step(action)
                done = term or trunc
                self.buffer.add(obs, action, reward, next_obs, done)
                rewards, dones, infos = np.array([reward]), np.array([done]), [info]
                if done:
                    next_obs, _ = self.env.reset()
            
            self.episode_tracker.step(rewards, dones, infos)
            obs = next_obs
            self.global_step += self.num_envs
        
        # Main training loop
        self.logger.info("Starting PD Gain Tuning SAC training...")
        pbar = tqdm(total=cfg.TRAIN.TOTAL_TIMESTEPS, initial=self.global_step, desc="Training")
        train_metrics = defaultdict(list)
        
        while self.global_step < cfg.TRAIN.TOTAL_TIMESTEPS:
            # Select action
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                if self.num_envs == 1 and obs_t.ndim == 1:
                    obs_t = obs_t.unsqueeze(0)
                
                # [FIXED] Unpack only 2 values (action, log_prob)
                actions_t, _ = self.actor.sample(obs_t)
                actions = actions_t.cpu().numpy()
            
            # Step environment
            if self.num_envs > 1:
                next_obs, rewards, dones, infos = self.env.step(actions)
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones)
            else:
                next_obs, reward, term, trunc, info = self.env.step(actions[0])
                done = term or trunc
                self.buffer.add(obs, actions[0], reward, next_obs, done)
                rewards, dones, infos = np.array([reward]), np.array([done]), [info]
                if done:
                    next_obs, _ = self.env.reset()
            
            self.episode_tracker.step(rewards, dones, infos)
            obs = next_obs
            self.global_step += self.num_envs
            pbar.update(self.num_envs)
            
            # Update networks
            if self.global_step % cfg.TRAIN.TRAIN_FREQUENCY == 0:
                num_updates = max(1, int(cfg.TRAIN.TRAIN_FREQUENCY * cfg.TRAIN.UTD_RATIO))
                for _ in range(num_updates):
                    batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
                    metrics = self.sac.update(batch)
                    for k, v in metrics.items():
                        train_metrics[k].append(v)
            
            # Log metrics
            if self.global_step % cfg.TRAIN.LOG_FREQ == 0:
                for k, v_list in train_metrics.items():
                    if v_list:
                        self.writer.add_scalar(f'Train/{k}', np.mean(v_list), self.global_step)
                train_metrics = defaultdict(list)
                
                ep_stats = self.episode_tracker.get_stats()
                if ep_stats:
                    self.writer.add_scalar('Episode/mean_reward', ep_stats['mean_reward'], self.global_step)
                    self.writer.add_scalar('Episode/mean_kp', ep_stats['mean_kp'], self.global_step)
                    self.writer.add_scalar('Episode/mean_kd', ep_stats['mean_kd'], self.global_step)
                    pbar.set_postfix({
                        'reward': f"{ep_stats['mean_reward']:.1f}",
                        'kp': f"{ep_stats['mean_kp']:.1f}"
                    })
            
            # Evaluation
            if self.global_step % self.eval_config.eval_interval == 0:
                eval_stats = self.evaluate()
                for k, v in eval_stats.items():
                    self.writer.add_scalar(f'Eval/{k}', v, self.global_step)
                
                self.eval_history.append({'step': self.global_step, **eval_stats})
                current_reward = eval_stats['eval_mean_reward']
                
                self.logger.info(
                    f"Step {self.global_step:,}: Eval={current_reward:.2f} "
                    f"(best={self.best_eval_reward:.2f}), "
                    f"PosErr={eval_stats['eval_mean_pos_error']:.4f}, "
                    f"Kp={eval_stats['eval_mean_kp']:.1f}, Kd={eval_stats['eval_mean_kd']:.1f}"
                )
                
                if current_reward > self.best_eval_reward:
                    self.best_eval_reward = current_reward
                    self.evals_without_improvement = 0
                    self.save_checkpoint('best_checkpoint.pth', is_best=True)
                    self.logger.info("New best!")
                else:
                    self.evals_without_improvement += 1
                
                if self.evals_without_improvement >= self.eval_config.early_stop_patience:
                    self.logger.info("Early stopping!")
                    break
        
        pbar.close()
        self.save_checkpoint('final_checkpoint.pth')
        
        # Save eval history
        with open(self.output_dir / 'eval_history.json', 'w') as f:
            json.dump(self.eval_history, f, indent=2)
        
        return self.best_eval_reward


def main():
    parser = argparse.ArgumentParser(description="PD Gain Tuning A-SAC Training")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Number of parallel environments (use 1 for --render)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", dest="delay_config", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Delay configuration: 0=NoDelay, 1=LowDelay, 2=HighDelay, 3=VarDelay")
    parser.add_argument("--eval-interval", type=int, default=cfg.TRAIN.EVAL_INTERVAL)
    parser.add_argument("--eval-episodes", type=int, default=10)
    
    # Render argument
    parser.add_argument("--render", action="store_true",
                        help="Enable MuJoCo visualization (requires --num-envs 1)")
    
    args = parser.parse_args()
    
    # Validate render + num_envs
    if args.render and args.num_envs > 1:
        print("[WARNING] --render requires --num-envs 1. Setting num_envs to 1.")
        args.num_envs = 1
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    delay_map = {0: "NoDelay", 1: "LowDelay", 2: "HighDelay", 3: "VarDelay"}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"GainTuningSAC_{delay_map[args.delay_config]}_{timestamp}"
    
    output_dir = cfg.ROBOT.CHECKPOINT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"PD Gain Tuning SAC Training: {run_name}")
    print("=" * 70)
    print(f"  Delay Config   : {delay_map[args.delay_config]}")
    print(f"  Num Envs       : {args.num_envs}")
    print(f"  Render         : {'ENABLED' if args.render else 'DISABLED'}")
    print(f"  Device         : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Kp Range       : [{cfg.PD_GAINS.KP_MIN[0]:.0f}, {cfg.PD_GAINS.KP_MAX[0]:.0f}]")
    print(f"  Kd Range       : [{cfg.PD_GAINS.KD_MIN[0]:.0f}, {cfg.PD_GAINS.KD_MAX[0]:.0f}]")
    print("=" * 70 + "\n")
    
    # Create environments
    if args.num_envs > 1:
        env = SubprocVecEnv([make_gain_tuning_env(i, args) for i in range(args.num_envs)])
    else:
        env = make_gain_tuning_env(0, args)()
    
    eval_env = make_gain_tuning_eval_env(args)
    eval_config = EvalConfig(num_eval_episodes=args.eval_episodes, eval_interval=args.eval_interval)
    
    trainer = GainTuningSACTrainer(env, eval_env, output_dir, eval_config)
    
    try:
        best_reward = trainer.train()
        print(f"\nTraining complete! Best reward: {best_reward:.2f}")
    except KeyboardInterrupt:
        print("\nInterrupted by user!")
        trainer.save_checkpoint('interrupted.pth')
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()