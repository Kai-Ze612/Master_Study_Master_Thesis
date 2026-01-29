"""
SBSP Training Script
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

import SBSP.config.robot_config as cfg
from SBSP.training_env import make_sbsp_env, make_sbsp_eval_env
from SBSP.sbsp_network import SBSPActor, SBSPCritic
from SBSP.sbsp_algorithm import SBSPAlgorithm
from SBSP.utils.replay_buffer import ReplayBuffer

@dataclass
class EvalConfig:
    num_eval_episodes: int = 10
    eval_interval: int = cfg.TRAIN.EVAL_INTERVAL
    early_stop_patience: int = cfg.TRAIN.EARLY_STOP_PATIENCE

class SBSPTrainer:
    def __init__(self, env, eval_env, output_dir, eval_config):
        self.env = env
        self.eval_env = eval_env
        self.output_dir = Path(output_dir)
        self.eval_config = eval_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_logging()
        
        # Networks
        obs_dim = cfg.ROBOT.RL_OBS_DIM
        self.actor = SBSPActor(input_dim=obs_dim).to(self.device)
        self.critic = SBSPCritic(obs_dim=obs_dim).to(self.device)
        self.critic_target = SBSPCritic(obs_dim=obs_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.TRAIN.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        
        self.log_alpha = torch.tensor([np.log(cfg.SAC.INITIAL_ALPHA)], requires_grad=True, device=self.device, dtype=torch.float32)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)
        
        self.sac = SBSPAlgorithm(
            self.actor, self.critic, self.critic_target,
            self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer,
            self.log_alpha
        )
        
        self.buffer = ReplayBuffer(cfg.TRAIN.BUFFER_SIZE, obs_dim, cfg.ROBOT.ACTION_DIM, self.device)
        self.global_step = 0
        self.best_eval_reward = -np.inf
        self.evals_without_improvement = 0

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SBSP")
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
    
    def evaluate(self):
        self.actor.eval()
        rewards, pos_errors = [], []
        
        for _ in range(self.eval_config.num_eval_episodes):
            obs, _ = self.eval_env.reset()
            ep_rew = 0
            ep_err = []
            done = False
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    mu, _, _ = self.actor(obs_t) # unpack 3
                    action = torch.tanh(mu).cpu().numpy().squeeze()
                
                obs, reward, term, trunc, info = self.eval_env.step(action)
                done = term or trunc
                ep_rew += reward
                if 'position_error' in info: ep_err.append(info['position_error'])
            
            rewards.append(ep_rew)
            if ep_err: pos_errors.append(np.mean(ep_err))
        
        self.actor.train()
        return np.mean(rewards), np.mean(pos_errors) if pos_errors else 0.0

    def save_checkpoint(self, filename, is_best=False):
        ckpt = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'best_eval_reward': self.best_eval_reward
        }
        torch.save(ckpt, self.output_dir / filename)
        if is_best: 
            torch.save(ckpt, self.output_dir / 'best_model.pth')
            torch.save(self.actor.state_dict(), self.output_dir / 'best_actor.pth')

    def train(self):
        obs = self.env.reset() if self.env.num_envs > 1 else self.env.reset()[0]
        self.logger.info("Starting SBSP Training...")
        
        # Warmup
        warmup = cfg.TRAIN.WARMUP_STEPS // getattr(self.env, "num_envs", 1)
        for _ in tqdm(range(warmup), desc="Warmup"):
            if getattr(self.env, "num_envs", 1) > 1:
                actions = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
                next_obs, rewards, dones, infos = self.env.step(actions)
                # Extract true states
                true_states = np.array([i['true_leader_q'] for i in infos])
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, true_states)
            else:
                action = self.env.action_space.sample()
                next_obs, reward, term, trunc, info = self.env.step(action)
                done = term or trunc
                self.buffer.add(obs, action, reward, next_obs, done, info['true_leader_q'])
                if done: next_obs, _ = self.env.reset()
            obs = next_obs
            self.global_step += getattr(self.env, "num_envs", 1)

        # Main Loop
        pbar = tqdm(total=cfg.TRAIN.TOTAL_TIMESTEPS, initial=self.global_step)
        while self.global_step < cfg.TRAIN.TOTAL_TIMESTEPS:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                if obs_t.ndim == 1: obs_t = obs_t.unsqueeze(0)
                actions_t, _, _ = self.actor.sample(obs_t) # unpack 3
                actions = actions_t.cpu().numpy()
            
            if getattr(self.env, "num_envs", 1) > 1:
                next_obs, rewards, dones, infos = self.env.step(actions)
                true_states = np.array([i['true_leader_q'] for i in infos])
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, true_states)
                step_inc = self.env.num_envs
            else:
                next_obs, reward, term, trunc, info = self.env.step(actions[0])
                done = term or trunc
                self.buffer.add(obs, actions[0], reward, next_obs, done, info['true_leader_q'])
                if done: next_obs, _ = self.env.reset()
                step_inc = 1
            
            obs = next_obs
            self.global_step += step_inc
            pbar.update(step_inc)
            
            if self.global_step % cfg.TRAIN.TRAIN_FREQUENCY == 0:
                batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
                metrics = self.sac.update(batch)
                if self.global_step % cfg.TRAIN.LOG_FREQ == 0:
                    self.writer.add_scalar("Train/loss", metrics['critic_loss'], self.global_step)
                    self.writer.add_scalar("Train/pred_loss", metrics['pred_loss'], self.global_step)

            if self.global_step % self.eval_config.eval_interval == 0:
                mean_r, mean_err = self.evaluate()
                self.logger.info(f"Step {self.global_step}: Reward={mean_r:.2f}, Error={mean_err:.4f}")
                self.writer.add_scalar("Eval/Reward", mean_r, self.global_step)
                
                if mean_r > self.best_eval_reward:
                    self.best_eval_reward = mean_r
                    self.evals_without_improvement = 0
                    self.save_checkpoint('best_checkpoint.pth', is_best=True)
                else:
                    self.evals_without_improvement += 1
                
                if self.evals_without_improvement >= self.eval_config.early_stop_patience:
                    self.logger.info("Early stopping!")
                    break
                    
        pbar.close()
        self.save_checkpoint('final.pth')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--config", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    
    if args.render: args.num_envs = 1
    
    run_name = f"SBSP_{datetime.now().strftime('%Y%m%d_%H%M')}"
    out_dir = cfg.ROBOT.CHECKPOINT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.num_envs > 1:
        env = SubprocVecEnv([make_sbsp_env(i, args) for i in range(args.num_envs)])
    else:
        env = make_sbsp_env(0, args)()
        
    trainer = SBSPTrainer(env, make_sbsp_eval_env(args), out_dir, EvalConfig())
    trainer.train()

if __name__ == "__main__":
    main()