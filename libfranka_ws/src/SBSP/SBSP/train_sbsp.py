"""
SBSP Training: Pre-train DCNN then Train SAC (PMDC Method)
"""
import argparse
import logging
import json
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv

import SBSP.config.robot_config as cfg
from SBSP.training_env import make_sbsp_env, make_sbsp_eval_env
from SBSP.sbsp_network import SBSPActor, SBSPCritic, DCNN
from SBSP.sbsp_algorithm import SBSPAlgorithm
from SBSP.utils.replay_buffer import ReplayBuffer
from SBSP.pmdc_wrapper import PMDCWrapper

@dataclass
class EvalConfig:
    num_eval_episodes: int = 10
    eval_interval: int = cfg.TRAIN.EVAL_INTERVAL
    early_stop_patience: int = cfg.TRAIN.EARLY_STOP_PATIENCE

class SBSPTrainer:
    def __init__(self, env, eval_env, output_dir, eval_config, dcnn_model):
        self.env = env
        self.eval_env = eval_env
        self.output_dir = Path(output_dir)
        self.eval_config = eval_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dcnn = dcnn_model # Pre-trained DCNN
        
        self._setup_logging()
        self._save_config()
        
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
        
        self.eval_history = {"step": [], "reward": [], "position_error": []}

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SBSP")
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
    
    def _save_config(self):
        config_data = {
            "ROBOT": asdict(cfg.ROBOT),
            "SBSP": asdict(cfg.SBSP),
            "TRAIN": asdict(cfg.TRAIN),
            "REWARD": asdict(cfg.REWARD),
            "SAC": asdict(cfg.SAC)
        }
        def serialize(obj):
            if isinstance(obj, Path): return str(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        clean_config = {}
        for section, params in config_data.items():
            clean_config[section] = {}
            for k, v in params.items():
                clean_config[section][k] = serialize(v)
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(clean_config, f, indent=4)

    def evaluate(self):
        self.actor.eval()
        rewards, pos_errors = [], []
        
        for _ in range(self.eval_config.num_eval_episodes):
            obs, _ = self.eval_env.reset() # PMDCWrapper handles prediction here automatically
            ep_rew = 0
            ep_err = []
            done = False
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    mu, _ = self.actor(obs_t) 
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
            'best_eval_reward': self.best_eval_reward,
            'step': self.global_step
        }
        torch.save(ckpt, self.output_dir / filename)
        if is_best: 
            torch.save(ckpt, self.output_dir / 'best_checkpoint.pth')
            torch.save(self.actor.state_dict(), self.output_dir / 'best_actor.pth')
            torch.save(ckpt, self.output_dir / 'best_model.pth')

    def train(self):
        num_envs = getattr(self.env, "num_envs", 1) # PMDCWrapper might hide num_envs if not vectorized properly
        # Note: If env is wrapped, access inner
        
        obs = self.env.reset() if num_envs > 1 else self.env.reset()[0]
        self.logger.info(f"Starting SBSP Training (PMDC Mode). Output: {self.output_dir}")
        
        # Warmup
        warmup = cfg.TRAIN.WARMUP_STEPS // num_envs
        for _ in tqdm(range(warmup), desc="Warmup"):
            if num_envs > 1:
                actions = np.array([self.env.action_space.sample() for _ in range(num_envs)])
                next_obs, rewards, dones, infos = self.env.step(actions)
                true_states = np.array([i['true_leader_q'] for i in infos]) # Not used for aux loss anymore
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, true_states)
            else:
                action = self.env.action_space.sample()
                next_obs, reward, term, trunc, info = self.env.step(action)
                done = term or trunc
                # true_leader_q still stored but unused by algo
                self.buffer.add(obs, action, reward, next_obs, done, info['true_leader_q']) 
                if done: next_obs, _ = self.env.reset()
            obs = next_obs
            self.global_step += num_envs

        # Main Loop
        pbar = tqdm(total=cfg.TRAIN.TOTAL_TIMESTEPS, initial=self.global_step)
        while self.global_step < cfg.TRAIN.TOTAL_TIMESTEPS:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                if obs_t.ndim == 1: obs_t = obs_t.unsqueeze(0)
                actions_t, _ = self.actor.sample(obs_t)
                actions = actions_t.cpu().numpy()
            
            if num_envs > 1:
                next_obs, rewards, dones, infos = self.env.step(actions)
                true_states = np.array([i['true_leader_q'] for i in infos])
                self.buffer.add_batch(obs, actions, rewards, next_obs, dones, true_states)
                step_inc = num_envs
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

            if self.global_step % self.eval_config.eval_interval == 0:
                mean_r, mean_err = self.evaluate()
                self.logger.info(f"Step {self.global_step}: Reward={mean_r:.2f}, Error={mean_err:.4f}")
                self.writer.add_scalar("Eval/Reward", mean_r, self.global_step)
                
                self.eval_history["step"].append(self.global_step)
                self.eval_history["reward"].append(float(mean_r))
                self.eval_history["position_error"].append(float(mean_err))
                with open(self.output_dir / "eval_history.json", "w") as f:
                    json.dump(self.eval_history, f, indent=4)
                
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
        self.save_checkpoint('final_checkpoint.pth')

def pretrain_dynamics(env, dcnn, steps=20000, batch_size=256, device="cuda"):
    """
    Collects data using random policy and trains DCNN
    Task: Predict Next Leader State (s_t+1) given Current Leader State (s_t)
    """
    print("--- Starting DCNN Pre-training ---")
    optimizer = optim.Adam(dcnn.parameters(), lr=1e-3)
    dcnn.train()
    
    # Collect Data
    # We need simple transitions: State -> Next State
    # Leader is autonomous, so Action is dummy.
    
    buffer = []
    obs, _ = env.reset()
    # obs is wrapped, we need to access underlying env for Raw Leader State?
    # Actually, the Wrapper expects the DCNN to predict. 
    # To train DCNN, we need GROUND TRUTH Leader States.
    # The 'info' dict contains 'true_leader_q'.
    
    # We will use the unwrapped env to collect raw data for training
    raw_env = env.env if isinstance(env, PMDCWrapper) else env
    
    l_state_dim = 14 # 7 Pos + 7 Vel
    
    obs, _ = raw_env.reset()
    prev_leader = None # We need to reconstruct state from obs or info?
    # Info gives true_leader_q (7). We also need QD?
    # Let's extract from obs (it has delayed state). Wait, pre-training needs TRUE dynamics.
    # We can just step the environment and record info['true_leader_q'] evolution.
    
    # Simpler: Just train on the observed transitions of the Leader.
    # Leader state is in obs indices [0:14] (if no delay).
    # With delay, it's [0:14] but old.
    # We rely on 'info' to get True state for Ground Truth labels.
    
    # Collecting data...
    curr_leader_q = None
    
    for _ in tqdm(range(steps), desc="Collecting Dynamics Data"):
        action = raw_env.action_space.sample()
        obs, _, _, _, info = raw_env.step(action)
        
        # info['true_leader_q'] is pos. We need vel too? 
        # SBSPEnv doesn't export true_leader_qd in info.
        # We can approximate or modify env.
        # OR: Just train DCNN to predict POS.
        # But DCNN output must match what Wrapper expects (14 dims).
        
        # HACK: Use the Delayed Leader State from Obs as 's_t' and Next Delayed as 's_t+1'.
        # The dynamics of the leader (Figure 8) are the same regardless of time (it's autonomous).
        # So s_t -> s_t+1 relationship holds for delayed data too (just time shifted).
        
        # Frame: [Leader(15) | Follower(14) | Action(14)]
        # Leader: [Q(7), QD(7), Delay(1)]
        leader_vec = obs[:14] # Q + QD
        
        if curr_leader_q is not None:
            buffer.append((curr_leader_q, leader_vec))
        
        curr_leader_q = leader_vec
        
    # Training
    buffer = np.array(buffer, dtype=object)
    states = np.array([x[0] for x in buffer], dtype=np.float32)
    next_states = np.array([x[1] for x in buffer], dtype=np.float32)
    
    states_t = torch.tensor(states, device=device)
    next_states_t = torch.tensor(next_states, device=device)
    dummy_actions = torch.zeros((len(states), 1), device=device) # Dummy
    
    dataset = torch.utils.data.TensorDataset(states_t, next_states_t, dummy_actions)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(5): # 5 Epochs
        total_loss = 0
        for s, s_next, a in loader:
            pred = dcnn(s, a)
            loss = F.mse_loss(pred, s_next)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss {total_loss/len(loader):.5f}")
    
    print("--- Pre-training Complete ---")
    dcnn.eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1) # Wrappers hard to vectorise simply
    parser.add_argument("--config", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay-config", type=int, default=3)
    
    args = parser.parse_args()
    
    run_name = f"SBSP_PMDC_{datetime.now().strftime('%Y%m%d_%H%M')}"
    out_dir = cfg.ROBOT.CHECKPOINT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Init DCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcnn = DCNN(input_dim=14, action_dim=1).to(device) # Leader State 14, Action 1 (Dummy)
    
    # 2. Init Env (Unwrapped for Pre-training)
    # Using 1 env for simplicity with wrapper logic
    env = make_sbsp_env(0, args)() 
    
    # 3. Pre-train DCNN
    pretrain_dynamics(env, dcnn, steps=10000, device=device)
    
    # 4. Wrap Env
    env_wrapped = PMDCWrapper(env, dcnn, device)
    eval_env_wrapped = PMDCWrapper(make_sbsp_eval_env(args), dcnn, device)
    
    # 5. Train RL
    trainer = SBSPTrainer(env_wrapped, eval_env_wrapped, out_dir, EvalConfig(), dcnn)
    trainer.train()

if __name__ == "__main__":
    main()