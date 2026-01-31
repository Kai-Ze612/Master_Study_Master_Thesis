"""
Pre-train Joint Actor using Behavior Cloning (BC) with Augmentations.
[UPDATED VERSION]
- Feature: Added Early Stopping to prevent overfitting/wasted time.
- Feature: Validates EVERY epoch to catch the best model precisely.
- Config: "Sweet Spot" parameters (Noise 0.15, Dropout 0.3).

Usage:
    python3 train_bc.py          (Fast training, no window)
    python3 train_bc.py --render (Opens MuJoCo window during validation)
"""

import os
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Tuple, List
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import mujoco 

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.e2e_network import JointActor
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import LeaderRobotSimulator, TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from E2E_Teleoperation.E2E_RL.follower_robot_simulator import FollowerRobotSimulator 

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class BCTrainingConfig:
    # Paths
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path: Path = Path(cfg.PRETRAINED_CHECKPOINT_PATH)
    
    # Flags
    render_val: bool = False  # Controlled by CLI argument
    
    # Hyperparameters
    batch_size: int = 4096
    lr: float = 1e-4 
    epochs: int = 50           # Max epochs (Early stopping will likely cut this short)
    early_stop_patience: int = 10  # Stop if no improvement for 10 epochs
    grad_clip: float = 1.0
    
    # Loss Weights
    velocity_weight: float = 10.0
    action_weight: float = 100.0 
    
    # Model Params
    ar_horizon: int = cfg.ROBOT.MAX_PREDICTION_ROLLOUT_STEPS
    rnn_seq_len: int = cfg.ROBOT.RNN_SEQ_LEN
    
    # Expert PD Gains
    kp_values: List[float] = field(default_factory=lambda: [300.0, 300.0, 300.0, 200.0, 120.0, 150.0, 10.0])
    kd_values: List[float] = field(default_factory=lambda: [30.0, 30.0, 30.0, 20.0, 12.0, 15.0, 3.0])
    
    # === [CRITICAL TUNING] ===
    # 1. Noise Scale: 0.15 (Enough to learn recovery, low enough to avoid saturation)
    noise_scale: float = 0.15
    
    # 2. History Strategy: 0.3 Dropout (Standard robustness)
    history_dropout_prob: float = 0.3 
    history_noise_prob: float = 0.4   
    
    # 3. Data Volume: 200k (Needed for higher noise variance)
    num_samples: int = 200000
    delay_config: ExperimentConfig = ExperimentConfig.HIGH_VARIANCE 

class RobotSequenceDataset(Dataset):
    def __init__(self, obs_data: np.ndarray, target_data: np.ndarray, config: BCTrainingConfig):
        self.obs_data = torch.FloatTensor(obs_data).to(config.device)
        self.target_data = torch.FloatTensor(target_data).to(config.device)
        self.cfg = config
        self.valid_indices = len(self.obs_data) - self.cfg.rnn_seq_len - self.cfg.ar_horizon

    def __len__(self):
        return max(0, self.valid_indices)

    def __getitem__(self, idx):
        hist_start = idx
        hist_end = idx + self.cfg.rnn_seq_len
        history = self.obs_data[hist_start : hist_end]
        
        curr_idx = hist_end
        target_state = self.target_data[curr_idx]
        
        future_start = curr_idx
        future_end = curr_idx + self.cfg.ar_horizon
        future_targets = self.target_data[future_start : future_end]
        
        return history, target_state, future_targets

class OfflineDataCollector:
    def __init__(self, config: BCTrainingConfig):
        self.cfg = config
        self.leader = LeaderRobotSimulator(trajectory_type=TrajectoryType.FIGURE_8, control_freq=cfg.CONTROL_FREQ)
        self.delay_sim = DelaySimulator(control_freq=cfg.CONTROL_FREQ, config=self.cfg.delay_config)
        
    def collect(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Collecting {self.cfg.num_samples} samples...")
        self.leader.reset()
        obs_list, target_list = [], []
        leader_hist = deque(maxlen=200)
        
        q_mean = cfg.ROBOT.Q_MEAN
        q_std = cfg.ROBOT.Q_STD
        qd_mean = cfg.ROBOT.QD_MEAN
        qd_std = cfg.ROBOT.QD_STD
        
        for _ in tqdm(range(self.cfg.num_samples), desc="Simulating"):
            t_q, t_qd, _, _, _, _, _ = self.leader.step()
            leader_hist.append((t_q, t_qd))
            d_q, d_qd, d_sec = self.delay_sim.get_delayed_state(leader_hist)
            
            x_vec = np.concatenate([(d_q - q_mean) / q_std, (d_qd - qd_mean) / qd_std, [d_sec]])
            obs_list.append(x_vec)
            target_list.append(np.concatenate([t_q, t_qd]))
            
        return np.array(obs_list), np.array(target_list)

class BCTrainer:
    def __init__(self, config: BCTrainingConfig):
        self.cfg = config
        self.actor = JointActor().to(self.cfg.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        
        self.q_mean = torch.tensor(cfg.ROBOT.Q_MEAN, device=self.cfg.device).float()
        self.q_std = torch.tensor(cfg.ROBOT.Q_STD, device=self.cfg.device).float()
        self.qd_mean = torch.tensor(cfg.ROBOT.QD_MEAN, device=self.cfg.device).float()
        self.qd_std = torch.tensor(cfg.ROBOT.QD_STD, device=self.cfg.device).float()
        self.kp_vec = torch.tensor(self.cfg.kp_values, device=self.cfg.device).float()
        self.kd_vec = torch.tensor(self.cfg.kd_values, device=self.cfg.device).float()
        
        # Validation Simulator (Ephemeral)
        self.val_leader = LeaderRobotSimulator(trajectory_type=TrajectoryType.FIGURE_8)
        self.val_follower = FollowerRobotSimulator(
            delay_config=self.cfg.delay_config, 
            render=self.cfg.render_val # Controlled by --render flag
        ) 
        self.val_delay = DelaySimulator(cfg.CONTROL_FREQ, config=self.cfg.delay_config)

    def _compute_ar_loss(self, history, future_targets, h, c):
        loss_ar = 0.0
        curr_pred_norm = history[:, -1, :14]
        curr_delay = history[:, -1, 14:15]
        
        for t in range(self.cfg.ar_horizon):
            lstm_in = torch.cat([curr_pred_norm, curr_delay], dim=1)
            h, c = self.actor.base_encoder.lstm_cell(lstm_in, (h, c))
            delta_vel = self.actor.base_encoder.predictor(h)
            
            curr_pos_norm = curr_pred_norm[:, :7]
            curr_vel_norm = curr_pred_norm[:, 7:]
            next_vel_norm = curr_vel_norm + delta_vel
            next_pos_norm = curr_pos_norm + (curr_vel_norm * self.actor.base_encoder.dt_scale)
            
            pred_pos = (next_pos_norm * self.q_std) + self.q_mean
            pred_vel = (next_vel_norm * self.qd_std) + self.qd_mean
            real_target = future_targets[:, t, :]
            
            loss_ar += F.mse_loss(pred_pos, real_target[:, :7]) + \
                       self.cfg.velocity_weight * F.mse_loss(pred_vel, real_target[:, 7:])
            
            curr_pred_norm = torch.cat([next_pos_norm, next_vel_norm], dim=1)
            curr_delay = torch.clamp(curr_delay - cfg.DT, min=0.0)
            
        return loss_ar / self.cfg.ar_horizon, h

    def _compute_bc_loss(self, target_state, latent, is_training: bool = True):
        t_q = target_state[:, :7]
        t_qd = target_state[:, 7:]
        
        # Noise Injection
        pos_noise = torch.randn_like(t_q) * self.cfg.noise_scale
        vel_noise = torch.randn_like(t_qd) * self.cfg.noise_scale * 2.0
        noisy_q = t_q + pos_noise
        noisy_qd = t_qd + vel_noise
        
        # Expert Calculation (Pure PD)
        pos_err = t_q - noisy_q
        vel_err = t_qd - noisy_qd
        expert_torque = (pos_err * self.kp_vec) + (vel_err * self.kd_vec)
        
        target_action_norm = expert_torque / self.actor.action_scale
        target_action_norm = torch.clamp(target_action_norm, -0.99, 0.99)
        
        noisy_q_n = (noisy_q - self.q_mean) / self.q_std
        noisy_qd_n = (noisy_qd - self.qd_mean) / self.qd_std
        
        batch_size = target_state.shape[0]
        
        if is_training:
            rand_val = np.random.random()
            if rand_val < self.cfg.history_dropout_prob:
                # 1. Total Dropout
                fake_history = torch.zeros(batch_size, 7 * cfg.ROBOT.ACTION_HISTORY_LEN, device=self.cfg.device)
                fake_prev = torch.zeros(batch_size, 7, device=self.cfg.device)
            elif rand_val < (self.cfg.history_dropout_prob + self.cfg.history_noise_prob):
                # 2. Noisy History
                hist = target_action_norm.repeat(1, cfg.ROBOT.ACTION_HISTORY_LEN)
                fake_history = hist + (torch.randn_like(hist) * 0.3) 
                fake_prev = target_action_norm + (torch.randn_like(target_action_norm) * 0.3)
            else:
                # 3. Perfect History
                fake_history = target_action_norm.repeat(1, cfg.ROBOT.ACTION_HISTORY_LEN)
                fake_prev = target_action_norm
        else:
            fake_history = target_action_norm.repeat(1, cfg.ROBOT.ACTION_HISTORY_LEN)
            fake_prev = target_action_norm
            
        policy_in = torch.cat([noisy_q_n, noisy_qd_n, latent.detach(), fake_history, fake_prev], dim=1)
        pred_action = torch.tanh(self.actor.res_mu(self.actor.residual_net(policy_in)))
        
        return F.mse_loss(pred_action, target_action_norm)

    def train_epoch(self, loader: DataLoader) -> float:
        self.actor.train()
        total_loss = 0.0
        for history, target_state, future_targets in tqdm(loader, desc="Train", leave=False):
            self.optimizer.zero_grad()
            h = torch.zeros(history.size(0), cfg.ROBOT.RNN_HIDDEN_DIM, device=self.cfg.device)
            c = torch.zeros(history.size(0), cfg.ROBOT.RNN_HIDDEN_DIM, device=self.cfg.device)
            for t in range(self.cfg.rnn_seq_len):
                h, c = self.actor.base_encoder.lstm_cell(history[:, t, :], (h, c))
            
            loss_ar, latent = self._compute_ar_loss(history, future_targets, h, c)
            loss_bc = self._compute_bc_loss(target_state, latent, is_training=True)
            
            batch_loss = loss_ar + (self.cfg.action_weight * loss_bc)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            total_loss += batch_loss.item()
        return total_loss / len(loader)

    # Real Environment Validation
    def run_env_validation(self):
        """Runs a simulation episode to verify tracking."""
        self.actor.eval()
        l_q, _ = self.val_leader.reset()
        f_q, _ = self.val_follower.reset()
        self.val_delay.reset()
        
        l_hist = deque(maxlen=200)
        for _ in range(200): l_hist.append((l_q, np.zeros(7)))
        
        # Init buffers
        f_hist_q = deque([f_q]*cfg.ROBOT.RNN_SEQ_LEN, maxlen=cfg.ROBOT.RNN_SEQ_LEN)
        f_hist_qd = deque([np.zeros(7)]*cfg.ROBOT.RNN_SEQ_LEN, maxlen=cfg.ROBOT.RNN_SEQ_LEN)
        act_hist = deque([np.zeros(7)]*cfg.ROBOT.ACTION_HISTORY_LEN, maxlen=cfg.ROBOT.ACTION_HISTORY_LEN)
        
        total_err = 0.0
        steps = 200 # Short horizon check
        
        for _ in range(steps):
            # Slow down simulation to match 200Hz if rendering is enabled
            if self.cfg.render_val:
                time.sleep(1.0 / cfg.CONTROL_FREQ)

            l_q_new, l_qd_new, _, _, _, _, _ = self.val_leader.step()
            l_hist.append((l_q_new, l_qd_new))
            
            # Update ghost for visual debugging
            if hasattr(self.val_follower, 'update_leader_view'):
                self.val_follower.update_leader_view(l_q_new)

            # Prepare Obs
            seq_data = []
            for i in range(cfg.ROBOT.RNN_SEQ_LEN):
                l_del, ld_del, d_sec = self.val_delay.get_delayed_state(l_hist, offset_indices=cfg.ROBOT.RNN_SEQ_LEN-1-i)
                l_norm = (l_del - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD
                ld_norm = (ld_del - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD
                f_norm = (f_hist_q[i] - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD
                fd_norm = (f_hist_qd[i] - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD
                a_norm = np.zeros(7) # Simplified history in sequence
                seq_data.append(np.concatenate([l_norm, ld_norm, [d_sec], f_norm, fd_norm, a_norm]))
            
            target_seq = np.array(seq_data, dtype=np.float32)
            
            # Current State
            f_curr_norm = (f_hist_q[-1] - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD
            fd_curr_norm = (f_hist_qd[-1] - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD
            curr_state = np.concatenate([f_curr_norm, fd_curr_norm])
            
            # Flatten Action History
            act_hist_arr = np.array(act_hist, dtype=np.float32) 
            act_hist_norm = act_hist_arr / cfg.ROBOT.MAX_ACTION_TORQUE
            act_flat = act_hist_norm.flatten()

            prev_act = act_hist[-1] / cfg.ROBOT.MAX_ACTION_TORQUE
            
            full_obs = np.concatenate([curr_state, target_seq.flatten(), act_flat, prev_act])
            
            # Inference
            with torch.no_grad():
                t_obs = torch.as_tensor(full_obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
                mu, _, _, _, _ = self.actor(t_obs)
                action = (torch.tanh(mu) * self.actor.action_scale).cpu().numpy()[0]
            
            # Step
            info = self.val_follower.step(action)
            f_q_new = info['q_follower']
            f_qd_new = info['qd_follower']
            
            # Update buffers
            f_hist_q.append(f_q_new)
            f_hist_qd.append(f_qd_new)
            act_hist.append(action)
            
            total_err += np.linalg.norm(l_q_new - f_q_new)
            
        return total_err / steps

    def run(self):
        collector = OfflineDataCollector(self.cfg)
        obs_data, target_data = collector.collect()
        dataset = RobotSequenceDataset(obs_data, target_data, self.cfg)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        
        logger.info(f"Training on {len(dataset)} samples. dropout={self.cfg.history_dropout_prob}")
        
        best_sim_err = float('inf')
        no_improvement_count = 0  # <--- [EARLY STOP LOGIC]
        
        for epoch in range(self.cfg.epochs):
            train_loss = self.train_epoch(loader)
            
            # Validate EVERY epoch to catch degradation immediately
            sim_err = self.run_env_validation()
            logger.info(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | SIM TRACK ERR: {sim_err:.4f}")
            
            if sim_err < best_sim_err:
                best_sim_err = sim_err
                no_improvement_count = 0  # Reset
                self.save_checkpoint(epoch, train_loss, is_best=True)
            else:
                no_improvement_count += 1
                if no_improvement_count >= self.cfg.early_stop_patience:
                    logger.info(f"EARLY STOPPING: No improvement for {self.cfg.early_stop_patience} epochs.")
                    break
            
            # Save latest checkpoint anyway for backup
            self.save_checkpoint(epoch, train_loss, is_best=False)

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        state = {
            'actor': self.actor.state_dict(),
            'norm': {'q_mean': cfg.ROBOT.Q_MEAN, 'q_std': cfg.ROBOT.Q_STD, 'qd_mean': cfg.ROBOT.QD_MEAN, 'qd_std': cfg.ROBOT.QD_STD},
            'epoch': epoch, 'loss': loss
        }
        if is_best:
            torch.save(state, self.cfg.output_path)
            logger.info(f"    --> Saved Best Checkpoint.")

if __name__ == "__main__":
    # CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render simulation during validation steps")
    args = parser.parse_args()

    # Pass args to config
    config = BCTrainingConfig(render_val=args.render)
    
    trainer = BCTrainer(config)
    trainer.run()