"""
Pre-train Joint Actor using Behavior Cloning (BC) with Augmentations.
[FIXED] Model saving logic - now correctly saves best model

Pipeline:
1. Data Collection: Simulates Leader trajectory and Delayed Observations.
2. Batch Sampling: Retrieval of sequential history and future targets.
3. Augmentation: 
   - State Noise Injection (Robustness)
   - History Dropout (Anti-Overfitting)
4. Optimization: Multi-objective loss (Action MSE + Future Prediction MSE).
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Tuple, List
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.e2e_network import JointActor
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import LeaderRobotSimulator, TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig

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
    
    # Hyperparameters
    batch_size: int = 4096
    lr: float = 1e-5
    epochs: int = 100
    grad_clip: float = 1.0
    
    # Early Stopping
    val_split: float = 0.2
    patience: int = 10
    min_delta: float = 1e-4
    
    # Loss Weights
    velocity_weight: float = 50.0
    action_weight: float = 5.0
    
    # Model Params
    ar_horizon: int = cfg.ROBOT.MAX_PREDICTION_ROLLOUT_STEPS
    rnn_seq_len: int = cfg.ROBOT.RNN_SEQ_LEN
    
    # Expert / Robustness
    kp_values: List[float] = field(default_factory=lambda: [300.0, 300.0, 300.0, 200.0, 120.0, 150.0, 10.0])
    kd_values: List[float] = field(default_factory=lambda: [30.0, 30.0, 30.0, 20.0, 12.0, 15.0, 3.0])
    noise_scale: float = 0.10
    history_dropout_prob: float = 0.5
    
    # Data Collection
    num_samples: int = 100000
    delay_config: ExperimentConfig = ExperimentConfig.HIGH_VARIANCE 


# [FIXED] Helper: Early Stopper - now returns whether this is a new best
class EarlyStopper:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss: float) -> Tuple[bool, bool]:
        """
        Returns:
            (should_stop, is_new_best)
        """
        is_new_best = False
        should_stop = False
        
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            is_new_best = True
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                should_stop = True
                
        return should_stop, is_new_best


# 2. Dataset
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


# 3. Data Collector
class OfflineDataCollector:
    def __init__(self, config: BCTrainingConfig):
        self.cfg = config
        
        self.leader = LeaderRobotSimulator(
            trajectory_type=TrajectoryType.FIGURE_8, 
            control_freq=cfg.CONTROL_FREQ
        )

        self.delay_sim = DelaySimulator(
            control_freq=cfg.CONTROL_FREQ, 
            config=self.cfg.delay_config
        )
        
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
            
            x_vec = np.concatenate([
                (d_q - q_mean) / q_std,
                (d_qd - qd_mean) / qd_std,
                [d_sec]
            ])
            
            obs_list.append(x_vec)
            target_list.append(np.concatenate([t_q, t_qd]))
            
        return np.array(obs_list), np.array(target_list)


# 4. Trainer
class BCTrainer:
    def __init__(self, config: BCTrainingConfig):
        self.cfg = config
        self.actor = JointActor().to(self.cfg.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        
        # Norm Constants
        self.q_mean = torch.tensor(cfg.ROBOT.Q_MEAN, device=self.cfg.device).float()
        self.q_std = torch.tensor(cfg.ROBOT.Q_STD, device=self.cfg.device).float()
        self.qd_mean = torch.tensor(cfg.ROBOT.QD_MEAN, device=self.cfg.device).float()
        self.qd_std = torch.tensor(cfg.ROBOT.QD_STD, device=self.cfg.device).float()
        
        # Expert Gains to Tensor
        self.kp_vec = torch.tensor(self.cfg.kp_values, device=self.cfg.device).float()
        self.kd_vec = torch.tensor(self.cfg.kd_values, device=self.cfg.device).float()
        
        self.early_stopper = EarlyStopper(patience=self.cfg.patience, min_delta=self.cfg.min_delta)
        
        self.best_val_loss = float('inf')  # [NEW] Track best loss separately

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
        
        # A. Noise Injection
        pos_noise = torch.randn_like(t_q) * self.cfg.noise_scale
        vel_noise = torch.randn_like(t_qd) * self.cfg.noise_scale * 2.0
        noisy_q = t_q + pos_noise
        noisy_qd = t_qd + vel_noise
        
        # B. Vectorized PD Expert (Pure PD, No Gravity)
        pos_err = t_q - noisy_q
        vel_err = t_qd - noisy_qd
        expert_torque = (pos_err * self.kp_vec) + (vel_err * self.kd_vec)
        
        target_action_norm = expert_torque / self.actor.action_scale
        target_action_norm = torch.clamp(target_action_norm, -0.99, 0.99)
        
        # C. Inputs
        noisy_q_n = (noisy_q - self.q_mean) / self.q_std
        noisy_qd_n = (noisy_qd - self.qd_mean) / self.qd_std
        
        # D. History Dropout
        batch_size = target_state.shape[0]
        if is_training and np.random.random() < self.cfg.history_dropout_prob:
            fake_history = torch.zeros(batch_size, 7 * cfg.ROBOT.ACTION_HISTORY_LEN, device=self.cfg.device)
            fake_prev = torch.zeros(batch_size, 7, device=self.cfg.device)
        else:
            fake_history = target_action_norm.repeat(1, cfg.ROBOT.ACTION_HISTORY_LEN)
            fake_prev = target_action_norm
            
        policy_in = torch.cat([noisy_q_n, noisy_qd_n, latent.detach(), fake_history, fake_prev], dim=1)
        pred_action = torch.tanh(self.actor.res_mu(self.actor.residual_net(policy_in)))
        
        return F.mse_loss(pred_action, target_action_norm)

    def train_epoch(self, loader: DataLoader, epoch_idx: int) -> float:
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

    def validate_epoch(self, loader: DataLoader) -> float:
        self.actor.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for history, target_state, future_targets in loader:
                h = torch.zeros(history.size(0), cfg.ROBOT.RNN_HIDDEN_DIM, device=self.cfg.device)
                c = torch.zeros(history.size(0), cfg.ROBOT.RNN_HIDDEN_DIM, device=self.cfg.device)
                for t in range(self.cfg.rnn_seq_len):
                    h, c = self.actor.base_encoder.lstm_cell(history[:, t, :], (h, c))
                
                loss_ar, latent = self._compute_ar_loss(history, future_targets, h, c)
                loss_bc = self._compute_bc_loss(target_state, latent, is_training=False)
                
                batch_loss = loss_ar + (self.cfg.action_weight * loss_bc)
                total_loss += batch_loss.item()
                
        return total_loss / len(loader)

    def run(self):
        collector = OfflineDataCollector(self.cfg)
        obs_data, target_data = collector.collect()
        full_dataset = RobotSequenceDataset(obs_data, target_data, self.cfg)
        
        val_size = int(len(full_dataset) * self.cfg.val_split)
        train_size = len(full_dataset) - val_size
        
        indices = list(range(len(full_dataset)))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_loader = DataLoader(Subset(full_dataset, train_indices), 
                                batch_size=self.cfg.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(full_dataset, val_indices), 
                              batch_size=self.cfg.batch_size, shuffle=False)
        
        logger.info(f"Training on {train_size} samples, Validating on {val_size} samples.")
        logger.info(f"PD Gains: KP={self.cfg.kp_values}, KD={self.cfg.kd_values}")
        
        for epoch in range(self.cfg.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate_epoch(val_loader)
            
            logger.info(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
            # [FIXED] Correct order: check for new best BEFORE updating early stopper
            # Or use the new early stopper that returns both values
            should_stop, is_new_best = self.early_stopper(val_loss)
            
            if is_new_best:
                self.save_checkpoint(epoch, val_loss)
                logger.info(f"--> Saved Best Model (Val Loss: {val_loss:.5f})")
            
            if should_stop:
                logger.info(f"!!! Early Stopping triggered at Epoch {epoch+1} !!!")
                break
        
        # [NEW] Always save final model as well
        self.save_checkpoint(epoch, val_loss, is_final=True)
        logger.info(f"==> Final Model Saved (Epoch {epoch+1}, Val Loss: {val_loss:.5f})")

    def save_checkpoint(self, epoch: int, loss: float, is_final: bool = False):
        state = {
            'actor': self.actor.state_dict(),
            'norm': {'q_mean': cfg.ROBOT.Q_MEAN, 'q_std': cfg.ROBOT.Q_STD, 'qd_mean': cfg.ROBOT.QD_MEAN, 'qd_std': cfg.ROBOT.QD_STD},
            'epoch': epoch,
            'loss': loss,
            'config': str(self.cfg)
        }
        
        # Save best model
        if not is_final:
            torch.save(state, self.cfg.output_path)
            logger.info(f"    Checkpoint saved to: {self.cfg.output_path}")
        else:
            # Save final model with different name
            final_path = self.cfg.output_path.parent / "final_checkpoint.pth"
            torch.save(state, final_path)
            logger.info(f"    Final checkpoint saved to: {final_path}")

if __name__ == "__main__":
    trainer = BCTrainer(BCTrainingConfig())
    trainer.run()