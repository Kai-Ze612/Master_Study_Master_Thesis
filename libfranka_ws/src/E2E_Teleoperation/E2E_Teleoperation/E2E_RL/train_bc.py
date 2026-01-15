"""
Behavior Cloning (BC) Pre-training Script
Integrated: Auto-Calibration -> Data Collection -> Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random

from E2E_Teleoperation.E2E_RL.e2e_network import JointActor
import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import LeaderRobotSimulator, TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perform_calibration():
    """
    1. Runs Figure-8 trajectories to find empirical Mean/Std.
    2. Saves to disk.
    3. Updates 'cfg.ROBOT' in memory so the rest of this script uses it.
    """
    print("\n[Calibration] Calculating Normalization for FIGURE-8...")
    
    # RESTRICTED TO FIGURE-8
    traj_types = [TrajectoryType.FIGURE_8] 
    
    all_q = []
    all_qd = []
    steps_per_traj = 10000 
    
    for t_type in traj_types:
        sim = LeaderRobotSimulator(trajectory_type=t_type, randomize_params=True)
        sim.reset()
        for _ in range(steps_per_traj):
            q, qd, _, _, _, _, _ = sim.step()
            all_q.append(q)
            all_qd.append(qd)
            
    data_q = np.array(all_q, dtype=np.float32)
    data_qd = np.array(all_qd, dtype=np.float32)
    
    mean_q = np.mean(data_q, axis=0)
    std_q = np.std(data_q, axis=0) + 1e-4 
    mean_qd = np.mean(data_qd, axis=0)
    std_qd = np.std(data_qd, axis=0) + 1e-4
    
    # 1. Save to Disk
    np.savez(cfg.ROBOT.NORMALIZATION_FILE_PATH, 
             q_mean=mean_q, q_std=std_q, 
             qd_mean=mean_qd, qd_std=std_qd)
    print(f"[Calibration] Stats saved to {cfg.ROBOT.NORMALIZATION_FILE_PATH}")
    
    # 2. Update Memory
    object.__setattr__(cfg.ROBOT, 'Q_MEAN', mean_q)
    object.__setattr__(cfg.ROBOT, 'Q_STD', std_q)
    object.__setattr__(cfg.ROBOT, 'QD_MEAN', mean_qd)
    object.__setattr__(cfg.ROBOT, 'QD_STD', std_qd)
    print("[Calibration] In-memory config updated.\n")


def get_undelayed_expert_action(f_q, f_qd, target_q, target_qd):
    Kp = cfg.BC_EXPERT.KP
    Kd = cfg.BC_EXPERT.KD
    pos_err = target_q - f_q
    vel_err = target_qd - f_qd
    tau = (Kp * pos_err) + (Kd * vel_err)
    return np.clip(tau, -cfg.ROBOT.MAX_ACTION_TORQUE, cfg.ROBOT.MAX_ACTION_TORQUE)

def collect_data(steps_per_config, noise_level=0.0):
    obs_data = []
    act_data = []
    
    # RESTRICTED TO FIGURE-8 + HIGH VARIANCE
    configs_to_run = [
        (ExperimentConfig.HIGH_VARIANCE, TrajectoryType.FIGURE_8),
    ]
    
    steps_per_sub_config = steps_per_config // len(configs_to_run)
    
    print(f">>> Collecting Data: Total Steps={steps_per_config} | Noise={noise_level}")

    for delay_conf, traj_type in configs_to_run:
        env = TeleoperationEnv(
            delay_config=delay_conf,
            trajectory_type=traj_type,
            randomize_trajectory=True, 
            render_mode=None
        )
        
        obs, info = env.reset()
        current_steps = 0
        
        pbar = tqdm(total=steps_per_sub_config, desc=f"Config {traj_type.name}", leave=False)
        
        while current_steps < steps_per_sub_config:
            true_leader_q = info['leader_q']
            true_leader_qd = np.zeros(7)
            
            expert_action = get_undelayed_expert_action(
                info['follower_q'], 
                np.zeros(7), 
                true_leader_q, 
                true_leader_qd
            )
            
            obs_data.append(obs)
            act_data.append(expert_action)
            
            noise = np.random.normal(0, noise_level, size=expert_action.shape)
            noisy_action = expert_action + noise
            noisy_action = np.clip(noisy_action, -cfg.ROBOT.MAX_ACTION_TORQUE, cfg.ROBOT.MAX_ACTION_TORQUE)
            
            next_obs, _, terminated, truncated, info = env.step(noisy_action)
            
            obs = next_obs
            current_steps += 1
            pbar.update(1)
            
            if terminated or truncated:
                obs, info = env.reset()
                
        pbar.close()
        env.close()
            
    # DIAGNOSTICS
    act_array = np.array(act_data)
    mean_act = np.mean(np.abs(act_array))
    saturation = np.mean(np.abs(act_array) > (cfg.ROBOT.MAX_ACTION_TORQUE * 0.95))
    print(f"   [Data Stats] Action Mean: {mean_act:.2f} | Saturation: {saturation*100:.1f}%")

    return np.array(obs_data, dtype=np.float32), np.array(act_data, dtype=np.float32)

def train_bc():
    # 1. AUTO-CALIBRATE
    perform_calibration()
    
    # 2. COLLECT DATA
    print("--- Phase 1: Collecting Noisy Data (Learning Recovery) ---")
    X_noisy, Y_noisy = collect_data(steps_per_config=30_000, noise_level=2.0) 
    
    print("--- Phase 2: Collecting Clean Data (Learning Precision) ---")
    X_clean, Y_clean = collect_data(steps_per_config=10_000, noise_level=0.0)
    
    X_all = np.concatenate([X_noisy, X_clean], axis=0)
    Y_all = np.concatenate([Y_noisy, Y_clean], axis=0)
    
    print(f">>> Total Dataset Size: {X_all.shape[0]}")
    
    # 3. TRAIN
    actor = JointActor().to(DEVICE)
    optimizer = optim.Adam(actor.parameters(), lr=cfg.BC.LR)
    loss_fn = nn.MSELoss()
    
    dataset = TensorDataset(torch.from_numpy(X_all).to(DEVICE), torch.from_numpy(Y_all).to(DEVICE))
    loader = DataLoader(dataset, batch_size=cfg.BC.BATCH_SIZE, shuffle=True)
    
    print(f">>> Starting BC Training on {DEVICE}...")
    actor.train()
    
    best_loss = float('inf')
    
    for epoch in range(cfg.BC.EPOCHS):
        total_loss = 0
        for batch_obs, batch_act in loader:
            optimizer.zero_grad()
            mu, _, _, _ = actor(batch_obs)
            pred_action = torch.tanh(mu) * actor.action_scale
            loss = loss_fn(pred_action, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:02d} | MSE Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(actor.state_dict(), cfg.ROBOT.PRETRAINED_ACTOR_PATH)
            print(f"    >>> Best Model Saved (Loss: {best_loss:.4f})")

if __name__ == "__main__":
    train_bc()