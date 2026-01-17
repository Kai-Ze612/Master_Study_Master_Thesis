"""
pretrained-LSTM script
[Optimized for < 0.1 rad error]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.e2e_network import JointActor
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import LeaderRobotSimulator, TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig

########################################################################
# Local Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SPLIT = 0.8

# [CHANGE 1] Aggressive Velocity Weight
# Increased from 10.0 -> 50.0 to force the network to match speed 
# and reduce the "lag" drift.
VELOCITY_LOSS_WEIGHT = 50.0 
########################################################################

def inject_fixed_normalization():
    """
    [Phase 1] Manual Normalization (The Fix)
    Instead of auto-calibrating (which breaks on joints that don't move much),
    we use a fixed 'Safe' normalization.
    
    Mean = Initial Pose (or 0)
    Std  = 1.0 (Preserves true scale of noise)
    """
    print(">>> [1/4] Injecting Fixed 'Safety' Normalization...")
    
    # 1. Mean: Centered on the robot's spawn configuration
    # This ensures "0" input = "Home Position"
    q_mean = cfg.INITIAL_JOINT_CONFIG.copy()
    
    # 2. Std: Force to 1.0
    # This prevents the 'Magnifying Glass' effect on wrist joints.
    q_std = np.ones(7, dtype=np.float32)
    
    # 3. Velocity: Zero mean, 1.0 Std
    qd_mean = np.zeros(7, dtype=np.float32)
    qd_std  = np.ones(7, dtype=np.float32)
    
    # Inject into Global Config for this run
    object.__setattr__(cfg.ROBOT, 'Q_MEAN', q_mean)
    object.__setattr__(cfg.ROBOT, 'Q_STD', q_std)
    object.__setattr__(cfg.ROBOT, 'QD_MEAN', qd_mean)
    object.__setattr__(cfg.ROBOT, 'QD_STD', qd_std)
    
    print(f"    Fixed Stats Set: Q_STD={q_std[0]}, QD_STD={qd_std[0]}")

def collect_dataset():
    """
    [Phase 2] Data Collection
    Uses TUNED PD controller gains (Vector) to prevent Wrist Jitter.
    """
    steps_to_collect = cfg.BC.STEPS_TO_COLLECT
    print(f">>> [2/4] Collecting Expert Data ({steps_to_collect} steps)...")
    
    env = TeleoperationEnv(
        delay_config=ExperimentConfig.HIGH_VARIANCE,
        trajectory_type=TrajectoryType.FIGURE_8,
        
        # [CHANGE 2] Disable Randomization
        # Gives the LSTM a consistent pattern to learn first.
        # This removes "Bad Workspace" noise.
        randomize_trajectory=False, 
        
        render_mode=None
    )
    
    obs_list = []
    state_list = []
    
    obs, info = env.reset()
    
    # TUNED GAINS: High for arm, Low for wrist to prevent vibration
    # J1-J4: Heavy (150) | J5-J7: Light (50, 20)
    Kp = np.array([150.0, 150.0, 150.0, 150.0, 50.0, 50.0, 20.0], dtype=np.float32)
    Kd = np.array([ 10.0,  10.0,  10.0,  10.0,  5.0,  5.0,  2.0], dtype=np.float32)
    
    for _ in tqdm(range(steps_to_collect)):
        curr_q = info['follower_q'] 
        curr_qd = info['follower_qd']
        targ_q = info['leader_q']   
        targ_qd = info['leader_qd']
        
        # Vectorized PD Control
        tau = Kp * (targ_q - curr_q) + Kd * (targ_qd - curr_qd)
        tau = np.clip(tau, -cfg.ROBOT.MAX_ACTION_TORQUE, cfg.ROBOT.MAX_ACTION_TORQUE)
        
        # Store Data
        obs_list.append(obs)
        state_list.append(info['true_state_vector']) 
        
        obs, _, term, trunc, info = env.step(tau)
        if term or trunc:
            obs, info = env.reset()
            
    env.close()
    
    X = torch.tensor(np.array(obs_list), dtype=torch.float32)
    Y = torch.tensor(np.array(state_list), dtype=torch.float32)
    return TensorDataset(X, Y)

def train_bc():
    # [CHANGE 3] Use Fixed Normalization instead of Auto-Calibrate
    inject_fixed_normalization()
    
    # 2. Collect
    full_dataset = collect_dataset()
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=cfg.BC.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.BC.BATCH_SIZE, shuffle=False)

    # 3. Setup Model
    print(">>> [3/4] Initializing Physically-Aware LSTM...")
    actor = JointActor().to(DEVICE)
    
    optimizer = optim.Adam(actor.base_encoder.parameters(), lr=cfg.BC.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    mse_loss = nn.MSELoss()

    # 4. Training Loop
    epochs = cfg.BC.EPOCHS
    print(f">>> [4/4] Starting Training ({epochs} Epochs)...")
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # --- TRAIN ---
        actor.train()
        batch_losses = []
        
        for b_obs, b_target in train_loader:
            b_obs, b_target = b_obs.to(DEVICE), b_target.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, _, pred_state, _ = actor(b_obs)
            
            # Split State
            pred_pos, pred_vel = pred_state[:, :7], pred_state[:, 7:14]
            true_pos, true_vel = b_target[:, :7], b_target[:, 7:14]
            
            # [PHYSICS LOSS]
            l_pos = mse_loss(pred_pos, true_pos)
            l_vel = mse_loss(pred_vel, true_vel)
            
            # Stronger Velocity Penalty
            loss = l_pos + (VELOCITY_LOSS_WEIGHT * l_vel)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.base_encoder.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())
            
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)

        # --- VALIDATION ---
        actor.eval()
        val_batch_losses = []
        with torch.no_grad():
            for b_obs, b_target in val_loader:
                b_obs, b_target = b_obs.to(DEVICE), b_target.to(DEVICE)
                _, _, pred_state, _ = actor(b_obs)
                
                pred_pos, pred_vel = pred_state[:, :7], pred_state[:, 7:14]
                true_pos, true_vel = b_target[:, :7], b_target[:, 7:14]
                
                loss = mse_loss(pred_pos, true_pos) + (VELOCITY_LOSS_WEIGHT * mse_loss(pred_vel, true_vel))
                val_batch_losses.append(loss.item())
                
        avg_val_loss = np.mean(val_batch_losses)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1:02d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(actor, "best")
            
    # Final Save
    save_checkpoint(actor, "final")
    print(">>> Training Complete.")
    
    # Plot Loss Curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.savefig('bc_training_curve.png')

def save_checkpoint(actor, tag):
    base_path = str(cfg.BC.SAVE_PATH)
    
    if base_path.endswith(".pth"):
        path = base_path.replace(".pth", f"_{tag}.pth")
    else:
        path = f"{base_path}_{tag}.pth"
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'actor': actor.state_dict(),
        'norm': {
            'q_mean': cfg.ROBOT.Q_MEAN,
            'q_std': cfg.ROBOT.Q_STD,
            'qd_mean': cfg.ROBOT.QD_MEAN,
            'qd_std': cfg.ROBOT.QD_STD
        }
    }
    torch.save(checkpoint, path)
    print(f"[Saved] Checkpoint: {path}")

if __name__ == "__main__":
    train_bc()