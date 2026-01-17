"""
pretrained-LSTM script
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
VELOCITY_LOSS_WEIGHT = 10.0
########################################################################

def auto_calibrate_stats():
    """
    [Phase 1] Auto-Calibration
    Runs the robot through a fast trajectory to calculate robust Mean/Std for normalization.
    """
    print(">>> [1/4] Auto-Calibrating Normalization Statistics...")
    
    sim = LeaderRobotSimulator(
        trajectory_type=TrajectoryType.FIGURE_8, 
        randomize_params=True,
        control_freq=cfg.CONTROL_FREQ
    )
    sim.reset()
    
    data_q, data_qd = [], []
    
    # Run for 5000 steps to get a good distribution
    for _ in range(50000):
        q, qd, _, _, _, _, _ = sim.step()
        data_q.append(q)
        data_qd.append(qd)
        
    q_arr = np.array(data_q)
    qd_arr = np.array(data_qd)
    
    # Calculate Stats
    q_mean = np.mean(q_arr, axis=0)
    q_std = np.maximum(np.std(q_arr, axis=0), 1e-3) 
    
    qd_mean = np.mean(qd_arr, axis=0)
    # [CRITICAL] Enforce a minimum floor on Velocity STD
    qd_std = np.maximum(np.std(qd_arr, axis=0), 0.5) 
    
    # Inject into Global Config for this run
    object.__setattr__(cfg.ROBOT, 'Q_MEAN', q_mean)
    object.__setattr__(cfg.ROBOT, 'Q_STD', q_std)
    object.__setattr__(cfg.ROBOT, 'QD_MEAN', qd_mean)
    object.__setattr__(cfg.ROBOT, 'QD_STD', qd_std)
    
    print(f"    Calibration Done. Velocity Std: {np.round(qd_std, 3)}")

def collect_dataset():
    """
    [Phase 2] Data Collection
    Uses TUNED PD controller gains (Vector) to prevent Wrist Jitter.
    """
    # [FIX] Correctly use config variable
    steps_to_collect = cfg.BC.STEPS_TO_COLLECT
    print(f">>> [2/4] Collecting Expert Data ({steps_to_collect} steps)...")
    
    env = TeleoperationEnv(
        delay_config=ExperimentConfig.HIGH_VARIANCE,
        trajectory_type=TrajectoryType.FIGURE_8,
        randomize_trajectory=True, 
        render_mode=None
    )
    
    obs_list = []
    state_list = []
    
    obs, info = env.reset()
    
    # [FIX] TUNED GAINS: High for arm, Low for wrist to prevent vibration
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
    # 1. Calibrate
    auto_calibrate_stats()
    
    # 2. Collect
    full_dataset = collect_dataset()
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # [FIX] Use Config Batch Size
    train_loader = DataLoader(train_data, batch_size=cfg.BC.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.BC.BATCH_SIZE, shuffle=False)

    # 3. Setup Model
    print(">>> [3/4] Initializing Physically-Aware LSTM...")
    actor = JointActor().to(DEVICE)
    
    # [FIX] Use Config LR
    optimizer = optim.Adam(actor.base_encoder.parameters(), lr=cfg.BC.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    mse_loss = nn.MSELoss()

    # 4. Training Loop
    # [FIX] Use Config Epochs
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
    # [FIX] Handle paths correctly
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