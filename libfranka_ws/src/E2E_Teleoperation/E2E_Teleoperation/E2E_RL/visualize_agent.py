import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Project Imports
import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.e2e_network import JointActor
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = cfg.PRETRAINED_DIR / "best_checkpoint_best.pth"

def visualize_all_joints():
    print(f"[Info] Loading Model: {CHECKPOINT_PATH}")
    
    if not CHECKPOINT_PATH.exists():
        print(f"[Error] File not found: {CHECKPOINT_PATH}")
        return

    # 1. Load Checkpoint & Restore Norm
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if 'norm' in checkpoint:
        stats = checkpoint['norm']
        object.__setattr__(cfg.ROBOT, 'Q_MEAN', stats['q_mean'])
        object.__setattr__(cfg.ROBOT, 'Q_STD', stats['q_std'])
        object.__setattr__(cfg.ROBOT, 'QD_MEAN', stats['qd_mean'])
        object.__setattr__(cfg.ROBOT, 'QD_STD', stats['qd_std'])

    # 2. Init Model
    actor = JointActor().to(DEVICE)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()

    # 3. Setup Env
    env = TeleoperationEnv(
        delay_config=ExperimentConfig.HIGH_VARIANCE,
        trajectory_type=TrajectoryType.FIGURE_8,
        randomize_trajectory=False, # [CRITICAL] Keep False for clean visualization
        render_mode=None 
    )
    
    obs, info = env.reset()
    
    # --- [FIX] Tuned Vector Gains (Matches Training) ---
    # Prevents "Hulk Smash" vibration on the lightweight wrist joints
    Kp = np.array([150.0, 150.0, 150.0, 150.0, 50.0, 50.0, 20.0], dtype=np.float32)
    Kd = np.array([ 10.0,  10.0,  10.0,  10.0,  5.0,  5.0,  2.0], dtype=np.float32)
    # ---------------------------------------------------
    
    # 4. Collect Data
    history = {'time': [], 'true_q': [], 'pred_q': [], 'delay': []}
    steps = 1000 # 10 seconds
    
    print(">>> Collecting Data (1000 steps)...")
    for i in tqdm(range(steps)):
        # Neural Net Prediction
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, _, pred_state_norm, _ = actor(obs_t)
            pred_q = (pred_state_norm.cpu().numpy()[0][:7] * cfg.ROBOT.Q_STD) + cfg.ROBOT.Q_MEAN

        # Ground Truth
        real_q = info['leader_q']
        
        # Expert Driver
        curr_q = info['follower_q']
        curr_qd = info['follower_qd']
        
        # Apply Vectorized Gains
        tau = Kp * (real_q - curr_q) + Kd * (info['leader_qd'] - curr_qd)
        tau = np.clip(tau, -cfg.ROBOT.MAX_ACTION_TORQUE, cfg.ROBOT.MAX_ACTION_TORQUE)
        
        history['time'].append(i * cfg.DT)
        history['true_q'].append(real_q)
        history['pred_q'].append(pred_q)
        history['delay'].append(info['delay'] * 1000)
        
        obs, _, term, trunc, info = env.step(tau)
        if term or trunc: break
        
    env.close()
    
    # 5. Plotting 7 Joints
    print(">>> Generating 7-Joint Plot...")
    time = np.array(history['time'])
    true_q = np.array(history['true_q'])
    pred_q = np.array(history['pred_q'])
    
    # Create a tall figure with 7 subplots
    fig, axes = plt.subplots(7, 1, figsize=(10, 18), sharex=True)
    
    joint_names = ["J1 (Base)", "J2 (Shoulder)", "J3 (Elbow)", "J4 (Wrist 1)", 
                   "J5 (Wrist 2)", "J6 (Wrist 3)", "J7 (End Effector)"]

    for j in range(7):
        ax = axes[j]
        # Plot True vs Pred
        ax.plot(time, true_q[:, j], 'k-', linewidth=2.5, alpha=0.4, label='True Leader')
        ax.plot(time, pred_q[:, j], 'r--', linewidth=1.5, label='LSTM Prediction')
        
        # Calculate Error for Title
        rmse = np.sqrt(np.mean((true_q[:, j] - pred_q[:, j])**2))
        
        ax.set_ylabel(f"J{j+1} (rad)")
        ax.set_title(f"{joint_names[j]} - RMSE: {rmse:.4f}", fontsize=10, loc='left')
        ax.grid(True, alpha=0.3)
        
        # Only legend on first plot
        if j == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    output_file = "all_joints_tracking.png"
    plt.savefig(output_file, dpi=150)
    print(f"[Output] Saved '{output_file}'")
    plt.show()

if __name__ == "__main__":
    visualize_all_joints()