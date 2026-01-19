import torch
import numpy as np
import argparse
import time
from collections import deque
from pathlib import Path

# --- Project Imports ---
import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.e2e_network import JointActor
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import LeaderRobotSimulator, TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # 1. Locate Checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Auto-find the latest run if specific path not provided
        print(f"[Info] Checkpoint '{checkpoint_path}' not found. Searching for latest in pretrain folder...")
        base_dir = cfg.CHECKPOINT_DIR / "pretrain"
        if base_dir.exists():
            runs = sorted(base_dir.glob("AR_LSTM_*"))
            if runs:
                checkpoint_path = runs[-1] / "best_model.pth"
                print(f"[Info] Found latest run: {runs[-1].name}")
            else:
                print("[Error] No AR_LSTM runs found in pretrain directory.")
                return
        else:
            print(f"[Error] Directory {base_dir} does not exist.")
            return

    print(f">>> Loading Model from: {checkpoint_path}")
    
    # 2. Load Model
    # We load the full JointActor wrapper because that is how we saved the weights
    actor = JointActor().to(DEVICE)
    try:
        actor.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    except Exception as e:
        print(f"[Warning] Strict loading failed. Retrying with strict=False... Error: {e}")
        actor.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)
        
    lstm = actor.base_encoder
    lstm.eval()

    # 3. Initialize Simulators
    print(">>> Initializing Simulators...")
    leader = LeaderRobotSimulator(
        trajectory_type=TrajectoryType.FIGURE_8,
        control_freq=cfg.CONTROL_FREQ
    )
    # Using High Variance to make the test interesting
    delay_sim = DelaySimulator(
        control_freq=cfg.CONTROL_FREQ, 
        config=ExperimentConfig.HIGH_VARIANCE 
    )
    
    # 4. Initialize Buffers
    leader.reset()
    delay_sim.reset()
    
    leader_hist = deque(maxlen=200)                   # History for delay simulator lookup
    lstm_buffer = deque(maxlen=cfg.ROBOT.RNN_SEQ_LEN) # History for LSTM input
    
    # Pre-fill LSTM buffer with zeros so we can start immediately
    for _ in range(cfg.ROBOT.RNN_SEQ_LEN):
        lstm_buffer.append(np.zeros(15, dtype=np.float32))

    # Header for the printout
    print("\n" + "="*95)
    print(f"{'STEP':<6} | {'DELAY':<8} | {'JOINT':<8} | {'TRUE (rad)':<12} | {'PRED (rad)':<12} | {'ERROR (rad)':<12} | {'STATUS'}")
    print("="*95)

    # 5. Main Test Loop
    try:
        for step in range(args.steps):
            # --- A. Ground Truth Generation ---
            # Step the physics to get the "Real" state now
            true_q, true_qd, _, _, _, _, _ = leader.step()
            leader_hist.append((true_q, true_qd))
            
            # --- B. Apply Delay ---
            # Ask the delay simulator: "What does the follower see right now?"
            obs_q, obs_qd, delay_sec = delay_sim.get_delayed_state(leader_hist)
            
            # --- C. Prepare LSTM Input (Normalization) ---
            # Normalize using the stats defined in config
            x_q = (obs_q - cfg.ROBOT.Q_MEAN) / cfg.ROBOT.Q_STD
            x_qd = (obs_qd - cfg.ROBOT.QD_MEAN) / cfg.ROBOT.QD_STD
            x_delay = np.array([delay_sec], dtype=np.float32)
            
            # 15D Vector: [7 Pos + 7 Vel + 1 Delay]
            input_vec = np.concatenate([x_q, x_qd, x_delay])
            lstm_buffer.append(input_vec)
            
            # --- D. Inference ---
            # Convert to tensor: [Batch=1, Seq_Len, Input_Dim]
            input_tensor = torch.tensor(np.array(lstm_buffer), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                # Forward pass returns: (hidden, pred_state_norm, (h,c))
                # pred_state_norm is [Batch, 14] -> 7 Pos, 7 Vel
                _, pred_norm, _ = lstm(input_tensor)
                pred_norm = pred_norm.cpu().numpy()[0]
            
            # --- E. Denormalize Prediction ---
            pred_q_norm = pred_norm[:7]
            pred_q = (pred_q_norm * cfg.ROBOT.Q_STD) + cfg.ROBOT.Q_MEAN
            
            # --- F. Calculate Error ---
            error = true_q - pred_q
            rmse = np.sqrt(np.mean(error**2))
            
            # --- G. Print Results ---
            # We print Joint 1 details + Overall RMSE
            if step % args.print_freq == 0:
                # Status indicator
                status = "OK"
                if rmse > 0.1: status = "DRIFT"
                if rmse > 0.5: status = "LOST"
                
                print(f"{step:<6} | {delay_sec*1000:5.0f} ms | J1       | {true_q[0]:12.4f} | {pred_q[0]:12.4f} | {error[0]:12.4f} | {status}")
                
                # Optional: Uncomment to see RMSE for the whole robot on the next line
                # print(f"{'':<6} | {'':<8} | {'ALL RMSE':<8} | {'---':<12} | {'---':<12} | {rmse:12.4f} |") 
                # print("-" * 95)

            # Optional: Add sleep to watch it unfold in real-time
            # time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Info] Test stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="auto", help="Path to best_model.pth (default: auto-find)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("--print-freq", type=int, default=1, help="Print every N steps")
    args = parser.parse_args()
    
    main(args)