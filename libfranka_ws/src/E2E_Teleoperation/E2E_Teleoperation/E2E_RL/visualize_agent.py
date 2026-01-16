"""
Visualization Script for E2E Teleoperation
Loads a trained actor and visualizes the follower robot tracking the leader trajectory in MuJoCo.
"""

import torch
import numpy as np
import time
import argparse
from pathlib import Path

from E2E_Teleoperation.E2E_RL.e2e_network import JointActor
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig
import E2E_Teleoperation.config.robot_config as cfg

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(model_path, delay_config, trajectory_type):
    print(f"\n[Info] Initializing Environment with Render Mode = 'human'...")
    print(f"[Info] Trajectory: {trajectory_type.name}")
    print(f"[Info] Delay Config: {delay_config.name}")

    # 1. Initialize Environment (Visualizer)
    env = TeleoperationEnv(
        delay_config=delay_config,
        trajectory_type=trajectory_type,
        randomize_trajectory=False, # Fixed trajectory for consistent visualization
        render_mode="human"         # <--- Enables MuJoCo Viewer
    )

    # 2. Initialize Actor & Load Weights
    print(f"[Info] Loading Model from: {model_path}")
    actor = JointActor().to(DEVICE)
    
    if not Path(model_path).exists():
        print(f"[Error] Model file not found at {model_path}")
        print("        Did you run 'train_bc.py' or 'train_agent.py' yet?")
        return

    try:
        # Load weights
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Handle saving differences (Direct state_dict vs Checkpoint dict)
        if isinstance(checkpoint, dict) and 'actor' in checkpoint:
            actor.load_state_dict(checkpoint['actor'])
            print("[Info] Loaded RL Checkpoint (Best Model).")
        else:
            actor.load_state_dict(checkpoint)
            print("[Info] Loaded BC/Pretrained State Dict.")
            
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    actor.eval()

    # 3. Simulation Loop
    print("\n>>> Starting Visualization (Press Ctrl+C to stop)...")
    obs, info = env.reset()
    
    try:
        while True:
            # Prepare observation
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                
                # Forward Pass
                mu, _, pred_leader, _ = actor(obs_tensor)
                
                # Deterministic Action for Visualization
                action = torch.tanh(mu) * actor.action_scale
                action_np = action.cpu().numpy()[0]
                
                # Get Prediction for Logging
                pred_leader_np = pred_leader.cpu().numpy()[0]

            # Step Environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            # --- Real-Time Logging ---
            # Denormalize Prediction for Comparison
            pred_q_real = (pred_leader_np[:7] * cfg.ROBOT.Q_STD) + cfg.ROBOT.Q_MEAN
            true_q_real = info['leader_q']
            follower_q_real = info['follower_q']
            
            # Print Errors to Console
            track_err = np.linalg.norm(true_q_real - follower_q_real)
            pred_err = np.linalg.norm(true_q_real - pred_q_real)
            
            print(f"\r[Step {env.step_count}] Track Err: {track_err:.4f} rad | Pred Err: {pred_err:.4f} rad | Action: {action_np[0]:.2f} Nm", end="")

            # Slow down to make visualization viewable (Real-time factor)
            time.sleep(cfg.DT) 

            if terminated or truncated:
                print("\n[Info] Episode finished. Resetting...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n[Info] Visualization stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Default to the PRETRAINED actor we just made
    default_path = cfg.ROBOT.PRETRAINED_ACTOR_PATH
    
    parser.add_argument("--model", type=str, default=str(default_path), 
                        help="Path to the model .pth file")
    
    args = parser.parse_args()

    # You can change ExperimentConfig.HIGH_VARIANCE to LOW_VARIANCE if you want to test easier conditions
    main(model_path=args.model, 
         delay_config=ExperimentConfig.HIGH_VARIANCE, 
         trajectory_type=TrajectoryType.FIGURE_8)