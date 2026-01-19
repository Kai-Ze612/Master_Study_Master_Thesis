import argparse
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.e2e_network import JointActor
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import LeaderRobotSimulator, TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_ee_position(model, data, q_pos):
    """
    Computes Forward Kinematics to find the End-Effector (EE) position
    for a given joint configuration `q_pos`.
    """
    # Copy q_pos to the data buffer
    data.qpos[:cfg.N_JOINTS] = q_pos
    # Forward Kinematics
    mujoco.mj_kinematics(model, data)
    # Get Site Position (Tip)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "panda_ee_site")
    return data.site_xpos[ee_id].copy()

def main(args):
    # 1. Load Model
    print(f">>> Loading Model...")
    # Auto-find latest checkpoint if not specified
    if args.checkpoint is None:
        base_dir = cfg.CHECKPOINT_DIR / "pretrain"
        runs = sorted(base_dir.glob("AR_LSTM_*"))
        if not runs:
            print("[Error] No checkpoints found in", base_dir)
            return
        ckpt_path = runs[-1] / "best_model.pth"
    else:
        ckpt_path = Path(args.checkpoint)
    
    print(f">>> Using Checkpoint: {ckpt_path}")
    
    actor = JointActor().to(DEVICE)
    try:
        actor.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    except Exception as e:
        print(f"[Warning] strict loading failed, trying loose loading... {e}")
        actor.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)
        
    lstm = actor.base_encoder
    lstm.eval()

    # 2. Setup Simulators
    # Map ID to Config
    config_map = {
        '1': ExperimentConfig.NO_DELAY,
        '2': ExperimentConfig.LOW_DELAY,
        '3': ExperimentConfig.HIGH_DELAY,
        '4': ExperimentConfig.HIGH_VARIANCE
    }
    delay_cfg = config_map[args.config]
    
    print(f">>> Simulating Config: {args.config} ({delay_cfg.name})")

    leader_sim = LeaderRobotSimulator(
        trajectory_type=TrajectoryType.FIGURE_8, # Good for visual tracking
        control_freq=cfg.CONTROL_FREQ
    )
    delay_sim = DelaySimulator(cfg.CONTROL_FREQ, config=delay_cfg)
    
    # 3. Setup MuJoCo Viewer
    # We use the leader's model for rendering
    model = leader_sim.model
    data = leader_sim.data
    
    # Create a secondary data object for FK calculations (Ghost logic)
    # This prevents the visualizer from flickering when we calculate prediction positions
    ghost_data = mujoco.MjData(model)

    # 4. Buffers
    leader_hist = deque(maxlen=200)
    lstm_buffer = deque(maxlen=cfg.ROBOT.RNN_SEQ_LEN)
    for _ in range(cfg.ROBOT.RNN_SEQ_LEN):
        lstm_buffer.append(np.zeros(15, dtype=np.float32))

    print(">>> Starting Visualization...")
    print("    [Robot]: True State")
    print("    [Green Sphere]: LSTM Prediction (Should track Robot)")
    print("    [Red Sphere]:   Delayed Input (Should lag behind)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()

            # --- A. Step Physics ---
            true_q, true_qd, _, _, _, _, _ = leader_sim.step()
            leader_hist.append((true_q, true_qd))
            
            # --- B. Get Delay ---
            obs_q, obs_qd, delay_sec = delay_sim.get_delayed_state(leader_hist)
            
            # --- C. LSTM Prediction ---
            # Norm
            x_q = (obs_q - cfg.ROBOT.Q_MEAN) / cfg.ROBOT.Q_STD
            x_qd = (obs_qd - cfg.ROBOT.QD_MEAN) / cfg.ROBOT.QD_STD
            x_del = np.array([delay_sec], dtype=np.float32)
            
            lstm_buffer.append(np.concatenate([x_q, x_qd, x_del]))
            
            # Predict
            inp = torch.tensor(np.array(lstm_buffer), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, pred_norm, _ = lstm(inp)
                pred_norm = pred_norm.cpu().numpy()[0]
                
            pred_q = (pred_norm[:7] * cfg.ROBOT.Q_STD) + cfg.ROBOT.Q_MEAN

            # --- D. Visualization Logic ---
            
            # 1. Update the main Robot to TRUE state
            data.qpos[:7] = true_q
            mujoco.mj_forward(model, data) # Update physics for viewer

            # 2. Calculate "Ghost" Positions (End Effector)
            pos_pred = get_ee_position(model, ghost_data, pred_q)
            pos_delayed = get_ee_position(model, ghost_data, obs_q)

            # 3. Draw Markers
            viewer.user_scn.ngeom = 0 # Clear previous markers
            
            # GREEN SPHERE: Prediction
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0], # Radius 2cm
                pos=pos_pred,
                mat=np.eye(3).flatten(),
                rgba=[0, 1, 0, 0.8] # Green, Opaque
            )
            
            # RED SPHERE: Delayed Input
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],
                pos=pos_delayed,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 0.5] # Red, Semi-transparent
            )
            
            viewer.user_scn.ngeom = 2 # Tell viewer to render 2 geoms
            
            # Sync
            viewer.sync()
            
            # Timing
            time_until_next_step = cfg.DT - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="4", choices=['1', '2', '3', '4'], help="Delay Config ID")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth file")
    args = parser.parse_args()
    
    main(args)