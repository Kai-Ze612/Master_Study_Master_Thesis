#!/usr/bin/env python3
"""
ASAC Visualization Script (Fixed Segfault)
------------------------------------------
Loads a trained model and visualizes the Leader (RED) vs Follower (BLUE) trajectory 
in the MuJoCo viewer using dynamic scene modification.
"""

import torch
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
from pathlib import Path
import time
import sys

# Add package root to path if needed (for module execution)
# sys.path.append(...) 

import ASAC.config.robot_config as cfg
from ASAC.leader_robot_simulator import LeaderRobotSimulator, TrajectoryType
from ASAC.follower_robot_simulator import FollowerRobotSimulator
from ASAC.utils.delay_simulator import DelaySimulator, ExperimentConfig
from ASAC.asac_network import GainTuningActor

# --- CONFIGURATION ---
MODEL_PATH = Path("/media/kai/NewDisk/Kai_thesis/Master_Thesis_E2E_RL_Teleop/libfranka_ws/src/ASAC/ASAC/trained_RL/high_var/best_actor.pth")
DELAY_CONFIG = ExperimentConfig.HIGH_VARIANCE
RENDER_FPS = 50
TRAIL_LENGTH = 1000  # How many points to keep in the trail
TRAIL_DENSITY = 5    # Add a point every N steps

def add_trail_to_scene(viewer, trail_points, rgba):
    """Draws a sequence of small spheres to simulate a trajectory line."""
    if viewer is None: return
    
    for pos in trail_points:
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
            
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.008, 0, 0], 
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=rgba
        )
        viewer.user_scn.ngeom += 1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load Model ---
    if not MODEL_PATH.exists():
        fallback = MODEL_PATH.parent / "final_checkpoint.pth"
        if fallback.exists():
            print(f"[WARN] 'best_actor.pth' not found. Using '{fallback.name}'")
            model_file = fallback
        else:
            print(f"[ERROR] Model not found at {MODEL_PATH}")
            return
    else:
        model_file = MODEL_PATH

    print(f"Loading model from {model_file}...")
    actor = GainTuningActor().to(device)
    
    try:
        checkpoint = torch.load(model_file, map_location=device)
        if 'actor_state_dict' in checkpoint:
            actor.load_state_dict(checkpoint['actor_state_dict'])
        else:
            actor.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        return
    actor.eval()

    # --- Setup Simulators ---
    leader_sim = LeaderRobotSimulator(trajectory_type=TrajectoryType.FIGURE_8, randomize_params=False)
    
    # [FIX] Set render=False here to prevent Segfault (Double Viewer)
    follower_sim = FollowerRobotSimulator(
        delay_config=DELAY_CONFIG,
        render=False,       # <--- THIS MUST BE FALSE
        render_fps=RENDER_FPS
    )
    delay_sim = DelaySimulator(cfg.CONTROL_FREQ, config=DELAY_CONFIG)

    # --- Setup Buffers ---
    obs_history = deque(maxlen=cfg.ROBOT.FRAME_STACK)
    leader_hist = deque(maxlen=cfg.ROBOT.LEADER_HISTORY_BUFFER_LEN)
    last_action = np.zeros(cfg.ROBOT.ACTION_DIM)
    
    red_trail = deque(maxlen=TRAIL_LENGTH)   # Leader
    blue_trail = deque(maxlen=TRAIL_LENGTH)  # Follower

    # Warmup
    l_q, _ = leader_sim.reset()
    f_q, _ = follower_sim.reset()
    delay_sim.reset()

    for _ in range(cfg.ROBOT.LEADER_HISTORY_BUFFER_LEN):
        leader_hist.append((l_q, np.zeros(7)))

    l_init = np.concatenate([(l_q - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, np.zeros(7), [0.0]])
    f_init = np.concatenate([(f_q - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, np.zeros(7)])
    act_init = np.zeros(cfg.ROBOT.ACTION_DIM)
    init_frame = np.concatenate([l_init, f_init, act_init])

    for _ in range(cfg.ROBOT.FRAME_STACK):
        obs_history.append(init_frame)

    # --- Main Loop ---
    print("\nStarting Visualization... (Press ESC in viewer to quit)")
    print("RED Line  = Leader Trajectory")
    print("BLUE Line = Follower Trajectory")
    
    step_counter = 0
    
    # Launch ONE passive viewer
    with mujoco.viewer.launch_passive(follower_sim.model, follower_sim.data) as viewer:
        while viewer.is_running():
            start_time = time.time()

            # 1. Update Leader
            l_q, l_qd, _, _, _, _, _ = leader_sim.step()
            leader_hist.append((l_q, l_qd))
            
            # Get Leader Position (Red)
            leader_sim.data.qpos[:7] = l_q
            mujoco.mj_forward(leader_sim.model, leader_sim.data)
            leader_ee_pos = leader_sim.data.site_xpos[leader_sim.model.site('panda_ee_site').id].copy()
            
            # 2. Delayed Reference
            ref_q, ref_qd, ref_delay = delay_sim.get_delayed_state(leader_hist)
            f_q, f_qd = follower_sim.get_joint_state()
            
            # Get Follower Position (Blue)
            # Note: We must query this BEFORE stepping physics if we want sync, but here is fine.
            follower_ee_pos = follower_sim.data.site_xpos[follower_sim.model.site('panda_ee_site').id].copy()

            # 3. Update Trails
            if step_counter % TRAIL_DENSITY == 0:
                red_trail.append(leader_ee_pos.copy())
                blue_trail.append(follower_ee_pos.copy())

            # 4. RL Observation
            l_vec = np.concatenate([
                (ref_q - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, 
                (ref_qd - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD, 
                [ref_delay]
            ])
            f_vec = np.concatenate([
                (f_q - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, 
                (f_qd - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD
            ])
            frame = np.concatenate([l_vec, f_vec, last_action])
            obs_history.append(frame)
            
            obs_tensor = torch.as_tensor(
                np.concatenate(obs_history).astype(np.float32), 
                device=device
            ).unsqueeze(0)

            # 5. Inference
            with torch.no_grad():
                mu, _ = actor(obs_tensor)
                action_norm = torch.tanh(mu).cpu().numpy().squeeze()

            # 6. Control Law
            kp = (action_norm[:7] + 1)/2 * (cfg.PD_GAINS.KP_MAX - cfg.PD_GAINS.KP_MIN) + cfg.PD_GAINS.KP_MIN
            kd = (action_norm[7:] + 1)/2 * (cfg.PD_GAINS.KD_MAX - cfg.PD_GAINS.KD_MIN) + cfg.PD_GAINS.KD_MIN

            follower_sim.data.qpos[:7] = f_q
            follower_sim.data.qvel[:7] = f_qd
            mujoco.mj_forward(follower_sim.model, follower_sim.data)
            gravity = follower_sim.data.qfrc_bias[:7].copy()

            tau = (kp * (ref_q - f_q)) + (kd * (ref_qd - f_qd)) + gravity
            
            # 7. Step Physics Manually
            follower_sim.data.ctrl[:7] = tau
            mujoco.mj_step(follower_sim.model, follower_sim.data)
            last_action = action_norm
            
            # 8. Render
            viewer.user_scn.ngeom = 0
            add_trail_to_scene(viewer, red_trail, rgba=[1, 0, 0, 1])
            add_trail_to_scene(viewer, blue_trail, rgba=[0, 0, 1, 1])
            viewer.sync()
            
            step_counter += 1

            # 9. Time Keeping
            elapsed = time.time() - start_time
            sleep_time = (1.0 / RENDER_FPS) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == '__main__':
    main()