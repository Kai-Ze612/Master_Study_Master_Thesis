"""
Minimal BC Inference Test
- Normalized action history (matches training)
- No gravity compensation (MuJoCo handles it)
- Clear debug output
"""

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
from E2E_Teleoperation.E2E_RL.follower_robot_simulator import FollowerRobotSimulator
from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ee_position(model, data, q_pos):
    data.qpos[:7] = q_pos
    mujoco.mj_kinematics(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "panda_ee_site")
    return data.site_xpos[ee_id].copy()


def main(args):
    # ==================== LOAD MODEL ====================
    print("=" * 80)
    print("BC INFERENCE TEST (Normalized Actions, No Gravity Comp)")
    print("=" * 80)
    
    checkpoint_path = Path("/media/kai/NewDisk/Kai_thesis/Master_Thesis_E2E_RL_Teleop/libfranka_ws/src/E2E_Teleoperation/E2E_Teleoperation/trained_RL/pre_trained_BC/best_checkpoint.pth")
    
    if not checkpoint_path.exists():
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Load normalization stats
    if 'norm' in checkpoint:
        stats = checkpoint['norm']
        Q_MEAN = np.array(stats['q_mean'], dtype=np.float32)
        Q_STD = np.array(stats['q_std'], dtype=np.float32)
        QD_MEAN = np.array(stats['qd_mean'], dtype=np.float32)
        QD_STD = np.array(stats['qd_std'], dtype=np.float32)
        print(f"[OK] Loaded normalization from checkpoint")
    else:
        Q_MEAN = cfg.ROBOT.Q_MEAN
        Q_STD = cfg.ROBOT.Q_STD
        QD_MEAN = cfg.ROBOT.QD_MEAN
        QD_STD = cfg.ROBOT.QD_STD
        print(f"[WARN] Using default normalization")

    print(f"  Q_MEAN: {Q_MEAN}")
    print(f"  Q_STD:  {Q_STD}")

    # Load actor
    actor = JointActor().to(DEVICE)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    
    # Get action scale (for denormalization)
    ACTION_SCALE = actor.action_scale.cpu().numpy()
    print(f"  ACTION_SCALE: {ACTION_SCALE}")

    # ==================== SETUP SIMULATION ====================
    config_map = {
        '1': ExperimentConfig.NO_DELAY, 
        '2': ExperimentConfig.LOW_DELAY, 
        '3': ExperimentConfig.HIGH_DELAY, 
        '4': ExperimentConfig.HIGH_VARIANCE
    }
    delay_cfg = config_map[args.config]
    print(f"\n[Config] Delay setting: {args.config} -> {delay_cfg}")

    leader_sim = LeaderRobotSimulator(
        trajectory_type=TrajectoryType.FIGURE_8, 
        control_freq=cfg.CONTROL_FREQ
    )
    follower_sim = FollowerRobotSimulator(delay_config=delay_cfg, render=False)
    delay_sim = DelaySimulator(cfg.CONTROL_FREQ, config=delay_cfg)

    # ==================== BUFFERS ====================
    leader_hist = deque(maxlen=200)
    lstm_buffer = deque(maxlen=cfg.ROBOT.RNN_SEQ_LEN)
    
    # ACTION HISTORY: stores NORMALIZED actions [-1, 1]
    action_hist_norm = deque(maxlen=cfg.ROBOT.ACTION_HISTORY_LEN)
    
    # Pre-fill with zeros
    for _ in range(cfg.ROBOT.RNN_SEQ_LEN):
        lstm_buffer.append(np.zeros(15, dtype=np.float32))
    for _ in range(cfg.ROBOT.ACTION_HISTORY_LEN):
        action_hist_norm.append(np.zeros(7, dtype=np.float32))

    # Previous action (NORMALIZED)
    prev_action_norm = np.zeros(7, dtype=np.float32)

    # Get initial follower state
    f_q, f_qd = follower_sim.get_joint_state()

    # Visualization data
    model = follower_sim.model
    data = follower_sim.data
    ghost_data = mujoco.MjData(model)

    # ==================== MAIN LOOP ====================
    print("\n" + "=" * 110)
    print(f"{'STEP':<6} | {'DELAY':<6} | {'TRUE':<8} | {'FOLL':<8} | {'PRED':<8} | "
          f"{'ERR':<7} | {'TAU':<8} | {'ACT_N':<8} | {'HIST[0]':<8}")
    print("=" * 110)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- A. Step Leader ---
            t_q, t_qd, _, _, _, _, _ = leader_sim.step()
            leader_hist.append((t_q, t_qd))

            # --- B. Get Delayed Observation ---
            d_q, d_qd, d_sec = delay_sim.get_delayed_state(leader_hist)

            # --- C. Build LSTM Input ---
            lstm_in = np.concatenate([
                (d_q - Q_MEAN) / Q_STD,
                (d_qd - QD_MEAN) / QD_STD,
                [d_sec]
            ]).astype(np.float32)
            lstm_buffer.append(lstm_in)

            inp_tensor = torch.tensor(
                np.array(lstm_buffer), device=DEVICE
            ).unsqueeze(0)

            # --- D. Model Inference ---
            with torch.no_grad():
                # LSTM forward
                _, pred_norm, (h_n, _) = actor.base_encoder(inp_tensor)
                latent = h_n  # [1, hidden_dim]

                # Normalize follower state
                f_q_norm = (f_q - Q_MEAN) / Q_STD
                f_qd_norm = (f_qd - QD_MEAN) / QD_STD
                
                robot_state = torch.tensor(
                    np.concatenate([f_q_norm, f_qd_norm]),
                    dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)

                # Action history (NORMALIZED, flattened)
                hist_flat = np.array(action_hist_norm).flatten()
                hist_tensor = torch.tensor(
                    hist_flat, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)

                # Previous action (NORMALIZED)
                prev_tensor = torch.tensor(
                    prev_action_norm, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)

                # Concatenate policy input
                policy_in = torch.cat([
                    robot_state,    # [1, 14]
                    latent,         # [1, hidden_dim]
                    hist_tensor,    # [1, 7 * ACTION_HISTORY_LEN]
                    prev_tensor     # [1, 7]
                ], dim=1)

                # Policy forward
                x_res = actor.residual_net(policy_in)
                mu = actor.res_mu(x_res)
                
                # Output is NORMALIZED action
                action_norm = torch.tanh(mu).cpu().numpy()[0]
                
                # Convert to actual torque
                action_torque = action_norm * ACTION_SCALE

                # Get prediction for display
                pred_np = pred_norm.cpu().numpy()[0]
                pred_q = (pred_np[:7] * Q_STD) + Q_MEAN

            # --- E. Step Follower (NO gravity compensation) ---
            f_info = follower_sim.step(action_torque)
            f_q = f_info['q_follower']
            f_qd = f_info['qd_follower']

            # --- F. Update History (NORMALIZED) ---
            action_hist_norm.append(action_norm.copy())
            prev_action_norm = action_norm.copy()

            # --- G. Metrics & Display ---
            trk_err = np.linalg.norm(t_q - f_q)
            
            if follower_sim._internal_tick % 10 == 0:
                # Show first element of action history for debugging
                hist_first = np.array(action_hist_norm)[0, 0]
                print(f"{follower_sim._internal_tick:<6} | {d_sec*1000:4.0f}ms | "
                      f"{t_q[0]:8.4f} | {f_q[0]:8.4f} | {pred_q[0]:8.4f} | "
                      f"{trk_err:7.4f} | {action_torque[0]:8.2f} | "
                      f"{action_norm[0]:8.4f} | {hist_first:8.4f}")

            # --- H. Visualization ---
            pos_leader = get_ee_position(model, ghost_data, t_q)
            pos_pred = get_ee_position(model, ghost_data, pred_q)
            pos_delayed = get_ee_position(model, ghost_data, d_q)

            viewer.user_scn.ngeom = 0
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0], 
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.025, 0, 0], pos=pos_leader,
                mat=np.eye(3).flatten(), rgba=[0, 1, 0, 0.8]
            )
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0], pos=pos_pred,
                mat=np.eye(3).flatten(), rgba=[1, 0, 0, 0.6]
            )
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[2],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.015, 0, 0], pos=pos_delayed,
                mat=np.eye(3).flatten(), rgba=[0, 0, 1, 0.4]
            )
            viewer.user_scn.ngeom = 3
            viewer.sync()

            time.sleep(max(0, cfg.DT - (time.time() - step_start)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="4", choices=['1', '2', '3', '4'])
    args = parser.parse_args()
    main(args)