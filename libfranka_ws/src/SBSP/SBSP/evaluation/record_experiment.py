#!/usr/bin/env python3
"""
SBSP Evaluation & Data Recording Node
----------------------------------------------------------
Features:
1. Runs SBSP Simulation with MuJoCo Visualization.
2. VISUALIZATION: Updates 'target' site (Red) to match Leader.
3. PREDICTION: Uses DCNN (if available) to correct delayed states (PMDC method).
4. Records all data to CSV in 'SBSP/evaluation/data/'.

Usage:
    python3 -m SBSP.evaluation.record_experiment --config 3
"""

import os
import sys
import argparse  # [ADDED]
import rclpy
from rclpy.node import Node
import torch
import numpy as np
import mujoco
import csv
from datetime import datetime
from collections import deque
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# ROS Messages
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

# Project Imports
import SBSP.config.robot_config as cfg
from SBSP.leader_robot_simulator import LeaderRobotSimulator, TrajectoryType
from SBSP.follower_robot_simulator import FollowerRobotSimulator
from SBSP.utils.delay_simulator import DelaySimulator, ExperimentConfig
from SBSP.sbsp_network import SBSPActor, DCNN

class DataRecorderNode(Node):
    def __init__(self, delay_config_int=3):
        super().__init__('sbsp_recorder_node')

        # --- 1. Settings ---
        self.control_rate = cfg.CONTROL_FREQ  # 200 Hz
        self.dt = cfg.DT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Map int to Config
        self.delay_config = self._get_delay_config(delay_config_int)
        self.get_logger().info(f"Delay Configuration: {self.delay_config.name} (ID: {delay_config_int})")

        # Directories
        self.package_root = cfg.PYTHON_PACKAGE_ROOT
        self.data_dir = self.package_root / "evaluation" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file_path = self.data_dir / f"sbsp_experiment_{timestamp}.csv"
        self.csv_file = open(self.csv_file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Define CSV Header
        header = ['time', 'ee_error']
        header += [f'leader_ee_pos_{x}' for x in ['x','y','z']]
        header += [f'follower_ee_pos_{x}' for x in ['x','y','z']]
        header += [f'leader_q_{i}' for i in range(7)]
        header += [f'follower_q_{i}' for i in range(7)]
        header += [f'kp_{i}' for i in range(7)]
        header += [f'kd_{i}' for i in range(7)]
        self.csv_writer.writerow(header)
        
        self.get_logger().info(f"Recording data to: {self.csv_file_path}")

        # --- 2. Load Models ---
        self.checkpoint_dir = cfg.CHECKPOINT_DIR
        self.run_dir = self._find_latest_run()
        
        # Initialize Networks
        self.actor = SBSPActor(input_dim=cfg.ROBOT.RL_OBS_DIM).to(self.device)
        self.dcnn = DCNN(input_dim=14, action_dim=1).to(self.device) 
        
        self.actor.eval()
        self.dcnn.eval()
        self._load_weights()

        # --- 3. Initialize Simulators ---
        self.leader = LeaderRobotSimulator(trajectory_type=TrajectoryType.FIGURE_8, randomize_params=True)
        
        # [VISUALIZATION] render=True enables the MuJoCo viewer window
        # We pass the dynamic delay config here
        self.follower = FollowerRobotSimulator(
            delay_config=self.delay_config, 
            render=True,       # <--- ENABLE VISUALIZATION
            render_fps=50      # Throttle rendering 
        )
        self.delay_sim = DelaySimulator(cfg.CONTROL_FREQ, config=self.delay_config)

        # [VISUALIZATION] Setup Red Ghost
        self.target_site_id = -1
        try:
            self.target_site_id = mujoco.mj_name2id(self.follower.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        except:
            pass

        # --- 4. State Buffers ---
        self.obs_history = deque(maxlen=cfg.ROBOT.FRAME_STACK)
        self.leader_hist = deque(maxlen=cfg.ROBOT.LEADER_HISTORY_BUFFER_LEN)
        self.last_action = np.zeros(cfg.ROBOT.ACTION_DIM)
        self._warmup_buffers()

        # --- 5. ROS Publishers ---
        self.pub_leader_pose = self.create_publisher(PoseStamped, '/sbsp/leader/ee_pose', 10)
        self.pub_follower_pose = self.create_publisher(PoseStamped, '/sbsp/follower/ee_pose', 10)
        self.pub_time = self.create_publisher(Float64, '/sbsp/time', 10)
        self.pub_err = self.create_publisher(Float64, '/sbsp/tracking_error', 10)

        # --- 6. Timer Loop ---
        self.step_count = 0
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(f"Starting SBSP Record Loop @ {self.control_rate}Hz...")

    def _get_delay_config(self, arg_value):
        if arg_value == 0: return ExperimentConfig.NO_DELAY
        if arg_value == 1: return ExperimentConfig.LOW_DELAY
        if arg_value == 2: return ExperimentConfig.HIGH_DELAY
        return ExperimentConfig.HIGH_VARIANCE 

    def _find_latest_run(self):
        """Finds the most recent training run in trained_RL"""
        if not self.checkpoint_dir.exists():
            self.get_logger().error(f"Checkpoint dir not found: {self.checkpoint_dir}")
            sys.exit(1)
            
        runs = sorted([d for d in self.checkpoint_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
        if not runs:
            self.get_logger().error(f"No training runs found in {self.checkpoint_dir}")
            sys.exit(1)
            
        latest = runs[-1]
        self.get_logger().info(f"Loading from latest run: {latest.name}")
        return latest

    def _load_weights(self):
        # Load Actor
        actor_path = self.run_dir / "best_actor.pth"
        if not actor_path.exists():
            actor_path = self.run_dir / "best_checkpoint.pth"
            
        try:
            ckpt = torch.load(actor_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'actor' in ckpt:
                self.actor.load_state_dict(ckpt['actor'])
            else:
                self.actor.load_state_dict(ckpt)
            self.get_logger().info("Actor weights loaded.")
        except Exception as e:
            self.get_logger().error(f"Failed to load Actor: {e}")

    def _warmup_buffers(self):
        l_q, _ = self.leader.reset()
        f_q, _ = self.follower.reset()
        self.delay_sim.reset()

        for _ in range(cfg.ROBOT.LEADER_HISTORY_BUFFER_LEN):
            self.leader_hist.append((l_q, np.zeros(7)))

        l_init = np.concatenate([(l_q - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, np.zeros(7), [0.0]])
        f_init = np.concatenate([(f_q - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, np.zeros(7)])
        act_init = np.zeros(cfg.ROBOT.ACTION_DIM)
        init_frame = np.concatenate([l_init, f_init, act_init])

        for _ in range(cfg.ROBOT.FRAME_STACK):
            self.obs_history.append(init_frame)

    def get_ee_pose(self, model, data):
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'panda_ee_site')
            pos = data.site_xpos[site_id].copy()
            return pos
        except:
            return np.zeros(3)

    def _predict_and_replace(self, obs_np):
        """PMDC Logic: Use DCNN to roll forward the leader state"""
        # Extract Leader Part from last frame
        leader_part = obs_np[-cfg.ROBOT.RL_OBS_DIM // cfg.ROBOT.FRAME_STACK : ][:15]
        
        delay_sec = leader_part[14]
        delay_steps = int(delay_sec * cfg.CONTROL_FREQ)
        
        if delay_steps > 0:
            current_state = torch.tensor(leader_part[:14], dtype=torch.float32, device=self.device).unsqueeze(0)
            dummy_action = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                pred_state = current_state
                # Limit prediction horizon 
                steps_to_predict = min(delay_steps, 50) 
                for _ in range(steps_to_predict):
                    pred_state = self.dcnn(pred_state, dummy_action)
            
            pred_np = pred_state.cpu().numpy().squeeze()
            
            # Replace in the observation (Last Frame only)
            start_idx = len(obs_np) - (43) 
            obs_np[start_idx : start_idx+14] = pred_np
            obs_np[start_idx+14] = 0.0 # Set delay to 0
            
        return obs_np

    def control_loop(self):
        # 1. Update Simulators
        l_q, l_qd, _, _, _, _, _ = self.leader.step()
        self.leader_hist.append((l_q, l_qd))
        
        # [VISUALIZATION] Update "Red Ghost"
        if hasattr(self.follower, 'update_leader_view'):
            self.follower.update_leader_view(l_q)
        
        ref_q, ref_qd, ref_delay = self.delay_sim.get_delayed_state(self.leader_hist)
        f_q, f_qd = self.follower.get_joint_state()

        # 2. Build Observation
        l_vec = np.concatenate([
            (ref_q - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, 
            (ref_qd - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD, 
            [ref_delay]
        ])
        f_vec = np.concatenate([
            (f_q - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD, 
            (f_qd - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD
        ])
        frame = np.concatenate([l_vec, f_vec, self.last_action])
        self.obs_history.append(frame)
        
        obs_flat = np.concatenate(self.obs_history).astype(np.float32)
        
        # [PMDC Step] Predict Undelayed State
        obs_corrected = self._predict_and_replace(obs_flat.copy())

        obs_tensor = torch.as_tensor(obs_corrected, device=self.device).unsqueeze(0)

        # 3. RL Inference
        with torch.no_grad():
            mu, _ = self.actor(obs_tensor)
            action_norm = torch.tanh(mu).cpu().numpy().squeeze()

        # 4. Calculate PD Torques
        kp = (action_norm[:7] + 1)/2 * (cfg.SBSP.KP_MAX - cfg.SBSP.KP_MIN) + cfg.SBSP.KP_MIN
        kd = (action_norm[7:] + 1)/2 * (cfg.SBSP.KD_MAX - cfg.SBSP.KD_MIN) + cfg.SBSP.KD_MIN

        # NOTE: Gravity is handled internally by FollowerRobotSimulator
        tau_pd = (kp * (ref_q - f_q)) + (kd * (ref_qd - f_qd))
        
        # 5. Step Follower
        self.follower.step(tau_pd)
        self.last_action = action_norm

        # 6. Logging & ROS
        self.leader.data.qpos[:7] = l_q
        mujoco.mj_kinematics(self.leader.model, self.leader.data)
        l_pos = self.get_ee_pose(self.leader.model, self.leader.data)
        f_pos = self.get_ee_pose(self.follower.model, self.follower.data)
        
        current_time = self.step_count * self.dt
        ee_error = np.linalg.norm(l_pos - f_pos)

        # CSV
        row = [current_time, ee_error]
        row.extend(l_pos.tolist())
        row.extend(f_pos.tolist())
        row.extend(l_q.tolist())
        row.extend(f_q.tolist())
        row.extend(kp.tolist())
        row.extend(kd.tolist())
        self.csv_writer.writerow(row)

        # ROS
        now = self.get_clock().now().to_msg()
        
        p_l = PoseStamped()
        p_l.header.stamp = now
        p_l.header.frame_id = "world"
        p_l.pose.position.x, p_l.pose.position.y, p_l.pose.position.z = l_pos
        self.pub_leader_pose.publish(p_l)

        p_f = PoseStamped()
        p_f.header.stamp = now
        p_f.header.frame_id = "world"
        p_f.pose.position.x, p_f.pose.position.y, p_f.pose.position.z = f_pos
        self.pub_follower_pose.publish(p_f)

        t_msg = Float64()
        t_msg.data = current_time
        self.pub_time.publish(t_msg)
        
        err_msg = Float64()
        err_msg.data = ee_error
        self.pub_err.publish(err_msg)

        self.step_count += 1
        
        if self.step_count % 50 == 0:
            print(f"Time: {current_time:.1f}s | EE Error: {ee_error:.4f}m", end='\r')

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    # [ADDED] ARGUMENT PARSING
    parser = argparse.ArgumentParser(description="SBSP Eval Node")
    parser.add_argument("--config", type=int, default=3, help="Delay Config ID (0=No, 1=Low, 2=High, 3=Variance)")
    
    # We use parse_known_args to ensure we don't break ROS2 args if mixed
    parsed_args, unknown_args = parser.parse_known_args()

    rclpy.init(args=args)
    
    # Pass config to node
    node = DataRecorderNode(delay_config_int=parsed_args.config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopping evaluation...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Data saved.")

if __name__ == '__main__':
    main()