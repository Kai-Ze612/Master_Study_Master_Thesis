#!/usr/bin/env python3
"""
ROS2 Evaluation & Data Recording Node for E2E BC/RL Model
---------------------------------------------------------
Frequency: 50 Hz (or matched to Config)
Usage:
    python3 evaluate_e2e_model.py --model path/to/model.pth --traj FIGURE_8 --delay HIGH_VARIANCE
"""

import os
import sys
import argparse
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
import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import LeaderRobotSimulator, TrajectoryType
from E2E_Teleoperation.E2E_RL.follower_robot_simulator import FollowerRobotSimulator
from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from E2E_Teleoperation.E2E_RL.e2e_network import JointActor

class E2EEvalNode(Node):
    def __init__(self, model_path, trajectory_type, delay_config, max_duration=40.0):
        super().__init__('e2e_data_recorder')

        # --- 1. Settings ---
        self.control_rate = float(cfg.CONTROL_FREQ)
        self.dt = 1.0 / self.control_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_duration = max_duration
        
        # Directories
        self.package_root = cfg.PACKAGE_ROOT
        self.data_dir = self.package_root / "evaluation" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_name = trajectory_type.name.lower()
        self.csv_file_path = self.data_dir / f"eval_{traj_name}_{timestamp}.csv"
        self.csv_file = open(self.csv_file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Define CSV Header
        header = ['time', 'tracking_error']
        header += [f'leader_ee_pos_{x}' for x in ['x','y','z']]
        header += [f'follower_ee_pos_{x}' for x in ['x','y','z']]
        header += [f'leader_q_{i}' for i in range(7)]
        header += [f'follower_q_{i}' for i in range(7)]
        header += [f'action_tau_{i}' for i in range(7)]
        self.csv_writer.writerow(header)
        
        self.get_logger().info(f"Recording data to: {self.csv_file_path}")

        # --- 2. Load Model ---
        self.model_path = Path(model_path)
        self.actor = JointActor().to(self.device)
        self.actor.eval()
        self._load_weights()

        # --- 3. Initialize Simulators ---
        self.leader = LeaderRobotSimulator(
            trajectory_type=trajectory_type, 
            randomize_params=False # Deterministic for evaluation comparison
        )
        
        self.follower = FollowerRobotSimulator(
            delay_config=delay_config, 
            render=True,       
            render_fps=int(self.control_rate)
        )
        self.delay_sim = DelaySimulator(cfg.CONTROL_FREQ, config=delay_config)
        
        # [VISUALIZATION] Find 'target' site for Red Ghost
        self.target_site_id = -1
        try:
            self.target_site_id = mujoco.mj_name2id(self.follower.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        except:
            self.get_logger().warn("No 'target' site found in XML. Leader visualization disabled.")

        # --- 4. Initialize History Buffers ---
        self._init_buffers()

        # --- 5. ROS Publishers ---
        self.pub_leader_pose = self.create_publisher(PoseStamped, '/leader/ee_pose', 10)
        self.pub_follower_pose = self.create_publisher(PoseStamped, '/follower/ee_pose', 10)
        self.pub_time = self.create_publisher(Float64, '/evaluation/time_step', 10)

        # --- 6. Timer Loop ---
        self.step_count = 0
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(f"Starting Evaluation (Max Duration: {self.max_duration}s)...")

    def _load_weights(self):
        if not self.model_path.exists():
            self.get_logger().error(f"Model not found at {self.model_path}")
            sys.exit(1)
            
        try:
            # Safe load for both full checkpoints and state_dicts
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                self.actor.load_state_dict(checkpoint['actor'])
                self.get_logger().info(f"Loaded 'actor' from dictionary (Step {checkpoint.get('global_step', '?')})")
            else:
                self.actor.load_state_dict(checkpoint)
                self.get_logger().info("Loaded raw state dictionary.")
        except Exception as e:
            self.get_logger().error(f"Failed to load weights: {e}")
            sys.exit(1)

    def _init_buffers(self):
        """Initialize all history buffers required by the Network."""
        l_q, _ = self.leader.reset()
        f_q, _ = self.follower.reset()
        self.delay_sim.reset()

        # 1. Leader History (for Delay calculation)
        self.leader_hist = deque(maxlen=200)
        for _ in range(200): 
            self.leader_hist.append((l_q, np.zeros(7)))

        # 2. Follower History (for Network Input)
        self.f_hist_q = deque([f_q]*cfg.ROBOT.RNN_SEQ_LEN, maxlen=cfg.ROBOT.RNN_SEQ_LEN)
        self.f_hist_qd = deque([np.zeros(7)]*cfg.ROBOT.RNN_SEQ_LEN, maxlen=cfg.ROBOT.RNN_SEQ_LEN)
        
        # 3. Action History
        # Initialize with zeros
        self.act_hist = deque(maxlen=cfg.ROBOT.ACTION_HISTORY_LEN)
        for _ in range(cfg.ROBOT.ACTION_HISTORY_LEN):
            self.act_hist.append(np.zeros(7))

    def get_ee_pose(self, model, data):
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'panda_ee_site')
            pos = data.site_xpos[site_id].copy()
            mat = data.site_xmat[site_id].reshape(3, 3)
            quat = R.from_matrix(mat).as_quat() 
            return pos, quat
        except:
            return np.zeros(3), np.array([0,0,0,1])

    def construct_observation(self):
        """Replicates observation logic from training_env.py"""
        seq_data = []
        
        # 1. Build RNN Sequence (History)
        for i in range(cfg.ROBOT.RNN_SEQ_LEN):
            # Get delayed leader state from history buffer
            offset = cfg.ROBOT.RNN_SEQ_LEN - 1 - i
            l_del, ld_del, d_sec = self.delay_sim.get_delayed_state(self.leader_hist, offset_indices=offset)
            
            # Normalize
            l_norm = (l_del - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD
            ld_norm = (ld_del - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD
            f_norm = (self.f_hist_q[i] - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD
            fd_norm = (self.f_hist_qd[i] - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD
            
            # In training_env.py, action history in sequence is taken from the deque
            # Here we simplify similarly to training
            if i < len(self.act_hist):
                a_norm = self.act_hist[i] / cfg.ROBOT.MAX_ACTION_TORQUE
            else:
                a_norm = np.zeros(7)
            
            seq_data.append(np.concatenate([l_norm, ld_norm, [d_sec], f_norm, fd_norm, a_norm]))
        
        target_seq = np.array(seq_data, dtype=np.float32)
        
        # 2. Current State
        f_curr_norm = (self.f_hist_q[-1] - cfg.ROBOT.Q_MEAN)/cfg.ROBOT.Q_STD
        fd_curr_norm = (self.f_hist_qd[-1] - cfg.ROBOT.QD_MEAN)/cfg.ROBOT.QD_STD
        curr_state = np.concatenate([f_curr_norm, fd_curr_norm])
        
        # 3. Action History (Flattened)
        act_hist_arr = np.array(self.act_hist, dtype=np.float32) 
        act_hist_norm = act_hist_arr / cfg.ROBOT.MAX_ACTION_TORQUE
        act_flat = act_hist_norm.flatten()

        # 4. Previous Action
        # [CRITICAL FIX] Matching training_env.py:
        # training_env.py uses self.action_hist[-2] if len > 1
        prev_act = self.act_hist[-2] / cfg.ROBOT.MAX_ACTION_TORQUE
        
        # Combine
        full_obs = np.concatenate([curr_state, target_seq.flatten(), act_flat, prev_act])
        return full_obs

    def control_loop(self):
        current_time = self.step_count * self.dt

        # Check Termination
        if current_time >= self.max_duration:
            self.get_logger().info("Max duration reached. Closing...")
            rclpy.shutdown()
            return

        # --- 1. Step Leader & Update Ghost ---
        l_q, l_qd, _, _, _, _, _ = self.leader.step()
        self.leader_hist.append((l_q, l_qd))
        
        # Visualization: Move Red Ghost to Leader Position
        self.leader.data.qpos[:7] = l_q
        mujoco.mj_forward(self.leader.model, self.leader.data)
        leader_ee_pos = self.leader.data.site_xpos[self.leader.model.site('panda_ee_site').id].copy()
        
        if self.target_site_id != -1:
            self.follower.data.site_xpos[self.target_site_id] = leader_ee_pos

        # --- 2. Inference ---
        obs_np = self.construct_observation()
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            mu, _, _, _, _ = self.actor(obs_tensor)
            # Apply Scaling to get Torque
            action_tau = (torch.tanh(mu) * self.actor.action_scale).cpu().numpy().squeeze()
            
        # [MODIFIED] No manual gravity compensation
        final_torque = action_tau 

        # --- 3. Step Follower ---
        info = self.follower.step(final_torque)
        f_q_new = info['q_follower']
        f_qd_new = info['qd_follower']
        
        # Update History Buffers
        self.f_hist_q.append(f_q_new)
        self.f_hist_qd.append(f_qd_new)
        self.act_hist.append(action_tau) # Store Network Output (PD), not total torque

        # --- 4. Logging & Publishing ---
        f_pos, f_quat = self.get_ee_pose(self.follower.model, self.follower.data)
        l_mat = self.leader.data.site_xmat[self.leader.model.site('panda_ee_site').id].reshape(3, 3)
        l_quat = R.from_matrix(l_mat).as_quat()

        ee_error = np.linalg.norm(leader_ee_pos - f_pos)

        # CSV
        row = [current_time, ee_error]
        row.extend(leader_ee_pos.tolist())
        row.extend(f_pos.tolist())
        row.extend(l_q.tolist())
        row.extend(f_q_new.tolist())
        row.extend(action_tau.tolist())
        self.csv_writer.writerow(row)

        # ROS
        now = self.get_clock().now().to_msg()
        
        def pub_pose(pub, pos, quat):
            p = PoseStamped()
            p.header.stamp = now
            p.header.frame_id = "world"
            p.pose.position.x, p.pose.position.y, p.pose.position.z = pos
            p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = quat
            pub.publish(p)

        pub_pose(self.pub_leader_pose, leader_ee_pos, l_quat)
        pub_pose(self.pub_follower_pose, f_pos, f_quat)

        t_msg = Float64()
        t_msg.data = current_time
        self.pub_time.publish(t_msg)

        self.step_count += 1
        
        if self.step_count % 50 == 0:
            print(f"Time: {current_time:.1f}s | EE Err: {ee_error:.4f}m | Action: {np.mean(np.abs(action_tau)):.2f}", end='\r')

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    # Arg Parsing
    parser = argparse.ArgumentParser(description="Evaluate E2E Model")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--traj", type=str, default="FIGURE_8", choices=["FIGURE_8", "SQUARE", "LISSAJOUS_COMPLEX"], help="Trajectory Type")
    parser.add_argument("--delay", type=str, default="HIGH_VARIANCE", choices=["NO_DELAY", "LOW_DELAY", "HIGH_DELAY", "HIGH_VARIANCE"], help="Delay Config")
    
    # Process args before initializing ROS
    # We filter out ROS-specific args (which start with --ros-args) manually if mixed
    cli_args, ros_args = parser.parse_known_args()
    
    # Setup Enums
    traj_enum = TrajectoryType[cli_args.traj]
    delay_enum = ExperimentConfig[cli_args.delay]
    
    rclpy.init(args=ros_args)
    
    node = E2EEvalNode(
        model_path=cli_args.model,
        trajectory_type=traj_enum,
        delay_config=delay_enum
    )
    
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(f"\nEvaluation Complete. Data saved to {node.csv_file_path}")

if __name__ == '__main__':
    main()