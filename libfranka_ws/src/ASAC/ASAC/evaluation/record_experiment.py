#!/usr/bin/env python3
"""
ROS2 Evaluation & Data Recording Node (With Visualization)
----------------------------------------------------------
Frequency: 50 Hz
Features:
1. Runs E2E Simulation with MuJoCo Visualization.
2. VISUALIZATION: Updates a 'target' site (Red) to match Leader Trajectory.
3. Publishes ROS2 topics.
4. Records all data to CSV in 'ASAC/ASAC/evaluation/data/'.

Usage:
    python3 -m ASAC.evaluation.record_experiment
"""

import os
import sys
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
import ASAC.config.robot_config as cfg
from ASAC.leader_robot_simulator import LeaderRobotSimulator, TrajectoryType
from ASAC.follower_robot_simulator import FollowerRobotSimulator
from ASAC.utils.delay_simulator import DelaySimulator, ExperimentConfig
from ASAC.asac_network import GainTuningActor

class DataRecorderNode(Node):
    def __init__(self):
        super().__init__('asac_data_recorder')

        # --- 1. Settings ---
        self.control_rate = 50.0  # Hz
        self.dt = 1.0 / self.control_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Directories
        self.package_root = cfg.PYTHON_PACKAGE_ROOT
        self.data_dir = self.package_root / "evaluation" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file_path = self.data_dir / f"experiment_{timestamp}.csv"
        self.csv_file = open(self.csv_file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Define CSV Header
        header = ['time']
        header += [f'leader_ee_pos_{x}' for x in ['x','y','z']]
        header += [f'leader_ee_quat_{x}' for x in ['x','y','z','w']]
        header += [f'follower_ee_pos_{x}' for x in ['x','y','z']]
        header += [f'follower_ee_quat_{x}' for x in ['x','y','z','w']]
        header += [f'leader_q_{i}' for i in range(7)]
        header += [f'follower_q_{i}' for i in range(7)]
        self.csv_writer.writerow(header)
        
        self.get_logger().info(f"Recording data to: {self.csv_file_path}")

        # --- 2. Load Model ---
        self.checkpoint_dir = cfg.ROBOT.CHECKPOINT_DIR
        self.model_path = self._find_latest_model()
        self.actor = GainTuningActor().to(self.device)
        self.actor.eval()
        self._load_weights()

        # --- 3. Initialize Simulators ---
        self.leader = LeaderRobotSimulator(trajectory_type=TrajectoryType.FIGURE_8, randomize_params=False)
        
        # [MODIFIED] render=True enables the MuJoCo viewer window
        self.follower = FollowerRobotSimulator(
            delay_config=ExperimentConfig.HIGH_VARIANCE, 
            render=True,       # <--- ENABLE VISUALIZATION
            render_fps=50      # Sync render FPS with control loop
        )
        self.delay_sim = DelaySimulator(cfg.CONTROL_FREQ, config=ExperimentConfig.LOW_DELAY)

        # [VISUALIZATION] Find the 'target' site ID to move the Red Ghost
        self.target_site_id = -1
        try:
            # Assumes your XML has a site named "target" (standard in many Franka XMLs)
            # If not, add: <site name="target" pos="0 0 0" size="0.03" rgba="1 0 0 0.5"/> to worldbody
            self.target_site_id = mujoco.mj_name2id(self.follower.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        except:
            self.get_logger().warn("No 'target' site found in XML. Leader visualization (Red Ghost) disabled.")

        # --- 4. State Buffers ---
        self.obs_history = deque(maxlen=cfg.ROBOT.FRAME_STACK)
        self.leader_hist = deque(maxlen=cfg.ROBOT.LEADER_HISTORY_BUFFER_LEN)
        self.last_action = np.zeros(cfg.ROBOT.ACTION_DIM)
        self._warmup_buffers()

        # --- 5. ROS Publishers ---
        self.pub_leader_pose = self.create_publisher(PoseStamped, '/leader/ee_pose', 10)
        self.pub_follower_pose = self.create_publisher(PoseStamped, '/follower/ee_pose', 10)
        self.pub_leader_joints = self.create_publisher(JointState, '/leader/joint_states', 10)
        self.pub_follower_joints = self.create_publisher(JointState, '/follower/joint_states', 10)
        self.pub_time = self.create_publisher(Float64, '/evaluation/time_step', 10)

        # --- 6. Timer Loop ---
        self.step_count = 0
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("Starting Simulation Loop @ 50Hz...")

    def _find_latest_model(self):
        # [UPDATED] Direct path to your 'high_delay' model directory
        model_dir = Path("/media/kai/NewDisk/Kai_thesis/Master_Thesis_E2E_RL_Teleop/libfranka_ws/src/ASAC/ASAC/trained_RL/low_delay")
        
        # Try best, fall back to final
        model_file = model_dir / "best_actor.pth"
        if not model_file.exists():
            model_file = model_dir / "final_checkpoint.pth"
        
        if not model_file.exists():
            self.get_logger().error(f"No model found in {model_dir}")
            sys.exit(1)
            
        self.get_logger().info(f"Using model: {model_file}")
        return model_file

    def _load_weights(self):
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'actor_state_dict' in checkpoint:
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
            else:
                self.actor.load_state_dict(checkpoint)
        except Exception as e:
            self.get_logger().error(f"Failed to load weights: {e}")
            sys.exit(1)

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
        """Extract EE Position (xyz) and Orientation (quat xyzw)"""
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'panda_ee_site')
            pos = data.site_xpos[site_id].copy()
            mat = data.site_xmat[site_id].reshape(3, 3)
            quat = R.from_matrix(mat).as_quat() # x, y, z, w
            return pos, quat
        except:
            return np.zeros(3), np.array([0,0,0,1])

    def control_loop(self):
        # 1. Update Simulators
        l_q, l_qd, _, _, _, _, _ = self.leader.step()
        self.leader_hist.append((l_q, l_qd))
        
        # [VISUALIZATION] Update "Red Ghost" Target Position
        # We compute where the leader IS (using leader sim kinematics)
        self.leader.data.qpos[:7] = l_q
        mujoco.mj_forward(self.leader.model, self.leader.data)
        leader_ee_pos = self.leader.data.site_xpos[self.leader.model.site('panda_ee_site').id].copy()
        
        # Update the 'target' site in the FOLLOWER'S simulation to match Leader
        if self.target_site_id != -1:
            self.follower.data.site_xpos[self.target_site_id] = leader_ee_pos
        
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
        
        obs_tensor = torch.as_tensor(
            np.concatenate(self.obs_history).astype(np.float32), 
            device=self.device
        ).unsqueeze(0)

        # 3. RL Inference
        with torch.no_grad():
            mu, _ = self.actor(obs_tensor)
            action_norm = torch.tanh(mu).cpu().numpy().squeeze()

        # 4. Apply Control
        kp = (action_norm[:7] + 1)/2 * (cfg.PD_GAINS.KP_MAX - cfg.PD_GAINS.KP_MIN) + cfg.PD_GAINS.KP_MIN
        kd = (action_norm[7:] + 1)/2 * (cfg.PD_GAINS.KD_MAX - cfg.PD_GAINS.KD_MIN) + cfg.PD_GAINS.KD_MIN

        self.follower.data.qpos[:7] = f_q
        self.follower.data.qvel[:7] = f_qd
        mujoco.mj_forward(self.follower.model, self.follower.data)
        gravity = self.follower.data.qfrc_bias[:7].copy()

        tau = (kp * (ref_q - f_q)) + (kd * (ref_qd - f_qd)) + gravity
        
        # 5. Step Follower (includes rendering)
        self.follower.step(tau)
        self.last_action = action_norm

        # 6. Get EE Poses for Logging
        # Note: leader_ee_pos is already computed above for visualization
        l_pos = leader_ee_pos 
        # But we need quat too
        l_mat = self.leader.data.site_xmat[self.leader.model.site('panda_ee_site').id].reshape(3, 3)
        l_quat = R.from_matrix(l_mat).as_quat()

        f_pos, f_quat = self.get_ee_pose(self.follower.model, self.follower.data)
        current_time = self.step_count * self.dt

        # --- CSV RECORDING ---
        row = [current_time]
        row.extend(l_pos.tolist())      # Leader xyz
        row.extend(l_quat.tolist())     # Leader quat
        row.extend(f_pos.tolist())      # Follower xyz
        row.extend(f_quat.tolist())     # Follower quat
        row.extend(l_q.tolist())        # Leader Joints
        row.extend(f_q.tolist())        # Follower Joints
        
        self.csv_writer.writerow(row)

        # --- ROS PUBLISHING ---
        now = self.get_clock().now().to_msg()
        
        # Leader Pose
        p_l = PoseStamped()
        p_l.header.stamp = now
        p_l.header.frame_id = "world"
        p_l.pose.position.x, p_l.pose.position.y, p_l.pose.position.z = l_pos
        p_l.pose.orientation.x, p_l.pose.orientation.y, p_l.pose.orientation.z, p_l.pose.orientation.w = l_quat
        self.pub_leader_pose.publish(p_l)

        # Follower Pose
        p_f = PoseStamped()
        p_f.header.stamp = now
        p_f.header.frame_id = "world"
        p_f.pose.position.x, p_f.pose.position.y, p_f.pose.position.z = f_pos
        p_f.pose.orientation.x, p_f.pose.orientation.y, p_f.pose.orientation.z, p_f.pose.orientation.w = f_quat
        self.pub_follower_pose.publish(p_f)

        # Joint States
        js_l = JointState()
        js_l.header.stamp = now
        js_l.name = [f"panda_joint{i+1}" for i in range(7)]
        js_l.position = l_q.tolist()
        self.pub_leader_joints.publish(js_l)

        js_f = JointState()
        js_f.header.stamp = now
        js_f.name = [f"panda_joint{i+1}" for i in range(7)]
        js_f.position = f_q.tolist()
        self.pub_follower_joints.publish(js_f)

        # Time
        t_msg = Float64()
        t_msg.data = current_time
        self.pub_time.publish(t_msg)

        self.step_count += 1
        
        # Optional: Print status every second
        if self.step_count % int(self.control_rate) == 0:
            err = np.linalg.norm(l_pos - f_pos)
            print(f"Time: {current_time:.1f}s | EE Error: {err:.4f}m", end='\r')

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopping recorder...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Data recording saved.")

if __name__ == '__main__':
    main()