#!/usr/bin/env python3
"""
Remote_Robot with RL-Powered Adaptive PD Control
This node loads a trained RL agent to adapt PD gains in real-time
to compensate for stochastic delays.
"""

import time
import threading
import os
import numpy as np
from collections import deque

# ROS 2 and MuJoCo libraries
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import mujoco
import mujoco.viewer

from stable_baselines3 import SAC

os.environ["MUJOCO_GL"] = "egl"

class StochasticDelaySimulator:
    def __init__(self):
        self.action_delay_steps = 8
        self.base_obs_delay = 3
        self.obs_jitter = 2
        self.current_obs_delay = self.base_obs_delay

    def update_observation_delay(self):
        jitter = np.random.randint(-self.obs_jitter, self.obs_jitter + 1)
        self.current_obs_delay = max(1, self.base_obs_delay + jitter)

    def get_delays(self):
        return self.action_delay_steps, self.current_obs_delay

class RemoteRobotRLController(Node):
    def __init__(self):
        super().__init__("remote_robot_rl_controller")
        self._init_parameters()
        self._init_rl_agent()
        self._init_delay_simulation()
        self._load_mujoco_model()
        self._init_ros_interfaces()
        self._start_simulation()
        self.get_logger().info("Remote Robot with RL Adaptive Control has started.")

    def _init_parameters(self):
        """ Initialize simulation and control parameters. """
        self.model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/rl_remote_ws/src/rl_remote_controller/models/franka_fr3/fr3.xml"
        self.control_frequency = 50
        self.step_time = 0.02

        # Default PD gains that the RL agent will scale
        self.default_kp = np.array([100.0] * 7)
        self.default_kd = np.array([10.0] * 7)
        self.kp = self.default_kp.copy() # Kp will be updated by RL
        self.kd = self.default_kd.copy() # Kd will be updated by RL
        self.force_limit = np.array([80.0, 80.0, 80.0, 60.0, 60.0, 40.0, 40.0])

    def _init_rl_agent(self):
        # IMPORTANT: Update this path to your best-performing model
        self.rl_model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/rl_remote_ws/src/rl_remote_controller/rl_remote_controller/models/sac_exp1_90-130ms_best/best_model.zip"
        
        if not os.path.exists(self.rl_model_path):
            self.get_logger().error(f"RL model not found at: {self.rl_model_path}")
            self.get_logger().error("Shutting down. Please update the rl_model_path.")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Loading trained RL model from: {self.rl_model_path}")
        self.rl_model = SAC.load(self.rl_model_path)
        
        # --- NEW: Buffers required for the RL agent's observation space ---
        self.rl_action_history = deque(maxlen=50) # To store Kp/Kd scalers
        self.error_history = deque(maxlen=10)     # To store sync error history

    def _init_delay_simulation(self):
        """ Initialize components for managing network delays. """
        self.delay_simulator = StochasticDelaySimulator()
        self.observation_buffer = deque(maxlen=200)
        self.command_buffer = deque(maxlen=50)
        self.current_step = 0
        
        # Initialize RL-related buffers with default values
        for _ in range(50): self.rl_action_history.append(np.array([1.0, 1.0]))
        for _ in range(10): self.error_history.append(0.0)

    def _load_mujoco_model(self):
        """ Load the MuJoCo model and initialize the simulation. """
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        initial_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.data.qpos[:7] = initial_qpos
        self.active_target_joints = initial_qpos.copy()
        mujoco.mj_forward(self.model, self.data)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _init_ros_interfaces(self):
        self.subscription = self.create_subscription(PoseStamped, "/local_robot/ee_pose", self.pose_callback, 10)
        self.ee_publisher = self.create_publisher(PoseStamped, "/remote_robot/ee_pose", 10)
        self.step_timer = self.create_timer(self.step_time, self.increment_step)
        self.state_publisher_timer = self.create_timer(1.0 / 30.0, self.publish_ee_state)

    def _start_simulation(self):
        self.simulation_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.simulation_thread.start()

    def pose_callback(self, msg):
        position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.observation_buffer.append(position)

    def increment_step(self):
        self.current_step += 1

    def publish_ee_state(self):
        ee_id = self.model.body("fr3_link7").id
        pos = self.data.xpos[ee_id]
        quat = self.data.xquat[ee_id]
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = float(pos[0]), float(pos[1]), float(pos[2])
        msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        self.ee_publisher.publish(msg)

    def process_observations(self):
        self.delay_simulator.update_observation_delay()
        action_delay_steps, obs_delay_steps = self.delay_simulator.get_delays()
        if len(self.observation_buffer) >= obs_delay_steps:
            delayed_position = self.observation_buffer.popleft()
            target_joints = self.solve_inverse_kinematics(delayed_position)
            if target_joints is not None:
                execution_step = self.current_step + action_delay_steps
                self.command_buffer.append({"joints": target_joints, "execute_at": execution_step})

    def execute_delayed_commands(self):
        if self.command_buffer and self.command_buffer[0]["execute_at"] <= self.current_step:
            command = self.command_buffer.popleft()
            self.active_target_joints = command["joints"] # Smoothing is removed as RL handles stability

    def solve_inverse_kinematics(self, target_pos):
        # IK solver remains the same
        def loss_function(q):
            temp_data = mujoco.MjData(self.model)
            temp_data.qpos[:7] = q
            mujoco.mj_fwdPosition(self.model, temp_data)
            return np.linalg.norm(temp_data.xpos[self.model.body("fr3_link7").id] - target_pos)
        initial_guess = self.data.qpos[:7].copy()
        bounds = [(-2.9, 2.9)] * 7
        result = minimize(loss_function, initial_guess, method="L-BFGS-B", bounds=bounds, options={"maxiter": 100, "ftol": 1e-6})
        return result.x if result.success and result.fun < 0.02 else None

    def compute_pd_torques(self):
        """ Computes torques using the gains adapted by the RL agent. """
        position_error = self.active_target_joints - self.data.qpos[:7]
        velocity_error = -self.data.qvel[:7]
        torques = self.kp * position_error - self.kd * velocity_error
        
        # Update error history for the next observation
        self.error_history.append(np.linalg.norm(position_error))
        
        return np.clip(torques, -self.force_limit, self.force_limit)

    def _get_rl_observation(self):
        """ --- NEW: Construct the exact observation the agent was trained on --- """
        remote_pos = self.data.qpos[:7]
        remote_vel = self.data.qvel[:7]
        
        # This requires a 'target' to be available. We use the active target.
        delayed_local_pos = self.active_target_joints
        position_error = delayed_local_pos - remote_pos

        # The stochastic range for exp 1 is 5-1=4. This needs to match the training env.
        stochastic_action_range = 4 
        action_hist_flat = np.array(list(self.rl_action_history)[-stochastic_action_range:]).flatten()

        observation = np.concatenate([
            remote_pos,
            remote_vel,
            delayed_local_pos,
            position_error,
            action_hist_flat,
            list(self.error_history)
        ])
        return observation.astype(np.float32)

    def _update_gains_with_rl(self):
        """ --- NEW: Get action from RL model and update PD gains --- """
        observation = self._get_rl_observation()
        action, _ = self.rl_model.predict(observation, deterministic=True)
        
        # Update Kp and Kd based on the agent's output
        kp_scale, kd_scale = action[0], action[1]
        self.kp = self.default_kp * kp_scale
        self.kd = self.default_kd * kd_scale

        # Store the action for the next observation
        self.rl_action_history.append(action)

    def simulation_loop(self):
        """ The main control loop, now with an RL step. """
        while rclpy.ok():
            self.process_observations()
            self.execute_delayed_commands()
            
            # --- CHANGE: Update gains using RL before computing torques ---
            self._update_gains_with_rl()
            
            self.data.ctrl[:7] = self.compute_pd_torques()
            mujoco.mj_step(self.model, self.data)
            
            if self.viewer.is_running():
                self.viewer.sync()
            
            time.sleep(1.0 / self.control_frequency)

def main(args=None):
    rclpy.init(args=args)
    try:
        remote_controller = RemoteRobotRLController()
        rclpy.spin(remote_controller)
    except KeyboardInterrupt:
        print("Shutting down remote controller.")
    finally:
        if 'remote_controller' in locals() and rclpy.ok():
            remote_controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()