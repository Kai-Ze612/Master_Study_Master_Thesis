"""Remote_Robot
Remote robot node that receives position commands from local robot, both robots apply traditional PD control under stochastic delays.
This is the baseline for comparison with RL controllers.

To run this node:
1. cd /media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller
2. colcon build --packages-select franka_mujoco_controller
3. source install/setup.bash
4. ros2 launch franka_mujoco_controller teleoperation_2.launch.py
5. In a separate terminal input position commands: ros2 topic pub --once /local_robot/cartesian_commands geometry_msgs/msg/Point "{x: 0.5, y: 0.2, z: 0.3}"
6. Record trajectory data: ros2 bag record /local_robot/ee_pose /remote_robot/ee_pose -o trajectory_data
"""

#!/usr/bin/env python3
import time
import threading # For running simulation in a separate thread
import os
import numpy as np
from scipy.optimize import minimize
from collections import deque

# Ros2 libraries
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point

# MuJoCo libraries
import mujoco
import mujoco.viewer

os.environ["MUJOCO_GL"] = "egl"

class StochasticDelaySimulator:
    def __init__(self):
        # Corresponds to the constant 80ms action delay (80ms / 10ms per step = 8 steps)
        self.action_delay_steps = 8

        # For a stochastic observation delay of 10ms-50ms (1-5 steps)
        # We set a base delay and a jitter range to achieve this.
        self.base_obs_delay = 3
        self.obs_jitter = 2

        self.current_obs_delay = self.base_obs_delay

    # Creating randomized observation delay
    def update_observation_delay(self):
        jitter = np.random.randint(-self.obs_jitter, self.obs_jitter + 1)
        self.current_obs_delay = max(1, self.base_obs_delay + jitter)

    def get_delays(self):
        return self.action_delay_steps, self.current_obs_delay

class RemoteRobotController(Node):
    """ Controls the remote robot with delay compensation and smoothing. """
    def __init__(self):
        super().__init__("remote_robot_controller")

        self._init_parameters()
        self._init_delay_simulation()
        self._load_mujoco_model()
        self._init_ros_interfaces()
        self._start_simulation()

    def _init_parameters(self):
        """ Initialize all control and simulation parameters. """
        self.model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller/models/franka_fr3/fr3_remote.xml"
        self.control_frequency = 50  # Hz for the physics simulation loop
        self.step_time = 0.02     # Seconds per logical step for the delay simulation

        # Smoothing factor for target commands (lower is smoother)
        self.smoothing_factor = 0.4

        # PD Controller Gains for joint control
        self.kp = np.array([120, 120, 120, 80, 80, 60, 60]) # Proportional gains for each joint (stiffness)
        self.kd = np.array([20, 20, 20, 15, 15, 12, 12]) # Derivative gains for each joint (damping)
        self.force_limit = np.array([80.0, 80.0, 80.0, 60.0, 60.0, 40.0, 40.0])

    def _init_delay_simulation(self):
        """ Initialize components for managing network delays. """
        self.delay_simulator = StochasticDelaySimulator()
        self.observation_buffer = deque(maxlen=200) # Buffer for incoming poses
        self.command_buffer = deque(maxlen=50)      # Buffer for pending commands to execute
        self.current_step = 0

    def _load_mujoco_model(self):
        """ Load the MuJoCo model, data, and launch the viewer. """
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Set a stable initial pose for the robot
        initial_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.data.qpos[:7] = initial_qpos

        # Initialize the target for the PD controller
        self.active_target_joints = initial_qpos.copy()
        mujoco.mj_forward(self.model, self.data)

        # Launch the passive viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _init_ros_interfaces(self):
        """ Initialize all ROS 2 publishers, subscribers, and timers. """
        self.subscription = self.create_subscription(
            PoseStamped, "/local_robot/ee_pose", self.pose_callback, 10
        )
        
        self.ee_publisher = self.create_publisher(PoseStamped, "/remote_robot/ee_pose", 10)

        # A timer to advance the logical step counter for delays
        self.step_timer = self.create_timer(self.step_time, self.increment_step)
        # A timer to publish the robot's state at 30 Hz
        self.state_publisher_timer = self.create_timer(1.0 / 30.0, self.publish_ee_state)

    def _start_simulation(self):
        """ Start the main simulation loop in a separate thread. """
        self.simulation_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.simulation_thread.start()

    # --- ROS 2 Callbacks ---
    def pose_callback(self, msg):
        """ Callback for receiving pose messages from the local robot. """
        position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.observation_buffer.append(position)

    def increment_step(self):
        """ Advances the simulation step counter. """
        self.current_step += 1

    def publish_ee_state(self):
        """ Publishes the current end-effector pose of the remote robot. """
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
        """ Applies delays, solves IK, and queues commands. """
        self.delay_simulator.update_observation_delay()
        action_delay_steps, obs_delay_steps = self.delay_simulator.get_delays()

        if len(self.command_buffer) >= self.command_buffer.maxlen:
            self.get_logger().warn("Command buffer full, dropping new command.", throttle_duration_sec=1)
            return

        if len(self.observation_buffer) >= obs_delay_steps:
            delayed_position = self.observation_buffer.popleft()
            target_joints = self.solve_inverse_kinematics(delayed_position)
            if target_joints is not None:
                execution_step = self.current_step + action_delay_steps
                self.command_buffer.append({"joints": target_joints, "execute_at": execution_step})

    def execute_delayed_commands(self):
        """ Executes queued commands and applies smoothing. """
        if self.command_buffer and self.command_buffer[0]["execute_at"] <= self.current_step:
            command = self.command_buffer.popleft()
            new_target = command["joints"]
            # Apply smoothing filter to the target
            self.active_target_joints = (self.smoothing_factor * new_target) + \
                                        ((1 - self.smoothing_factor) * self.active_target_joints)

    def solve_inverse_kinematics(self, target_pos):
        """ Solves IK to find joint angles for a Cartesian target. """
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
        """ Computes PD control torques to move towards the smoothed target. """
        position_error = self.active_target_joints - self.data.qpos[:7]
        velocity_error = -self.data.qvel[:7]
        torques = self.kp * position_error + self.kd * velocity_error
        return np.clip(torques, -self.force_limit, self.force_limit)

    # --- Main Simulation Thread ---
    def simulation_loop(self):
        """ The main control and physics loop. """
        while rclpy.ok():
            self.process_observations()
            self.execute_delayed_commands()
            self.data.ctrl[:7] = self.compute_pd_torques()
            mujoco.mj_step(self.model, self.data)
            if self.viewer.is_running():
                self.viewer.sync()
            time.sleep(1.0 / self.control_frequency)

def main(args=None):
    rclpy.init(args=args)
    try:
        remote_controller = RemoteRobotController()
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