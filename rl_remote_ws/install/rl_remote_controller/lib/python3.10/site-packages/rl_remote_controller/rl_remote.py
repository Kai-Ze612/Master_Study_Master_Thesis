"""
Remote Robot Node with RL-based Adaptive PD Control
Integrates trained RL agent for delay-aware control
"""

import time
import threading
import os
import numpy as np
from scipy.optimize import minimize
from collections import deque

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String

# MuJoCo imports
import mujoco
import mujoco.viewer

# RL imports
from stable_baselines3 import PPO, SAC

os.environ['MUJOCO_GL'] = 'egl'

class RL_remote(Node):
    def __init__(self, rl_model_path: str = None):
        super().__init__('simple_rl_remote_robot')
       
        self._init_parameters()
        self._load_mujoco_model()
        self._find_joint_indices()
        self._load_rl_model(rl_model_path)
        self._init_ros_interfaces()
        self._start_simulation()
    
    def _init_parameters(self):
        """Initialize parameters following the paper's approach."""
        
        # Joint names for FR3 robot
        self.joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        
        # Control parameters
        self.control_freq = 500  # 500 Hz control
        self.publish_freq = 100
        
        # Default PD Control gains (will be adapted by RL)
        self.default_kp = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        self.default_kd = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        
        # Current adaptive gains (updated by RL agent)
        self.current_kp = self.default_kp.copy()
        self.current_kd = self.default_kd.copy()
        
        # Force limits
        self.force_limit = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        
        # Target positions (received from local robot)
        self.target_positions = np.zeros(7)
        self.local_positions = np.zeros(7)  # Simulated local robot positions
        
        # Joint limits
        self.joint_limits_lower = np.array([
            -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
        ])
        self.joint_limits_upper = np.array([
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
        ])

        # Delay simulation
        self.action_delay = 3   # 30ms action delay (3 timesteps at 100Hz)
        self.observation_delay = 2  # 20ms observation delay (2 timesteps at 100Hz)
        self.max_delay = 10  # Maximum 100ms as in paper

        # Action buffer: stores incoming actions that have not yet been applied
        # Delayed actions: maintains sequence of actions that will be applied with proper timing
        # maxlen = self.max_delay will keep the buffer size manageable 
        self.action_buffer = deque(maxlen=self.max_delay)
        self.delayed_actions = deque(maxlen=self.max_delay)

        # State history for RL observation
        self.position_error_history = deque(maxlen=10)
        self.control_effort_history = deque(maxlen=10)
        
        # Initialize buffers
        for _ in range(self.max_delay):
            self.action_buffer.append(np.zeros(2))  # [Kp_scale, Kd_scale]
            self.delayed_actions.append(np.zeros(2))
        
        for _ in range(10):
            self.position_error_history.append(0.0)
            self.control_effort_history.append(0.0)
        
        # RL agent
        self.rl_model = None
        self.use_rl_control = False
        
        # Task tracking
        self.current_task = "idle"
        self.task_completion_threshold = 0.05
        
        # Synchronization error tracking (main objective from paper)
        self.sync_error_history = deque(maxlen=100)
        
        # Model path
        self.model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/rl_remote_ws/src/rl_remote_controller/models/franka_fr3/fr3.xml"
        
    def _load_mujoco_model(self):
        """Load and initialize the MuJoCo model."""
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _find_joint_indices(self):
        """Find the indices of the specified joint names in the MuJoCo model."""
        
        self.joint_indices = []
        
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.joint_indices.append(joint_id)
    
    def _load_rl_model(self, model_path: str):
        """Load trained RL model for adaptive PD control."""
        if model_path and os.path.exists(model_path):
            try:
                if "ppo" in model_path.lower():
                    self.rl_model = PPO.load(model_path)
                elif "sac" in model_path.lower():
                    self.rl_model = SAC.load(model_path)
                else:
                    self.get_logger().warn("Unknown RL model type, trying SAC...")
                    self.rl_model = SAC.load(model_path)
                
                self.use_rl_control = True
                self.get_logger().info(f"RL model loaded successfully: {model_path}")
                
            except Exception as e:
                self.get_logger().error(f"Failed to load RL model: {e}")
                self.use_rl_control = False
        else:
            self.get_logger().warn("No RL model provided, using default PD control")
            self.use_rl_control = False

    def _init_ros_interfaces(self):
        """Initialize ROS2 publishers and subscribers."""
        
        # Subscribers - Receive commands from local operator
        self.position_cmd_sub = self.create_subscription(
            Point, '/local_to_remote/position_commands',
            self.position_command_callback, 10)
        
        # Publishers - Send feedback to local operator
        self.status_pub = self.create_publisher(
            String, '/remote_to_local/status_feedback', 10)
        self.pose_pub = self.create_publisher(
            Point, '/remote_to_local/current_pose', 10)
        
        # Publishers for monitoring
        self.joint_state_pub = self.create_publisher(
            JointState, '/remote_robot/joint_states', 10)
        self.ee_pose_pub = self.create_publisher(
            PoseStamped, '/remote_robot/ee_pose', 10)
        
        # Publisher for RL debugging info
        self.rl_debug_pub = self.create_publisher(
            String, '/remote_robot/rl_debug', 10)
        
        # Timers
        self.publish_timer = self.create_timer(1.0/self.publish_freq, self.publish_states)
        self.rl_update_timer = self.create_timer(0.02, self.update_rl_control)  # 50Hz RL updates
    
    def position_command_callback(self, msg):
        """Handle position commands from local operator (simulates local robot position)."""
        target_position = np.array([msg.x, msg.y, msg.z])
        
        self.get_logger().info(f'REMOTE: Received local position: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]')
        
        # Convert Cartesian to joint space (this represents the "local robot" state)
        target_joints = self.inverse_kinematics(target_position)
        
        if target_joints is not None:
            # This simulates receiving the local robot's joint positions
            self.local_positions = target_joints
            
            # For simplicity, we also set this as target (in real system, 
            # remote robot tries to follow local robot)
            self.target_positions = target_joints
            
            self.get_logger().info(f'REMOTE: Following local robot with RL adaptive control')
        else:
            self.get_logger().warn('REMOTE: Could not convert local position')
    
    def update_rl_control(self):
        """Update RL agent to adapt PD gains (core of the paper)."""
        if not self.use_rl_control:
            return
        
        try:
            # Get observation for RL agent (following paper's state representation)
            observation = self._get_rl_observation()
            
            # Get action from RL agent (Kp and Kd scaling factors)
            action, _ = self.rl_model.predict(observation, deterministic=True)
            
            # Apply delay to action (action delay simulation)
            self._apply_action_delay(action)
            
            # Update PD gains based on (possibly delayed) RL action
            delayed_action = self.delayed_actions[0]  # Get oldest action
            self._update_pd_gains(delayed_action)
            
            # Publish debug info
            self._publish_rl_debug(observation, action, delayed_action)
            
        except Exception as e:
            self.get_logger().warn(f"RL update failed: {e}")
    
    def _get_rl_observation(self):
        """
        Get observation vector for RL agent following paper's approach.
        State includes: remote robot state, local robot state (delayed), and action history.
        """
        # Remote robot state (current, no delay)
        remote_positions = self.data.qpos[:7]
        remote_velocities = self.data.qvel[:7]
        
        # Local robot state (with observation delay)
        # In real system, this would be the delayed local robot state
        local_positions_delayed = self.local_positions  # Simplified: no delay simulation here
        
        # Position error (main signal for synchronization)
        position_error = local_positions_delayed - remote_positions
        
        # Action history buffer (augmented state from paper)
        action_history = []
        for action in list(self.action_buffer):
            action_history.extend(action)  # Flatten [Kp_scale, Kd_scale] pairs
        
        # Construct observation following paper's augmented state approach
        observation = np.concatenate([
            remote_positions,                    # Remote robot joint positions (7)
            remote_velocities,                   # Remote robot joint velocities (7)
            local_positions_delayed,             # Local robot positions (delayed) (7)
            position_error,                      # Position synchronization error (7)
            action_history,                      # Action history buffer (2 * max_delay)
            list(self.position_error_history),   # Recent error history (10)
        ])
        
        return observation.astype(np.float32)
    
    def _apply_action_delay(self, action):
        """Apply action delay as described in the paper."""
        # Add current action to buffer
        self.action_buffer.append(action.copy())
        
        # Simulate action delay by shifting the delayed_actions buffer
        self.delayed_actions.popleft()  # Remove oldest
        
        # Add action with delay
        if len(self.action_buffer) >= self.action_delay:
            delayed_action = list(self.action_buffer)[-self.action_delay]
            self.delayed_actions.append(delayed_action.copy())
        else:
            # If buffer not full, use current action (no delay)
            self.delayed_actions.append(action.copy())
    
    def _update_pd_gains(self, action):
        """Update PD gains based on RL agent action (core adaptation mechanism)."""
        # Action should be [Kp_scale, Kd_scale] - scaling factors for gains
        if len(action) >= 2:
            kp_scale = np.clip(action[0], 0.1, 5.0)  # Reasonable bounds
            kd_scale = np.clip(action[1], 0.1, 3.0)
            
            # Apply uniform scaling to all joints (simplified approach)
            self.current_kp = self.default_kp * kp_scale
            self.current_kd = self.default_kd * kd_scale
        else:
            self.get_logger().warn(f"Invalid action size: {len(action)}")
    
    def _publish_rl_debug(self, observation, current_action, delayed_action):
        """Publish RL debugging information."""
        debug_msg = String()
        
        # Calculate synchronization error (main metric from paper)
        sync_error = np.linalg.norm(self.local_positions - self.data.qpos[:7])
        self.sync_error_history.append(sync_error)
        
        avg_sync_error = np.mean(list(self.sync_error_history)[-10:])
        
        debug_msg.data = (
            f"SyncError: {sync_error:.4f}, "
            f"AvgSyncError: {avg_sync_error:.4f}, "
            f"Kp_scale: {delayed_action[0]:.3f}, "
            f"Kd_scale: {delayed_action[1]:.3f}, "
            f"ActionDelay: {self.action_delay}, "
            f"ObsDelay: {self.observation_delay}"
        )
        
        self.rl_debug_pub.publish(debug_msg)
    
    def compute_adaptive_pd_torques(self):
        """Compute PD control torques with RL-adapted gains."""
        current_positions = self.data.qpos[:7]
        current_velocities = self.data.qvel[:7]
        
        # Follow local robot positions (main objective from paper)
        position_error = self.local_positions - current_positions
        
        # Apply adaptive PD control
        torques = self.current_kp * position_error - self.current_kd * current_velocities
        
        # Apply force limits
        torques = np.clip(torques, -self.force_limit, self.force_limit)
        
        # Update history for RL observation
        error_norm = np.linalg.norm(position_error)
        effort_norm = np.linalg.norm(torques)
        
        self.position_error_history.append(error_norm)
        self.control_effort_history.append(effort_norm)
        
        return torques
    
    def inverse_kinematics(self, target_position):
        """Solve inverse kinematics using optimization."""
        
        def objective_function(joint_angles):
            temp_data = mujoco.MjData(self.model)
            temp_data.qpos[:7] = joint_angles
            mujoco.mj_fwdPosition(self.model, temp_data)
            
            try:
                ee_id = self.model.body('fr3_link7').id
                current_ee_pos = temp_data.xpos[ee_id]
                error = np.linalg.norm(current_ee_pos - target_position)
                return error
            except:
                return 1000.0
        
        initial_guess = self.data.qpos[:7].copy()
        bounds = list(zip(self.joint_limits_lower, self.joint_limits_upper))
        
        try:
            result = minimize(objective_function, initial_guess, method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 100})
            
            if result.success and result.fun < 0.02:
                return result.x
            else:
                return None
        except Exception as e:
            self.get_logger().error(f'IK optimization failed: {e}')
            return None

    def publish_states(self):
        """Publish robot states and current pose."""
        current_time = self.get_clock().now().to_msg()
        
        # Publish joint states
        joint_state = JointState()
        joint_state.header.stamp = current_time
        joint_state.name = self.joint_names
        joint_state.position = self.data.qpos[:7].tolist()
        joint_state.velocity = self.data.qvel[:7].tolist()
        joint_state.effort = self.data.qfrc_applied[:7].tolist()
        
        self.joint_state_pub.publish(joint_state)
        
        # Publish current pose feedback to local operator
        try:
            ee_id = self.model.body('fr3_link7').id
            ee_pos = self.data.xpos[ee_id]
            
            pose_msg = Point()
            pose_msg.x = float(ee_pos[0])
            pose_msg.y = float(ee_pos[1])
            pose_msg.z = float(ee_pos[2])
            self.pose_pub.publish(pose_msg)
            
            # Publish detailed EE pose
            ee_quat = self.data.xquat[ee_id]
            
            ee_pose = PoseStamped()
            ee_pose.header.stamp = current_time
            ee_pose.header.frame_id = "world"
            
            ee_pose.pose.position.x = float(ee_pos[0])
            ee_pose.pose.position.y = float(ee_pos[1])
            ee_pose.pose.position.z = float(ee_pos[2])
            
            ee_pose.pose.orientation.w = float(ee_quat[0])
            ee_pose.pose.orientation.x = float(ee_quat[1])
            ee_pose.pose.orientation.y = float(ee_quat[2])
            ee_pose.pose.orientation.z = float(ee_quat[3])
            
            self.ee_pose_pub.publish(ee_pose)
            
        except Exception as e:
            pass

    def _start_simulation(self):
        """Start the MuJoCo simulation thread."""
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def simulation_loop(self):
        """Main simulation loop with RL-adaptive PD control."""
        while rclpy.ok():
            # Compute and apply torques with RL-adaptive PD control
            torques = self.compute_adaptive_pd_torques()
            self.data.ctrl[:7] = torques
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Update viewer
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()
            
            time.sleep(1.0/self.control_freq)


def main(args=None):
    """Simple RL Remote Robot main entry point."""
    rclpy.init(args=args)
    
    rl_model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/rl_remote_ws/src/rl_remote_controller/models/sac_simple_adaptive_pd_final.zip"  # Update this path
    
    remote_robot = RL_remote(rl_model_path)
    rclpy.spin(remote_robot)

    if 'remote_robot' in locals():
        remote_robot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()