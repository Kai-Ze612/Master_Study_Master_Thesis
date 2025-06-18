"""_summary_
Remote robot node that receives position commands from local robot
This is where your RL adaptive PD control and delay compensation will be implemented
"""

#!/usr/bin/env python3

# Python libraries
import time
import threading
import os
import numpy as np
from scipy.optimize import minimize

# Ros2 libraries
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseStamped, Point

# MuJoCo libraries
import mujoco
import mujoco.viewer

# Set different rendering options
os.environ['MUJOCO_GL'] = 'egl'

class RemoteRobotController(Node):
    def __init__(self):
        super().__init__('remote_robot_controller')
        
        self._init_parameters()
        self._load_mujoco_model()
        self._init_ros_interfaces()
        self._start_simulation()
    
    def _init_parameters(self):
        """Initialize controller parameters."""
        # Joint names for remote robot
        self.joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        
        self.control_freq = 500
        self.publish_freq = 100
        
        # PD Control gains - THESE WILL BE ADAPTED BY YOUR RL SYSTEM
        self.kp = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        self.kd = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        
        # Target joint positions
        self.target_positions = np.zeros(7)
        
        # Force limits
        self.force_limit = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        
        # Joint limits
        self.joint_limits_lower = np.array([
            -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
        ])
        self.joint_limits_upper = np.array([
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
        ])
        
        # Remote robot model path (single robot)
        self.model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller/models/franka_fr3/fr3.xml"
        
        # Delay compensation variables (for your RL system)
        self.last_command_time = time.time()
        self.delay_pattern = "unknown"
        self.adaptive_gains_enabled = True
    
    def _load_mujoco_model(self):
        """Load and initialize the MuJoCo model."""
        self.get_logger().info(f'Loading REMOTE robot model from: {self.model_path}')
        
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as e:
            self.get_logger().warn(f"Could not create viewer: {e}")
            self.viewer = None
    
    def _init_ros_interfaces(self):
        """Initialize ROS2 publishers and subscribers."""
        
        # Subscriber - Receive position commands FROM LOCAL robot
        self.position_cmd_sub = self.create_subscription(
            Point, '/local_to_remote/position_commands',
            self.position_command_callback, 10)
        
        # Publishers for REMOTE robot state
        self.joint_state_pub = self.create_publisher(
            JointState, '/remote_robot/joint_states', 10)
        self.ee_pose_pub = self.create_publisher(
            PoseStamped, '/remote_robot/ee_pose', 10)
        
        # Publisher for feedback TO LOCAL robot (optional)
        self.status_pub = self.create_publisher(
            String, '/remote_to_local/status_feedback', 10)
        
        # Timer
        self.timer = self.create_timer(1.0/self.publish_freq, self.publish_states)
    
    def position_command_callback(self, msg):
        """
        Receive position commands from local robot (with potential network delays)
        THIS IS WHERE YOUR RL DELAY COMPENSATION HAPPENS
        """
        current_time = time.time()
        command_delay = current_time - self.last_command_time
        self.last_command_time = current_time
        
        target_position = np.array([msg.x, msg.y, msg.z])
        
        self.get_logger().info(f'REMOTE: Received position command: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]')
        self.get_logger().info(f'REMOTE: Command delay: {command_delay:.3f}s')
        
        # === YOUR NEURAL NETWORK DELAY CLASSIFICATION GOES HERE ===
        self.delay_pattern = self.classify_delay_pattern(command_delay)
        
        # === YOUR RL ADAPTIVE PD CONTROL GOES HERE ===
        if self.adaptive_gains_enabled:
            self.adapt_pd_gains(self.delay_pattern, command_delay)
        
        # Solve IK for remote robot
        target_joints = self.inverse_kinematics(target_position)
        
        if target_joints is not None:
            self.target_positions = target_joints
            self.get_logger().info(f'REMOTE: Moving to target with adapted gains: kp_avg={np.mean(self.kp):.1f}, kd_avg={np.mean(self.kd):.1f}')
            
            # Send status feedback to local robot
            self.publish_status("command_received")
        else:
            self.get_logger().warn('REMOTE: IK solution failed')
            self.publish_status("ik_failed")
    
    def classify_delay_pattern(self, command_delay):
        """
        Classify delay patterns using neural network
        THIS IS WHERE YOUR THESIS RESEARCH GOES
        """
        # TODO: Implement your neural network delay classifier here
        # For now, simple threshold-based classification
        
        if command_delay < 0.05:  # < 50ms
            return "low_delay"
        elif command_delay < 0.15:  # 50-150ms
            return "medium_delay"
        elif command_delay < 0.3:  # 150-300ms
            return "high_delay"
        else:
            return "very_high_delay"
    
    def adapt_pd_gains(self, delay_pattern, command_delay):
        """
        Adapt PD control gains based on delay pattern using RL
        THIS IS WHERE YOUR THESIS RESEARCH GOES
        """
        # TODO: Implement your RL-based adaptive PD control here
        # For now, simple rule-based adaptation
        
        base_kp = 100.0
        base_kd = 10.0
        
        if delay_pattern == "low_delay":
            # Normal gains for low delay
            gain_multiplier_p = 1.0
            gain_multiplier_d = 1.0
        elif delay_pattern == "medium_delay":
            # Reduce gains slightly for medium delay
            gain_multiplier_p = 0.8
            gain_multiplier_d = 1.2
        elif delay_pattern == "high_delay":
            # Reduce proportional, increase derivative for high delay
            gain_multiplier_p = 0.6
            gain_multiplier_d = 1.5
        else:  # very_high_delay
            # Conservative gains for very high delay
            gain_multiplier_p = 0.4
            gain_multiplier_d = 2.0
        
        # Apply adaptive gains
        self.kp = np.full(7, base_kp * gain_multiplier_p)
        self.kd = np.full(7, base_kd * gain_multiplier_d)
        
        self.get_logger().info(f'REMOTE: Adapted gains for {delay_pattern}: kp={base_kp * gain_multiplier_p:.1f}, kd={base_kd * gain_multiplier_d:.1f}')
    
    def publish_status(self, status):
        """Send status feedback to local robot."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
    
    def inverse_kinematics(self, target_position):
        """Solve inverse kinematics using optimization"""
        def objective_function(joint_angles):
            temp_data = mujoco.MjData(self.model)
            temp_data.qpos[:7] = joint_angles
            mujoco.mj_fwdPosition(self.model, temp_data)
            
            ee_id = self.model.body('fr3_link7').id
            current_ee_pos = temp_data.xpos[ee_id]
            
            error = np.linalg.norm(current_ee_pos - target_position)
            return error
        
        initial_guess = self.data.qpos[:7].copy()
        bounds = [  
            (-2.8973, 2.8973),   # Joint 1
            (-1.7628, 1.7628),   # Joint 2
            (-2.8973, 2.8973),   # Joint 3
            (-3.0718, -0.0698),  # Joint 4
            (-2.8973, 2.8973),   # Joint 5
            (-0.0175, 3.7525),   # Joint 6
            (-2.8973, 2.8973)    # Joint 7
        ]
     
        try:
            result = minimize(objective_function, initial_guess, method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 100})
            
            if result.success and result.fun < 0.02:
                self.get_logger().info(f'REMOTE: IK solved! Error: {result.fun:.4f}m')
                return result.x
            else:
                self.get_logger().warn(f'REMOTE: IK failed. Error: {result.fun:.4f}m')
                return None
        except Exception as e:
            self.get_logger().error(f'REMOTE: IK optimization failed: {e}')
            return None
    
    def compute_pd_torques(self):
        """
        Compute PD control torques with adaptive gains
        THIS IS WHERE YOUR ADAPTIVE CONTROL IS APPLIED
        """
        current_positions = self.data.qpos[:7]
        current_velocities = self.data.qvel[:7]
        
        # PD control equation with adaptive gains
        position_error = self.target_positions - current_positions
        torques = self.kp * position_error - self.kd * current_velocities
        
        # Apply force limits
        torques = np.clip(torques, -self.force_limit, self.force_limit)
        return torques
    
    def publish_states(self):
        """Publish joint states and end-effector pose."""
        current_time = self.get_clock().now().to_msg()
        self._publish_joint_states(current_time)
        self._publish_ee_pose(current_time)
    
    def _publish_joint_states(self, timestamp):
        """Publish joint states."""
        joint_state = JointState()
        joint_state.header.stamp = timestamp
        joint_state.name = self.joint_names
        joint_state.position = self.data.qpos[:7].tolist()
        joint_state.velocity = self.data.qvel[:7].tolist()
        joint_state.effort = self.data.qfrc_applied[:7].tolist()
        
        self.joint_state_pub.publish(joint_state)
    
    def _publish_ee_pose(self, timestamp):
        """Publish end-effector pose."""
        ee_pose = PoseStamped()
        ee_pose.header.stamp = timestamp
        ee_pose.header.frame_id = "world"
        
        try:
            ee_id = self.model.body('fr3_link7').id
            ee_pos = self.data.xpos[ee_id]
            ee_quat = self.data.xquat[ee_id]
            
            ee_pose.pose.position.x = float(ee_pos[0])
            ee_pose.pose.position.y = float(ee_pos[1])
            ee_pose.pose.position.z = float(ee_pos[2])
            
            ee_pose.pose.orientation.w = float(ee_quat[0])
            ee_pose.pose.orientation.x = float(ee_quat[1])
            ee_pose.pose.orientation.y = float(ee_quat[2])
            ee_pose.pose.orientation.z = float(ee_quat[3])
            
            self.ee_pose_pub.publish(ee_pose)
        except Exception as e:
            self.get_logger().warn(f'Could not publish REMOTE EE pose: {e}')
    
    def _start_simulation(self):
        """Start the MuJoCo simulation thread."""
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def simulation_loop(self):
        """Main simulation loop with adaptive PD control."""
        while rclpy.ok():
            # Compute torques with adaptive PD control
            torques = self.compute_pd_torques()
            self.data.ctrl[:7] = torques
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Update viewer
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()
            
            time.sleep(1.0/self.control_freq)

def main(args=None):
    """ROS2 main entry point."""
    rclpy.init(args=args)
    
    try:
        controller = RemoteRobotController() 
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()