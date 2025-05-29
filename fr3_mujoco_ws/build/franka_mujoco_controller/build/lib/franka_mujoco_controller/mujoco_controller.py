"""
Franka FR3 MuJoCo Simulation Controller

This ROS2 node provides an interface to simulate a Franka FR3 robot in MuJoCo and
exposes ROS2 publishers and subscribers for control and state monitoring.
"""

import os
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped


class FrankaMuJoCoController(Node):
    """
    ROS2 node for controlling a Franka robot in MuJoCo simulation.
    
    This node loads a Franka FR3 model from MuJoCo Menagerie, simulates it in MuJoCo,
    and provides ROS2 interfaces for joint control and state monitoring.
    """
    
    def __init__(self):
        """Initialize the Franka MuJoCo controller node."""
        super().__init__('franka_mujoco_controller')
        
        # Initialize parameters
        self._init_parameters()
        
        # Load MuJoCo model
        self._load_mujoco_model()
        
        # Initialize ROS2 publishers and subscribers
        self._init_ros_interfaces()
        
        # Start simulation thread
        self._start_simulation()
    
    def _init_parameters(self):
        """Initialize controller parameters."""
        # The joint names in the MuJoCo Menagerie FR3 model
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4', 
            'joint5', 'joint6', 'joint7'
        ]
        self.control_freq = 500  # Hz
        self.publish_freq = 100  # Hz
        
        # Model path
        master_thesis_path = "/media/kai/Kai_Backup/Study/Master_Thesis/My_Master_Thesis"
        self.model_path = os.path.join(
            master_thesis_path, 
            "franka_mujoco_ws", 
            "src", 
            "mujoco_menagerie", 
            "franka_fr3", 
            "fr3.xml"
        )
    
    def _load_mujoco_model(self):
        """Load and initialize the MuJoCo model."""
        # Log the path for debugging
        self.get_logger().info(f'Loading MuJoCo model from: {self.model_path}')
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            self.get_logger().error(f'Model file not found: {self.model_path}')
            self.get_logger().info('Make sure mujoco_menagerie is cloned in your workspace')
            raise FileNotFoundError(f'MuJoCo model not found at {self.model_path}')
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def _init_ros_interfaces(self):
        """Initialize ROS2 publishers and subscribers."""
        # Publishers
        self.joint_state_pub = self.create_publisher(
            JointState, '/joint_states', 10)
        self.ee_pose_pub = self.create_publisher(
            PoseStamped, '/ee_pose', 10)
        
        # Subscribers
        self.cmd_sub = self.create_subscription(
            Float64MultiArray, 
            '/joint_commands', 
            self.joint_command_callback, 
            10)
        
        # Timer for publishing
        self.timer = self.create_timer(1.0/self.publish_freq, self.publish_states)
    
    def _start_simulation(self):
        """Start the MuJoCo simulation thread."""
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def joint_command_callback(self, msg):
        """
        Handle joint command messages.
        
        Args:
            msg (Float64MultiArray): Joint position commands
        """
        if len(msg.data) == 7:
            self.data.ctrl[:7] = msg.data
        else:
            self.get_logger().warn(f'Expected 7 joint commands, got {len(msg.data)}')
    
    def simulation_loop(self):
        """Main simulation loop."""
        while rclpy.ok():
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Update viewer
            if self.viewer.is_running():
                self.viewer.sync()
            
            # Control frequency
            time.sleep(1.0/self.control_freq)
    
    def publish_states(self):
        """Publish joint states and end-effector pose."""
        current_time = self.get_clock().now().to_msg()
        
        # Publish joint states
        self._publish_joint_states(current_time)
        
        # Publish end-effector pose
        self._publish_ee_pose(current_time)
    
    def _publish_joint_states(self, timestamp):
        """
        Publish joint states.
        
        Args:
            timestamp: Current ROS time
        """
        joint_state = JointState()
        joint_state.header.stamp = timestamp
        joint_state.name = self.joint_names
        joint_state.position = self.data.qpos[:7].tolist()
        joint_state.velocity = self.data.qvel[:7].tolist()
        joint_state.effort = self.data.qfrc_applied[:7].tolist()
        
        self.joint_state_pub.publish(joint_state)
    
    def _publish_ee_pose(self, timestamp):
        """
        Publish end-effector pose.
        
        Args:
            timestamp: Current ROS time
        """
        ee_pose = PoseStamped()
        ee_pose.header.stamp = timestamp
        ee_pose.header.frame_id = "world"
        
        # Get end-effector position and orientation
        ee_id = self.model.body('panda_hand').id
        ee_pos = self.data.xpos[ee_id]
        ee_quat = self.data.xquat[ee_id]
        
        # Set position
        ee_pose.pose.position.x = float(ee_pos[0])
        ee_pose.pose.position.y = float(ee_pos[1])
        ee_pose.pose.position.z = float(ee_pos[2])
        
        # Set orientation
        ee_pose.pose.orientation.w = float(ee_quat[0])
        ee_pose.pose.orientation.x = float(ee_quat[1])
        ee_pose.pose.orientation.y = float(ee_quat[2])
        ee_pose.pose.orientation.z = float(ee_quat[3])
        
        self.ee_pose_pub.publish(ee_pose)


def main(args=None):
    """ROS2 main entry point."""
    rclpy.init(args=args)
    
    try:
        controller = FrankaMuJoCoController()
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