"""_summary_
In this file, we explore how to implement a simplest force and position PD controller
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
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, WrenchStamped, Point

# MuJoCo libraries
import mujoco
import mujoco.viewer


# Set different rendering options
# 'egl' is GPU rendering
# 'glfw' is desktop rendering
# 'osmesa' is software CPU rendering
os.environ['MUJOCO_GL'] = 'egl'

class force_position_control(Node):
    def __init__(self):
        super().__init__('franka_mujoco_controller')
        
        self._init_parameters()
        self._load_mujoco_model()
        self._init_ros_interfaces()
        self._start_simulation()
    
    def _init_parameters(self):
        """Initialize controller parameters."""
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7'
        ]
        self.control_freq = 500
        self.publish_freq = 100
        
        # PD Control gains
        self.kp = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        self.kd = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        
        # Fixed: consistent naming
        self.target_positions = np.zeros(7)  # Target joint positions
        
        # Fixed: consistent variable name
        self.force_limit = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        
        # Joint limits
        # It defines the physical constraints that how far each joint can move
        self.joint_limits_lower = np.array([
            -2.8973,   # Joint 1
            -1.7628,   # Joint 2
            -2.8973,   # Joint 3
            -3.0718,   # Joint 4
            -2.8973,   # Joint 5
            -0.0175,   # Joint 6, this value is checked
            -2.8973    # Joint 7
        ])

        self.joint_limits_upper = np.array([
            2.8973,   # Joint 1
            1.7628,   # Joint 2
            2.8973,   # Joint 3
            3.0718,   # Joint 4
            2.8973,   # Joint 5
            3.7525,   # Joint 6
            2.8973    # Joint 7
        ])
        
        self.ee_id = 'fr3_link7'
        self.model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller/models/franka_fr3/fr3_with_moveable_box.xml"
    
    def _load_mujoco_model(self):
        """Load and initialize the MuJoCo model."""
        self.get_logger().info(f'Loading MuJoCo model from: {self.model_path}')
        
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def _init_ros_interfaces(self):
        """Initialize ROS2 publishers and subscribers."""
        
        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.ee_pose_pub = self.create_publisher(PoseStamped, '/ee_pose', 10)
        self.box_pose_pub = self.create_publisher(PoseStamped, '/box_pose', 10)
        self.contact_force_pub = self.create_publisher(WrenchStamped, '/contact_force', 10)
        
        # Subscribers
        self.position_cmd_sub = self.create_subscription(
            Float64MultiArray, '/joint_commands', self.joint_command_callback, 10)
        
        self.cartesian_cmd_sub = self.create_subscription(
            Point, '/cartesian_position_commands', self.cartesian_command_callback, 10)
        
        self.torque_cmd_sub = self.create_subscription(
            Float64MultiArray, '/joint_torque_commands', self.torque_command_callback, 10)
        
        # Timer
        self.timer = self.create_timer(1.0/self.publish_freq, self.publish_states)
    
    def joint_command_callback(self, msg):
        """Receive joint position commands"""
        if len(msg.data) == 7:
            self.target_positions = np.array(msg.data)
        else:
            self.get_logger().warn(f'Expected 7 joint commands, got {len(msg.data)}')
    
    def cartesian_command_callback(self, msg):
        """Receive Cartesian position commands and compute IK"""
        target_position = np.array([msg.x, msg.y, msg.z])
        self.get_logger().info(f'Cartesian target: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]')
        
        target_joint_positions = self.inverse_kinematics(target_position)
        
        ## check if IK solution is found
        if target_joint_positions is not None:
            self.target_positions = target_joint_positions
            self.get_logger().info(f'target_joint_position: {target_joint_positions}')
        else:
            self.get_logger().warn('solutions out of bounds, failed')
    
    # Using optimization to solve inverse kinematics
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
        bounds = [  (-2.8973, 2.8973),  # Joint 1
                    (-1.7628, 1.7628),  # Joint 2
                    (-2.8973, 2.8973),  # Joint 3
                    (-3.0718, -0.0698),  # Joint 4
                    (-2.8973, 2.8973),  # Joint 5
                    (-0.0175, 3.7525),  # Joint 6
                    (-2.8973, 2.8973)]  # Joint 7
     
        try:
            result = minimize(objective_function, initial_guess, method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 100})
            
            if result.success and result.fun < 0.02:
                self.get_logger().info(f'IK solved! Error: {result.fun:.4f}m')
                return result.x
            else:
                self.get_logger().warn(f'IK failed. Error: {result.fun:.4f}m')
                return None
        except Exception as e:
            self.get_logger().error(f'IK optimization failed: {e}')
            return None

    def torque_command_callback(self, msg):
        """Receive direct torque commands"""
        if len(msg.data) == 7:
            torques = np.clip(msg.data, -self.force_limit, self.force_limit)
            self.data.ctrl[:7] = torques
        else:
            self.get_logger().warn(f'Expected 7 torque commands, got {len(msg.data)}')
    
    def compute_pd_torques(self):
        """Compute PD control torques"""
        current_positions = self.data.qpos[:7]
        current_velocities = self.data.qvel[:7]
        
        # PD control equation
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
    
    def _start_simulation(self):
        """Start the MuJoCo simulation thread."""
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def simulation_loop(self):
        """Main simulation loop with PD control."""
        while rclpy.ok():

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
        controller = force_position_control()
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