"""_Local Robot Controller
Local robot will receive Cartesian position commands and apply PD control to move to the target position.
Will publish the end-effector pose in real- time
"""
#!/usr/bin/env python3

# Python libraries
import time
import threading # For running simulation in a separate thread
import os
import numpy as np
from scipy.optimize import minimize

# Ros2 libraries
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point

# MuJoCo libraries
import mujoco
import mujoco.viewer

# Set different rendering options
os.environ['MUJOCO_GL'] = 'egl'

class LocalRobotController(Node):
    def __init__(self):
        super().__init__('local_robot_controller')

        self._init_parameters()
        self._load_mujoco_model()
        self._init_ros_interfaces()
        self._start_simulation()
    
    def _init_parameters(self):
        """Initialize controller parameters."""
        self.joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
       
        # Control frequency for PD control, this will run the control loop every 0.02 seconds
        self.control_freq = 50 # Hz

        # Publish frequency for joint states and EE pose, this will publish command very 0.1 seconds
        self.publish_freq = 30 # Hz

        # PD Control gains
        self.kp = np.array([120, 120, 120, 80, 80, 60, 60]) # Proportional gains for each joint (stiffness)
        self.kd = np.array([20, 20, 20, 15, 15, 12, 12]) # Derivative gains for each joint (damping)

        # Target joint positions
        # After solving IK, the parameters will be stored here
        self.target_positions = np.zeros(7)
      
        # Force limits
        self.force_limit = np.array([80.0, 80.0, 80.0, 60.0, 60.0, 40.0, 40.0]) # Unit:N
        
        # Joint limits
        self.joint_limits_lower = np.array([
            -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
        ])
        self.joint_limits_upper = np.array([
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
        ])
        
        self.ee_id = 'fr3_link7'
        self.model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller/models/franka_fr3/fr3_local.xml"
    
    # Mujoco model loading and initialization
    def _load_mujoco_model(self):
       
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Set initial joint positions for better starting pose
        initial_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.data.qpos[:7] = initial_joints
        self.target_positions = initial_joints.copy()
        mujoco.mj_forward(self.model, self.data)
        
    # Initialize ROS2 publishers and subscribers
    def _init_ros_interfaces(self):
        
        # publisher for local robot ee position
        # 100 is the queue size, which determines how many messages can be buffered before they are dropped
        # PoseStamped is a message type that contains position and orientation information with a timestamp
        self.ee_pose_pub = self.create_publisher(
            PoseStamped, '/local_robot/ee_pose', 100)
        
        # subscriber for local robot
        # unit is in cartesian coordinates (x, y ,z)
        self.local_cartesian_cmd_sub = self.create_subscription(
            Point, '/local_robot/cartesian_commands',
            self.local_cartesian_command_callback, 100)
        
        # Timer is used to control publish frequency
        # Timer is automatically started when the node is created, we don't need to call it manually
        self.timer = self.create_timer(1.0/self.publish_freq, self.publish_states)
    
    # (subscriber) Receive Cartesian position commands and compute inverse kinematics
    def local_cartesian_command_callback(self, msg):
        target_position = np.array([msg.x, msg.y, msg.z])
        self.get_logger().info(f'target position: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]')
        
        # Solve IK for local robot
        target_joint_positions = self.inverse_kinematics(target_position)
        
        if target_joint_positions is not None:
            self.target_positions = target_joint_positions
            self.get_logger().info(f'LOCAL: Moving to target position')
            
        else:
            self.get_logger().warn('LOCAL: IK solution failed')
    
    # Inverse Kinematics using optimization method
    def inverse_kinematics(self, target_position):
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
                self.get_logger().info(f'LOCAL: IK solved! Error: {result.fun:.4f}m')
                return result.x
            else:
                self.get_logger().warn(f'LOCAL: IK failed. Error: {result.fun:.4f}m')
                return None
        except Exception as e:
            self.get_logger().error(f'LOCAL: IK optimization failed: {e}')
            return None
    
    # Compute PD control torques for local robot
    def compute_pd_torques(self):
        current_positions = self.data.qpos[:7]
        current_velocities = self.data.qvel[:7]
        
        # PD control equation
        position_error = self.target_positions - current_positions
        torques = self.kp * position_error - self.kd * current_velocities
        
        # Apply force limits
        torques = np.clip(torques, -self.force_limit, self.force_limit)
        return torques
    
    # (publisher) Publish ee position in frequency of timer setting
    def publish_states(self):
        current_time = self.get_clock().now().to_msg()
        self._publish_ee_pose(current_time)
    
    # Message structure for end-effector pose:
    # - header: contains timestamp and frame_id
    # - pose: contains position (x, y, z) and orientation (quaternion w, x, y, z)
        # - position: Cartesian coordinates of the end-effector
        # - orientation: Quaternion representation of the end-effector's orientation
    def _publish_ee_pose(self, timestamp):
        """Publish end-effector pose."""
        ee_pose = PoseStamped()
        ee_pose.header.stamp = timestamp
        ee_pose.header.frame_id = "world"
        
        ee_id = self.model.body('fr3_link7').id
        ee_pos = self.data.xpos[ee_id]
        ee_quat = self.data.xquat[ee_id]
        
        # Cartesian coordinates - location
        ee_pose.pose.position.x = float(ee_pos[0])
        ee_pose.pose.position.y = float(ee_pos[1])
        ee_pose.pose.position.z = float(ee_pos[2])

        # Quaternion coordinates - orientation
        ee_pose.pose.orientation.w = float(ee_quat[0]) # Scalar part
        ee_pose.pose.orientation.x = float(ee_quat[1]) # Vector part
        ee_pose.pose.orientation.y = float(ee_quat[2]) # Vector part
        ee_pose.pose.orientation.z = float(ee_quat[3]) # Vector part

        self.ee_pose_pub.publish(ee_pose)
    
    # Start the MuJoCo simulation thread
    # A thread is a separate execution that runs at the same time as the main program, which means it will perform the simulation in the background
    def _start_simulation(self):
        """Start the MuJoCo simulation thread."""
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    # Main simulation loop that will run continuously
    def simulation_loop(self):
        while rclpy.ok():
            torques = self.compute_pd_torques()
            self.data.ctrl[:7] = torques
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Update viewer
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()

            time.sleep(1.0/self.control_freq)

# Main function to initialize ROS2 and start the controller node
def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = LocalRobotController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()

## We don't need to call the main function because setup.py will automatically call it when the package is run
## But we keep it for testing purposes
# if __name__ == '__main__':
#     main()