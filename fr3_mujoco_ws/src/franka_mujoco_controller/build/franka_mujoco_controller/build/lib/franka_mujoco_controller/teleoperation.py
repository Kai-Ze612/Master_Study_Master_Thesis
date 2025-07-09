"""
In this file, we explore how to set a teleoperation of two fr3 robots in the same scene
the method is to publish the target postion to the local robot and then copy the end-effector position, make a coordinate transformation to the remote robot
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
from geometry_msgs.msg import PoseStamped, Point

# MuJoCo libraries
import mujoco
import mujoco.viewer

# Set different rendering options
os.environ['MUJOCO_GL'] = 'egl'

class DualRobots(Node):
    def __init__(self):
        super().__init__('dual_robot_controller')
        
        self._init_parameters()
        self._load_mujoco_model()
        self._find_joint_indices()
        self._coordinate_transformation()
        self._init_ros_interfaces()
        self._start_simulation()
    
    def _init_parameters(self):
        """Initialize parameters for the dual robot controller."""
        
        # Joint names for both robots (fr3_joint naming)
        self.local_joint_names = [
            'fr3_joint1_robot1', 'fr3_joint2_robot1', 'fr3_joint3_robot1', 'fr3_joint4_robot1',
            'fr3_joint5_robot1', 'fr3_joint6_robot1', 'fr3_joint7_robot1'
        ]
        
        self.remote_joint_names = [
            'fr3_joint1_robot2', 'fr3_joint2_robot2', 'fr3_joint3_robot2', 'fr3_joint4_robot2',
            'fr3_joint5_robot2', 'fr3_joint6_robot2', 'fr3_joint7_robot2'
        ]
        
        # Control parameters
        self.control_freq = 500
        self.publish_freq = 100
        
        # PD Control gains (same for both robots)
        self.kp = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        self.kd = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        
        # Force limits for both robots
        self.force_limit = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        
        # Target positions for both robots
        self.local_target_positions = np.zeros(7)   # Local robot (leader)
        self.remote_target_positions = np.zeros(7)  # Remote robot (follower)
        
        # Joint limits
        self.joint_limits_lower = np.array([
            -2.8973,   # Joint 1
            -1.7628,   # Joint 2
            -2.8973,   # Joint 3
            -3.0718,   # Joint 4
            -2.8973,   # Joint 5
            -0.0175,   # Joint 6
            -2.8973    # Joint 7
        ])

        self.joint_limits_upper = np.array([
            2.8973,   # Joint 1
            1.7628,   # Joint 2
            2.8973,   # Joint 3
            -0.0698,   # Joint 4
            2.8973,   # Joint 5
            3.7525,   # Joint 6
            2.8973    # Joint 7
        ])
        
        self.action_delay = 3 #delay in seconds before applying actions, only for better visualization
        
        self.model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller/models/franka_fr3/dual_fr3.xml"   
        
    def _load_mujoco_model(self):
        """Load and initialize the MuJoCo model."""
        self.get_logger().info(f'Loading MuJoCo model from: {self.model_path}')
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as e:
            self.get_logger().warn(f"Could not create viewer: {e}")
            self.viewer = None
        
    def _find_joint_indices(self):
        """Find the indices of the specified joint names in the MuJoCo model."""
        
        self.local_joint_indices = []
        self.remote_joint_indices = []
        
        # Find local robot joint indices
        for joint_name in self.local_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.local_joint_indices.append(joint_id)
        
        self.get_logger().info(f'Local joint indices: {self.local_joint_indices}')
        
        # Find remote robot joint indices
        for joint_name in self.remote_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.remote_joint_indices.append(joint_id)
            
        self.get_logger().info(f'Remote joint indices: {self.remote_joint_indices}')
    
    def _coordinate_transformation(self):
        """Calculate the 3D coordinate transformation between robot bases."""
        
        # Step simulation once to get valid positions
        mujoco.mj_step(self.model, self.data)
        
        try:
            # Get 3D base positions of both robots (use correct names from error message)
            local_base_id = self.model.body('base_robot1').id
            remote_base_id = self.model.body('base_robot2').id
            
            local_base_position = self.data.xpos[local_base_id]
            remote_base_position = self.data.xpos[remote_base_id]
            
            # Calculate 3D offset between robot bases
            self.robot_offset = remote_base_position - local_base_position
            
            self.get_logger().info(f'Local robot base position: {local_base_position}')
            self.get_logger().info(f'Remote robot base position: {remote_base_position}')
            self.get_logger().info(f'Robot 3D offset: {self.robot_offset}')
            
        except Exception as e:
            self.get_logger().error(f'Could not find robot base bodies: {e}')
            # Fallback to no offset
            self.robot_offset = np.array([0.0, 0.0, 0.0])

    def _init_ros_interfaces(self):
        """Initialize ROS2 publishers and subscribers for the dual robot controller."""
        
        # Publishers for local robot (leader)
        self.local_joint_state_pub = self.create_publisher(
            JointState, '/local_robot/joint_states', 10)
        self.local_ee_pose_pub = self.create_publisher(
            PoseStamped, '/local_robot/ee_pose', 10)

        # Publishers for remote robot (follower) - for monitoring
        self.remote_joint_state_pub = self.create_publisher(
            JointState, '/remote_robot/joint_states', 10)
        self.remote_ee_pose_pub = self.create_publisher(
            PoseStamped, '/remote_robot/ee_pose', 10)

        # Subscribers for local robot (leader)
        self.local_cartesian_cmd_sub = self.create_subscription(
            Point, '/local_robot/cartesian_commands',
            self.local_cartesian_command_callback, 10)
        
        self.local_joint_cmd_sub = self.create_subscription(
            Float64MultiArray, '/local_robot/joint_commands',
            self.local_joint_command_callback, 10)

        # Timer for publishing states
        self.timer = self.create_timer(1.0/self.publish_freq, self.publish_states)
    
    def local_cartesian_command_callback(self, msg):
        """Handle local robot Cartesian commands."""
        target_position = np.array([msg.x, msg.y, msg.z])
        self.get_logger().info(f'Local robot Cartesian target: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]')
        
        # Store the original Cartesian target for coordinate transformation
        self.current_cartesian_target = target_position
        
        # Solve the inverse kinematics for local robot
        target_joints = self.inverse_kinematics(target_position, robot='local')
        
        if target_joints is not None:
            self.local_target_positions = target_joints
            self.copy_motion_to_remote_via_coordinates()
        else:
            self.get_logger().warn('IK solution not found for local robot')
    
    def local_joint_command_callback(self, msg):
        """Handle local robot joint commands."""
        if len(msg.data) == 7:
            self.local_target_positions = np.array(msg.data)
            # For joint commands, just copy the same joint angles
            self.remote_target_positions = np.array(msg.data)
        else:
            self.get_logger().warn(f'Expected 7 joint commands, got {len(msg.data)}')
   
    def copy_motion_to_remote_via_coordinates(self):
        """Copy local robot motion to remote robot via coordinate transformation."""
        
        # Use the stored Cartesian target directly
        if hasattr(self, 'current_cartesian_target'):
            local_target_position = self.current_cartesian_target
            
            # Apply coordinate transformation (add 3D base offset)
            remote_target_position = local_target_position + self.robot_offset
            
            self.get_logger().info(f'Local target: {local_target_position}')
            self.get_logger().info(f'Remote target: {remote_target_position}')
            
            # Solve IK for remote robot
            remote_target_joints = self.inverse_kinematics(remote_target_position, robot='remote')
            
            if remote_target_joints is not None:
                self.remote_target_positions = remote_target_joints
            else:
                self.get_logger().warn('IK solution not found for remote robot')
        else:
            self.get_logger().warn('No current Cartesian target available')
    
    def inverse_kinematics(self, target_position, robot='local'):
        """Solve inverse kinematics using optimization."""
        
        def objective_function(joint_angles):
            temp_data = mujoco.MjData(self.model)
            
            # Set joint positions based on robot
            if robot == 'local':
                temp_data.qpos[:7] = joint_angles  # For local robot, use first 7 joints
                ee_body_name = 'fr3_link7_robot1'
            else:
                # For remote robot, we need to set the correct joint indices
                temp_data.qpos[:] = self.data.qpos[:]  # Copy current state
                for i, joint_idx in enumerate(self.remote_joint_indices):
                    temp_data.qpos[joint_idx] = joint_angles[i]
                ee_body_name = 'fr3_link7_robot2'
            
            mujoco.mj_fwdPosition(self.model, temp_data)
            
            try:
                ee_id = self.model.body(ee_body_name).id
                current_ee_pos = temp_data.xpos[ee_id]
                error = np.linalg.norm(current_ee_pos - target_position)
                return error
            except Exception as e:
                return 1000.0
        
        # Use current joint positions as initial guess
        if robot == 'local':
            initial_guess = self.data.qpos[:7].copy()
        else:
            initial_guess = np.array([self.data.qpos[i] for i in self.remote_joint_indices])
        
        # Use the same bounds as your working version
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
                self.get_logger().info(f'IK solved for {robot}! Error: {result.fun:.4f}m')
                return result.x
            else:
                self.get_logger().warn(f'IK failed for {robot}. Error: {result.fun:.4f}m')
                return None
        except Exception as e:
            self.get_logger().error(f'IK optimization failed for {robot}: {e}')
            return None

    def compute_pd_torques(self):
        """Compute PD control torques for both robots."""
        
        # Local robot torques
        local_current_pos = np.array([self.data.qpos[i] for i in self.local_joint_indices])
        local_current_vel = np.array([self.data.qvel[i] for i in self.local_joint_indices])
        
        local_pos_error = self.local_target_positions - local_current_pos
        local_torques = self.kp * local_pos_error - self.kd * local_current_vel
        local_torques = np.clip(local_torques, -self.force_limit, self.force_limit)
        
        # Remote robot torques
        remote_current_pos = np.array([self.data.qpos[i] for i in self.remote_joint_indices])
        remote_current_vel = np.array([self.data.qvel[i] for i in self.remote_joint_indices])
        
        remote_pos_error = self.remote_target_positions - remote_current_pos
        remote_torques = self.kp * remote_pos_error - self.kd * remote_current_vel
        remote_torques = np.clip(remote_torques, -self.force_limit, self.force_limit)
        
        return local_torques, remote_torques

    def publish_states(self):
        """Publish states for both robots."""
        current_time = self.get_clock().now().to_msg()
        
        # Publish local robot states
        self._publish_robot_states(current_time, 'local')
        
        # Publish remote robot states
        self._publish_robot_states(current_time, 'remote')

    def _publish_robot_states(self, timestamp, robot_type):
        """Publish joint states and end-effector pose for specified robot."""
        
        if robot_type == 'local':
            joint_indices = self.local_joint_indices
            joint_names = self.local_joint_names
            joint_pub = self.local_joint_state_pub
            ee_pub = self.local_ee_pose_pub
            ee_body_name = 'fr3_link7_robot1'
        else:
            joint_indices = self.remote_joint_indices
            joint_names = self.remote_joint_names
            joint_pub = self.remote_joint_state_pub
            ee_pub = self.remote_ee_pose_pub
            ee_body_name = 'fr3_link7_robot2'
        
        # Publish joint states
        joint_state = JointState()
        joint_state.header.stamp = timestamp
        joint_state.name = joint_names
        joint_state.position = [self.data.qpos[i] for i in joint_indices]
        joint_state.velocity = [self.data.qvel[i] for i in joint_indices]
        joint_state.effort = [self.data.qfrc_applied[i] for i in joint_indices]
        
        joint_pub.publish(joint_state)
        
        # Publish end-effector pose
        try:
            ee_id = self.model.body(ee_body_name).id
            ee_pos = self.data.xpos[ee_id]
            ee_quat = self.data.xquat[ee_id]
            
            ee_pose = PoseStamped()
            ee_pose.header.stamp = timestamp
            ee_pose.header.frame_id = "world"
            
            ee_pose.pose.position.x = float(ee_pos[0])
            ee_pose.pose.position.y = float(ee_pos[1])
            ee_pose.pose.position.z = float(ee_pos[2])
            
            ee_pose.pose.orientation.w = float(ee_quat[0])
            ee_pose.pose.orientation.x = float(ee_quat[1])
            ee_pose.pose.orientation.y = float(ee_quat[2])
            ee_pose.pose.orientation.z = float(ee_quat[3])
            
            ee_pub.publish(ee_pose)
            
        except Exception as e:
            self.get_logger().warn(f'Could not publish {robot_type} EE pose: {e}')

    def _start_simulation(self):
        """Start the MuJoCo simulation thread."""
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def simulation_loop(self):
        """Main simulation loop with PD control for both robots."""
        while rclpy.ok():
            # Compute torques for both robots
            local_torques, remote_torques = self.compute_pd_torques()
            
            # Apply torques to local robot
            for i, joint_idx in enumerate(self.local_joint_indices):
                if joint_idx < len(self.data.ctrl):
                    self.data.ctrl[joint_idx] = local_torques[i]
            
            # Apply torques to remote robot
            for i, joint_idx in enumerate(self.remote_joint_indices):
                if joint_idx < len(self.data.ctrl):
                    self.data.ctrl[joint_idx] = remote_torques[i]
            
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
        controller = DualRobots()
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