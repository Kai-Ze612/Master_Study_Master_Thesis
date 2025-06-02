#!/usr/bin/env python3
#!/usr/bin/env python3

# Loading python modules
import os
import time
import threading

# Loading ROS2 modules 
import rclpy
from rclpy.node import Node

# Loading MuJoCo modules
import mujoco
import mujoco.viewer

# Loading ROS2 message types
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, Point

# Loading numerical computation
import numpy as np


class CartesianPositionController(Node):
    def __init__(self):
        super().__init__('cartesian_position_controller')
        
        model_path = "/media/kai/Kai_Backup/Master_Study/Practical_Project/Practical_Courses/Master_Thesis/Master_Study_Master_Thesis/MuJoCo_Creating_Scene/FR3_MuJoCo/franka_fr3/fr3_with_moveable_box.xml"
        
        self.get_logger().info(f'Loading MuJoCo model from: {model_path}')
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Robot parameters
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.n_joints = len(self.joint_names)
        self.ee_body_name = 'fr3_link7'
        
        # Control parameters
        self.control_freq = 1000
        self.dt = 1.0 / self.control_freq
       
        # PD Gains - Joint Space (Reduced for stability)
        self.kp_joint = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 30.0, 15.0])
        self.kd_joint = np.array([8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 2.0])
        
        # Cartesian PD Gains (Much lower for smooth motion)
        self.kp_cartesian = np.array([200.0, 200.0, 200.0])  # X, Y, Z position gains
        self.kd_cartesian = np.array([40.0, 40.0, 40.0])     # X, Y, Z velocity gains
        
        # Control mode: 'joint' or 'cartesian'
        self.control_mode = 'cartesian'
        
        # Target states
        self.q_desired = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # Home position
        self.qd_desired = np.zeros(self.n_joints)
        
        # Cartesian targets with trajectory planning
        self.pos_desired = np.array([0.5, 0.0, 0.5])  # Current target
        self.pos_target = np.array([0.5, 0.0, 0.5])   # Final target
        self.vel_desired = np.zeros(3)
        
        # Trajectory parameters
        self.trajectory_active = False
        self.trajectory_start_time = 0.0
        self.trajectory_duration = 3.0  # 3 seconds to reach target
        self.pos_start = np.array([0.5, 0.0, 0.5])
        
        # Motion smoothing
        self.max_velocity = 0.2  # m/s
        self.max_acceleration = 0.5  # m/s²
        
        # Initialize robot to home position
        self.data.qpos[:self.n_joints] = self.q_desired.copy()
        mujoco.mj_forward(self.model, self.data)
        
        # Get initial end-effector position
        body_id = self.model.body(self.ee_body_name).id
        self.pos_desired = self.data.xpos[body_id].copy()
        
        # ROS2 publishers and subscribers
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.ee_pose_pub = self.create_publisher(PoseStamped, '/ee_pose', 10)
        
        # Subscribers for targets
        self.joint_target_sub = self.create_subscription(
            Float64MultiArray, '/joint_target', self.joint_target_callback, 10)
        
        self.cartesian_target_sub = self.create_subscription(
            Point, '/cartesian_target', self.cartesian_target_callback, 10)
            
        self.pose_target_sub = self.create_subscription(
            PoseStamped, '/pose_target', self.pose_target_callback, 10)
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Timer for publishing
        self.timer = self.create_timer(1.0/100.0, self.publish_states)
        
        self.get_logger().info('🚀 Cartesian Position Controller Started!')
        self.get_logger().info(f'Control mode: {self.control_mode}')
        self.get_logger().info(f'Initial EE position: {np.round(self.pos_desired, 3)}')
        
    def joint_target_callback(self, msg):
        """Set joint space target"""
        if len(msg.data) == self.n_joints:
            self.control_mode = 'joint'
            self.q_desired = np.array(msg.data)
            self.qd_desired = np.zeros(self.n_joints)
            self.get_logger().info(f'Switched to joint control. Target: {np.round(self.q_desired, 3)}')
    
    def cartesian_target_callback(self, msg):
        """Set Cartesian position target with smooth trajectory"""
        self.control_mode = 'cartesian'
        
        # Get current end-effector position as starting point
        body_id = self.model.body(self.ee_body_name).id
        self.pos_start = self.data.xpos[body_id].copy()
        
        # Set new target
        self.pos_target = np.array([msg.x, msg.y, msg.z])
        
        # Calculate trajectory duration based on distance
        distance = np.linalg.norm(self.pos_target - self.pos_start)
        self.trajectory_duration = max(2.0, distance / self.max_velocity)
        
        # Start trajectory
        self.trajectory_active = True
        self.trajectory_start_time = self.data.time
        
        self.get_logger().info(f'New Cartesian target: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]')
        self.get_logger().info(f'Distance: {distance:.3f}m, Duration: {self.trajectory_duration:.2f}s')
    
    def pose_target_callback(self, msg):
        """Set Cartesian pose target with smooth trajectory"""
        self.control_mode = 'cartesian'
        
        # Get current end-effector position as starting point
        body_id = self.model.body(self.ee_body_name).id
        self.pos_start = self.data.xpos[body_id].copy()
        
        # Set new target
        self.pos_target = np.array([
            msg.pose.position.x,
            msg.pose.position.y, 
            msg.pose.position.z
        ])
        
        # Calculate trajectory duration based on distance
        distance = np.linalg.norm(self.pos_target - self.pos_start)
        self.trajectory_duration = max(2.0, distance / self.max_velocity)
        
        # Start trajectory
        self.trajectory_active = True
        self.trajectory_start_time = self.data.time
        
        self.get_logger().info(f'New pose target: [{msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}]')
        self.get_logger().info(f'Distance: {distance:.3f}m, Duration: {self.trajectory_duration:.2f}s')
    
    def get_jacobian(self):
        """Compute end-effector Jacobian matrix"""
        # Get body ID
        body_id = self.model.body(self.ee_body_name).id
        
        # Initialize Jacobian matrices
        jacp = np.zeros((3, self.model.nv))  # Position Jacobian
        jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
        
        # Compute Jacobian
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.xpos[body_id], body_id)
        
        # Return only position Jacobian for position control
        return jacp[:, :self.n_joints]
    
    def compute_inverse_kinematics(self, target_pos, current_q, max_iterations=50, tolerance=1e-4):
        """
        Simple inverse kinematics using Jacobian pseudo-inverse method
        """
        q_solution = current_q.copy()
        
        for i in range(max_iterations):
            # Update model with current joint angles
            self.data.qpos[:self.n_joints] = q_solution
            mujoco.mj_forward(self.model, self.data)
            
            # Get current end-effector position
            body_id = self.model.body(self.ee_body_name).id
            current_pos = self.data.xpos[body_id].copy()
            
            # Position error
            pos_error = target_pos - current_pos
            
            # Check convergence
            if np.linalg.norm(pos_error) < tolerance:
                self.get_logger().info(f'IK converged in {i+1} iterations')
                break
            
            # Get Jacobian
            J = self.get_jacobian()
            
            # Compute pseudo-inverse
            J_pinv = np.linalg.pinv(J)
            
            # Update joint angles
            dq = J_pinv @ pos_error
            
            # Apply step size and joint limits
            step_size = 0.1
            q_solution += step_size * dq
            
            # Apply joint limits (Franka FR3 limits)
            q_limits_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            q_limits_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
            q_solution = np.clip(q_solution, q_limits_low, q_limits_high)
        
        return q_solution
    
    def update_trajectory(self):
        """Update trajectory interpolation"""
        if not self.trajectory_active:
            return
        
        # Calculate trajectory progress
        elapsed_time = self.data.time - self.trajectory_start_time
        progress = min(elapsed_time / self.trajectory_duration, 1.0)
        
        # S-curve (smooth) interpolation
        if progress <= 0.5:
            # Acceleration phase
            s = 2.0 * progress * progress
        else:
            # Deceleration phase  
            s = 1.0 - 2.0 * (1.0 - progress) * (1.0 - progress)
        
        # Interpolate position
        self.pos_desired = self.pos_start + s * (self.pos_target - self.pos_start)
        
        # Calculate desired velocity (numerical derivative)
        if progress < 1.0:
            # Compute velocity for smooth motion
            dt = 0.001  # Small time step
            if progress <= 0.5:
                ds_dt = 4.0 * progress / self.trajectory_duration
            else:
                ds_dt = 4.0 * (1.0 - progress) / self.trajectory_duration
            
            self.vel_desired = ds_dt * (self.pos_target - self.pos_start)
        else:
            # Trajectory complete
            self.pos_desired = self.pos_target.copy()
            self.vel_desired = np.zeros(3)
            self.trajectory_active = False
            self.get_logger().info('✅ Trajectory completed!')
    
    def cartesian_pd_control(self):
        """Cartesian-space PD control with smooth trajectory"""
        # Update trajectory interpolation
        self.update_trajectory()
        
        # Get current end-effector state
        body_id = self.model.body(self.ee_body_name).id
        current_pos = self.data.xpos[body_id].copy()
        
        # Position error
        pos_error = self.pos_desired - current_pos
        
        # Get Jacobian for velocity computation
        J = self.get_jacobian()
        current_vel = J @ self.data.qvel[:self.n_joints]
        vel_error = self.vel_desired - current_vel
        
        # Cartesian PD control with lower gains
        force_cartesian = self.kp_cartesian * pos_error + self.kd_cartesian * vel_error
        
        # Convert to joint torques using Jacobian transpose
        J_T = J.T
        tau_cartesian = J_T @ force_cartesian
        
        # Add damping for stability
        damping = -5.0 * self.data.qvel[:self.n_joints]
        tau_total = tau_cartesian + damping
        
        return tau_total, pos_error, vel_error
    
    def joint_pd_control(self):
        """Joint-space PD control"""
        # Current state
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()
        
        # Position and velocity errors
        q_error = self.q_desired - q_current
        qd_error = self.qd_desired - qd_current
        
        # PD control law
        tau_pd = self.kp_joint * q_error + self.kd_joint * qd_error
        
        return tau_pd, q_error, qd_error
    
    def compute_gravity_compensation(self):
        """Compute gravity compensation torques"""
        return self.data.qfrc_bias[:self.n_joints].copy()
    
    def matrix_to_quaternion(self, rotation_matrix):
        """Convert rotation matrix to quaternion [w, x, y, z]"""
        R = rotation_matrix
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return np.array([qw, qx, qy, qz])
    
    def simulation_loop(self):
        """Main simulation and control loop"""
        while rclpy.ok():
            # Compute control torques based on mode
            if self.control_mode == 'cartesian':
                tau, pos_error, vel_error = self.cartesian_pd_control()
                
                # Log Cartesian errors periodically
                if int(self.data.time * 10) % 20 == 0:  # Every 2 seconds
                    self.get_logger().info(f'Cartesian errors: pos={np.round(pos_error, 3)}, vel={np.round(vel_error, 3)}')
            else:
                tau, q_error, qd_error = self.joint_pd_control()
                
                # Log joint errors periodically
                if int(self.data.time * 10) % 20 == 0:  # Every 2 seconds
                    self.get_logger().info(f'Joint errors: pos={np.round(q_error, 3)}, vel={np.round(qd_error, 3)}')
            
            # Add gravity compensation
            tau_gravity = self.compute_gravity_compensation()
            tau_total = tau + tau_gravity
            
            # Apply torque limits (safety)
            tau_max = np.array([87, 87, 87, 87, 12, 12, 12])  # Franka limits
            tau_total = np.clip(tau_total, -tau_max, tau_max)
            
            # Apply control torques
            self.data.ctrl[:self.n_joints] = tau_total
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Update viewer
            if self.viewer.is_running():
                self.viewer.sync()
            
            # Control frequency
            time.sleep(self.dt)
    
    def publish_states(self):
        """Publish robot states"""
        # Joint states
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.joint_names
        joint_state.position = self.data.qpos[:self.n_joints].tolist()
        joint_state.velocity = self.data.qvel[:self.n_joints].tolist()
        joint_state.effort = self.data.ctrl[:self.n_joints].tolist()
        self.joint_state_pub.publish(joint_state)
        
        # End-effector pose
        try:
            body_id = self.model.body(self.ee_body_name).id
            ee_pos = self.data.xpos[body_id]
            ee_quat = self.matrix_to_quaternion(self.data.xmat[body_id].reshape(3, 3))
            
            ee_pose = PoseStamped()
            ee_pose.header.stamp = joint_state.header.stamp
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

def main(args=None):
    rclpy.init(args=args)
    controller = CartesianPositionController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()