import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import numpy as np
import os
import time
import threading
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped

# Set MuJoCo backend
os.environ['MUJOCO_GL'] = 'egl'

class AdvancedPDController(Node):
    def __init__(self):
        super().__init__('advanced_pd_controller')
        
        # Load MuJoCo model
        master_thesis_path = "/media/kai/Kai_Backup/Study/Master_Thesis/My_Master_Thesis"
        model_path = os.path.join(master_thesis_path, "franka_mujoco_ws", "src", 
                                 "mujoco_menagerie", "franka_fr3", "fr3.xml")
        
        self.get_logger().info(f'Loading MuJoCo model from: {model_path}')
        
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model file not found: {model_path}')
            raise FileNotFoundError(f'MuJoCo model not found at {model_path}')
        
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
        
        # PD Gains - Joint Space
        self.kp_joint = np.array([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 25.0])
        self.kd_joint = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 2.5])
        
        # Control mode
        self.control_mode = 'joint'
        
        # Target states
        self.q_desired = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.qd_desired = np.zeros(self.n_joints)
        
        # ROS2 publishers and subscribers
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.ee_pose_pub = self.create_publisher(PoseStamped, '/ee_pose', 10)
        
        # Subscribers for targets
        self.joint_target_sub = self.create_subscription(
            Float64MultiArray, '/joint_target', self.joint_target_callback, 10)
        
        # Initialize robot to home position
        self.data.qpos[:self.n_joints] = self.q_desired.copy()
        mujoco.mj_forward(self.model, self.data)
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Timer for publishing
        self.timer = self.create_timer(1.0/100.0, self.publish_states)
        
        self.get_logger().info('🚀 Advanced PD Controller Started!')
        self.get_logger().info(f'Control mode: {self.control_mode}')
        
    def joint_target_callback(self, msg):
        """Set joint space target"""
        if len(msg.data) == self.n_joints:
            self.q_desired = np.array(msg.data)
            self.qd_desired = np.zeros(self.n_joints)
            self.get_logger().info(f'New joint target: {np.round(self.q_desired, 3)}')
    
    def matrix_to_quaternion(self, rotation_matrix):
        """Convert rotation matrix to quaternion [w, x, y, z] - Simple implementation"""
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
    
    def compute_gravity_compensation(self):
        """Compute gravity compensation torques"""
        return self.data.qfrc_bias[:self.n_joints].copy()
    
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
        
        # Add gravity compensation
        tau_gravity = self.compute_gravity_compensation()
        
        # Total torque
        tau_total = tau_pd + tau_gravity
        
        return tau_total, q_error, qd_error
    
    def simulation_loop(self):
        """Main simulation and control loop"""
        while rclpy.ok():
            # Compute control torques
            tau, q_error, qd_error = self.joint_pd_control()
            
            # Log joint errors periodically
            if int(self.data.time * 10) % 10 == 0:  # Every 1 second
                self.get_logger().info(f'Joint errors: pos={np.round(q_error, 3)}, vel={np.round(qd_error, 3)}')
            
            # Apply torque limits (safety)
            tau_max = np.array([87, 87, 87, 87, 12, 12, 12])  # Franka limits
            tau = np.clip(tau, -tau_max, tau_max)
            
            # Apply control torques
            self.data.ctrl[:self.n_joints] = tau
            
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
        
        # End-effector pose (simplified - only publish if needed)
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
            # Don't spam logs with quaternion errors
            pass

def main(args=None):
    rclpy.init(args=args)
    controller = AdvancedPDController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()