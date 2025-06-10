#!/usr/bin/env python3

# Load ROS2 Python libraries
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseStamped, Point, Vector3

# Load MuJoCo libraries
import mujoco
import mujoco.viewer

# Python standard libraries
import numpy as np
import os
import time
import threading

# Set MuJoCo backend
os.environ['MUJOCO_GL'] = 'egl'

class DualRobotController(Node):
    def __init__(self):
        super().__init__('dual_robot_controller')
        
        model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller/models/franka_fr3/dual_fr3.xml"
                
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
                   
        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # DEBUG: Print model structure
        self.debug_model_structure()
        
        # Robot parameters
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.n_joints = len(self.joint_names)
        
        # Local and remote robot parameters
        self.local_ee_body_name = 'fr3_link7'
        self.remote_ee_body_name = 'remote_fr3_link7'
        self.local_object_name = 'local_box'
        
        # Control parameters
        self.control_freq = 1000
        self.dt = 1.0 / self.control_freq
        
        # PD Gains - Stronger for remote robot
        self.kp_joint = np.array([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 25.0])
        self.kd_joint = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 2.5])
        self.kp_cart_pos = np.array([2000.0, 2000.0, 2000.0])
        self.kd_cart_pos = np.array([200.0, 200.0, 200.0])
        
        # Much stronger gains for remote robot copying
        self.kp_remote = self.kp_joint * 3.0  # Triple the gains
        self.kd_remote = self.kd_joint * 2.0
        
        # Task state machine
        self.task_state = 'IDLE'
        self.tolerance = 0.08
        
        # Local robot waypoints
        self.local_waypoints = [
            np.array([0.3, 0, 0.75]),
            np.array([0.5, 0, 0.1]),
            np.array([0.7, 0, 0.1])
        ]
        
        self.current_local_target = self.local_waypoints[0].copy()
        
        # Home positions
        self.q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        # Control modes
        self.local_control_mode = 'cartesian'
        self.remote_control_mode = 'copy'
        
        # Desired positions
        self.local_q_desired = self.q_home.copy()
        self.remote_q_desired = self.q_home.copy()
        
        # Direct copying instead of ROS2 communication
        self.use_direct_copy = True
        
        # ROS2 Publishers
        self.local_joint_state_pub = self.create_publisher(JointState, '/local/joint_states', 10)
        self.local_ee_pose_pub = self.create_publisher(PoseStamped, '/local/ee_pose', 10)
        self.remote_joint_state_pub = self.create_publisher(JointState, '/remote/joint_states', 10)
        self.remote_ee_pose_pub = self.create_publisher(PoseStamped, '/remote/ee_pose', 10)
        self.task_state_pub = self.create_publisher(String, '/task_state', 10)
        self.sync_status_pub = self.create_publisher(String, '/sync_status', 10)
        
        # Subscribers
        self.start_task_sub = self.create_subscription(
            String, '/start_dual_task', self.start_dual_task, 10)
        
        # Initialize both robots to home position
        self.data.qpos[:self.n_joints] = self.q_home.copy()
        self.data.qpos[self.n_joints:2*self.n_joints] = self.q_home.copy()
        
        mujoco.mj_forward(self.model, self.data)
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Timer for publishing
        self.timer = self.create_timer(1.0/50.0, self.publish_states)
        
        # Auto-start timer
        self.create_timer(3.0, self.auto_start_task)
        
        self.get_logger().info('Dual Robot Controller Started!')
        self.get_logger().info('Using DIRECT copying mode for better synchronization')
    
    def debug_model_structure(self):
        """Debug the actual model structure"""
        self.get_logger().info(f"=== MODEL STRUCTURE DEBUG ===")
        self.get_logger().info(f"Total joints: {self.model.njnt}")
        self.get_logger().info(f"Total actuators: {self.model.nu}")
        self.get_logger().info(f"Total DOF: {self.model.nv}")
        
        # Print first 20 joints
        self.get_logger().info("JOINTS:")
        for i in range(min(20, self.model.njnt)):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.get_logger().info(f"  Joint {i}: {joint_name}")
        
        # Print first 20 actuators
        self.get_logger().info("ACTUATORS:")
        for i in range(min(20, self.model.nu)):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.get_logger().info(f"  Actuator {i}: {actuator_name}")
        
        self.get_logger().info("=== END DEBUG ===")
    
    def auto_start_task(self):
        """Auto-start the dual robot task"""
        if self.task_state == 'IDLE':
            self.start_dual_task_internal()
    
    def start_dual_task(self, msg):
        """Start dual robot task (ROS callback)"""
        if msg.data == "start":
            self.start_dual_task_internal()
    
    def start_dual_task_internal(self):
        """Internal method to start dual robot task"""
        self.task_state = 'MOVE_TO_START'
        self.local_control_mode = 'cartesian'
        self.current_local_target = self.local_waypoints[0].copy()
        
        self.get_logger().info('Starting dual robot task!')
        self.get_logger().info('Local robot will lead, remote robot will copy directly')
    
    def get_local_ee_position(self):
        """Get local robot end-effector position"""
        try:
            ee_id = self.model.body(self.local_ee_body_name).id
            return self.data.xpos[ee_id].copy()
        except:
            return np.zeros(3)
    
    def get_remote_ee_position(self):
        """Get remote robot end-effector position"""
        try:
            ee_id = self.model.body(self.remote_ee_body_name).id
            return self.data.xpos[ee_id].copy()
        except:
            return np.zeros(3)
    
    def compute_local_jacobian(self):
        """Compute local robot Jacobian"""
        try:
            ee_id = self.model.body(self.local_ee_body_name).id
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, ee_id)
            return np.vstack([jacp[:, :self.n_joints], jacr[:, :self.n_joints]])
        except:
            return np.zeros((6, self.n_joints))
    
    def local_task_state_machine(self):
        """Task state machine for local robot (leader)"""
        local_ee_pos = self.get_local_ee_position()
        
        if self.task_state == 'IDLE':
            self.local_control_mode = 'joint'
            self.local_q_desired = self.q_home.copy()
            
        elif self.task_state == 'MOVE_TO_START':
            self.local_control_mode = 'cartesian'
            self.current_local_target = self.local_waypoints[0].copy()
            
            distance = np.linalg.norm(local_ee_pos - self.current_local_target)
            if distance < self.tolerance:
                self.task_state = 'APPROACH_OBJECT'
                self.get_logger().info('Local robot reached start, remote copying')
                
        elif self.task_state == 'APPROACH_OBJECT':
            self.current_local_target = self.local_waypoints[1].copy()
            
            distance = np.linalg.norm(local_ee_pos - self.current_local_target)
            if distance < self.tolerance:
                self.task_state = 'PUSH_OBJECT'
                self.get_logger().info('Local robot approaching object')
                
        elif self.task_state == 'PUSH_OBJECT':
            self.current_local_target = self.local_waypoints[2].copy()
            
            distance = np.linalg.norm(local_ee_pos - self.current_local_target)
            if distance < self.tolerance:
                self.task_state = 'DONE'
                self.get_logger().info('Task completed! Remote should have copied all movements')
    
    def direct_copy_remote(self):
        """Directly copy local robot position to remote robot (no ROS2 delay)"""
        if self.use_direct_copy:
            # Get current local robot joint positions
            local_q = self.data.qpos[:self.n_joints].copy()
            
            # Set remote robot desired position to exactly match local
            self.remote_q_desired = local_q.copy()
    
    def local_cartesian_control(self):
        """Cartesian control for local robot"""
        ee_pos = self.get_local_ee_position()
        jacobian = self.compute_local_jacobian()
        
        pos_error = self.current_local_target - ee_pos
        ee_vel = jacobian[:3, :] @ self.data.qvel[:self.n_joints]
        vel_error = np.zeros(3) - ee_vel
        
        F_cartesian = self.kp_cart_pos * pos_error + self.kd_cart_pos * vel_error
        tau_cartesian = jacobian[:3, :].T @ F_cartesian
        tau_gravity = self.data.qfrc_bias[:self.n_joints].copy()
        
        return tau_cartesian + tau_gravity
    
    def local_joint_control(self):
        """Joint control for local robot"""
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()
        
        q_error = self.local_q_desired - q_current
        qd_error = np.zeros(self.n_joints) - qd_current
        
        tau_pd = self.kp_joint * q_error + self.kd_joint * qd_error
        tau_gravity = self.data.qfrc_bias[:self.n_joints].copy()
        
        return tau_pd + tau_gravity
    
    def remote_joint_control(self):
        """Strong joint control for remote robot to copy local exactly"""
        q_current = self.data.qpos[self.n_joints:2*self.n_joints].copy()
        qd_current = self.data.qvel[self.n_joints:2*self.n_joints].copy()
        
        q_error = self.remote_q_desired - q_current
        qd_error = np.zeros(self.n_joints) - qd_current
        
        # Use very strong gains for exact copying
        tau_pd = self.kp_remote * q_error + self.kd_remote * qd_error
        tau_gravity = self.data.qfrc_bias[self.n_joints:2*self.n_joints].copy()
        
        return tau_pd + tau_gravity
    
    def simulation_loop(self):
        """Main simulation and control loop"""
        while rclpy.ok():
            try:
                # Local robot task state machine
                self.local_task_state_machine()
                
                # Direct copying (no ROS2 delay)
                self.direct_copy_remote()
                
                # Compute control torques
                if self.local_control_mode == 'cartesian':
                    tau_local = self.local_cartesian_control()
                else:
                    tau_local = self.local_joint_control()
                
                tau_remote = self.remote_joint_control()
                
                # Apply torque limits
                tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
                tau_local = np.clip(tau_local, -tau_max, tau_max)
                tau_remote = np.clip(tau_remote, -tau_max, tau_max)
                
                # Apply control torques to correct actuators
                if self.model.nu >= 14:
                    self.data.ctrl[:7] = tau_local        # Local robot: actuators 0-6
                    self.data.ctrl[7:14] = tau_remote     # Remote robot: actuators 7-13
                else:
                    self.get_logger().error(f"Expected 14 actuators, got {self.model.nu}")
                    self.data.ctrl[:self.n_joints] = tau_local
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update viewer
                if self.viewer.is_running():
                    self.viewer.sync()
                
                time.sleep(self.dt)
                
            except Exception as e:
                self.get_logger().error(f'Simulation loop error: {e}')
                time.sleep(self.dt)
    
    def publish_states(self):
        """Publish states for both robots"""
        try:
            timestamp = self.get_clock().now().to_msg()
            
            # Local robot joint states
            local_joint_state = JointState()
            local_joint_state.header.stamp = timestamp
            local_joint_state.name = self.joint_names
            local_joint_state.position = self.data.qpos[:self.n_joints].tolist()
            local_joint_state.velocity = self.data.qvel[:self.n_joints].tolist()
            local_joint_state.effort = self.data.ctrl[:self.n_joints].tolist()
            self.local_joint_state_pub.publish(local_joint_state)
            
            # Remote robot joint states
            remote_joint_state = JointState()
            remote_joint_state.header.stamp = timestamp
            remote_joint_state.name = [f'remote_{name}' for name in self.joint_names]
            remote_joint_state.position = self.data.qpos[self.n_joints:2*self.n_joints].tolist()
            remote_joint_state.velocity = self.data.qvel[self.n_joints:2*self.n_joints].tolist()
            if self.model.nu >= 14:
                remote_joint_state.effort = self.data.ctrl[7:14].tolist()
            else:
                remote_joint_state.effort = [0.0] * 7
            self.remote_joint_state_pub.publish(remote_joint_state)
            
            # End-effector poses
            local_ee_pos = self.get_local_ee_position()
            remote_ee_pos = self.get_remote_ee_position()
            
            # Task state
            state_msg = String()
            state_msg.data = self.task_state
            self.task_state_pub.publish(state_msg)
            
            # Enhanced synchronization status
            sync_distance = np.linalg.norm(local_ee_pos - remote_ee_pos)
            joint_error = np.linalg.norm(self.data.qpos[:7] - self.data.qpos[7:14])
            
            # Individual joint errors for debugging
            joint_errors = np.abs(self.data.qpos[:7] - self.data.qpos[7:14])
            max_joint_error = np.max(joint_errors)
            worst_joint = np.argmax(joint_errors)
            
            sync_status = String()
            sync_status.data = f"sync_dist:{sync_distance:.4f}m,joint_err:{joint_error:.4f}rad,max_joint_err:{max_joint_error:.4f}rad_at_joint{worst_joint}"
            self.sync_status_pub.publish(sync_status)
            
            # Detailed periodic logging
            if int(self.data.time * 2) % 10 == 0:  # Every 5 seconds
                self.get_logger().info(f'=== COPY STATUS ===')
                self.get_logger().info(f'Task: {self.task_state} | Copy quality: {sync_distance:.4f}m')
                self.get_logger().info(f'Local EE:  {np.round(local_ee_pos, 3)}')
                self.get_logger().info(f'Remote EE: {np.round(remote_ee_pos, 3)}')
                self.get_logger().info(f'Joint error: {joint_error:.4f} rad (max: {max_joint_error:.4f} at joint {worst_joint})')
                self.get_logger().info(f'Local joints:  {np.round(self.data.qpos[:7], 3)}')
                self.get_logger().info(f'Remote joints: {np.round(self.data.qpos[7:14], 3)}')
                self.get_logger().info(f'Remote desired: {np.round(self.remote_q_desired, 3)}')
                
        except Exception as e:
            self.get_logger().error(f'Publishing error: {e}')

def main(args=None):
    rclpy.init(args=args)
    controller = DualRobotController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()