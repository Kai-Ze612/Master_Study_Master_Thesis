#!/usr/bin/env python3
## import ROS2 Python libraries
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Vector3

## import MuJoCo libraries
import mujoco
import mujoco.viewer

## Import python libraries
import numpy as np
import os
import time
import threading

## Set different rendering options
## 'egl' is GPU rendering
## 'glfw' is desktop rendering
## 'osmesa' is software CPU rendering
os.environ['MUJOCO_GL'] = 'egl'

class position_control(Node):
    def __init__(self):
        super().__init__('fixed_object_push_controller')
"
        model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/MuJoCo_Creating_Scene/FR3_MuJoCo/franka_fr3/fr3_with_moveable_box.xml"
                
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
                   
        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Robot parameters
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.n_joints = len(self.joint_names)
        self.ee_body_name = 'fr3_link7'
        self.object_name = 'box'
        
        # Control parameters
        self.control_freq = 1000
        self.dt = 1.0 / self.control_freq
        
        # MUCH HIGHER PD Gains - This was the main issue!
        self.kp_joint = np.array([2000.0, 2000.0, 2000.0, 2000.0, 1000.0, 1000.0, 500.0])
        self.kd_joint = np.array([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 25.0])
        
        # Enhanced Cartesian PD Gains for pushing task
        self.kp_cart_pos = np.array([8000.0, 8000.0, 8000.0])  # Slightly reduced for stability
        self.kd_cart_pos = np.array([800.0, 800.0, 800.0])     # Slightly reduced for stability
        
        # Push force parameters - tuned for effective object manipulation
        self.push_force_max = 30.0   # Moderate force to avoid instability
        self.contact_force = 10.0    # Desired contact force during pushing
        
        # Task state machine
        self.task_state = 'IDLE'
        self.state_start_time = 0.0
        self.waypoint_index = 0
        
        # Fixed target - doesn't change once set!
        self.target_set = False
        self.current_target = np.zeros(3)
        
        # Optimized push waypoints for effective pushing
        self.push_waypoints = [
            np.array([0.5, 0.0, 0.2]),   # Behind object (approach position)
            np.array([0.7, 0.0, 0.2]), # Contact with object (slightly above object)
            np.array([0.8, 0.0, 0.2])    # Push through to end position
        ]
        
        # Task tolerances
        self.tolerance = 0.03  # 3cm tolerance for waypoints
        self.push_tolerance = 0.05  # 5cm tolerance for final object position
        
        # Object tracking - object starts at 0.5,0,0.05 and should end at 0.7,0,0.05
        self.object_start_pos = np.array([0.7, 0.0, 0.05])
        self.object_target_pos = np.array([0.8, 0.0, 0.05])
        
        # Home position
        self.q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.q_desired = self.q_home.copy()
        self.qd_desired = np.zeros(self.n_joints)
        
        # Control mode
        self.control_mode = 'joint'
        
        # Progress tracking
        self.last_distance = float('inf')
        self.stuck_counter = 0
        
        # ROS2 publishers and subscribers
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.ee_pose_pub = self.create_publisher(PoseStamped, '/ee_pose', 10)
        self.object_pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)
        self.task_state_pub = self.create_publisher(String, '/task_state', 10)
        self.push_force_pub = self.create_publisher(Vector3, '/push_force', 10)
        
        # Subscribers
        self.start_task_sub = self.create_subscription(
            String, '/start_push', self.start_push_task, 10)
        self.reset_task_sub = self.create_subscription(
            String, '/reset_task', self.reset_task, 10)
        
        # Initialize robot position
        self.data.qpos[:self.n_joints] = self.q_home.copy()
        self.setup_object_position()
        mujoco.mj_forward(self.model, self.data)
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Timer for publishing
        self.timer = self.create_timer(1.0/50.0, self.publish_states)
        
        self.get_logger().info('🚀 Fixed Object Push Controller Started!')
        self.get_logger().info(f'Push strategy: {self.object_start_pos} -> {self.object_target_pos}')
        self.get_logger().info(f'Waypoints: Behind({self.push_waypoints[0]}) -> Contact({self.push_waypoints[1]}) -> Push({self.push_waypoints[2]})')
        self.get_logger().info(f'Control gains: kp_cart={self.kp_cart_pos[0]}, kd_cart={self.kd_cart_pos[0]}')
        self.get_logger().info('Send "start" to /start_push to begin pushing task')
        
    def setup_object_position(self):
        """Setup object at starting position"""
        try:
            object_id = self.model.body(self.object_name).id
            body = self.model.body(self.object_name)
            if body.jntadr[0] >= 0:
                joint_start = body.jntadr[0]
                joint_num = body.jntnum[0]
                
                if joint_num >= 3:
                    self.data.qpos[self.n_joints + joint_start:self.n_joints + joint_start + 3] = self.object_start_pos
                    self.get_logger().info(f'Set object position to: {self.object_start_pos}')
                    
        except Exception as e:
            self.get_logger().warn(f"Could not setup object position: {e}")
    
    def start_push_task(self, msg):
        """Start pushing task - ONLY ONCE per command"""
        if msg.data == "start" and self.task_state == 'IDLE':  # Only start if IDLE
            self.task_state = 'MOVE_TO_START'
            self.state_start_time = self.data.time
            self.waypoint_index = 0
            
            # Set target ONCE and don't change it!
            self.current_target = self.push_waypoints[0].copy()
            self.target_set = True
            self.control_mode = 'cartesian'
            
            self.get_logger().info('Starting push task!')
            self.get_logger().info(f'Target set to: {self.current_target}')
            self.get_logger().info('Target will NOT change until waypoint is reached!')
    
    def reset_task(self, msg):
        """Reset task to initial state"""
        self.task_state = 'IDLE'
        self.q_desired = self.q_home.copy()
        self.control_mode = 'joint'
        self.waypoint_index = 0
        self.target_set = False
        self.stuck_counter = 0
        
        self.setup_object_position()
        self.get_logger().info('Task reset to IDLE')
    
    def get_object_position(self):
        """Get current object position"""
        try:
            object_id = self.model.body(self.object_name).id
            return self.data.xpos[object_id].copy()
        except:
            return self.object_start_pos.copy()
    
    def get_ee_position(self):
        """Get current end-effector position"""
        try:
            ee_id = self.model.body(self.ee_body_name).id
            return self.data.xpos[ee_id].copy()
        except:
            return np.zeros(3)
    
    def compute_jacobian(self):
        """Compute end-effector Jacobian"""
        try:
            ee_id = self.model.body(self.ee_body_name).id
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, ee_id)
            return jacp[:, :self.n_joints]
        except:
            return np.zeros((3, self.n_joints))
    
    def get_contact_force(self):
        """Get contact force between end-effector and object"""
        total_force = 0.0
        contact_normal = np.zeros(3)
        
        for i in range(self.data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            force_magnitude = np.linalg.norm(force[:3])
            
            if force_magnitude > 0.1:
                total_force += force_magnitude
                if force_magnitude > np.linalg.norm(contact_normal):
                    contact_normal = force[:3]
        
        return total_force, contact_normal
    
    def task_state_machine(self):
        """Fixed task state machine - targets don't change randomly!"""
        current_time = self.data.time
        ee_pos = self.get_ee_position()
        object_pos = self.get_object_position()
        
        if self.task_state == 'IDLE':
            self.control_mode = 'joint'
            self.q_desired = self.q_home.copy()
            
        elif self.task_state == 'MOVE_TO_START':
            self.control_mode = 'cartesian'
            # Target was set in start_push_task() and NEVER changes here!
            
            distance_to_target = np.linalg.norm(ee_pos - self.current_target)
            
            # Check for progress
            if distance_to_target < self.last_distance - 0.001:  # Making progress
                self.stuck_counter = 0
                self.last_distance = distance_to_target
            else:
                self.stuck_counter += 1
            
            # Log progress every 3 seconds for cleaner output
            if int(current_time * 0.33) % 1 == 0:
                self.get_logger().info(f'MOVE_TO_START: Distance = {distance_to_target:.4f}m (tolerance: {self.tolerance:.4f}m)')
                self.get_logger().info(f'Moving to position behind object: {np.round(ee_pos, 3)} -> {np.round(self.current_target, 3)}')
            
            # Success condition
            if distance_to_target < self.tolerance:
                self.task_state = 'APPROACH_OBJECT'
                self.waypoint_index = 1
                self.current_target = self.push_waypoints[1].copy()  # Set next target
                self.state_start_time = current_time
                self.last_distance = float('inf')
                self.stuck_counter = 0
                self.get_logger().info('SUCCESS: Reached approach position behind object!')
                self.get_logger().info(f'Now approaching object for contact: {self.current_target}')
                
            # Timeout or stuck detection
            elif current_time - self.state_start_time > 30.0 or self.stuck_counter > 5000:
                self.get_logger().warn('MOVE_TO_START: Taking too long, proceeding anyway')
                self.task_state = 'APPROACH_OBJECT'
                self.waypoint_index = 1
                self.current_target = self.push_waypoints[1].copy()
                
        elif self.task_state == 'APPROACH_OBJECT':
            self.control_mode = 'cartesian'
            
            distance_to_target = np.linalg.norm(ee_pos - self.current_target)
            contact_force, _ = self.get_contact_force()
            
            if int(current_time * 0.33) % 1 == 0:
                self.get_logger().info(f'APPROACH_OBJECT: Distance = {distance_to_target:.4f}m, Force = {contact_force:.2f}N')
                self.get_logger().info(f'Approaching object for contact: EE at {np.round(ee_pos, 3)}')
            
            if distance_to_target < self.tolerance or contact_force > 5.0:
                self.task_state = 'PUSH_OBJECT'
                self.waypoint_index = 2
                self.current_target = self.push_waypoints[2].copy()
                self.state_start_time = current_time
                self.get_logger().info('SUCCESS: Made contact with object!')
                self.get_logger().info(f'Now pushing object to target: {self.current_target}')
                self.get_logger().info(f'Object should move from {self.object_start_pos[:2]} to {self.object_target_pos[:2]}')
                
        elif self.task_state == 'PUSH_OBJECT':
            self.control_mode = 'cartesian'
            
            object_distance_to_target = np.linalg.norm(object_pos[:2] - self.object_target_pos[:2])
            ee_distance_to_target = np.linalg.norm(ee_pos - self.current_target)
            contact_force, _ = self.get_contact_force()
            
            if int(current_time * 0.33) % 1 == 0:
                self.get_logger().info(f'PUSH_OBJECT: EE dist = {ee_distance_to_target:.4f}m, Obj dist = {object_distance_to_target:.4f}m')
                self.get_logger().info(f'Contact force: {contact_force:.2f}N')
                self.get_logger().info(f'Object position: {np.round(object_pos, 3)} (target: {np.round(self.object_target_pos, 3)})')
            
            if object_distance_to_target < self.push_tolerance or ee_distance_to_target < self.tolerance:
                self.task_state = 'DONE'
                self.get_logger().info('SUCCESS: Object successfully pushed to target!')
                self.get_logger().info(f'Final object position: {np.round(object_pos, 3)}')
                self.get_logger().info(f'Push distance achieved: {np.linalg.norm(object_pos[:2] - self.object_start_pos[:2]):.3f}m')
                
        elif self.task_state == 'DONE':
            self.control_mode = 'joint'
            self.q_desired = self.q_home.copy()
    
    def joint_pd_control(self):
        """Enhanced joint-space PD control"""
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()
        
        q_error = self.q_desired - q_current
        qd_error = self.qd_desired - qd_current
        
        tau_pd = self.kp_joint * q_error + self.kd_joint * qd_error
        tau_gravity = self.data.qfrc_bias[:self.n_joints].copy()
        
        return tau_pd + tau_gravity
    
    def cartesian_pd_control(self):
        """MUCH MORE AGGRESSIVE Cartesian control"""
        ee_pos = self.get_ee_position()
        jacobian = self.compute_jacobian()
        
        # Position error
        pos_error = self.current_target - ee_pos
        
        # Velocity error
        try:
            ee_vel = jacobian @ self.data.qvel[:self.n_joints]
            vel_error = np.zeros(3) - ee_vel
        except:
            vel_error = np.zeros(3)
        
        # MUCH HIGHER Cartesian PD control
        F_cartesian = self.kp_cart_pos * pos_error + self.kd_cart_pos * vel_error
        
        # Enhanced pushing forces during contact
        if self.task_state == 'PUSH_OBJECT':
            # Apply strong forward pushing force
            push_direction = np.array([1.0, 0.0, 0.0])  # +X direction (toward 0.7)
            additional_force = push_direction * self.contact_force * 2  # Stronger push
            F_cartesian += additional_force
            
            # Apply moderate downward force to maintain contact
            F_cartesian[2] -= 8.0  # Downward force for contact maintenance
        
        # More robust pseudo-inverse
        try:
            lambda_damping = 0.001  # Smaller damping for stronger control
            J_T = jacobian.T
            JJT = jacobian @ J_T
            tau_cartesian = J_T @ np.linalg.solve(JJT + lambda_damping * np.eye(3), F_cartesian)
        except:
            tau_cartesian = jacobian.T @ F_cartesian
        
        # Add gravity compensation
        tau_gravity = self.data.qfrc_bias[:self.n_joints].copy()
        
        return tau_cartesian + tau_gravity
    
    def simulation_loop(self):
        """Main simulation and control loop"""
        loop_counter = 0
        
        while rclpy.ok():
            try:
                self.task_state_machine()
                
                if self.control_mode == 'joint':
                    tau = self.joint_pd_control()
                else:
                    tau = self.cartesian_pd_control()
                
                # Apply torque limits
                tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
                tau_clipped = np.clip(tau, -tau_max, tau_max)
                
                # Log when torques are clipped (less frequently)
                if not np.allclose(tau, tau_clipped) and loop_counter % 2000 == 0:
                    self.get_logger().warn(f"⚠️  Torques clipped from {np.round(np.max(np.abs(tau)), 1)} to {np.round(np.max(np.abs(tau_clipped)), 1)}")
                
                self.data.ctrl[:self.n_joints] = tau_clipped
                
                mujoco.mj_step(self.model, self.data)
                
                if self.viewer.is_running():
                    self.viewer.sync()
                
                loop_counter += 1
                time.sleep(self.dt)
                
            except Exception as e:
                self.get_logger().error(f'❌ Simulation loop error: {e}')
                time.sleep(self.dt)
    
    def publish_states(self):
        """Publish robot and object states"""
        try:
            timestamp = self.get_clock().now().to_msg()
            
            # Joint states
            joint_state = JointState()
            joint_state.header.stamp = timestamp
            joint_state.name = self.joint_names
            joint_state.position = self.data.qpos[:self.n_joints].tolist()
            joint_state.velocity = self.data.qvel[:self.n_joints].tolist()
            joint_state.effort = self.data.ctrl[:self.n_joints].tolist()
            self.joint_state_pub.publish(joint_state)
            
            # End-effector pose
            try:
                ee_pos = self.get_ee_position()
                ee_pose = PoseStamped()
                ee_pose.header.stamp = timestamp
                ee_pose.header.frame_id = "world"
                ee_pose.pose.position.x = float(ee_pos[0])
                ee_pose.pose.position.y = float(ee_pos[1])
                ee_pose.pose.position.z = float(ee_pos[2])
                ee_pose.pose.orientation.w = 1.0
                self.ee_pose_pub.publish(ee_pose)
            except Exception as e:
                self.get_logger().error(f'EE pose publishing error: {e}')
            
            # Object pose
            try:
                object_pos = self.get_object_position()
                object_pose = PoseStamped()
                object_pose.header.stamp = timestamp
                object_pose.header.frame_id = "world"
                object_pose.pose.position.x = float(object_pos[0])
                object_pose.pose.position.y = float(object_pos[1])
                object_pose.pose.position.z = float(object_pos[2])
                object_pose.pose.orientation.w = 1.0
                self.object_pose_pub.publish(object_pose)
            except Exception as e:
                self.get_logger().error(f'Object pose publishing error: {e}')
            
            # Task state
            try:
                state_msg = String()
                state_msg.data = self.task_state
                self.task_state_pub.publish(state_msg)
            except Exception as e:
                self.get_logger().error(f'Task state publishing error: {e}')
            
            # Contact force
            try:
                contact_force, contact_normal = self.get_contact_force()
                force_msg = Vector3()
                force_msg.x = float(contact_normal[0]) if len(contact_normal) > 0 else 0.0
                force_msg.y = float(contact_normal[1]) if len(contact_normal) > 1 else 0.0
                force_msg.z = float(contact_force)
                self.push_force_pub.publish(force_msg)
            except Exception as e:
                self.get_logger().error(f'Force publishing error: {e}')
                    
        except Exception as e:
            self.get_logger().error(f'❌ Publishing error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    controller = FixedObjectPushController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()