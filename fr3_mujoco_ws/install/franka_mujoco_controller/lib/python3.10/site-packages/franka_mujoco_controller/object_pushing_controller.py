#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import numpy as np
import os
import time
import threading
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseStamped, Point, Vector3
import math

# Set MuJoCo backend
os.environ['MUJOCO_GL'] = 'egl'

class ObjectPushController(Node):
    def __init__(self):
        super().__init__('object_push_controller')
        
        # Load MuJoCo model with table scene
        master_thesis_path = "/media/kai/Kai_Backup/Study/Master_Thesis/My_Master_Thesis"
        
        # Use your specific scene file
        scene_path = os.path.join(master_thesis_path, "MuJoCo_Operation", "franka_fr3_scene", "franka_with_table_scene.xml")
        
        # Fallback options if the main scene doesn't work
        scene_files = [
            scene_path,
            os.path.join(master_thesis_path, "MuJoCo_Operation", "franka_fr3_scene", "scene.xml"),
            os.path.join(master_thesis_path, "MuJoCo_Operation", "franka_fr3_scene", "fr3_with_table.xml"),
            os.path.join(master_thesis_path, "franka_mujoco_ws", "src", "mujoco_menagerie", "franka_fr3", "fr3.xml")
        ]
        
        model_loaded = False
        for scene_file in scene_files:
            if os.path.exists(scene_file):
                try:
                    self.get_logger().info(f'Trying to load: {scene_file}')
                    self.model = mujoco.MjModel.from_xml_path(scene_file)
                    self.data = mujoco.MjData(self.model)
                    self.get_logger().info(f'✅ Successfully loaded: {scene_file}')
                    model_loaded = True
                    break
                except Exception as e:
                    self.get_logger().warn(f'❌ Failed to load {scene_file}: {e}')
                    continue
        
        if not model_loaded:
            raise RuntimeError("Could not load any scene file!")
        
        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Robot parameters
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.n_joints = len(self.joint_names)
        self.ee_body_name = 'fr3_link7'
        self.object_name = 'box'  # Name of the object to push
        
        # Control parameters
        self.control_freq = 1000
        self.dt = 1.0 / self.control_freq
        
        # PD Gains - Joint Space
        self.kp_joint = np.array([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 25.0])
        self.kd_joint = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 2.5])
        
        # Cartesian PD Gains
        self.kp_cart_pos = np.array([1000.0, 1000.0, 1000.0])
        self.kd_cart_pos = np.array([100.0, 100.0, 100.0])
        
        # Push force parameters
        self.push_force_max = 50.0  # Increased maximum push force (N)
        self.contact_force = 15.0   # Desired contact force (N)
        
        # Task state machine
        self.task_state = 'IDLE'  # IDLE, MOVE_TO_START, APPROACH_OBJECT, PUSH_OBJECT, MOVE_TO_END, DONE
        self.state_start_time = 0.0
        self.waypoint_index = 0
        
        # Push waypoints: move EE through these positions to push the box
        self.push_waypoints = [
            np.array([0.3, 0.0, 0.1]),  # Start position (behind object)
            np.array([0.5, 0.0, 0.1]),  # Object position (make contact)
            np.array([0.7, 0.0, 0.1])   # End position (push object here)
        ]
        
        self.current_target = self.push_waypoints[0].copy()
        self.tolerance = 0.02  # Position tolerance (2cm)
        
        # Object tracking
        self.object_start_pos = np.array([0.5, 0.0, 0.05])  # Expected object start
        self.object_target_pos = np.array([0.7, 0.0, 0.05])  # Where we want to push it
        
        # Home position
        self.q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.q_desired = self.q_home.copy()
        self.qd_desired = np.zeros(self.n_joints)
        
        # Control mode
        self.control_mode = 'cartesian'  # 'joint' or 'cartesian'
        
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
        
        # Try to set object to start position if it exists
        self.setup_object_position()
        
        mujoco.mj_forward(self.model, self.data)
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Timer for publishing
        self.timer = self.create_timer(1.0/50.0, self.publish_states)
        
        self.get_logger().info('🚀 Object Push Controller Started!')
        self.get_logger().info(f'Task state: {self.task_state}')
        self.get_logger().info('Send "start" to /start_push to begin pushing task')
        self.get_logger().info(f'Push waypoints: {self.push_waypoints}')
        
    def setup_object_position(self):
        """Setup object at starting position"""
        try:
            # Find object body
            object_id = self.model.body(self.object_name).id
            self.get_logger().info(f'Found object "{self.object_name}" with ID: {object_id}')
            
            # Check if object has joints (moveable)
            if hasattr(self.model.body(self.object_name), 'jntadr') and self.model.body(self.object_name).jntadr[0] >= 0:
                object_joint_id = self.model.body(self.object_name).jntadr[0]
                # Set object position and orientation
                self.data.qpos[self.n_joints + object_joint_id:self.n_joints + object_joint_id + 7] = [
                    self.object_start_pos[0], self.object_start_pos[1], self.object_start_pos[2],
                    1.0, 0.0, 0.0, 0.0  # quaternion (w, x, y, z)
                ]
                self.get_logger().info(f'Set object position to: {self.object_start_pos}')
            else:
                self.get_logger().warn("Object appears to be fixed (no joints)")
                
        except Exception as e:
            self.get_logger().warn(f"Could not setup object position: {e}")
    
    def start_push_task(self, msg):
        """Start pushing task"""
        if msg.data == "start":
            self.task_state = 'MOVE_TO_START'
            self.state_start_time = self.data.time
            self.waypoint_index = 0
            self.current_target = self.push_waypoints[0].copy()
            self.control_mode = 'cartesian'
            
            self.get_logger().info('🎯 Starting push task!')
            self.get_logger().info(f'Moving to first waypoint: {self.current_target}')
    
    def reset_task(self, msg):
        """Reset task to initial state"""
        self.task_state = 'IDLE'
        self.q_desired = self.q_home.copy()
        self.control_mode = 'joint'
        self.waypoint_index = 0
        
        # Reset object position
        self.setup_object_position()
        
        self.get_logger().info('🔄 Task reset to IDLE')
    
    def get_object_position(self):
        """Get current object position"""
        try:
            object_id = self.model.body(self.object_name).id
            return self.data.xpos[object_id].copy()
        except:
            return self.object_start_pos.copy()
    
    def get_object_velocity(self):
        """Get current object velocity"""
        try:
            object_id = self.model.body(self.object_name).id
            object_joint_id = self.model.body(self.object_name).jntadr[0]
            if object_joint_id >= 0:
                return self.data.qvel[self.n_joints + object_joint_id:self.n_joints + object_joint_id + 3].copy()
        except:
            pass
        return np.zeros(3)
    
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
            return np.vstack([jacp[:, :self.n_joints], jacr[:, :self.n_joints]])
        except:
            return np.zeros((6, self.n_joints))
    
    def get_contact_force(self):
        """Get contact force between end-effector and object"""
        total_force = 0.0
        contact_normal = np.zeros(3)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get contact force
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            force_magnitude = np.linalg.norm(force[:3])
            
            if force_magnitude > 0.1:  # Only consider significant forces
                total_force += force_magnitude
                if force_magnitude > np.linalg.norm(contact_normal):
                    contact_normal = force[:3]
        
        return total_force, contact_normal
    
    def task_state_machine(self):
        """Main task state machine for object pushing"""
        current_time = self.data.time
        ee_pos = self.get_ee_position()
        object_pos = self.get_object_position()
        
        if self.task_state == 'IDLE':
            # Do nothing, wait for command
            self.control_mode = 'joint'
            self.q_desired = self.q_home.copy()
            
        elif self.task_state == 'MOVE_TO_START':
            # Move to first waypoint (behind object)
            self.control_mode = 'cartesian'
            self.current_target = self.push_waypoints[0].copy()
            
            distance_to_target = np.linalg.norm(ee_pos - self.current_target)
            
            if distance_to_target < self.tolerance:
                self.task_state = 'APPROACH_OBJECT'
                self.waypoint_index = 1
                self.state_start_time = current_time
                self.get_logger().info('📍 Reached start position, approaching object')
                
        elif self.task_state == 'APPROACH_OBJECT':
            # Move to object position (make contact)
            self.control_mode = 'cartesian'
            self.current_target = self.push_waypoints[1].copy()
            
            distance_to_target = np.linalg.norm(ee_pos - self.current_target)
            contact_force, _ = self.get_contact_force()
            
            # Check if we've reached the object or made contact
            if distance_to_target < self.tolerance or contact_force > 5.0:
                self.task_state = 'PUSH_OBJECT'
                self.waypoint_index = 2
                self.state_start_time = current_time
                self.get_logger().info('🤝 Contact with object, starting push')
                self.get_logger().info(f'Contact force: {contact_force:.2f}N')
                
        elif self.task_state == 'PUSH_OBJECT':
            # Push object to end position
            self.control_mode = 'cartesian'
            self.current_target = self.push_waypoints[2].copy()
            
            # Check object progress
            object_distance_to_target = np.linalg.norm(object_pos[:2] - self.object_target_pos[:2])
            ee_distance_to_target = np.linalg.norm(ee_pos - self.current_target)
            
            # Success condition: object close to target OR end-effector reached end position
            if object_distance_to_target < 0.05 or ee_distance_to_target < self.tolerance:
                self.task_state = 'MOVE_TO_END'
                self.state_start_time = current_time
                self.get_logger().info('🎯 Push completed, moving to final position')
                self.get_logger().info(f'Final object position: {object_pos}')
                
        elif self.task_state == 'MOVE_TO_END':
            # Move to final position (away from object)
            self.control_mode = 'cartesian'
            final_pos = self.push_waypoints[2].copy()
            final_pos[2] += 0.1  # Lift up 10cm
            self.current_target = final_pos
            
            distance_to_target = np.linalg.norm(ee_pos - self.current_target)
            
            if distance_to_target < self.tolerance:
                self.task_state = 'DONE'
                self.get_logger().info('✅ Push task completed successfully!')
                
        elif self.task_state == 'DONE':
            # Task completed, return to home
            self.control_mode = 'joint'
            self.q_desired = self.q_home.copy()
    
    def joint_pd_control(self):
        """Joint-space PD control"""
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()
        
        q_error = self.q_desired - q_current
        qd_error = self.qd_desired - qd_current
        
        tau_pd = self.kp_joint * q_error + self.kd_joint * qd_error
        tau_gravity = self.data.qfrc_bias[:self.n_joints].copy()
        
        return tau_pd + tau_gravity
    
    def cartesian_pd_control(self):
        """Enhanced Cartesian-space PD control with pushing force"""
        ee_pos = self.get_ee_position()
        jacobian = self.compute_jacobian()
        
        # Position error
        pos_error = self.current_target - ee_pos
        
        # Velocity error (assume zero target velocity)
        ee_vel = jacobian[:3, :] @ self.data.qvel[:self.n_joints]
        vel_error = np.zeros(3) - ee_vel
        
        # Basic Cartesian PD control
        F_cartesian = self.kp_cart_pos * pos_error + self.kd_cart_pos * vel_error
        
        # Add extra pushing force during PUSH_OBJECT state
        if self.task_state == 'PUSH_OBJECT':
            # Apply additional force in the pushing direction
            push_direction = np.array([1.0, 0.0, 0.0])  # Push in +X direction
            additional_force = push_direction * self.contact_force
            F_cartesian += additional_force
            
            # Add downward force to maintain contact
            F_cartesian[2] -= 5.0  # 5N downward force
        
        # Map to joint torques (position only)
        tau_cartesian = jacobian[:3, :].T @ F_cartesian
        
        # Add gravity compensation
        tau_gravity = self.data.qfrc_bias[:self.n_joints].copy()
        
        return tau_cartesian + tau_gravity
    
    def simulation_loop(self):
        """Main simulation and control loop"""
        while rclpy.ok():
            try:
                # Update task state machine
                self.task_state_machine()
                
                # Compute control torques
                if self.control_mode == 'joint':
                    tau = self.joint_pd_control()
                else:
                    tau = self.cartesian_pd_control()
                
                # Apply torque limits
                tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
                tau = np.clip(tau, -tau_max, tau_max)
                
                # Apply control torques
                self.data.ctrl[:self.n_joints] = tau
                
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
            ee_pos = self.get_ee_position()
            ee_pose = PoseStamped()
            ee_pose.header.stamp = timestamp
            ee_pose.header.frame_id = "world"
            ee_pose.pose.position.x = float(ee_pos[0])
            ee_pose.pose.position.y = float(ee_pos[1])
            ee_pose.pose.position.z = float(ee_pos[2])
            ee_pose.pose.orientation.w = 1.0
            self.ee_pose_pub.publish(ee_pose)
            
            # Object pose
            object_pos = self.get_object_position()
            object_pose = PoseStamped()
            object_pose.header.stamp = timestamp
            object_pose.header.frame_id = "world"
            object_pose.pose.position.x = float(object_pos[0])
            object_pose.pose.position.y = float(object_pos[1])
            object_pose.pose.position.z = float(object_pos[2])
            object_pose.pose.orientation.w = 1.0
            self.object_pose_pub.publish(object_pose)
            
            # Task state
            state_msg = String()
            state_msg.data = self.task_state
            self.task_state_pub.publish(state_msg)
            
            # Contact force
            contact_force, contact_normal = self.get_contact_force()
            force_msg = Vector3()
            force_msg.x = float(contact_normal[0])
            force_msg.y = float(contact_normal[1])
            force_msg.z = float(contact_force)
            self.push_force_pub.publish(force_msg)
            
            # Log state periodically
            if int(self.data.time * 4) % 20 == 0:  # Every 5 seconds
                ee_pos = self.get_ee_position()
                obj_pos = self.get_object_position()
                self.get_logger().info(f'State: {self.task_state} | EE: {np.round(ee_pos, 3)} | Obj: {np.round(obj_pos, 3)} | Target: {np.round(self.current_target, 3)}')
                
        except Exception as e:
            self.get_logger().error(f'Publishing error: {e}')

def main(args=None):
    rclpy.init(args=args)
    controller = ObjectPushController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()