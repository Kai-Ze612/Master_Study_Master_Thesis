"""Remote_Robot
Remote robot node that receives position commands from local robot, both robots apply traditional PD control under stochastic delays.
This is the baseline for comparison with RL controllers.

To run this node:
1. cd /media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller
2. colcon build --packages-select franka_mujoco_controller
3. source install/setup.bash
4. ros2 launch franka_mujoco_controller teleoperation_2.launch.py
5. In a separate terminal input position commands: ros2 topic pub --once /local_robot/cartesian_commands geometry_msgs/msg/Point "{x: 0.5, y: 0.2, z: 0.3}"
6. Record trajectory data: ros2 bag record /local_robot/ee_pose /remote_robot/ee_pose -o trajectory_data
"""

#!/usr/bin/env python3

# Python libraries
import time
import threading
import os
import numpy as np
from scipy.optimize import minimize
from collections import deque

# Ros2 libraries
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# MuJoCo libraries
import mujoco
import mujoco.viewer

# Set different rendering options
os.environ['MUJOCO_GL'] = 'egl'

# Generate time delay pattern
class StochasticDelaySimulator:
    def __init__(self, experiment_config=1):
        self.experiment_config = experiment_config
        
        if experiment_config == 1:
            self.base_action_delay = 9    # 90ms base
            self.base_obs_delay = 5       # 50ms base  
            self.stochastic_range = 4     # ±40ms variation
            self.experiment_name = "90-130ms"
            
        elif experiment_config == 2:
            self.base_action_delay = 17   # 170ms base
            self.base_obs_delay = 10      # 100ms base
            self.stochastic_range = 4     # ±40ms variation
            self.experiment_name = "170-210ms"
            
        elif experiment_config == 3:
            self.base_action_delay = 25   # 250ms base
            self.base_obs_delay = 15      # 150ms base
            self.stochastic_range = 4     # ±40ms variation
            self.experiment_name = "250-290ms"
        
        self.current_action_delay = self.base_action_delay
        self.current_obs_delay = self.base_obs_delay
    
    def update_delays(self):
        """Update delays with stochastic variation."""
        # add random variation to delays
        action_variation = np.random.randint(-self.stochastic_range, self.stochastic_range + 1)
        obs_variation = np.random.randint(-self.stochastic_range, self.stochastic_range + 1)
        
        # ensure delays are at least 1 step
        self.current_action_delay = max(1, self.base_action_delay + action_variation)
        self.current_obs_delay = max(1, self.base_obs_delay + obs_variation)
    
    def get_current_delays(self):
        """Get current delay values."""
        return {
            'action_delay_steps': self.current_action_delay,
            'obs_delay_steps': self.current_obs_delay,
            'action_delay_ms': self.current_action_delay * 10,
            'obs_delay_ms': self.current_obs_delay * 10,
            'total_delay_ms': (self.current_action_delay + self.current_obs_delay) * 10
        }

class RemoteRobotController(Node):
    def __init__(self, experiment_config=1):
        super().__init__('remote_robot_controller')
        
        self.experiment_config = experiment_config
        self._init_parameters()
        self._init_stochastic_delays()
        self._load_mujoco_model()
        self._init_ros_interfaces()
        self._start_simulation()

    def _init_parameters(self):
        """Initialize controller parameters."""
        self.joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        
        self.control_freq = 100  # Hz
        self.publish_freq = 20  # Hz
        
        self.kp = np.array([120, 120, 120, 80, 80, 60, 60]) # Proportional gains for each joint (stiffness)
        self.kd = np.array([20, 20, 20, 15, 15, 12, 12]) # Derivative gains for each joint (damping)

        # Target joint positions
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
        self.model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller/models/franka_fr3/fr3_remote.xml"
    
    # initialize stochastic delays for trajectory following
    def _init_stochastic_delays(self):
        self.delay_simulator = StochasticDelaySimulator(self.experiment_config)

        # Initialize step counter for action delay timing
        self.current_step = 0
        
        self.step_timer = self.create_timer(1/20, self.increment_step)
        max_delay_buffer = 100
        self.local_waypoint_buffer = deque(maxlen=max_delay_buffer)
        self.delayed_waypoint_buffer = deque()
    
    def increment_step(self):
        self.current_step += 1
        if self.current_step % 10 == 0:  # Log every 10 steps
            self.get_logger().info(f"REMOTE: Current step: {self.current_step}")

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
        
        # publisher for remote robot ee position
        self.ee_pose_pub = self.create_publisher(
            PoseStamped, '/remote_robot/ee_pose', 10)

        # subscriber for remote robot
        # subscribe to ee pose of local robot
        self.local_ee_pose_sub = self.create_subscription(
            PoseStamped, '/local_robot/ee_pose',
            self.local_ee_pose_callback, 10)

        self.timer = self.create_timer(1.0/self.publish_freq, self.publish_states)
    
    # apply observation delay
    def _apply_observation_delay(self):
        delay_steps = self.delay_simulator.current_obs_delay
        
        # Check if we have enough data in buffer
        if len(self.local_waypoint_buffer) >= delay_steps:
            # Get position from delay_steps ago
            delayed_position = self.local_waypoint_buffer[-delay_steps]
            return delayed_position
        else:
            # Not enough data yet, return None
            return None
    
    def _apply_action_delay(self, position):
        """Queue command for future execution based on action delay."""
        delay_steps = self.delay_simulator.current_action_delay
        execute_at_step = self.current_step + delay_steps
        
        delayed_command = {
            'position': position.copy(),
            'execute_at_step': execute_at_step,
            'queued_at_step': self.current_step  # For debugging
        }
        
        self.delayed_waypoint_buffer.append(delayed_command)
    
    # (subscriber) Receive Cartesian position from local robot and compute inverse kinematics
    def local_ee_pose_callback(self, msg):
        target_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    
        # # Add to observation buffer
        # self.local_waypoint_buffer.append(target_position.copy())
        
        # # Update stochastic delays
        # self.delay_simulator.update_delays()
        
        # # Apply observation delay
        # delayed_position = self._apply_observation_delay()
        
        # if delayed_position is not None:
        #     self._apply_action_delay(delayed_position)
            
        #     delay_info = self.delay_simulator.get_current_delays()
        #     execute_at_step = self.current_step + delay_info['action_delay_steps']
        #     self.get_logger().info(
        #         f"REMOTE: Current step {self.current_step} - Queued for step {execute_at_step}"
        #     )
        # else:
        #     self.get_logger().info(f"REMOTE: Step {self.current_step} - Waiting for observation data")
        
        self.get_logger().info(f"REMOTE: Received position [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
    
        target_joint_positions = self.inverse_kinematics(target_position)
        
        if target_joint_positions is not None:
            self.target_positions = target_joint_positions
            self.get_logger().info('REMOTE: Moving to target position (DIRECT - no delays)')
        else:
            self.get_logger().warn('REMOTE: IK failed')
        
    def _execute_delayed_commands(self):
        """Execute commands that have completed their action delay."""
        executed_count = 0
        
        # Process all ready commands (there might be multiple)
        while (len(self.delayed_waypoint_buffer) > 0 and 
            self.delayed_waypoint_buffer[0]['execute_at_step'] <= self.current_step):
            
            delayed_command = self.delayed_waypoint_buffer.popleft()
            target_position = delayed_command['position']
            
            # Execute the delayed command
            target_joint_positions = self.inverse_kinematics(target_position)
            
            if target_joint_positions is not None:
                self.target_positions = target_joint_positions
                executed_count += 1
                
                self.get_logger().info(
                    f'REMOTE: Executed delayed command from step {delayed_command["queued_at_step"]} '
                    f'at step {self.current_step} (delay: {self.current_step - delayed_command["queued_at_step"]} steps)'
                )
            else:
                self.get_logger().warn('REMOTE: IK failed for delayed command')
        
        return executed_count
    
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
                self.get_logger().info(f'Remote: IK solved! Error: {result.fun:.4f}m')
                return result.x
            else:
                self.get_logger().warn(f'REMOTE: IK failed. Error: {result.fun:.4f}m')
                return None
        except Exception as e:
            self.get_logger().error(f'REMOTE: IK optimization failed: {e}')
            return None

    # Compute PD control torques for remote robot
    def compute_pd_torques(self):
        current_positions = self.data.qpos[:7]
        current_velocities = self.data.qvel[:7]
        
        # PD control equation
        position_error = self.target_positions - current_positions
        torques = self.kp * position_error - self.kd * current_velocities
        
        # Apply force limits
        torques = np.clip(torques, -self.force_limit, self.force_limit)
        return torques
    
    def publish_states(self):
        current_time = self.get_clock().now().to_msg()
        self._publish_ee_pose(current_time)
    
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
        """Main simulation loop with delay processing."""
        while rclpy.ok():
            
            # # Execute any delayed commands that are ready
            # self._execute_delayed_commands()
            
            # Compute and apply torques
            torques = self.compute_pd_torques()
            self.data.ctrl[:7] = torques
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Update viewer
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()

            time.sleep(1.0/self.control_freq)

def main(args=None):
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

# if __name__ == '__main__':
#     main()