"""
This script creates a bridge between ROS2 and MuJoCo.
Specifically, it loads a Franka FR3 model, and use a ROS2 node to control the robot in simulation.
Specifically, it subscribes to joint commands and publishes joint states and end-effector poses.

In this file, we explore how to create a ROS2 node that integrates with MuJoCo for simulating a Franka robot.
And try to use publisher and subscriber to control the robot in simulation.
"""

# Loading python modules
import time # for control frequency
import threading # for running multiple things at once

# Loading ROS2 modules
import rclpy
from rclpy.node import Node

# Loading MuJoCo modules
import mujoco
import mujoco.viewer

# Loading ROS2 message types
from sensor_msgs.msg import JointState # standard format for robot joint information
from std_msgs.msg import Float64MultiArray # simple list of numbers
from geometry_msgs.msg import PoseStamped # position and orientation in space

class Bridge_ROS2_MUJOCO(Node):
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
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7'
        ]
        self.control_freq = 500  # Hz
        self.publish_freq = 100  # Hz
        
        # Model path
        self.model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller/models/franka_fr3/fr3_with_moveable_box.xml"
            
    def _load_mujoco_model(self):
        """Load and initialize the MuJoCo model."""
        # get_logger is a ROS2 function to print messages to the console
        # Basic syntax: get_logger().info('message')
        self.get_logger().info(f'Loading MuJoCo model from: {self.model_path}')
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def _init_ros_interfaces(self):
        """Initialize ROS2 publishers and subscribers."""
        
        # Create ROS2 publishers
        # create_publisher is a ROS2 function to send data to other program
        # Basic syntax: create_publisher(MessageType, topic_name, queue_size)
        # queue_size is the number of messages that can be buffered before sending
        self.joint_state_pub = self.create_publisher(
            JointState, '/joint_states', 10)
        self.ee_pose_pub = self.create_publisher(
            PoseStamped, '/ee_pose', 10)
        
        ## Create ROS2 subscribers
        self.cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/joint_commands',
            self.joint_command_callback,
            10)
       
        ## Timer for publishing
        ## basic syntax: create_timer(frequency, publish_function)
        # The timer will call the publish_states function at the specified frequency
        self.timer = self.create_timer(1.0/self.publish_freq, self.publish_states)
    
    ## The subscriber automatically calls this function when a message is received
    def joint_command_callback(self, msg):
        # Trigger when someone publishes to /joint_commands topic
        # check if the message has the expected number of joint commands
        if len(msg.data) == 7:
            self.data.ctrl[:7] = msg.data
        else:
            self.get_logger().warn(f'Expected 7 joint commands, got {len(msg.data)}')
    
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
        ee_id = self.model.body('fr3_link7').id
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
   
    def _start_simulation(self):
        """Start the MuJoCo simulation thread."""
        self.simulation_thread = threading.Thread(target=self.simulation_loop) # create a separate thread to run physics simulation
        self.simulation_thread.daemon = True # Thread will stop when main program stops
        self.simulation_thread.start() # Begin running the simulation loop
    
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

def main(args=None):
    """ROS2 main entry point."""
    rclpy.init(args=args)
    
    try:
        controller = Bridge_ROS2_MUJOCO() # create a ROS2 node
        rclpy.spin(controller) # Keep the node running
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
