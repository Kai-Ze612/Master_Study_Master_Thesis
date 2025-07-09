"""
Create a trajectory script for local robot
The trajectory is 8 shaped and continuous
"""

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import numpy as np
import time

class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('continuous_trajectory_publisher')
        
        # Trajectory parameters
        # This will publish command very 0.1 seconds
        self.publish_freq = 10.0  # Hz
        
        self._init_ROS_interfaces()
        
        # Timer for continuous publishing
        self.timer = self.create_timer(1.0/self.publish_freq, self.publish_trajectory_point)

        # Trajectory state
        # Time step is used to count how many points have been published
        self.time_step = 0
        # will get current time when trajectory starts
        self.trajectory_start_time = time.time()
       
    def _init_ROS_interfaces(self):
        """Initialize ROS interfaces."""
        
        # Create a publisher
        self.publisher = self.create_publisher(
            Point,
            '/local_robot/cartesian_commands', 
            100 # Queue size for the publisher
        )
    
    def publish_trajectory_point(self):
        """Publish a single trajectory point based on current time."""
        current_time = time.time() - self.trajectory_start_time
        
        if current_time < 5:
            position = self.generate_figure8_trajectory(0)
        
        else:
            position = self.generate_figure8_trajectory(current_time - 5)
        
        # Create and publish message
        msg = Point()
        msg.x = float(position[0])
        msg.y = float(position[1])
        msg.z = float(position[2])
        
        self.publisher.publish(msg)
        
        # Log every 50 points to avoid spam
        if self.time_step % 50 == 0:
            self.get_logger().info(
                f'Published point {self.time_step}: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]'
            )
        
        # Increment time step, this is used to track how many points have been published
        self.time_step += 1
   
    def generate_figure8_trajectory(self, t):
        """Generate figure-8 trajectory."""
        # Figure-8 parameters
        center = np.array([0.5, 0, 0.3])
        scale = 0.2  # Scale factor for the figure-8 size
        period = 10  # seconds for one complete figure-8
        
        angle = 2 * np.pi * t / period
        
        x = center[0] + scale * np.sin(angle)
        y = center[1] + scale * np.sin(angle) * np.cos(angle)
        z = center[2]
        
        return np.array([x, y, z])

def main(args=None):
    rclpy.init(args=args)
    
    try:
        trajectory_publisher = TrajectoryGenerator()
        rclpy.spin(trajectory_publisher)

    except KeyboardInterrupt:
        print("Trajectory publishing stops")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        trajectory_publisher.destroy_node()
        rclpy.shutdown()
        print("Trajectory publisher shutdown complete")

if __name__ == '__main__':
    main()