#!/usr/bin/env python3
"""
Fixed trajectory generator with proper frequency and smooth trajectory
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
import numpy as np
import time


class ImprovedTrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('improved_trajectory_generator')
        
        self.publish_freq = 50  # Hz - matches robot control frequency
        
        # Trajectory parameters
        self.trajectory_type = 'figure8'  # or 'step' for testing
        self.figure8_period = 20.0  # seconds for one complete figure-8
        self.figure8_scale = 0.15  # smaller for better tracking
        
        # Center position (home position of robot)
        self.center = np.array([0.5, 0.0, 0.3])
        
        # Publishers
        self.traj_pub = self.create_publisher(
            PoseStamped,  # Use PoseStamped for consistency
            '/trajectory/target_pose', 
            10
        )
        
        # Also publish as Point for backward compatibility
        self.point_pub = self.create_publisher(
            Point,
            '/local_robot/cartesian_commands',
            10
        )
        
        # Start time
        self.start_time = self.get_clock().now()
        
        # Timer for publishing
        self.timer = self.create_timer(1.0/self.publish_freq, self.publish_trajectory)
        
        self.get_logger().info(f'Trajectory generator started at {self.publish_freq} Hz')
        
    def publish_trajectory(self):
        """Publish trajectory point."""
        # Calculate elapsed time
        current_time = self.get_clock().now()
        elapsed = (current_time - self.start_time).nanoseconds / 1e9
        
        # Generate position based on trajectory type
        if self.trajectory_type == 'figure8':
            position = self.generate_figure8(elapsed)
        else:
            position = self.generate_step(elapsed)
        
        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time.to_msg()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        
        # Set orientation to default (pointing forward)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        
        # Publish as PoseStamped
        self.traj_pub.publish(pose_msg)
        
        # Also publish as Point
        point_msg = Point()
        point_msg.x = float(position[0])
        point_msg.y = float(position[1])
        point_msg.z = float(position[2])
        self.point_pub.publish(point_msg)
        
        # Log periodically
        if int(elapsed * self.publish_freq) % self.publish_freq == 0:  # Log once per second
            self.get_logger().info(
                f'Time: {elapsed:.1f}s, Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]'
            )
    
    def generate_figure8(self, t):
        """Generate smooth figure-8 trajectory."""
        # Angular frequency
        omega = 2 * np.pi / self.figure8_period
        
        # Figure-8 parametric equations
        x = self.center[0] + self.figure8_scale * np.sin(omega * t)
        y = self.center[1] + self.figure8_scale * np.sin(2 * omega * t) / 2
        z = self.center[2]  # Keep Z constant
        
        return np.array([x, y, z])
    
    def generate_step(self, t):
        """Generate step trajectory for testing."""
        # Change position every 5 seconds
        step_duration = 5.0
        step_num = int(t / step_duration) % 4
        
        offsets = [
            [0.1, 0.0, 0.0],   # Right
            [0.0, 0.1, 0.0],   # Forward
            [-0.1, 0.0, 0.0],  # Left
            [0.0, -0.1, 0.0]   # Back
        ]
        
        return self.center + np.array(offsets[step_num])


def main(args=None):
    rclpy.init(args=args)
    
    try:
        generator = ImprovedTrajectoryGenerator()
        rclpy.spin(generator)
    except KeyboardInterrupt:
        print("\nTrajectory generator stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'generator' in locals():
            generator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()