#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseStamped, Twist

class ControllerTester(Node):
    def __init__(self):
        super().__init__('controller_tester')
        
        # Publishers
        self.joint_target_pub = self.create_publisher(Float64MultiArray, '/joint_target', 10)
        self.cartesian_target_pub = self.create_publisher(PoseStamped, '/cartesian_target', 10)
        self.cartesian_vel_pub = self.create_publisher(Twist, '/cartesian_velocity', 10)
        self.mode_pub = self.create_publisher(String, '/control_mode', 10)
        
        # Wait for controller to start
        time.sleep(2.0)
        
        self.get_logger().info('🧪 Controller Tester Ready!')
    
    def test_joint_control(self):
        """Test joint space control"""
        self.get_logger().info('Testing Joint Space Control...')
        
        # Switch to joint mode
        mode_msg = String()
        mode_msg.data = 'joint'
        self.mode_pub.publish(mode_msg)
        time.sleep(0.5)
        
        # Test positions
        test_positions = [
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],  # Home
            [0.5, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],  # Joint 1 moved
            [0.0, -0.5, 0.0, -2.0, 0.0, 1.2, 0.785],        # Different config
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],  # Back to home
        ]
        
        for i, pos in enumerate(test_positions):
            self.get_logger().info(f'Moving to position {i+1}: {np.round(pos, 3)}')
            
            msg = Float64MultiArray()
            msg.data = pos
            self.joint_target_pub.publish(msg)
            
            time.sleep(3.0)  # Wait for movement
    
    def test_cartesian_control(self):
        """Test Cartesian space control"""
        self.get_logger().info('Testing Cartesian Space Control...')
        
        # Switch to Cartesian mode
        mode_msg = String()
        mode_msg.data = 'cartesian'
        self.mode_pub.publish(mode_msg)
        time.sleep(0.5)
        
        # Test positions
        test_poses = [
            ([0.4, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0]),     # Center
            ([0.4, 0.2, 0.4], [1.0, 0.0, 0.0, 0.0]),     # Move right
            ([0.4, -0.2, 0.4], [1.0, 0.0, 0.0, 0.0]),    # Move left
            ([0.4, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]),     # Move up
            ([0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0]),     # Move down
            ([0.4, 0.0, 0.4], [0.707, 0.0, 0.0, 0.707]), # Rotate
        ]
        
        for i, (pos, ori) in enumerate(test_poses):
            self.get_logger().info(f'Moving to Cartesian pose {i+1}: pos={np.round(pos, 3)}, ori={np.round(ori, 3)}')
            
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position.x = pos[0]
            pose_msg.pose.position.y = pos[1] 
            pose_msg.pose.position.z = pos[2]
            pose_msg.pose.orientation.w = ori[0]
            pose_msg.pose.orientation.x = ori[1]
            pose_msg.pose.orientation.y = ori[2]
            pose_msg.pose.orientation.z = ori[3]
            
            self.cartesian_target_pub.publish(pose_msg)
            time.sleep(4.0)  # Wait for movement
    
    def test_cartesian_velocity(self):
        """Test Cartesian velocity control"""
        self.get_logger().info('Testing Cartesian Velocity Control...')
        
        # Switch to Cartesian mode
        mode_msg = String()
        mode_msg.data = 'cartesian'
        self.mode_pub.publish(mode_msg)
        time.sleep(0.5)
        
        # Circular motion
        self.get_logger().info('Performing circular motion...')
        
        for t in np.linspace(0, 4*np.pi, 200):  # 2 full circles
            vel_msg = Twist()
            vel_msg.linear.x = 0.1 * np.cos(t)
            vel_msg.linear.y = 0.1 * np.sin(t)
            vel_msg.linear.z = 0.0
            vel_msg.angular.x = 0.0
            vel_msg.angular.y = 0.0
            vel_msg.angular.z = 0.1 * np.cos(t)
            
            self.cartesian_vel_pub.publish(vel_msg)
            time.sleep(0.05)
        
        # Stop motion
        vel_msg = Twist()
        self.cartesian_vel_pub.publish(vel_msg)

def main(args=None):
    rclpy.init(args=args)
    tester = ControllerTester()
    
    try:
        # Test sequence
        tester.test_joint_control()
        time.sleep(2.0)
        tester.test_cartesian_control()
        time.sleep(2.0)
        tester.test_cartesian_velocity()
        
        tester.get_logger().info('✅ All tests completed!')
        
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()