#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import String
from geometry_msgs.msg import Point

class ManualPushControl(Node):
    def __init__(self):
        super().__init__('manual_push_control')
        
        # Publishers
        self.push_target_pub = self.create_publisher(Point, '/push_target', 10)
        self.reset_task_pub = self.create_publisher(String, '/reset_task', 10)
        
        time.sleep(1.0)
        self.get_logger().info('🎮 Manual Push Control Ready!')
        self.print_commands()
    
    def print_commands(self):
        print("\n🎮 Manual Push Control Commands:")
        print("1 - Push to (0.3, 0.3)   [Northeast]")
        print("2 - Push to (0.7, 0.0)   [East]") 
        print("3 - Push to (0.3, -0.3)  [Southeast]")
        print("4 - Push to (0.5, 0.0)   [Center]")
        print("r - Reset task")
        print("q - Quit")
        print("Enter command: ", end='', flush=True)
    
    def send_push_command(self, x, y, z=0.05):
        """Send push command"""
        target = Point()
        target.x = x
        target.y = y  
        target.z = z
        self.push_target_pub.publish(target)
        self.get_logger().info(f'📤 Sent push target: ({x:.1f}, {y:.1f}, {z:.1f})')
    
    def reset_task(self):
        """Reset the task"""
        reset_msg = String()
        reset_msg.data = 'reset'
        self.reset_task_pub.publish(reset_msg)
        self.get_logger().info('# Object Pushing PD Controller for Franka FR3