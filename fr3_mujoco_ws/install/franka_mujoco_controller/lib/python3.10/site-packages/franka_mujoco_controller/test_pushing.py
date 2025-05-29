#!/usr/bin/env python3

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from std_msgs.msg import String
from geometry_msgs.msg import Point, PoseStamped

class PushingTester(Node):
    def __init__(self):
        super().__init__('pushing_tester')
        
        # Publishers
        self.point_a_pub = self.create_publisher(Point, '/target_point_a', 10)
        self.point_b_pub = self.create_publisher(Point, '/target_point_b', 10)
        self.start_pub = self.create_publisher(String, '/start_pushing', 10)
        
        # Subscribers for monitoring
        self.phase_sub = self.create_subscription(
            String, '/pushing_phase', self.phase_callback, 10)
        self.object_sub = self.create_subscription(
            PoseStamped, '/object_pose', self.object_callback, 10)
        
        self.current_phase = "UNKNOWN"
        self.object_pos = np.zeros(3)
        
        # Wait for controller to start
        time.sleep(2.0)
        
        self.get_logger().info('🧪 Pushing Tester Ready!')
    
    def phase_callback(self, msg):
        """Monitor pushing phase"""
        if self.current_phase != msg.data:
            self.current_phase = msg.data
            self.get_logger().info(f'Phase changed to: {self.current_phase}')
    
    def object_callback(self, msg):
        """Monitor object position"""
        self.object_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
    
    def test_basic_push(self):
        """Test basic pushing from A to B"""
        self.get_logger().info('🔧 Testing Basic Push...')
        
        # Set points A and B
        point_a = Point()
        point_a.x, point_a.y, point_a.z = 0.4, -0.2, 0.05
        
        point_b = Point()
        point_b.x, point_b.y, point_b.z = 0.4, 0.2, 0.05
        
        self.point_a_pub.publish(point_a)
        time.sleep(0.5)
        self.point_b_pub.publish(point_b)
        time.sleep(0.5)
        
        # Start pushing
        start_msg = String()
        start_msg.data = "start"
        self.start_pub.publish(start_msg)
        
        self.get_logger().info(f'Pushing from A{[point_a.x, point_a.y, point_a.z]} to B{[point_b.x, point_b.y, point_b.z]}')
        
        # Monitor progress
        start_time = time.time()
        while time.time() - start_time < 30.0:  # 30 second timeout
            if self.current_phase == "COMPLETE":
                self.get_logger().info('✅ Push completed successfully!')
                break
            time.sleep(1.0)
            
            # Log progress every 5 seconds
            if int(time.time() - start_time) % 5 == 0:
                self.get_logger().info(f'Current object position: {np.round(self.object_pos, 3)}')
    
    def test_multiple_pushes(self):
        """Test multiple pushing tasks"""
        self.get_logger().info('🔧 Testing Multiple Pushes...')
        
        push_sequences = [
            # (point_A, point_B)
            ([0.4, -0.2, 0.05], [0.4, 0.0, 0.05]),   # Push to center
            ([0.4, 0.0, 0.05], [0.4, 0.2, 0.05]),    # Push to right
            ([0.4, 0.2, 0.05], [0.3, 0.2, 0.05]),    # Push forward
            ([0.3, 0.2, 0.05], [0.3, -0.2, 0.05]),   # Push left
        ]
        
        for i, (a, b) in enumerate(push_sequences):
            self.get_logger().info(f'Push sequence {i+1}: {a} → {b}')
            
            # Set points
            point_a = Point()
            point_a.x, point_a.y, point_a.z = a
            
            point_b = Point()
            point_b.x, point_b.y, point_b.z = b
            
            self.point_a_pub.publish(point_a)
            time.sleep(0.5)
            self.point_b_pub.publish(point_b)
            time.sleep(0.5)
            
            # Start push
            start_msg = String()
            start_msg.data = "start"
            self.start_pub.publish(start_msg)
            
            # Wait for completion
            start_time = time.time()
            while time.time() - start_time < 20.0:
                if self.current_phase == "COMPLETE":
                    break
                time.sleep(0.5)
            
            time.sleep(2.0)  # Pause between pushes
    
    def test_precision_pushing(self):
        """Test precision pushing with small movements"""
        self.get_logger().info('🔧 Testing Precision Pushing...')
        
        # Small, precise movements
        current_pos = [0.4, 0.0, 0.05]
        
        for i in range(5):
            target_pos = current_pos.copy()
            target_pos[1] += 0.05  # 5cm movements
            
            point_a = Point()
            point_a.x, point_a.y, point_a.z = current_pos
            
            point_b = Point()
            point_b.x, point_b.y, point_b.z = target_pos
            
            self.point_a_pub.publish(point_a)
            time.sleep(0.5)
            self.point_b_pub.publish(point_b)
            time.sleep(0.5)
            
            start_msg = String()
            start_msg.data = "start"
            self.start_pub.publish(start_msg)
            
            # Wait for completion
            start_time = time.time()
            while time.time() - start_time < 15.0:
                if self.current_phase == "COMPLETE":
                    break
                time.sleep(0.5)
            
            current_pos = target_pos.copy()
            time.sleep(1.0)

def main(args=None):
    rclpy.init(args=args)
    tester = PushingTester()
    
    try:
        # Run tests
        tester.test_basic_push()
        time.sleep(3.0)
        
        # Reset
        reset_msg = String()
        reset_msg.data = "reset"
        tester.start_pub.publish(reset_msg)
        time.sleep(2.0)
        
        # More tests
        tester.test_precision_pushing()
        
        tester.get_logger().info('🎉 All pushing tests completed!')
        
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()