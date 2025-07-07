"""
This module monitors the trajectory following performance of a remote robot.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointStates
from std_msgs.msg import String
from geometry_msgs.msg import Point
import numpy as np
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
import numpy as np
import time

class TrajectoryTestController(Node):
    def __init__(self):
        super().__init__('trajectory_test_controller')
        
        # Test trajectories (A→B pairs)
        self.test_trajectories = [
            # Format: [start_x, start_y, start_z, end_x, end_y, end_z]
            [0.5, 0.2, 0.3, 0.6, 0.1, 0.4],      # Trajectory 1
            [0.6, 0.1, 0.4, 0.4, 0.3, 0.35],     # Trajectory 2  
            [0.4, 0.3, 0.35, 0.55, 0.0, 0.32],   # Trajectory 3
            [0.55, 0.0, 0.32, 0.45, 0.25, 0.38], # Trajectory 4
            [0.45, 0.25, 0.38, 0.52, 0.15, 0.33] # Trajectory 5
        ]
        
        # Test state
        self.current_test = 0
        self.trajectories_sent = 0
        self.trajectories_completed = 0
        self.is_trajectory_running = False
        self.start_time = time.time()
        
        # System status
        self.remote_robot_responding = False
        self.local_robot_active = False
        
        # Publishers and subscribers
        self.trajectory_cmd_pub = self.create_publisher(
            String, '/start_trajectory_test', 10)
        
        self.comparison_debug_sub = self.create_subscription(
            String, '/remote_robot/trajectory_comparison_debug',
            self.comparison_debug_callback, 10)
        
        self.remote_status_sub = self.create_subscription(
            String, '/remote_to_local/status_feedback',
            self.remote_status_callback, 10)
        
        # Timers
        self.status_timer = self.create_timer(3.0, self.print_status)
        self.test_timer = self.create_timer(8.0, self.send_next_trajectory)  # 8 seconds between tests
        self.connectivity_timer = self.create_timer(5.0, self.check_system_health)
        
        self.get_logger().info("🎯 Trajectory Test Controller started")
        print("🎯 TRAJECTORY COMPARISON TEST CONTROLLER")
        print("=" * 60)
        print("📊 Tests point-by-point trajectory following performance")
        print("🔄 Process: Sends A→B trajectories → Monitors waypoint tracking")
        print("📈 Analyzes how well remote robot follows trajectory waypoints")
        print("=" * 60)
    
    def check_system_health(self):
        """Check if required systems are running."""
        if not self.remote_robot_responding and (time.time() - self.start_time) > 10:
            self.get_logger().warn("⚠️  Remote robot not responding - check trajectory_comparison_system.py")
    
    def comparison_debug_callback(self, msg):
        """Monitor trajectory comparison debug messages."""
        self.remote_robot_responding = True
        
        try:
            data_str = msg.data
            
            if "TRAJECTORY_COMPARISON" in data_str:
                # Extract trajectory status
                if "Status: TRACKING" in data_str:
                    if not self.is_trajectory_running:
                        self.is_trajectory_running = True
                        self.get_logger().info("🚀 Trajectory tracking STARTED")
                
                elif "Status: IDLE" in data_str:
                    if self.is_trajectory_running:
                        self.is_trajectory_running = False
                        self.trajectories_completed += 1
                        self.get_logger().info("✅ Trajectory tracking COMPLETED")
                
                # Extract performance data
                if "WaypointError:" in data_str:
                    try:
                        error_part = data_str.split("WaypointError: ")[1].split(",")[0]
                        waypoint_error = float(error_part)
                        
                        # Extract progress
                        if "Progress:" in data_str:
                            progress_part = data_str.split("Progress: ")[1].split(",")[0]
                            progress = float(progress_part)
                            
                            # Log every 25% progress
                            if progress in [0.25, 0.5, 0.75]:
                                self.get_logger().info(f"📈 Progress: {progress:.0%}, Waypoint Error: {waypoint_error:.4f} rad")
                    
                    except Exception:
                        pass  # Ignore parsing errors
                
        except Exception as e:
            pass  # Ignore parsing errors
    
    def remote_status_callback(self, msg):
        """Monitor remote robot status."""
        status = msg.data
        
        if status == "trajectory_started":
            self.get_logger().info("🚀 Remote robot started trajectory execution")
        elif status == "trajectory_completed":
            self.get_logger().info("🎯 Remote robot completed trajectory")
        elif status == "ik_failed":
            self.get_logger().warn("❌ IK solution failed for trajectory")
        elif status == "trajectory_error":
            self.get_logger().error("❌ Trajectory execution error")
    
    def send_next_trajectory(self):
        """Send the next test trajectory."""
        # Don't send if trajectory is still running
        if self.is_trajectory_running:
            self.get_logger().info("⏸️  Trajectory still running - waiting...")
            return
        
        # Don't send if no remote robot response
        if not self.remote_robot_responding:
            self.get_logger().warn("⚠️  Remote robot not responding - not sending trajectory")
            return
        
        # Check if we have more test trajectories
        if self.current_test >= len(self.test_trajectories):
            self.get_logger().info("✅ All test trajectories completed!")
            return
        
        # Get current trajectory
        trajectory = self.test_trajectories[self.current_test]
        start_pos = trajectory[:3]
        end_pos = trajectory[3:]
        
        # Format trajectory command
        # Format: "start_x,start_y,start_z;end_x,end_y,end_z"
        cmd_msg = String()
        cmd_msg.data = f"{start_pos[0]},{start_pos[1]},{start_pos[2]};{end_pos[0]},{end_pos[1]},{end_pos[2]}"
        
        # Send trajectory command
        self.trajectory_cmd_pub.publish(cmd_msg)
        self.trajectories_sent += 1
        self.current_test += 1
        
        self.get_logger().info(f"🎯 Sent trajectory #{self.trajectories_sent}:")
        self.get_logger().info(f"   A: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
        self.get_logger().info(f"   B: [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
        self.get_logger().info(f"   Expected: 5s trajectory with ~100 waypoints")
    
    def print_status(self):
        """Print comprehensive status."""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        # Calculate success rate
        success_rate = (self.trajectories_completed / max(1, self.trajectories_sent)) * 100
        
        status = f"""
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║                           🎯 TRAJECTORY TEST CONTROLLER STATUS                               ║
╠════════════════════════════════════════════════════════════════════════════════════════════╣
║ 🤖 Remote System:       {'✅ RESPONDING' if self.remote_robot_responding else '❌ NOT RESPONDING':<15}                           ║
║ 🎯 Trajectory Status:   {'🚀 TRACKING' if self.is_trajectory_running else '⏸️  IDLE':<15}                              ║
║                                                                                                ║
║ 📊 TEST PROGRESS:                                                                             ║
║   📝 Trajectories Sent:     {self.trajectories_sent:<8}                                      ║
║   ✅ Trajectories Done:     {self.trajectories_completed:<8}                                 ║
║   📈 Success Rate:          {success_rate:<8.1f}%                                            ║
║   📋 Tests Remaining:       {len(self.test_trajectories) - self.current_test:<8}            ║
║                                                                                                ║
║ ⏱️  Runtime:                {runtime:<8.1f} seconds                                           ║
║ 🔄 Test Mode:              Point-by-Point Trajectory Comparison                              ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝
        """
        
        print(status)
        
        # Show upcoming tests
        if self.current_test < len(self.test_trajectories):
            next_traj = self.test_trajectories[self.current_test]
            print(f"🔮 Next trajectory: A[{next_traj[0]:.3f},{next_traj[1]:.3f},{next_traj[2]:.3f}] → B[{next_traj[3]:.3f},{next_traj[4]:.3f},{next_traj[5]:.3f}]")
        
        # System diagnostics
        if not self.remote_robot_responding:
            print("🔧 DIAGNOSTIC: Remote robot not responding")
            print("   💡 Start: /usr/bin/python3 trajectory_comparison_system.py")
        
        if self.trajectories_sent > 0 and self.trajectories_completed == 0:
            print("🔧 DIAGNOSTIC: Trajectories sent but none completed")
            print("   💡 Check remote robot trajectory execution")
        
        print()
    
    def send_manual_trajectory(self, start_pos, end_pos):
        """Send a manual trajectory for testing."""
        cmd_msg = String()
        cmd_msg.data = f"{start_pos[0]},{start_pos[1]},{start_pos[2]};{end_pos[0]},{end_pos[1]},{end_pos[2]}"
        
        self.trajectory_cmd_pub.publish(cmd_msg)
        self.trajectories_sent += 1
        
        self.get_logger().info(f"🎯 Manual trajectory sent:")
        self.get_logger().info(f"   A: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
        self.get_logger().info(f"   B: [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")

def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        print("🎯 Starting Trajectory Test Controller...")
        print("📊 This sends A→B trajectory commands and monitors point-by-point tracking")
        print("🔄 Process: A→B command → Trajectory generation → Waypoint tracking → Error analysis")
        print("=" * 80)
        
        controller = TrajectoryTestController()
        
        print("\n🎯 Test controller running...")
        print("📝 Will automatically send trajectory tests every 8 seconds")
        print("📈 Watch for trajectory tracking status and waypoint errors")
        print("⏸️  Press Ctrl+C to stop\n")
        
        # Optional: Send immediate test
        response = input("🚀 Send immediate test trajectory? (y/n): ")
        if response.lower() == 'y':
            controller.send_manual_trajectory([0.5, 0.2, 0.3], [0.6, 0.1, 0.4])
        
        rclpy.spin(controller)
        
    except KeyboardInterrupt:
        print("\n🛑 Test controller stopped")
        if 'controller' in locals():
            print(f"\n📊 FINAL TEST SUMMARY:")
            print(f"Trajectories sent: {controller.trajectories_sent}")
            print(f"Trajectories completed: {controller.trajectories_completed}")
            if controller.trajectories_sent > 0:
                success_rate = controller.trajectories_completed / controller.trajectories_sent * 100
                print(f"Success rate: {success_rate:.1f}%")
    except Exception as e:
        print(f"❌ Test controller error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()