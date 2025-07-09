#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import deque
import time

class Clean3DTracker(Node):
    def __init__(self):
        super().__init__('clean_3d_tracker')
        
        # Data storage
        self.max_points = 500
        
        # Position data for 3D trajectory
        self.local_positions = deque(maxlen=self.max_points)   
        self.remote_positions = deque(maxlen=self.max_points)  
        
        # Error tracking
        self.error_times = deque(maxlen=self.max_points)
        self.errors = deque(maxlen=self.max_points)
        
        # ROS2 subscribers
        self.local_sub = self.create_subscription(
            PoseStamped, '/local_robot/ee_pose', self.local_callback, 10)
        self.remote_sub = self.create_subscription(
            PoseStamped, '/remote_robot/ee_pose', self.remote_callback, 10)
        
        # Timer for plot updates
        self.plot_timer = self.create_timer(0.5, self.update_plots)  # 2 Hz updates
        
        self.start_time = time.time()
        self.plot_initialized = False
        
        self.get_logger().info('Clean 3D Trajectory tracker started')
        self.get_logger().info('Waiting for robot data...')
    
    def local_callback(self, msg):
        """Handle local robot position data."""
        position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.local_positions.append(position)
        
        self.calculate_error()
        
        # Initialize plot after first data
        if not self.plot_initialized:
            self.initialize_plots()
    
    def remote_callback(self, msg):
        """Handle remote robot position data."""
        position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.remote_positions.append(position)
        
        self.calculate_error()
    
    def calculate_error(self):
        """Calculate tracking error."""
        if len(self.local_positions) > 0 and len(self.remote_positions) > 0:
            local_pos = np.array(self.local_positions[-1])
            remote_pos = np.array(self.remote_positions[-1])
            
            error = np.linalg.norm(local_pos - remote_pos)
            current_time = time.time() - self.start_time
            
            self.error_times.append(current_time)
            self.errors.append(error)
    
    def initialize_plots(self):
        """Initialize matplotlib 3D plots."""
        try:
            plt.ion()  # Interactive mode
            
            # Create figure with 3D subplot and error subplot
            self.fig = plt.figure(figsize=(16, 8))
            
            # Left subplot: 3D trajectory
            self.ax_3d = self.fig.add_subplot(121, projection='3d')
            
            # Right subplot: Error over time
            self.ax_error = self.fig.add_subplot(122)
            
            # Setup 3D plot
            self.ax_3d.set_title('3D Robot Trajectory Tracking\n(Blue=Local, Orange=Remote)', 
                               fontsize=14, fontweight='bold')
            self.ax_3d.set_xlabel('X Position (m)', fontsize=12)
            self.ax_3d.set_ylabel('Y Position (m)', fontsize=12)
            self.ax_3d.set_zlabel('Z Position (m)', fontsize=12)
            
            # Setup error plot
            self.ax_error.set_title('Position Tracking Error', fontsize=14, fontweight='bold')
            self.ax_error.set_xlabel('Time (s)', fontsize=12)
            self.ax_error.set_ylabel('Tracking Error (m)', fontsize=12)
            self.ax_error.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show(block=False)
            
            self.plot_initialized = True
            self.get_logger().info('3D trajectory plots initialized!')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize plots: {e}')
    
    def update_plots(self):
        """Update 3D trajectory and error plots - CLEAN VERSION."""
        if not self.plot_initialized:
            return
        
        try:
            # Only clear error plot, keep 3D plot accumulating
            self.ax_error.clear()
            
            # Only clear and redraw 3D plot occasionally for performance
            if len(self.local_positions) % 25 == 0 and len(self.local_positions) > 0:  # Every 25 points
                self.ax_3d.clear()
                
                # Re-setup 3D axes
                self.ax_3d.set_title('3D Robot Trajectory Tracking\n(Blue=Local, Orange=Remote)', 
                                   fontsize=14, fontweight='bold')
                self.ax_3d.set_xlabel('X Position (m)', fontsize=12)
                self.ax_3d.set_ylabel('Y Position (m)', fontsize=12)
                self.ax_3d.set_zlabel('Z Position (m)', fontsize=12)
                
                # Plot LOCAL robot trajectory - BLUE SOLID LINE
                if len(self.local_positions) > 1:
                    local_array = np.array(list(self.local_positions))
                    
                    self.ax_3d.plot(local_array[:, 0], local_array[:, 1], local_array[:, 2],
                                  'b-', linewidth=4, label='Local Robot (Operator)', alpha=0.9)
                    
                    # Current local position - BLUE CIRCLE
                    current_local = local_array[-1]
                    self.ax_3d.scatter(current_local[0], current_local[1], current_local[2],
                                     color='blue', s=150, marker='o', alpha=1.0, 
                                     edgecolors='darkblue', linewidth=2)
                
                # Plot REMOTE robot trajectory - ORANGE DASHED LINE
                if len(self.remote_positions) > 1:
                    remote_array = np.array(list(self.remote_positions))
                    
                    self.ax_3d.plot(remote_array[:, 0], remote_array[:, 1], remote_array[:, 2],
                                  'orange', linewidth=4, label='Remote Robot (PMDC)', 
                                  alpha=0.9, linestyle='--')
                    
                    # Current remote position - ORANGE TRIANGLE
                    current_remote = remote_array[-1]
                    self.ax_3d.scatter(current_remote[0], current_remote[1], current_remote[2],
                                     color='orange', s=150, marker='^', alpha=1.0,
                                     edgecolors='darkorange', linewidth=2)
                
                # Add error markers - simple red dots for high error points
                if len(self.errors) > 10:
                    errors = np.array(list(self.errors))
                    if len(self.local_positions) == len(errors):
                        local_array = np.array(list(self.local_positions))
                        
                        # Show only high error points (above average + std)
                        if len(errors) > 0:
                            high_error_threshold = np.mean(errors) + 0.5 * np.std(errors)
                            high_error_indices = errors > high_error_threshold
                            
                            if np.any(high_error_indices):
                                high_error_positions = local_array[high_error_indices]
                                self.ax_3d.scatter(high_error_positions[:, 0], 
                                                 high_error_positions[:, 1], 
                                                 high_error_positions[:, 2],
                                                 c='red', s=80, alpha=0.7, marker='x',
                                                 label='High Error Points')
                
                # Set proper axis limits
                if len(self.local_positions) > 10:
                    all_positions = list(self.local_positions) + list(self.remote_positions)
                    if all_positions:
                        all_array = np.array(all_positions)
                        
                        # Calculate limits with margin
                        margin = 0.05
                        x_min, x_max = all_array[:, 0].min() - margin, all_array[:, 0].max() + margin
                        y_min, y_max = all_array[:, 1].min() - margin, all_array[:, 1].max() + margin
                        z_min, z_max = all_array[:, 2].min() - margin, all_array[:, 2].max() + margin
                        
                        self.ax_3d.set_xlim([x_min, x_max])
                        self.ax_3d.set_ylim([y_min, y_max])
                        self.ax_3d.set_zlim([z_min, z_max])
                
                # Add legend
                self.ax_3d.legend(loc='upper left', fontsize=11)
                
                # Add info text
                duration = time.time() - self.start_time
                info_text = f'Duration: {duration:.1f}s | Points: {len(self.local_positions)}'
                self.ax_3d.text2D(0.02, 0.98, info_text, transform=self.ax_3d.transAxes,
                                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # ALWAYS update error plot (right side)
            self.ax_error.set_title('Position Tracking Error Over Time', fontsize=14, fontweight='bold')
            self.ax_error.set_xlabel('Time (s)', fontsize=12)
            self.ax_error.set_ylabel('Tracking Error (m)', fontsize=12)
            self.ax_error.grid(True, alpha=0.3)
            
            # Plot tracking error
            if len(self.error_times) > 0:
                self.ax_error.plot(list(self.error_times), list(self.errors),
                                 'red', linewidth=3, label='Tracking Error', alpha=0.8)
                
                # Add statistics
                errors_list = list(self.errors)
                if len(errors_list) > 5:
                    current_error = errors_list[-1]
                    mean_error = np.mean(errors_list)
                    max_error = np.max(errors_list)
                    rms_error = np.sqrt(np.mean(np.array(errors_list)**2))
                    
                    # Simple delay estimation
                    if mean_error > 0.05:
                        delay_status = "High Delay"
                    elif mean_error > 0.02:
                        delay_status = "Medium Delay"
                    else:
                        delay_status = "Low Delay"
                    
                    stats_text = (f'Current: {current_error:.4f}m\n'
                                f'Mean: {mean_error:.4f}m\n'
                                f'RMS: {rms_error:.4f}m\n'
                                f'Max: {max_error:.4f}m\n'
                                f'Status: {delay_status}\n'
                                f'Points: {len(errors_list)}')
                    
                    self.ax_error.text(0.02, 0.98, stats_text, transform=self.ax_error.transAxes,
                                     verticalalignment='top', fontsize=11,
                                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
                
                self.ax_error.legend(loc='upper right')
            
            # Refresh display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            # Log progress
            if len(self.local_positions) % 50 == 0 and len(self.local_positions) > 0:
                current_error = self.errors[-1] if self.errors else 0
                self.get_logger().info(
                    f'3D Tracking: {len(self.local_positions)} points, Error: {current_error:.4f}m'
                )
        
        except Exception as e:
            self.get_logger().warn(f'Plot update error: {e}')
    
    def save_data(self, filename="clean_3d_trajectory"):
        """Save 3D trajectory and error data."""
        import json
        
        data = {
            'experiment_info': {
                'timestamp': time.time(),
                'duration_seconds': time.time() - (self.start_time + time.time()),
                'total_points': len(self.local_positions)
            },
            'local_trajectory': [list(pos) for pos in self.local_positions],
            'remote_trajectory': [list(pos) for pos in self.remote_positions],
            'tracking_error': {
                'timestamps': list(self.error_times),
                'errors': list(self.errors)
            }
        }
        
        filename_with_timestamp = f"{filename}_{int(time.time())}.json"
        with open(filename_with_timestamp, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.get_logger().info(f'3D trajectory data saved to {filename_with_timestamp}')
        return filename_with_timestamp

def main(args=None):
    rclpy.init(args=args)
    
    try:
        tracker = Clean3DTracker()
        
        print("3D Trajectory Tracking Visualization")
        print("===================================")
        print("Left: 3D trajectory (Blue=Local, Orange=Remote)")
        print("Right: Real-time tracking error")
        print("Red X marks = High tracking error locations")
        print("Matplotlib window will open when data arrives")
        print("Press Ctrl+C to stop and save data")
        print("===================================")
        
        rclpy.spin(tracker)
        
    except KeyboardInterrupt:
        print("\nStopping 3D tracker and saving data...")
        if 'tracker' in locals():
            tracker.save_data("clean_3d_experiment")
        print("3D trajectory data saved successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'tracker' in locals():
            tracker.destroy_node()
        rclpy.shutdown()
        
        try:
            plt.close('all')
        except:
            pass
        
        print("3D tracker shutdown complete")

if __name__ == '__main__':
    main()