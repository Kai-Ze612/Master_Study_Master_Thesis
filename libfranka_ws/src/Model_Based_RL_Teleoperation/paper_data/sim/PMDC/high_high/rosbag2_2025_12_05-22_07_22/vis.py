#!/usr/bin/env python3
"""
Visualize ROS 2 Teleoperation Data (XY Plane Only).
- Reads directly from ROS 2 bag (.mcap or .db3).
- Auto-detects topics.
- Plots:
    1. 2D XY Trajectories (Top Down View)
    2. XY Time Series
    3. XY Tracking Error (Euclidean Distance in 2D)
    4. System Delays
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
# 3D projection import removed as we are doing 2D only
import pandas as pd
import numpy as np

try:
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message
    from rclpy.serialization import deserialize_message
except ImportError as e:
    print(f"Error: Required ROS 2 Python packages not found: {e}")
    sys.exit(1)

# ---------------- CONFIGURATION ---------------- #
LEADER_CANDIDATES = ['/leader/ee_pose', '/leader/ee_position', '/leader/current_pose']
FOLLOWER_CANDIDATES = ['/remote/ee_pose', '/remote/ee_position', '/follower/ee_pose']
DELAY_OBS_TOPIC = '/agent/obs_delay_steps'
DELAY_ACT_TOPIC = '/agent/act_delay_steps'
MAX_DURATION_SEC = 50
# ----------------------------------------------- #

def extract_xyz_from_msg(msg):
    """Robustly extract x,y,z from Point, Pose, or PoseStamped messages."""
    if hasattr(msg, 'pose') and hasattr(msg.pose, 'position'):
        return msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
    if hasattr(msg, 'position') and hasattr(msg.position, 'x'):
        return msg.position.x, msg.position.y, msg.position.z
    if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
        return msg.x, msg.y, msg.z
    if hasattr(msg, 'point') and hasattr(msg.point, 'x'):
        return msg.point.x, msg.point.y, msg.point.z
    return None, None, None

def extract_float(msg):
    if hasattr(msg, 'data'):
        return msg.data
    return None

def get_actual_topics(reader):
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    all_topics = list(type_map.keys())
    
    target_map = {}
    print("Found topics:", all_topics)

    for cand in LEADER_CANDIDATES:
        if cand in all_topics:
            target_map['leader'] = cand
            break
    for cand in FOLLOWER_CANDIDATES:
        if cand in all_topics:
            target_map['follower'] = cand
            break
            
    if DELAY_OBS_TOPIC in all_topics: target_map['obs_delay'] = DELAY_OBS_TOPIC
    if DELAY_ACT_TOPIC in all_topics: target_map['act_delay'] = DELAY_ACT_TOPIC
    
    return target_map, type_map

def read_bag_data(bag_path):
    bag_path_obj = Path(bag_path).resolve()
    if not bag_path_obj.exists():
        print(f"Error: Path {bag_path_obj} does not exist.")
        return None

    try:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=str(bag_path_obj), storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)
    except Exception:
        # Fallback to sqlite3
        try:
            reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(uri=str(bag_path_obj), storage_id='sqlite3')
            converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
            reader.open(storage_options, converter_options)
        except Exception as e:
            print(f"Could not open bag: {e}")
            return None

    targets, type_map = get_actual_topics(reader)
    if 'leader' not in targets:
        print("Could not find leader topic.")
        return None

    data = {k: [] for k in targets.keys()}
    reverse_map = {v: k for k, v in targets.items()}
    
    start_time = None
    
    while reader.has_next():
        topic, raw_data, ts = reader.read_next()
        if start_time is None: start_time = ts
        
        rel_time = (ts - start_time) / 1e9
        if rel_time > MAX_DURATION_SEC: break

        if topic in reverse_map:
            key = reverse_map[topic]
            try:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(raw_data, msg_type)
                
                if key in ['leader', 'follower']:
                    x, y, z = extract_xyz_from_msg(msg)
                    if x is not None:
                        data[key].append({'t': rel_time, 'x': x, 'y': y, 'z': z})
                elif 'delay' in key:
                    val = extract_float(msg)
                    if val is not None:
                        data[key].append({'t': rel_time, 'val': val})
            except:
                pass

    dfs = {k: pd.DataFrame(v) for k, v in data.items() if v}
    return dfs

def plot_data(dfs):
    fig = plt.figure(figsize=(16, 12))
    
    # ----------------------------------------
    # 1. 2D Trajectory (Top Left) - XY Plane Only
    # ----------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    if 'leader' in dfs:
        ax1.plot(dfs['leader']['x'], dfs['leader']['y'], label='Leader', color='blue', alpha=0.8)
    if 'follower' in dfs:
        ax1.plot(dfs['follower']['x'], dfs['follower']['y'], label='Follower', color='red', alpha=0.8)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('2D Trajectory (XY Plane)')
    ax1.axis('equal') # Important for spatial plots to keep aspect ratio
    ax1.grid(True)
    ax1.legend()

    # ----------------------------------------
    # 2. XY Components (Top Right) - No Z
    # ----------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    if 'leader' in dfs:
        l = dfs['leader']
        ax2.plot(l['t'], l['x'], 'b-', label='Leader X', alpha=0.5)
        ax2.plot(l['t'], l['y'], 'b--', label='Leader Y', alpha=0.5)
        # Z removed
    
    if 'follower' in dfs:
        f = dfs['follower']
        ax2.plot(f['t'], f['x'], 'r-', label='Follower X', alpha=0.5)
        ax2.plot(f['t'], f['y'], 'r--', label='Follower Y', alpha=0.5)
        # Z removed
    
    ax2.set_title('XY Position vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.legend(loc='upper right', fontsize='small', ncol=2)
    ax2.grid(True)

    # ----------------------------------------
    # 3. XY Tracking Error (Bottom Left)
    # ----------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    if 'leader' in dfs and 'follower' in dfs:
        # Sort values just in case
        leader_df = dfs['leader'].sort_values('t')
        follower_df = dfs['follower'].sort_values('t')
        
        # Merge closest timestamps to calculate error
        # We match each leader point to the closest follower point
        merged = pd.merge_asof(
            leader_df, 
            follower_df, 
            on='t', 
            suffixes=('_l', '_f'), 
            direction='nearest', 
            tolerance=0.1 # 100ms tolerance
        )
        merged = merged.dropna()
        
        if not merged.empty:
            dx = merged['x_l'] - merged['x_f']
            dy = merged['y_l'] - merged['y_f']
            # Z ignored in error calculation
            merged['error'] = np.sqrt(dx**2 + dy**2)
            
            mae = merged['error'].mean()
            rmse = np.sqrt((merged['error']**2).mean())
            
            ax3.plot(merged['t'], merged['error'], 'k-', label=f'XY Error', linewidth=1.5)
            ax3.set_title(f'XY Tracking Error (MAE: {mae:.3f}m, RMSE: {rmse:.3f}m)')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Error (m)')
            ax3.grid(True)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Timestamps disjoint (max 100ms gap)', ha='center')
    else:
        ax3.text(0.5, 0.5, 'Need both Leader & Follower data', ha='center')

    # ----------------------------------------
    # 4. Delays (Bottom Right)
    # ----------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    has_delay = False
    if 'obs_delay' in dfs:
        ax4.plot(dfs['obs_delay']['t'], dfs['obs_delay']['val'], label='Obs Delay', color='purple')
        has_delay = True
    if 'act_delay' in dfs:
        ax4.plot(dfs['act_delay']['t'], dfs['act_delay']['val'], label='Act Delay', color='orange')
        has_delay = True
    
    if has_delay:
        ax4.set_title('System Delays')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Steps')
        ax4.legend()
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, 'No Delay Data Found', ha='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_rosbag.py <path_to_bag_folder>")
        sys.exit(1)
        
    dfs = read_bag_data(sys.argv[1])
    if dfs:
        print("Data extraction complete. Plotting...")
        plot_data(dfs)
    else:
        print("No data found to plot.")