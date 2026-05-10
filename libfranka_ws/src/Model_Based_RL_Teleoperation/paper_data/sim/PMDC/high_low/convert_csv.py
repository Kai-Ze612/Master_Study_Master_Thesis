#!/usr/bin/env python3
"""
Robust ROS 2 Bag Converter.
- Auto-detects leader/remote pose topics.
- Handles PoseStamped, Pose, and Point messages.
- Extracts XYZ and Delay metrics only.
- Filters first 50 seconds.
"""

import sys
from pathlib import Path
import pandas as pd

try:
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message
    from rclpy.serialization import deserialize_message
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    sys.exit(1)

# ---------------- CONFIGURATION ---------------- #
# We will try to find topics matching these patterns
LEADER_CANDIDATES = ['/leader/ee_pose', '/leader/ee_position', '/leader/current_pose']
FOLLOWER_CANDIDATES = ['/remote/ee_pose', '/remote/ee_position', '/follower/ee_pose']
DELAY_OBS_TOPIC = '/agent/obs_delay_steps'
DELAY_ACT_TOPIC = '/agent/act_delay_steps'

MAX_DURATION_SEC = 50
# ----------------------------------------------- #

def extract_xyz_from_msg(msg):
    """
    Robustly extract x,y,z from Point, Pose, or PoseStamped messages.
    Returns (x, y, z) tuple or (None, None, None).
    """
    # 1. Check for PoseStamped (has header and pose)
    if hasattr(msg, 'pose') and hasattr(msg.pose, 'position'):
        return msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
    
    # 2. Check for Pose (has position and orientation)
    if hasattr(msg, 'position') and hasattr(msg.position, 'x'):
        return msg.position.x, msg.position.y, msg.position.z

    # 3. Check for Point (has x, y, z directly)
    if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
        return msg.x, msg.y, msg.z

    # 4. Check for PointStamped (has header and point)
    if hasattr(msg, 'point') and hasattr(msg.point, 'x'):
        return msg.point.x, msg.point.y, msg.point.z

    return None, None, None

def extract_float(msg):
    """Extract float value from Float32 or similar."""
    if hasattr(msg, 'data'):
        return msg.data
    return None

def get_actual_topics(reader):
    """
    Scan the bag for available topics and map them to our targets.
    Returns a dict of { internal_name: actual_topic_name }.
    """
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    all_topics = list(type_map.keys())

    print("\n=== Topics Found in Rosbag ===")
    for t in sorted(all_topics):
        print(f"  {t}  [{type_map[t]}]")
    print("==============================\n")

    target_map = {}

    # Find Leader Topic
    for candidate in LEADER_CANDIDATES:
        if candidate in all_topics:
            target_map['leader_pose'] = candidate
            print(f"-> Selected Leader Topic: {candidate}")
            break
    
    # Find Follower Topic
    for candidate in FOLLOWER_CANDIDATES:
        if candidate in all_topics:
            target_map['remote_pose'] = candidate
            print(f"-> Selected Remote Topic: {candidate}")
            break
    
    # Delay Topics
    if DELAY_OBS_TOPIC in all_topics:
        target_map['obs_delay'] = DELAY_OBS_TOPIC
    if DELAY_ACT_TOPIC in all_topics:
        target_map['act_delay'] = DELAY_ACT_TOPIC

    return target_map, type_map

def read_rosbag2(bag_path, max_duration_sec=50):
    bag_path = str(Path(bag_path).resolve())
    
    # Setup Reader
    try:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening rosbag (trying sqlite3 fallback): {e}")
        try:
            reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'
            )
            reader.open(storage_options, converter_options)
        except Exception:
            print("CRITICAL ERROR: Could not open rosbag.")
            return None

    # Auto-detect topics
    target_topics, type_map = get_actual_topics(reader)
    if 'leader_pose' not in target_topics:
        print("Error: Could not find a valid Leader Pose topic.")
        return None

    # Prepare storage
    data_store = {k: [] for k in target_topics.keys()}
    reverse_mapping = {v: k for k, v in target_topics.items()}
    
    start_time_ns = None
    max_duration_ns = max_duration_sec * 1e9
    
    print(f"\nReading messages (max {max_duration_sec}s)...")
    
    while reader.has_next():
        topic, data, ts = reader.read_next()
        
        if start_time_ns is None:
            start_time_ns = ts
        
        if (ts - start_time_ns) > max_duration_ns:
            break

        if topic in reverse_mapping:
            name = reverse_mapping[topic]
            try:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)

                if 'pose' in name:
                    x, y, z = extract_xyz_from_msg(msg)
                    if x is not None:
                        data_store[name].append({
                            'timestamp_ns': ts,
                            'x': x, 'y': y, 'z': z
                        })
                elif 'delay' in name:
                    val = extract_float(msg)
                    if val is not None:
                        data_store[name].append({
                            'timestamp_ns': ts,
                            'value': int(val) # Cast delays to int
                        })
            except Exception as e:
                # pass on serialization errors for individual messages
                pass

    # Convert to DataFrames
    dfs = {}
    for k, v in data_store.items():
        if v:
            dfs[k] = pd.DataFrame(v)
            print(f"  {k}: {len(v)} records")
        else:
            print(f"  Warning: No data extracted for {k}")
    
    return dfs

def merge_dataframes(dataframes):
    if not dataframes or 'leader_pose' not in dataframes:
        return None
    
    # Base is leader
    combined = dataframes['leader_pose'].sort_values('timestamp_ns')
    combined = combined.rename(columns={
        'x': 'leader_ee_pose_point.x',
        'y': 'leader_ee_pose_point.y',
        'z': 'leader_ee_pose_point.z'
    })

    # Merge Remote
    if 'remote_pose' in dataframes:
        remote = dataframes['remote_pose'].sort_values('timestamp_ns')
        remote = remote.rename(columns={
            'x': 'follower_ee_pose_point.x',
            'y': 'follower_ee_pose_point.y',
            'z': 'follower_ee_pose_point.z'
        })
        combined = pd.merge_asof(
            combined, remote,
            on='timestamp_ns',
            direction='nearest',
            tolerance=int(100e6) # 100ms
        )

    # Merge Delays
    if 'obs_delay' in dataframes:
        obs = dataframes['obs_delay'].rename(columns={'value': 'obs_delay_steps'})
        combined = pd.merge_asof(combined, obs, on='timestamp_ns', direction='nearest', tolerance=int(100e6))
        
    if 'act_delay' in dataframes:
        act = dataframes['act_delay'].rename(columns={'value': 'act_delay_steps'})
        combined = pd.merge_asof(combined, act, on='timestamp_ns', direction='nearest', tolerance=int(100e6))

    # Drop timestamp for final clean CSV
    if 'timestamp_ns' in combined.columns:
        combined = combined.drop(columns=['timestamp_ns'])
        
    return combined

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 rosbag2_to_csv_values.py <bag_path> [output_csv]")
        sys.exit(1)

    bag_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else Path(bag_path).parent / "teleoperation_values_only.csv"

    dfs = read_rosbag2(bag_path, MAX_DURATION_SEC)
    
    if dfs:
        final_df = merge_dataframes(dfs)
        if final_df is not None and not final_df.empty:
            final_df.to_csv(output_path, index=False)
            print(f"\nSuccess! CSV saved to: {output_path}")
            print("Columns:", list(final_df.columns))
        else:
            print("Error: Merged DataFrame is empty.")
    else:
        print("Error: No dataframes created.")

if __name__ == '__main__':
    main()