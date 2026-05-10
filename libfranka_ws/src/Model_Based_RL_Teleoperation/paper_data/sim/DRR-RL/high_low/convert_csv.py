#!/usr/bin/env python3
"""
Convert rosbag2 teleoperation data to a single combined CSV file.
Extracts leader and follower ee_pose (XYZ only) plus observation and action delay steps.
Excludes timestamps and headers from the final output.
Keeps only the first 50 seconds of data.
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


def message_to_dict(msg):
    """Convert ROS message to dictionary."""
    result = {}
    if not hasattr(msg, 'get_fields_and_field_types'):
        return result
    
    for field_name, field_type in msg.get_fields_and_field_types().items():
        value = getattr(msg, field_name)
        if hasattr(value, 'get_fields_and_field_types'):
            result[field_name] = message_to_dict(value)
        elif isinstance(value, (list, tuple)):
            result[field_name] = str(value)
        else:
            result[field_name] = value
    
    return result


def extract_pose_data(msg):
    """Extract timestamp and point coordinates from PointStamped message."""
    try:
        msg_dict = message_to_dict(msg)
        
        timestamp_sec = msg_dict.get('header', {}).get('stamp', {}).get('sec', None)
        x = msg_dict.get('point', {}).get('x', None)
        y = msg_dict.get('point', {}).get('y', None)
        z = msg_dict.get('point', {}).get('z', None)
        
        return {
            'timestamp_sec': timestamp_sec,
            'point_x': x,
            'point_y': y,
            'point_z': z
        }
    except Exception as e:
        print(f"Error extracting pose data: {e}")
        return None


def extract_float32_data(msg):
    """Extract data from Float32 message."""
    try:
        return {'value': msg.data}
    except Exception as e:
        print(f"Error extracting Float32 data: {e}")
        return None


def read_rosbag2(bag_path, target_topics, max_duration_sec=50):
    """
    Read rosbag2 and extract specific topics.
    
    Args:
        bag_path: Path to rosbag2 directory
        target_topics: Dict mapping {name: topic_path}
        max_duration_sec: Maximum duration in seconds to extract
    
    Returns:
        Dict mapping {name: list of dicts}
    """
    bag_path = str(Path(bag_path).resolve())
    
    try:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening rosbag2 (trying sqlite3 fallback): {e}")
        try:
            reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'
            )
            reader.open(storage_options, converter_options)
        except Exception as e2:
            print(f"Error opening rosbag2: {e2}")
            return None
    
    # Get topic type mappings
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    
    print("\nAvailable topics in rosbag2:")
    for topic_name in sorted(type_map.keys()):
        print(f"  {topic_name}: {type_map[topic_name]}")
    
    # Data storage
    data_store = {name: [] for name in target_topics.keys()}
    reverse_mapping = {v: k for k, v in target_topics.items()}
    
    message_count = 0
    start_time_ns = None
    max_duration_ns = max_duration_sec * 1e9
    
    print(f"\nReading messages (max {max_duration_sec} seconds)...")
    
    try:
        while reader.has_next():
            topic, data, ts = reader.read_next()
            
            # Set start time on first message
            if start_time_ns is None:
                start_time_ns = ts
            
            # Check if we have exceeded the duration limit
            elapsed_ns = ts - start_time_ns
            if elapsed_ns > max_duration_ns:
                print(f"  Reached {max_duration_sec} second limit, stopping extraction...")
                break
            
            if topic in reverse_mapping:
                name = reverse_mapping[topic]
                
                try:
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                    
                    # Handle different message types
                    if 'pose' in name:
                        extracted_data = extract_pose_data(msg)
                    elif 'delay' in name:
                        extracted_data = extract_float32_data(msg)
                    else:
                        extracted_data = None
                    
                    if extracted_data:
                        extracted_data['timestamp_ns'] = ts
                        data_store[name].append(extracted_data)
                    
                    message_count += 1
                    
                    if message_count % 1000 == 0:
                        elapsed_sec = elapsed_ns / 1e9
                        print(f"  Processed {message_count} messages ({elapsed_sec:.2f} seconds)...")
                
                except Exception as e:
                    print(f"Error deserializing message from {topic}: {e}")
                    pass
    
    except Exception as e:
        print(f"Error reading messages: {e}")
        return None
    
    print(f"Total messages extracted: {message_count}")
    
    # Convert to DataFrames
    dataframes = {}
    for name, records in data_store.items():
        if records:
            dataframes[name] = pd.DataFrame(records)
            print(f"  {name}: {len(records)} records")
        else:
            print(f"Warning: No data for topic '{name}'")
            dataframes[name] = pd.DataFrame()
    
    return dataframes


def merge_dataframes(dataframes):
    """
    Merge leader, follower, and delay dataframes by timestamp.
    
    Args:
        dataframes: Dict of {name: DataFrame}
    
    Returns:
        Combined DataFrame with renamed columns
    """
    if not dataframes:
        return None
    
    # Get non-empty dataframes
    dfs_available = {name: df for name, df in dataframes.items() if not df.empty}
    
    if not dfs_available:
        print("Error: All dataframes are empty")
        return None
    
    # Sort all dataframes by timestamp
    for name in dfs_available:
        dfs_available[name] = dfs_available[name].sort_values('timestamp_ns').reset_index(drop=True)
    
    # Start with leader pose as base
    if 'leader_pose' not in dfs_available:
        print("Error: Leader pose data not found")
        return None
    
    df_combined = dfs_available['leader_pose'].copy()
    df_combined = df_combined.rename(columns={
        'point_x': 'leader_ee_pose_point.x',
        'point_y': 'leader_ee_pose_point.y',
        'point_z': 'leader_ee_pose_point.z'
    })
    
    # Merge follower pose
    if 'remote_pose' in dfs_available:
        follower_df = dfs_available['remote_pose'].rename(columns={
            'point_x': 'follower_ee_pose_point.x',
            'point_y': 'follower_ee_pose_point.y',
            'point_z': 'follower_ee_pose_point.z'
        })
        # Drop columns we don't want to merge (like the sec timestamp)
        follower_cols = ['timestamp_ns', 'follower_ee_pose_point.x', 'follower_ee_pose_point.y', 'follower_ee_pose_point.z']
        follower_df = follower_df[[c for c in follower_cols if c in follower_df.columns]]
        
        df_combined = pd.merge_asof(
            df_combined, follower_df,
            on='timestamp_ns',
            direction='nearest',
            tolerance=int(100e6)  # 100ms tolerance
        )
    else:
        print("Warning: Follower pose data not found")
    
    # Merge obs_delay_steps
    if 'obs_delay' in dfs_available:
        obs_delay_df = dfs_available['obs_delay'].rename(columns={
            'value': 'obs_delay_steps'
        })
        obs_delay_df = obs_delay_df[['timestamp_ns', 'obs_delay_steps']]
        
        df_combined = pd.merge_asof(
            df_combined, obs_delay_df,
            on='timestamp_ns',
            direction='nearest',
            tolerance=int(100e6)
        )
    else:
        print("Warning: Observation delay data not found")
    
    # Merge act_delay_steps
    if 'act_delay' in dfs_available:
        act_delay_df = dfs_available['act_delay'].rename(columns={
            'value': 'act_delay_steps'
        })
        act_delay_df = act_delay_df[['timestamp_ns', 'act_delay_steps']]
        
        df_combined = pd.merge_asof(
            df_combined, act_delay_df,
            on='timestamp_ns',
            direction='nearest',
            tolerance=int(100e6)
        )
    else:
        print("Warning: Action delay data not found")
    
    # Convert delay steps to integer (they are float from Float32 msg)
    if 'obs_delay_steps' in df_combined.columns:
        df_combined['obs_delay_steps'] = df_combined['obs_delay_steps'].round().astype('Int64')
    if 'act_delay_steps' in df_combined.columns:
        df_combined['act_delay_steps'] = df_combined['act_delay_steps'].round().astype('Int64')
    
    # CLEANUP: Remove timestamps and headers
    # We only want x, y, z and delays
    cols_to_drop = ['timestamp_ns', 'timestamp_sec']
    df_combined = df_combined.drop(columns=[c for c in cols_to_drop if c in df_combined.columns])
    
    return df_combined


def write_combined_csv(df_combined, output_file):
    """Write combined dataframe to CSV file."""
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df_combined.to_csv(output_file, index=False)
        print(f"\nCombined CSV created: {output_file}")
        print(f"Total rows: {len(df_combined)}")
        print(f"Total columns: {len(df_combined.columns)}")
        print(f"\nFinal Column List:")
        for col in df_combined.columns:
            print(f"  - {col}")
        return True
    
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 rosbag2_to_csv_values.py <rosbag2_path> [output_file]")
        print("\nExample:")
        print("  python3 rosbag2_to_csv_values.py ./rosbag2_2025_12_05-13_42_41")
        print("  python3 rosbag2_to_csv_values.py ./rosbag2_2025_12_05-13_42_41 ./data_values_only.csv")
        sys.exit(1)
    
    bag_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else Path(bag_path).parent / "teleoperation_values_only.csv"
    
    # Define topics to extract (including delay metrics)
    target_topics = {
        'leader_pose': '/leader/ee_pose',
        'remote_pose': '/remote/ee_pose',
        'obs_delay': '/agent/obs_delay_steps',
        'act_delay': '/agent/act_delay_steps'
    }
    
    print(f"Reading rosbag2 from: {bag_path}")
    print(f"Target topics:")
    for name, topic in target_topics.items():
        print(f"  {name}: {topic}")
    print(f"Duration limit: 50 seconds")
    
    dataframes = read_rosbag2(bag_path, target_topics, max_duration_sec=50)
    
    if dataframes:
        print("\nMerging dataframes and filtering columns...")
        df_combined = merge_dataframes(dataframes)
        
        if df_combined is not None:
            write_combined_csv(df_combined, output_file)
        else:
            print("Failed to merge dataframes")
            sys.exit(1)
    else:
        print("Failed to read rosbag2 data")
        sys.exit(1)


if __name__ == '__main__':
    main()