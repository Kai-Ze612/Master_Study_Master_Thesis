import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# ---------------- CONFIGURATION ---------------- #
# PATH: Point to the FOLDER containing the metadata.yaml and .mcap file
BAG_PATH = '/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/experiment_data/SBSP/Low_delay/rosbag2_2025_12_05-21_00_28'

TOPIC_LEADER = '/leader/ee_position'
TOPIC_REMOTE = '/remote/ee_position'
# ----------------------------------------------- #

def get_rosbag_options(path, serialization_format='cdr'):
    # CHANGED: storage_id set to 'mcap' instead of 'sqlite3'
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)
    return storage_options, converter_options

def extract_xyz(msg):
    """
    Helper to extract x,y,z from different message types 
    """
    # 1. Try accessing as Point (msg.x)
    if hasattr(msg, 'x') and hasattr(msg, 'y'):
        return msg.x, msg.y, msg.z
    
    # 2. Try accessing as Pose (msg.position.x)
    if hasattr(msg, 'position'):
        return msg.position.x, msg.position.y, msg.position.z

    # 3. Try accessing as PoseStamped (msg.pose.position.x)
    if hasattr(msg, 'pose') and hasattr(msg.pose, 'position'):
        return msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        
    raise AttributeError(f"Could not find x,y,z in message type: {type(msg)}")

def read_topic_data(bag_path, target_topic):
    data_list = []
    print(f"Reading topic: {target_topic}...")
    
    storage_options, converter_options = get_rosbag_options(bag_path)
    reader = rosbag2_py.SequentialReader()
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"CRITICAL ERROR opening bag: {e}")
        print(f"Ensure that 'rosbag2-storage-mcap' is installed and BAG_PATH is correct.")
        return pd.DataFrame()

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    if target_topic not in type_map:
        print(f"WARNING: Topic '{target_topic}' not found in bag.")
        print(f"Available topics: {list(type_map.keys())}")
        return pd.DataFrame()

    msg_type = get_message(type_map[target_topic])

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == target_topic:
            msg = deserialize_message(data, msg_type)
            try:
                x, y, z = extract_xyz(msg)
                data_list.append({
                    'timestamp': t,
                    'x': x,
                    'y': y,
                    'z': z
                })
            except AttributeError:
                pass 

    df = pd.DataFrame(data_list)
    if not df.empty:
        start_time = df['timestamp'].iloc[0]
        df['time_sec'] = (df['timestamp'] - start_time) / 1e9
        df['timestamp'] = pd.to_datetime(df['timestamp']) 
    return df

def main():
    # 1. Read Data
    df_leader = read_topic_data(BAG_PATH, TOPIC_LEADER)
    df_remote = read_topic_data(BAG_PATH, TOPIC_REMOTE)

    if df_leader.empty or df_remote.empty:
        print("Error: One or both topics are empty. Cannot plot.")
        return

    # 2. Synchronize Data
    df_remote = df_remote.sort_values('timestamp')
    df_leader = df_leader.sort_values('timestamp')
    
    merged = pd.merge_asof(
        df_remote, 
        df_leader, 
        on='timestamp', 
        direction='nearest', 
        suffixes=('_remote', '_leader'),
        tolerance=pd.Timedelta('50ms') 
    )
    merged.dropna(inplace=True)

    # 3. Calculate Error
    merged['err_x'] = merged['x_leader'] - merged['x_remote']
    merged['err_y'] = merged['y_leader'] - merged['y_remote']
    merged['err_z'] = merged['z_leader'] - merged['z_remote']
    
    merged['euclidean_error'] = np.sqrt(
        merged['err_x']**2 + merged['err_y']**2 + merged['err_z']**2
    )
    
    mae = merged['euclidean_error'].mean()
    rmse = np.sqrt((merged['euclidean_error']**2).mean())
    print(f"Stats -> Mean Error: {mae:.4f} m | RMSE: {rmse:.4f} m")

    # 4. Plot
    plt.figure(figsize=(10, 8))

    # Top Plot: Euclidean Error
    plt.subplot(2, 1, 1)
    plt.plot(merged['time_sec_remote'], merged['euclidean_error'], label='Position Error', color='red')
    plt.title(f'Teleoperation Tracking Error\n(MAE: {mae:.4f} m)')
    plt.ylabel('Euclidean Distance [m]')
    plt.grid(True)
    plt.legend()

    # Bottom Plot: XYZ Components
    plt.subplot(2, 1, 2)
    plt.plot(merged['time_sec_remote'], merged['err_x'], label='X Error', alpha=0.6)
    plt.plot(merged['time_sec_remote'], merged['err_y'], label='Y Error', alpha=0.6)
    plt.plot(merged['time_sec_remote'], merged['err_z'], label='Z Error', alpha=0.6)
    plt.xlabel('Time [s]')
    plt.ylabel('Component Error [m]')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()