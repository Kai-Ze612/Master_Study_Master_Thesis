import argparse
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32

import rosbag2_py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_bag_data(bag_path, topics_dict):
    """
    Reads bag and extracts messages based on a dictionary of {name: topic_string}.
    Returns a dictionary of DataFrames.
    """
    reader = rosbag2_py.SequentialReader()
    # Ensure 'mcap' is used
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag: {e}")
        return None

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    topic_to_name = {v: k for k, v in topics_dict.items()}
    data_store = {name: [] for name in topics_dict.keys()}

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic in topic_to_name:
            name = topic_to_name[topic]
            try:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                row = {'timestamp': t}
                
                if 'pose' in topic: 
                    row['x'] = msg.point.x
                    row['y'] = msg.point.y
                    row['z'] = msg.point.z
                elif 'delay' in topic or 'time' in topic:
                    row['value'] = msg.data
                
                data_store[name].append(row)
            except Exception:
                pass

    return {k: pd.DataFrame(v) for k, v in data_store.items()}

def process_and_align_data(dfs):
    """
    Aligns Leader, Remote, and Delay data streams.
    Handles short recordings gracefully.
    """
    # 1. Check basic data existence
    if dfs['leader'].empty or dfs['remote'].empty:
        print("Error: Leader or Remote trajectory data is missing/empty.")
        return None

    df_local = dfs['leader'].sort_values('timestamp')
    df_remote = dfs['remote'].sort_values('timestamp')

    # 2. Smart Trimming Logic
    t_min = min(df_local['timestamp'].iloc[0], df_remote['timestamp'].iloc[0])
    t_max = max(df_local['timestamp'].iloc[-1], df_remote['timestamp'].iloc[-1])
    total_duration_ns = t_max - t_min
    
    # Only trim if we have enough data (> 5 seconds)
    if total_duration_ns > 5 * 1e9:
        print("  -> Trimming 2s from start/end (Standard Mode)")
        trim_ns = 2 * 1e9
        df_local = df_local[(df_local['timestamp'] >= t_min + trim_ns) & (df_local['timestamp'] <= t_max - trim_ns)]
        df_remote = df_remote[(df_remote['timestamp'] >= t_min + trim_ns) & (df_remote['timestamp'] <= t_max - trim_ns)]
    else:
        print(f"  -> Short recording ({total_duration_ns/1e9:.2f}s). Skipping trim.")

    if df_local.empty or df_remote.empty:
        print("Error: Data became empty after trimming.")
        return None

    # 3. Merge Pose Data
    # Use a generous tolerance (100ms) to ensure we find matches even if sim lagged
    df_main = pd.merge_asof(
        df_local, df_remote, 
        on='timestamp', suffixes=('_loc', '_rem'), 
        direction='nearest', tolerance=int(100e6)
    ).dropna()

    if df_main.empty:
        print("Error: No matching timestamps found between Leader and Remote.")
        return None

    # 4. Merge Delays (Optional)
    for delay_type in ['obs_delay', 'act_delay']:
        if not dfs[delay_type].empty:
            df_delay = dfs[delay_type].sort_values('timestamp')
            df_main = pd.merge_asof(
                df_main, df_delay,
                on='timestamp', direction='nearest', tolerance=int(100e6)
            ).rename(columns={'value': delay_type})
        else:
            df_main[delay_type] = 0.0

    # 5. Calculate Metrics
    # Safe normalization: ensure we align to 0.0s start
    df_main['time_sec'] = (df_main['timestamp'] - df_main['timestamp'].iloc[0]) / 1e9
    df_main['error_xy'] = np.sqrt((df_main['x_loc'] - df_main['x_rem'])**2 + (df_main['y_loc'] - df_main['y_rem'])**2)
    df_main['total_delay'] = df_main['obs_delay'] + df_main['act_delay']
    
    return df_main

def visualize_analysis(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    # --- PLOT 1: TRACKING ERROR vs TOTAL DELAY ---
    ax1 = axes[0]
    ax1_r = ax1.twinx()
    
    # Delay Area
    ax1_r.fill_between(df['time_sec'], 0, df['total_delay'], color='gray', alpha=0.2, label='Total Delay')
    ax1_r.set_ylabel("Delay (steps)", color='gray', fontsize=12)
    # Dynamic Y-limit for delay
    delay_max = df['total_delay'].max()
    ax1_r.set_ylim(0, max(delay_max * 1.5, 10))

    # Error Line
    ax1.plot(df['time_sec'], df['error_xy'], color='#d62728', linewidth=2, label='Tracking Error')
    ax1.set_title("Tracking Error vs. Network Delay", fontsize=16, pad=15)
    ax1.set_ylabel("Error (m)", color='#d62728', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # --- PLOT 2: TRAJECTORY COMPONENTS ---
    ax2 = axes[1]
    ax2.plot(df['time_sec'], df['x_loc'], label='Leader X', color='tab:blue')
    ax2.plot(df['time_sec'], df['x_rem'], label='Remote X', color='tab:blue', linestyle='--')
    ax2.plot(df['time_sec'], df['y_loc'], label='Leader Y', color='tab:green')
    ax2.plot(df['time_sec'], df['y_rem'], label='Remote Y', color='tab:green', linestyle='--')
    
    ax2.set_title("Trajectory Components (X/Y)", fontsize=14)
    ax2.set_ylabel("Position (m)", fontsize=12)
    ax2.legend(loc='upper right', ncol=2)

    # --- PLOT 3: SPATIAL PATH ---
    ax3 = axes[2]
    ax3.plot(df['x_loc'], df['y_loc'], label='Leader', color='#1f77b4', linewidth=2, alpha=0.8)
    ax3.plot(df['x_rem'], df['y_rem'], label='Remote', color='#ff7f0e', linewidth=2, linestyle='--')
    ax3.set_title("Spatial Path (X-Y)", fontsize=14)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.axis('equal')
    ax3.legend()

    # Stats
    rmse = np.sqrt((df['error_xy']**2).mean())
    avg_delay = df['total_delay'].mean()
    stats = f"RMSE: {rmse:.4f}m | Avg Delay: {avg_delay:.1f} steps"
    ax1.text(0.5, 0.95, stats, transform=ax1.transAxes, ha='center', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    output_file = 'delay_impact_analysis.png'
    plt.savefig(output_file, dpi=300)
    print(f"Analysis saved to {output_file}")
    plt.show()

def main():
    import sys
    # Quick arg parsing
    if len(sys.argv) < 2:
        print("Usage: python3 visual.py <path_to_bag>")
        return
    bag_path = sys.argv[1]

    topics = {
        'leader': '/leader/ee_pose',
        'remote': '/remote/ee_pose',
        'obs_delay': '/agent/obs_delay_steps',
        'act_delay': '/agent/act_delay_steps'
    }

    print(f"Reading bag: {bag_path}...")
    dfs = read_bag_data(bag_path, topics)
    
    if dfs:
        print("Aligning data streams...")
        df_aligned = process_and_align_data(dfs)
        if df_aligned is not None:
            visualize_analysis(df_aligned)

if __name__ == "__main__":
    main()