# analyze_full_paper_data.py — CSV 版本 (real DRR-RL 從 CSV 讀)

import os
import csv
import numpy as np
import pandas as pd
from mcap_ros2.reader import read_ros2_messages

T_CUTOFF = 50.0
PAPER_DATA = os.path.dirname(os.path.abspath(__file__))

TOPIC_CONFIG = {
    "DRR-RL": {"leader": "/leader/ee_pose",     "follower": "/remote/ee_pose"},
    "PMDC":   {"leader": "/leader/ee_position", "follower": "/remote/ee_position"},
    "PD":     {"leader": "/leader/ee_pose",     "follower": "/remote/ee_pose"},
}

CONDITIONS = ["low_low", "high_low", "high_high"]

EXPERIMENTS = [
    ("sim",  "DRR-RL"),
    ("sim",  "PMDC"),
    ("sim",  "PD"),
    ("real", "DRR-RL"),
]


# === Helpers ===
def find_mcap(bag_dir):
    for f in os.listdir(bag_dir):
        if f.endswith(".mcap"):
            return os.path.join(bag_dir, f)
    return None


def find_csv(bag_dir):
    """從 bag_dir 找 tracking_data.csv"""
    for f in os.listdir(bag_dir):
        if f.endswith(".csv"):
            return os.path.join(bag_dir, f)
    return None


def find_bag_dir(env, method, condition):
    parent = os.path.join(PAPER_DATA, env, method, condition)
    if not os.path.isdir(parent):
        return None
    for d in os.listdir(parent):
        full = os.path.join(parent, d)
        if os.path.isdir(full) and d.startswith("rosbag2_"):
            return full
    return None


def extract_xyz(ros_msg):
    if hasattr(ros_msg, 'point'):
        return ros_msg.point.x, ros_msg.point.y, ros_msg.point.z
    elif hasattr(ros_msg, 'x'):
        return ros_msg.x, ros_msg.y, ros_msg.z
    raise ValueError(f"Unknown message type: {type(ros_msg)}")


def load_tracking_error_from_mcap(env, method, condition):
    bag_dir = find_bag_dir(env, method, condition)
    if bag_dir is None:
        return None
    mcap = find_mcap(bag_dir)
    if mcap is None:
        return None
    
    leader_topic = TOPIC_CONFIG[method]["leader"]
    follower_topic = TOPIC_CONFIG[method]["follower"]
    
    leader, follower = [], []
    for msg in read_ros2_messages(mcap):
        topic = msg.channel.topic
        ts_ns = msg.log_time_ns
        ros_msg = msg.ros_msg
        if topic == leader_topic:
            x, y, z = extract_xyz(ros_msg)
            leader.append((ts_ns, x, y, z))
        elif topic == follower_topic:
            x, y, z = extract_xyz(ros_msg)
            follower.append((ts_ns, x, y, z))
    
    if not leader or not follower:
        return None
    
    n = min(len(leader), len(follower))
    t0 = leader[0][0]
    times = np.array([(leader[i][0] - t0) / 1e9 for i in range(n)])
    lpos = np.array([(leader[i][1], leader[i][2], leader[i][3]) for i in range(n)])
    fpos = np.array([(follower[i][1], follower[i][2], follower[i][3]) for i in range(n)])
    error = np.linalg.norm(lpos - fpos, axis=1)
    
    mask = times <= T_CUTOFF
    return error[mask]


def load_tracking_error_from_csv(env, method, condition):
    """從 tracking_data.csv 讀 leader/follower 算 error"""
    bag_dir = find_bag_dir(env, method, condition)
    if bag_dir is None:
        return None
    csv_path = find_csv(bag_dir)
    if csv_path is None:
        return None
    
    df = pd.read_csv(csv_path)
    
    # CSV column names: time, /leader/ee_pose/point/x, ...
    df['t_sec'] = df['time'] - df['time'].iloc[0]
    df = df[df['t_sec'] <= T_CUTOFF].reset_index(drop=True)
    
    lx = df['/leader/ee_pose/point/x'].values
    ly = df['/leader/ee_pose/point/y'].values
    lz = df['/leader/ee_pose/point/z'].values
    fx = df['/remote/ee_pose/point/x'].values
    fy = df['/remote/ee_pose/point/y'].values
    fz = df['/remote/ee_pose/point/z'].values
    
    error = np.sqrt((lx - fx)**2 + (ly - fy)**2 + (lz - fz)**2)
    return error


def load_tracking_error(env, method, condition):
    """Real DRR-RL 從 CSV 讀,其他從 mcap 讀"""
    if env == "real" and method == "DRR-RL":
        # 優先從 CSV 讀,fallback 到 mcap
        e = load_tracking_error_from_csv(env, method, condition)
        if e is not None:
            return e
    return load_tracking_error_from_mcap(env, method, condition)


def metrics(error):
    return {
        "mean":   np.mean(error),
        "median": np.median(error),
        "p95":    np.percentile(error, 95),
        "max":    np.max(error),
        "std":    np.std(error),
        "tv":     np.sum(np.abs(np.diff(error))) / len(error),
        "n":      len(error),
    }


# === Load all data ===
results = {}
for env, method in EXPERIMENTS:
    for cond in CONDITIONS:
        e = load_tracking_error(env, method, cond)
        if e is None:
            continue
        results[(env, method, cond)] = e
        print(f"Loaded {env}/{method}/{cond}: n={len(e)}")


# === Print: SIM comparison table ===
print("\n" + "=" * 110)
print("TABLE 1: SIMULATION COMPARISON (first 50 s, three delay conditions)")
print("=" * 110)
print(f"{'Method':<10} {'Condition':<12} {'Mean':>8} {'Median':>8} {'P95':>8} {'Max':>8} {'Std':>8} {'TV/n':>8}")
print("-" * 110)
for cond in CONDITIONS:
    for method in ["DRR-RL", "PMDC", "PD"]:
        if ("sim", method, cond) not in results:
            print(f"{method:<10} {cond:<12} {'(missing)':>30}")
            continue
        m = metrics(results[("sim", method, cond)])
        print(f"{method:<10} {cond:<12} "
              f"{m['mean']:>8.4f} {m['median']:>8.4f} {m['p95']:>8.4f} "
              f"{m['max']:>8.4f} {m['std']:>8.4f} {m['tv']:>8.4f}")
    print("-" * 110)


# === Print: REAL DRR-RL table ===
print("\n" + "=" * 110)
print("TABLE 2: REAL-ROBOT DRR-RL DEPLOYMENT (first 50 s, from CSV)")
print("=" * 110)
print(f"{'Condition':<12} {'Mean':>8} {'Median':>8} {'P95':>8} {'Max':>8} {'Std':>8} {'TV/n':>8}")
print("-" * 110)
for cond in CONDITIONS:
    if ("real", "DRR-RL", cond) not in results:
        print(f"{cond:<12} {'(missing)':>30}")
        continue
    m = metrics(results[("real", "DRR-RL", cond)])
    print(f"{cond:<12} "
          f"{m['mean']:>8.4f} {m['median']:>8.4f} {m['p95']:>8.4f} "
          f"{m['max']:>8.4f} {m['std']:>8.4f} {m['tv']:>8.4f}")


# === Print: SIM-to-REAL gap ===
print("\n" + "=" * 110)
print("TABLE 3: SIM-TO-REAL GAP for DRR-RL")
print("=" * 110)
print(f"{'Condition':<12} {'Sim Mean':>10} {'Real Mean':>10} {'Δ Mean':>9} {'Ratio':>7} | "
      f"{'Sim Max':>9} {'Real Max':>9} {'Δ Max':>9}")
print("-" * 110)
for cond in CONDITIONS:
    sim_key = ("sim", "DRR-RL", cond)
    real_key = ("real", "DRR-RL", cond)
    if sim_key not in results or real_key not in results:
        print(f"{cond:<12} (missing)")
        continue
    s = metrics(results[sim_key])
    r = metrics(results[real_key])
    gap_mean = r['mean'] - s['mean']
    ratio = r['mean'] / s['mean']
    gap_max = r['max'] - s['max']
    print(f"{cond:<12} "
          f"{s['mean']:>10.4f} {r['mean']:>10.4f} {gap_mean:>+9.4f} {ratio:>6.2f}x | "
          f"{s['max']:>9.4f} {r['max']:>9.4f} {gap_max:>+9.4f}")


# === DRR-RL vs PMDC head-to-head (sim) ===
print("\n" + "=" * 110)
print("DRR-RL vs PMDC HEAD-TO-HEAD (simulation only)")
print("=" * 110)
print(f"{'Condition':<12} {'DRR Mean':>10} {'PMDC Mean':>10} {'Improvement':>13} | "
      f"{'DRR P95':>9} {'PMDC Mean':>10} {'P95 dom?':>10}")
print("-" * 110)
for cond in CONDITIONS:
    drr_key = ("sim", "DRR-RL", cond)
    pmdc_key = ("sim", "PMDC", cond)
    if drr_key not in results or pmdc_key not in results:
        print(f"{cond:<12} (missing)")
        continue
    drr = results[drr_key]
    pmdc = results[pmdc_key]
    imp = (np.mean(pmdc) - np.mean(drr)) / np.mean(pmdc) * 100
    p95_dom = np.percentile(drr, 95) < np.mean(pmdc)
    print(f"{cond:<12} "
          f"{np.mean(drr):>10.4f} {np.mean(pmdc):>10.4f} {imp:>+12.1f}% | "
          f"{np.percentile(drr, 95):>9.4f} {np.mean(pmdc):>10.4f} {str(p95_dom):>10}")


# Save CSV
rows = []
for (env, method, cond), e in results.items():
    m = metrics(e)
    rows.append({"env": env, "method": method, "condition": cond, **m})
df = pd.DataFrame(rows)
out_csv = os.path.join(PAPER_DATA, "full_metrics_summary.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved to {out_csv}")