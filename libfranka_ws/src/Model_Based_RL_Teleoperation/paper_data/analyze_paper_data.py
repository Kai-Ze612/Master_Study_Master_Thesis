# analyze_paper_data.py
# 針對 paper_data/ 乾淨結構的最終分析腳本
# 產出: tracking error metrics, worst-case dominance check, paper-ready summary

import os
import csv
import numpy as np
import pandas as pd
from mcap_ros2.reader import read_ros2_messages

# === Settings ===
T_CUTOFF = 50.0  # 取前 50 秒
PAPER_DATA = os.path.dirname(os.path.abspath(__file__))  # 假設 script 在 paper_data/ 裡

# Topic naming per method (因為 PMDC 用了不同 topic 名稱)
TOPIC_CONFIG = {
    "DRR-RL": {"leader": "/leader/ee_pose",     "follower": "/remote/ee_pose"},
    "PMDC":   {"leader": "/leader/ee_position", "follower": "/remote/ee_position"},
    "PD":     {"leader": "/leader/ee_pose",     "follower": "/remote/ee_pose"},
}

CONDITIONS = ["low_low", "high_low", "high_high"]
METHODS = ["DRR-RL", "PMDC", "PD"]


# === Helpers ===
def find_mcap(bag_dir):
    for f in os.listdir(bag_dir):
        if f.endswith(".mcap"):
            return os.path.join(bag_dir, f)
    return None


def find_method_condition_dir(method, condition):
    """從 paper_data/<method>/<condition>/ 找到實際 bag 資料夾"""
    parent = os.path.join(PAPER_DATA, method, condition)
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


def load_tracking_error(method, condition):
    """讀指定 method × condition 的 bag,return (times, error) array"""
    bag_dir = find_method_condition_dir(method, condition)
    if bag_dir is None:
        return None, None
    
    mcap = find_mcap(bag_dir)
    if mcap is None:
        return None, None
    
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
        return None, None
    
    n = min(len(leader), len(follower))
    t0 = leader[0][0]
    times = np.array([(leader[i][0] - t0) / 1e9 for i in range(n)])
    lpos = np.array([(leader[i][1], leader[i][2], leader[i][3]) for i in range(n)])
    fpos = np.array([(follower[i][1], follower[i][2], follower[i][3]) for i in range(n)])
    error = np.linalg.norm(lpos - fpos, axis=1)
    
    mask = times <= T_CUTOFF
    return times[mask], error[mask]


def compute_metrics(error):
    return {
        "mean":   np.mean(error),
        "median": np.median(error),
        "p95":    np.percentile(error, 95),
        "max":    np.max(error),
        "min":    np.min(error),
        "std":    np.std(error),
        "tv_per_n": np.sum(np.abs(np.diff(error))) / len(error),
        "n":      len(error),
    }


# === Main ===
print("=" * 100)
print(f"PAPER DATA ANALYSIS — first {T_CUTOFF:.0f} seconds")
print("=" * 100)

results = {}  # {(method, condition): error array}
metrics_table = []

for method in METHODS:
    for cond in CONDITIONS:
        times, error = load_tracking_error(method, cond)
        if error is None:
            print(f"  MISSING: {method}/{cond}")
            continue
        results[(method, cond)] = error
        m = compute_metrics(error)
        metrics_table.append({
            "method": method, "condition": cond, **m
        })

# === Print main table ===
print("\n" + "=" * 100)
print(f"{'Method':<10} {'Condition':<12} {'Mean':>8} {'Median':>8} {'P95':>8} {'Max':>8} {'Min':>8} {'Std':>8} {'TV/n':>8}")
print("-" * 100)

for cond in CONDITIONS:
    for method in METHODS:
        rows = [r for r in metrics_table if r["method"] == method and r["condition"] == cond]
        if not rows:
            print(f"{method:<10} {cond:<12} {'(missing)':>30}")
            continue
        r = rows[0]
        print(f"{method:<10} {cond:<12} "
              f"{r['mean']:>8.4f} {r['median']:>8.4f} "
              f"{r['p95']:>8.4f} {r['max']:>8.4f} {r['min']:>8.4f} "
              f"{r['std']:>8.4f} {r['tv_per_n']:>8.4f}")
    print("-" * 100)


# === DRR-RL vs PMDC head-to-head ===
print("\n" + "=" * 100)
print("DRR-RL vs PMDC — HEAD-TO-HEAD")
print("=" * 100)
print(f"{'Condition':<12} {'DRR Mean':>10} {'PMDC Mean':>10} {'Mean Δ%':>9} | "
      f"{'DRR P95':>9} {'PMDC Mean':>10} {'P95 dom?':>10} | "
      f"{'DRR Max':>9} {'PMDC Min':>9} {'Strong dom?':>13}")
print("-" * 100)

for cond in CONDITIONS:
    if ("DRR-RL", cond) not in results or ("PMDC", cond) not in results:
        print(f"{cond:<12} (missing)")
        continue
    
    drr = results[("DRR-RL", cond)]
    pmdc = results[("PMDC", cond)]
    
    drr_mean = np.mean(drr)
    drr_p95 = np.percentile(drr, 95)
    drr_max = np.max(drr)
    pmdc_mean = np.mean(pmdc)
    pmdc_min = np.min(pmdc)
    
    improvement = (pmdc_mean - drr_mean) / pmdc_mean * 100
    p95_dom = drr_p95 < pmdc_mean
    strong_dom = drr_max < pmdc_min
    
    print(f"{cond:<12} "
          f"{drr_mean:>10.4f} {pmdc_mean:>10.4f} {improvement:>+8.1f}% | "
          f"{drr_p95:>9.4f} {pmdc_mean:>10.4f} {str(p95_dom):>10} | "
          f"{drr_max:>9.4f} {pmdc_min:>9.4f} {str(strong_dom):>13}")


# === DRR-RL vs PD head-to-head ===
print("\n" + "=" * 100)
print("DRR-RL vs PD — HEAD-TO-HEAD")
print("=" * 100)
print(f"{'Condition':<12} {'DRR Mean':>10} {'PD Mean':>10} {'Mean Δ%':>9} | "
      f"{'DRR Max':>9} {'PD Max':>9} {'Max Δ%':>9}")
print("-" * 100)

for cond in CONDITIONS:
    if ("DRR-RL", cond) not in results or ("PD", cond) not in results:
        print(f"{cond:<12} (missing)")
        continue
    
    drr = results[("DRR-RL", cond)]
    pd_e = results[("PD", cond)]
    
    drr_mean = np.mean(drr)
    drr_max = np.max(drr)
    pd_mean = np.mean(pd_e)
    pd_max = np.max(pd_e)
    
    mean_delta = (pd_mean - drr_mean) / pd_mean * 100
    max_delta = (pd_max - drr_max) / pd_max * 100
    
    print(f"{cond:<12} "
          f"{drr_mean:>10.4f} {pd_mean:>10.4f} {mean_delta:>+8.1f}% | "
          f"{drr_max:>9.4f} {pd_max:>9.4f} {max_delta:>+8.1f}%")


# === Save metrics CSV ===
df = pd.DataFrame(metrics_table)
out_csv = os.path.join(PAPER_DATA, "metrics_summary.csv")
df.to_csv(out_csv, index=False)
print(f"\nMetrics saved to {out_csv}")


# === Generate LaTeX-ready table ===
print("\n" + "=" * 100)
print("LATEX TABLE (paper-ready)")
print("=" * 100)
print(r"""\begin{table}[t]
\centering
\caption{Tracking error metrics across delay conditions.}
\label{tab:tracking_metrics}
\small
\begin{tabular}{llcccc}
\toprule
Condition & Method & Mean (m) & P95 (m) & Max (m) & Std (m) \\
\midrule""")

cond_label = {
    "low_low":   r"Low / low var.",
    "high_low":  r"High / low var.",
    "high_high": r"High / high var.",
}

for cond in CONDITIONS:
    for method in METHODS:
        rows = [r for r in metrics_table if r["method"] == method and r["condition"] == cond]
        if not rows:
            continue
        r = rows[0]
        cond_str = cond_label[cond] if method == "DRR-RL" else ""  # only first row of group
        print(f"{cond_str} & {method} & {r['mean']:.4f} & {r['p95']:.4f} & {r['max']:.4f} & {r['std']:.4f} \\\\")
    print(r"\midrule")

print(r"""\bottomrule
\end{tabular}
\end{table}""")