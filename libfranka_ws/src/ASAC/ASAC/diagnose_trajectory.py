"""
Trajectory & IK Diagnostic Script
----------------------------------
Run this to visualize and debug why the robot gets stuck.

Usage:
    python3 -m ASAC.diagnose_trajectory --render
"""

import argparse
import numpy as np
import time

import ASAC.config.robot_config as cfg
from ASAC.leader_robot_simulator import LeaderRobotSimulator
from ASAC.follower_robot_simulator import FollowerRobotSimulator
from ASAC.utils.delay_simulator import ExperimentConfig


def test_leader_trajectory(num_steps=4000, verbose=True):
    """Test leader trajectory generation and IK."""
    print("\n" + "=" * 60)
    print("Testing Leader Trajectory (IK)")
    print("=" * 60)
    
    leader = LeaderRobotSimulator(
        trajectory_type="figure_8",
        randomize_params=False,
        verbose=True
    )
    
    leader.reset()
    
    ik_failures = 0
    q_history = []
    qd_history = []
    pos_history = []
    
    print(f"\nTrajectory: {leader._trajectory_type.value}")
    print(f"Center: {leader._params.center}")
    print(f"Scale: {leader._params.scale}")
    print(f"Frequency: {leader._params.frequency} Hz")
    print(f"\nRunning {num_steps} steps...")
    
    for step in range(num_steps):
        q, qd, _, _, _, _, info = leader.step()
        
        q_history.append(q.copy())
        qd_history.append(qd.copy())
        
        if not info.get('ik_success', True):
            ik_failures += 1
        
        if verbose and step % 500 == 0:
            qd_norm = np.linalg.norm(qd)
            print(f"  Step {step:4d}: |qd|={qd_norm:.4f}, IK failures={info.get('ik_failures', 0)}")
    
    q_history = np.array(q_history)
    qd_history = np.array(qd_history)
    
    print("\n" + "-" * 40)
    print("Results:")
    print(f"  Total IK failures: {ik_failures} / {num_steps} ({100*ik_failures/num_steps:.1f}%)")
    print(f"  Joint position range:")
    for j in range(7):
        print(f"    Joint {j}: [{q_history[:,j].min():.3f}, {q_history[:,j].max():.3f}]")
    print(f"  Max joint velocity: {np.abs(qd_history).max():.4f} rad/s")
    
    # Check for "stuck" periods (qd ≈ 0 for extended time)
    qd_norms = np.linalg.norm(qd_history, axis=1)
    stuck_threshold = 0.01
    stuck_steps = np.sum(qd_norms < stuck_threshold)
    print(f"  Steps with |qd| < {stuck_threshold}: {stuck_steps} ({100*stuck_steps/num_steps:.1f}%)")
    
    return q_history, qd_history


def test_pd_tracking(num_steps=4000, render=False, delay_config=0):
    """Test PD tracking with fixed gains."""
    print("\n" + "=" * 60)
    print("Testing PD Tracking (Fixed Gains)")
    print("=" * 60)
    
    leader = LeaderRobotSimulator(
        trajectory_type="figure_8",
        randomize_params=False,
        verbose=False
    )
    
    follower = FollowerRobotSimulator(
        delay_config=ExperimentConfig(delay_config),
        seed=42,
        render=render,
        render_fps=60,
        verbose=True
    )
    
    # Fixed PD gains
    kp = cfg.PD_GAINS.KP_BASE.copy()
    kd = cfg.PD_GAINS.KD_BASE.copy()
    
    print(f"\nKp: {kp}")
    print(f"Kd: {kd}")
    print(f"Delay config: {delay_config}")
    
    # Reset
    l_q, _ = leader.reset()
    f_q, _ = follower.reset()
    f_qd = np.zeros(7, dtype=np.float32)
    
    pos_errors = []
    
    print(f"\nRunning {num_steps} steps...")
    start_time = time.time()
    
    for step in range(num_steps):
        # Leader step
        l_q, l_qd, _, _, _, _, _ = leader.step()
        
        # PD control (no delay for this test)
        q_error = l_q - f_q
        qd_error = l_qd - f_qd
        
        pd_torque = kp * q_error + kd * qd_error
        pd_torque = np.clip(pd_torque, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)
        
        # Follower step
        follower_info = follower.step(pd_torque)
        f_q_new = follower_info['q_follower']
        f_qd = (f_q_new - f_q) / cfg.DT
        f_q = f_q_new
        
        pos_error = np.linalg.norm(l_q - f_q)
        pos_errors.append(pos_error)
        
        if step % 500 == 0:
            print(f"  Step {step:4d}: Pos Error = {pos_error:.4f} rad")
    
    elapsed = time.time() - start_time
    follower.close()
    
    pos_errors = np.array(pos_errors)
    
    print("\n" + "-" * 40)
    print("Results:")
    print(f"  Mean position error: {np.mean(pos_errors):.4f} rad")
    print(f"  Max position error: {np.max(pos_errors):.4f} rad")
    print(f"  Time: {elapsed:.1f}s ({num_steps/elapsed:.0f} steps/sec)")
    
    # Find when errors spike (potential stuck points)
    high_error_threshold = np.mean(pos_errors) + 2 * np.std(pos_errors)
    high_error_steps = np.where(pos_errors > high_error_threshold)[0]
    if len(high_error_steps) > 0:
        print(f"\n  High error steps (>{high_error_threshold:.4f}):")
        # Group consecutive steps
        groups = np.split(high_error_steps, np.where(np.diff(high_error_steps) > 10)[0] + 1)
        for g in groups[:5]:  # Show first 5 groups
            if len(g) > 0:
                print(f"    Steps {g[0]}-{g[-1]} (duration: {len(g)} steps)")
    
    return pos_errors


def main():
    parser = argparse.ArgumentParser(description="Diagnose Trajectory and IK")
    parser.add_argument("--render", action="store_true", help="Enable visualization")
    parser.add_argument("--steps", type=int, default=4000, help="Number of steps")
    parser.add_argument("--delay", type=int, default=0, choices=[0,1,2,3],
                        help="Delay config (0=NoDelay)")
    args = parser.parse_args()
    
    # Test 1: Leader trajectory alone
    q_hist, qd_hist = test_leader_trajectory(num_steps=args.steps, verbose=True)
    
    # Test 2: PD tracking
    pos_errors = test_pd_tracking(
        num_steps=args.steps, 
        render=args.render, 
        delay_config=args.delay
    )
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    
    # Interpretation
    mean_error = np.mean(pos_errors)
    if mean_error > 0.1:
        print("\n[!] High tracking error detected. Possible causes:")
        print("    - IK failures at trajectory corners")
        print("    - Trajectory outside reachable workspace")
        print("    - PD gains too low")
    elif mean_error > 0.05:
        print("\n[i] Moderate tracking error. Consider:")
        print("    - Increasing PD gains")
        print("    - Reducing trajectory speed (frequency)")
    else:
        print("\n[OK] Tracking looks good!")


if __name__ == "__main__":
    main()