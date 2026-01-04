"""
Training Monitor
Real-time monitoring tool for End-to-End Teleoperation training.
Usage: python monitor.py --log-dir ./trained_RL/E2E_.../logs
"""

import argparse
from pathlib import Path
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

class TrainingMonitor:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        # Assumes structure: checkpoint_dir/logs/ -> parent is checkpoint_dir
        self.checkpoint_dir = self.log_dir.parent 
        
        if not self.log_dir.exists():
            # Fallback: check if user provided checkpoint root instead of logs/
            if (self.log_dir / "logs").exists():
                self.checkpoint_dir = self.log_dir
                self.log_dir = self.log_dir / "logs"
            else:
                raise ValueError(f"Log directory not found: {self.log_dir}")
        
        print(f"[INFO] Monitoring directory: {self.checkpoint_dir.name}")

    def check_training_progress(self) -> Dict:
        """Check checkpoints and metadata JSON files."""
        progress = {}
        
        # 1. Check metadata (JSON)
        metadata_files = list(self.checkpoint_dir.glob("metadata_*.json"))
        if metadata_files:
            latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_metadata) as f:
                    data = json.load(f)
                    progress.update(data)
            except json.JSONDecodeError:
                pass

        # 2. Check Checkpoints
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        if checkpoints:
            latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
            progress['latest_checkpoint'] = latest_ckpt.name
            
        return progress

    def parse_log_file(self, log_file: Path) -> List[Dict]:
        """
        Parses log files for both Phase 1 (Pre-training) and Phase 2 (RL).
        Expected Formats:
        Phase 1: "Phase 1 Step 100 | Encoder Loss: 0.123"
        Phase 2: "Step 100 | CritLoss: 0.123 | ActLoss: 0.123 | ..."
        """
        metrics = []
        
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                step_data = {}
                
                # Case A: Phase 1 (Encoder Pre-training)
                if "Phase 1 Step" in line:
                    try:
                        parts = line.split('|')
                        # Part 0: "Phase 1 Step 100"
                        step_str = parts[0].replace("Phase 1 Step", "").strip()
                        step_data['step'] = int(step_str)
                        
                        # Part 1: "Encoder Loss: 0.123"
                        if len(parts) > 1 and "Encoder Loss" in parts[1]:
                            val_str = parts[1].split(":")[1].strip()
                            step_data['encoder_loss'] = float(val_str)
                            metrics.append(step_data)
                    except (ValueError, IndexError):
                        continue

                # Case B: Phase 2 (RL Training)
                elif line.startswith("Step"):
                    try:
                        parts = line.split('|')
                        # Part 0: "Step 100"
                        step_data['step'] = int(parts[0].replace('Step', '').strip())
                        
                        for part in parts[1:]:
                            if ':' in part:
                                key, val = part.split(':')
                                key = key.strip()
                                val = val.strip()
                                
                                # Map abbreviated keys from UnifiedTrainer
                                if key == 'CritLoss': step_data['critic_loss'] = float(val)
                                elif key == 'ActLoss': step_data['actor_loss'] = float(val)
                                elif key == 'PredLoss': step_data['pred_loss'] = float(val)
                                elif key == 'Alpha': step_data['alpha'] = float(val)
                        
                        if len(step_data) > 1:
                            metrics.append(step_data)
                    except (ValueError, IndexError):
                        continue
                    
        return metrics

    def estimate_time_remaining(self, current_step: int, total_steps: int, 
                                start_time: datetime) -> str:
        if current_step == 0: return "Calculating..."
        
        elapsed = (datetime.now() - start_time).total_seconds()
        # Avoid division by zero
        sps = current_step / (elapsed + 1e-6)
        remaining_sec = (total_steps - current_step) / (sps + 1e-6)
        
        return str(timedelta(seconds=int(remaining_sec)))

    def check_for_issues(self, metrics: List[Dict]) -> List[str]:
        issues = []
        if len(metrics) < 10: return issues
        
        recent = metrics[-20:]
        last_metric = recent[-1]

        # Phase 2 Checks
        if 'critic_loss' in last_metric:
            crit_losses = [m.get('critic_loss', 0) for m in recent]
            if any(l > 5000 for l in crit_losses):
                issues.append("[CRITICAL] Critic loss is exploding (>5000). Possible gradient explosion.")
            if np.mean(crit_losses) < 1e-5:
                issues.append("[WARNING] Critic loss vanished (<1e-5). Possible collapse.")

            pred_losses = [m.get('pred_loss', 0) for m in recent]
            if np.mean(pred_losses) > 20.0:
                 issues.append("[WARNING] High Prediction Loss (>20.0). LSTM may not be converging.")
        
        # Phase 1 Checks
        if 'encoder_loss' in last_metric:
             enc_losses = [m.get('encoder_loss', 0) for m in recent]
             if enc_losses[-1] > enc_losses[0] * 2.0:
                 issues.append("[WARNING] Phase 1 Divergence: Encoder loss is increasing.")

        return issues

    def continuous_monitor(self, refresh_rate=5, total_steps=1_000_000):
        # Clear screen command
        print("\033[2J") 
        start_time = datetime.now()
        
        try:
            while True:
                # 1. Fetch Data
                progress = self.check_training_progress()
                
                log_files = list(self.log_dir.glob("training_*.log"))
                metrics = []
                phase = "Initializing..."
                
                if log_files:
                    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                    metrics = self.parse_log_file(latest_log)
                    if metrics:
                        last_m = metrics[-1]
                        if 'encoder_loss' in last_m: 
                            phase = "Phase 1: Pre-training (Supervised)"
                        elif 'critic_loss' in last_m: 
                            phase = "Phase 2: RL Training (SAC)"

                # 2. Display Header
                # Move cursor to top left
                print("\033[H", end="") 
                print("-" * 60)
                print(f" TRAINING MONITOR | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 60)
                
                # 3. Display Status
                curr_step = 0
                if metrics:
                    curr_step = metrics[-1]['step']
                    
                pct = (curr_step / total_steps) * 100
                eta = self.estimate_time_remaining(curr_step, total_steps, start_time)
                
                print(f" Current Phase: {phase}")
                print(f" Progress:      {curr_step:,} / {total_steps:,} steps ({pct:.2f}%)")
                print(f" Estimated Time:{eta}")
                print(f" Best Reward:   {progress.get('best_eval_reward', -np.inf):.4f}")
                print("-" * 60)

                # 4. Display Recent Metrics
                if metrics:
                    last = metrics[-1]
                    print(" LATEST METRICS:")
                    if "Phase 1" in phase:
                        print(f"  * Encoder Loss: {last.get('encoder_loss', 0):.6f}")
                    elif "Phase 2" in phase:
                        print(f"  * Critic Loss:  {last.get('critic_loss', 0):.4f}")
                        print(f"  * Actor Loss:   {last.get('actor_loss', 0):.4f}")
                        print(f"  * Physics Loss: {last.get('pred_loss', 0):.4f}")
                        print(f"  * Alpha (Ent):  {last.get('alpha', 0):.4f}")
                
                # 5. Display Warnings
                issues = self.check_for_issues(metrics)
                if issues:
                    print("-" * 60)
                    print(" SYSTEM ALERTS:")
                    for issue in issues: 
                        print(f"  {issue}")
                else:
                    print("\n  [System Normal]")
                
                time.sleep(refresh_rate)
                
        except KeyboardInterrupt:
            print("\n[INFO] Monitoring terminated by user.")

    def plot_curves(self):
        """Generates static plots for thesis documentation."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[ERROR] Matplotlib not installed. Cannot generate plots.")
            return

        log_files = list(self.log_dir.glob("training_*.log"))
        if not log_files: 
            print("[ERROR] No log files found.")
            return
        
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        metrics = self.parse_log_file(latest_log)
        
        if not metrics or 'critic_loss' not in metrics[-1]:
            print("[INFO] Insufficient data or still in Phase 1. Skipping plot.")
            return

        steps = [m['step'] for m in metrics if 'critic_loss' in m]
        crit = [m['critic_loss'] for m in metrics if 'critic_loss' in m]
        act = [m['actor_loss'] for m in metrics if 'actor_loss' in m]
        pred = [m['pred_loss'] for m in metrics if 'pred_loss' in m]

        fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        
        # Plot 1: Critic
        ax[0].plot(steps, crit, color='black', linewidth=1)
        ax[0].set_title("Critic Loss", fontsize=10)
        ax[0].grid(True, linestyle='--', alpha=0.5)
        
        # Plot 2: Actor
        ax[1].plot(steps, act, color='black', linewidth=1)
        ax[1].set_title("Actor Loss", fontsize=10)
        ax[1].grid(True, linestyle='--', alpha=0.5)

        # Plot 3: Prediction
        ax[2].plot(steps, pred, color='black', linewidth=1)
        ax[2].set_title("State Prediction Error (MSE)", fontsize=10)
        ax[2].grid(True, linestyle='--', alpha=0.5)
        ax[2].set_xlabel("Training Steps")

        output_path = self.checkpoint_dir / "training_summary_plot.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"[INFO] Training plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the logs directory")
    parser.add_argument("--plot", action="store_true", help="Generate a static plot instead of monitoring")
    args = parser.parse_args()

    monitor = TrainingMonitor(args.log_dir)
    
    if args.plot:
        monitor.plot_curves()
    else:
        monitor.continuous_monitor()