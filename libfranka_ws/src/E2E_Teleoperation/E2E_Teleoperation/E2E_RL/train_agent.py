"""
The starting point for training.

Supports both single and vectorized environments.

argparse Arguments:
--num-envs: Number of parallel environments to run (default: 1)
--config: Delay configuration to use (1, 2, or 3) (default: 3)
--trajectory-type: Type of trajectory to train on (default: FIGURE_8)
--seed: Random seed for reproducibility (default: 42)
--render: Whether to render the environment during training (default: False)
--load-dir: Directory to load pre-trained models from (default: None)
"""

import argparse
from pathlib import Path
from datetime import datetime
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
from E2E_Teleoperation.E2E_RL.unified_trainer import UnifiedTrainer
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig

def make_env(rank, args):
    """Utility function for multiprocessed env."""
    def _init():
        env = TeleoperationEnv(
            delay_config=ExperimentConfig.HIGH_VARIANCE,
            trajectory_type=TrajectoryType.FIGURE_8,
            randomize_trajectory=True, 
            seed=args.seed + rank,
            render_mode=None 
        )
        return env
    return _init

def train_agent(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PureE2E_SAC_{args.num_envs}Envs_{timestamp}"
    output_dir = cfg.ROBOT.CHECKPOINT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
   
    # --- 1. Initialize Training Environments (Vectorized) ---
    if args.num_envs > 1:
        print(f"\n[INFO] Initializing {args.num_envs} Parallel Environments (SubprocVecEnv)...")
        env_fns = [make_env(i, args) for i in range(args.num_envs)]
        env = SubprocVecEnv(env_fns, start_method='fork') 
    else:
        print("\n[INFO] Initializing Single Training Environment...")
        render_mode = "human" if args.render else None
        env = TeleoperationEnv(
            delay_config=ExperimentConfig.HIGH_VARIANCE,
            trajectory_type=TrajectoryType.FIGURE_8,
            randomize_trajectory=True,
            seed=args.seed,
            render_mode=render_mode
        )

    # --- 2. Initialize Evaluation Environment (Always Single) ---
    # This prevents the shape mismatch error during evaluation loops
    print("[INFO] Initializing Separate Evaluation Environment...")
    eval_env = TeleoperationEnv(
        delay_config=ExperimentConfig.HIGH_VARIANCE,
        trajectory_type=TrajectoryType.FIGURE_8,
        randomize_trajectory=True, # Randomize to test generalization
        seed=args.seed + 1000,     # Different seed for evaluation
        render_mode=None
    )
    
    # --- 3. Initialize Trainer ---
    print(f"[INFO] Output Directory: {output_dir}")
    trainer = UnifiedTrainer(
        env=env,
        eval_env=eval_env,  # <--- PASS EVAL ENV HERE
        output_dir=output_dir
    )

    # --- 4. Start Training ---
    try:
        trainer.train_e2e()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user. Saving context...")
    finally:
        env.close()
        eval_env.close()
        print("[INFO] Environment closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="3", choices=['1', '2', '3']) # Kept for compatibility
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render (only works with num-envs=1)")
    
    args = parser.parse_args()
    
    if args.num_envs > 1 and args.render:
        print("[WARNING] Rendering is disabled for vectorized environments. Setting render=False.")
        args.render = False

    train_agent(args)