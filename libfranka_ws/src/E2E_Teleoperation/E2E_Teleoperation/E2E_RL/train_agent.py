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
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
from E2E_Teleoperation.E2E_RL.unified_trainer import UnifiedTrainer
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig

def make_env(rank, args):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = TeleoperationEnv(
            delay_config=args.config,
            trajectory_type=args.trajectory_type,
            randomize_trajectory=args.randomize_trajectory,
            seed=args.seed + rank,
            render_mode=None 
        )
        return env
    return _init

def train_agent(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"E2E_{args.config.name}_{args.trajectory_type.name}_{timestamp}"
    output_dir = cfg.ROBOT.CHECKPOINT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
   
    if args.num_envs > 1:
        print(f"INITIALIZING {args.num_envs} PARALLEL ENVIRONMENTS")
        env_fns = [make_env(i, args) for i in range(args.num_envs)]
        env = SubprocVecEnv(env_fns)
    else:
        print("INITIALIZING SINGLE ENVIRONMENT")
        from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
        
        render_mode = "human" if args.render else None
        
        env = TeleoperationEnv(
            delay_config=args.config,
            trajectory_type=args.trajectory_type,
            randomize_trajectory=args.randomize_trajectory,
            seed=args.seed,
            render_mode=render_mode
        )
    
    trainer = UnifiedTrainer(
        env=env, 
        output_dir=output_dir
    )

    trainer.train_e2e()
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="3", choices=['1', '2', '3'])
    parser.add_argument("--trajectory-type", type=str, default="FIGURE_8", choices=[t.name for t in TrajectoryType])
    parser.add_argument("--randomize-trajectory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load-dir", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    
    args = parser.parse_args()
    CONFIG_MAP = {'1': ExperimentConfig.LOW_DELAY, '2': ExperimentConfig.HIGH_DELAY, '3': ExperimentConfig.HIGH_VARIANCE}
    args.config = CONFIG_MAP[args.config]
    args.trajectory_type = TrajectoryType[args.trajectory_type.upper()]
    
    train_agent(args)