"""
The starting point for training.

Features:
- Supports both single and vectorized environments.

argparse Arguments:
--num-envs: Number of parallel environments to run (default: 1)
--config: Delay configuration to use (1, 2, or 3) (default: 3)
--trajectory-type: Type of trajectory to train on (default: FIGURE_8)
--seed: Random seed for reproducibility (default: 42)
--render: Whether to render the environment during training (default: False)
--start-stage: Training stage to start from (1, 2, or 3) (default: 1), if 3, skips to SAC training
--load-dir: Directory to load pre-trained models from (default: None)
"""

import argparse
from pathlib import Path
from datetime import datetime
import torch
import gymnasium as gym

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
from E2E_Teleoperation.E2E_RL.unified_trainer import UnifiedTrainer
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig

def load_checkpoint(trainer, load_dir):
    p_enc = Path(load_dir) / "stage1_encoder.pth"
    p_act = Path(load_dir) / "stage1_actor.pth"
    if p_enc.exists() and p_act.exists():
        print(f"Loading BC weights from {load_dir}")
        trainer.encoder.load_state_dict(torch.load(p_enc, map_location=trainer.device))
        trainer.actor.load_state_dict(torch.load(p_act, map_location=trainer.device))
    else:
        print(f"Checkpoints not found in {load_dir}")

def train_agent(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"RL_{args.config.name}_{args.trajectory_type.name}_{timestamp}"
    output_dir = cfg.ROBOT.CHECKPOINT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== TRAINING: {run_name} ===")
    
    env = TeleoperationEnv(
        delay_config=args.config,
        trajectory_type=args.trajectory_type,
        randomize_trajectory=args.randomize_trajectory,
        seed=args.seed,
        render_mode="human" if args.render else None
    )

    trainer = UnifiedTrainer(env, str(output_dir), is_vector_env=False)
    
    # === STAGE 1: BC (Behavioral Cloning) ===
    if args.start_stage == 1:
        trainer.train_stage1_bc()
    else:
        if args.load_dir:
            load_checkpoint(trainer, args.load_dir)
        else:
            print("WARNING: Skipping Stage 1 without loading weights!")

    # === STAGE 2: E2E (End-to-End SAC) ===
    trainer.train_stage2_e2e()
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="3", choices=['1', '2', '3'])
    parser.add_argument("--trajectory-type", type=str, default="FIGURE_8", choices=[t.name for t in TrajectoryType])
    parser.add_argument("--randomize-trajectory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--start-stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--load-dir", type=str, default=None)

    args = parser.parse_args()
    CONFIG_MAP = {'1': ExperimentConfig.LOW_DELAY, '2': ExperimentConfig.HIGH_DELAY, '3': ExperimentConfig.HIGH_VARIANCE}
    args.config = CONFIG_MAP[args.config]
    args.trajectory_type = TrajectoryType[args.trajectory_type.upper()]
    
    train_agent(args)