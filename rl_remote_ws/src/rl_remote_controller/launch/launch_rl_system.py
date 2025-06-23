#!/usr/bin/env python3
"""
Launch RL training model
"""

import os
import sys
import argparse

# Add path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
rl_controller_dir = os.path.join(parent_dir, 'rl_remote_controller')
sys.path.insert(0, rl_controller_dir)

def setup_directories():
    """Create necessary directories."""
    directories = [
        os.path.join(parent_dir, 'models'),
        os.path.join(parent_dir, 'logs'), 
        os.path.join(parent_dir, 'results')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def train_rl_model(model_path, timesteps=50000, algorithm='SAC'):
    """Train the RL model."""
    print(f"Starting RL training: {timesteps} timesteps with {algorithm}")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"Error: MuJoCo model not found: {model_path}")
        return False
    
    try:
        import rl_training
        
        # Change to parent directory for training
        original_dir = os.getcwd()
        os.chdir(parent_dir)
        
        model = rl_training.train_simple_rl_agent(
            model_path=model_path,
            total_timesteps=timesteps,
            algorithm=algorithm
        )
        
        # Change back to original directory
        os.chdir(original_dir)
        
        print("="*60)
        print("✓ Training completed successfully!")
        print(f"✓ Model saved: ./models/{algorithm.lower()}_simple_adaptive_pd_best/best_model.zip")
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='RL Training for Adaptive PD Control')
    parser.add_argument('--model-path', type=str, 
                       default="/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/rl_remote_ws/src/rl_remote_controller/models/franka_fr3/fr3.xml",
                       help='Path to MuJoCo robot model')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Training timesteps')
    parser.add_argument('--algorithm', type=str, default='SAC',
                       choices=['SAC', 'PPO'],
                       help='RL algorithm')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Train
    success = train_rl_model(args.model_path, args.timesteps, args.algorithm)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())