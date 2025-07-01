#!/usr/bin/env python3
"""
Launch RL training model with stochastic delays (Paper-accurate)
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

def train_rl_model(model_path, timesteps=50000, algorithm='SAC', experiment_config=1, use_stochastic=True):
    """Train the RL model with options for stochastic or constant delays."""
    
    experiment_names = {1: "90-130ms", 2: "170-210ms", 3: "250-290ms"}
    delay_type = "stochastic" if use_stochastic else "constant"
    exp_name = experiment_names.get(experiment_config, "unknown")
    
    print(f"Starting RL training: {timesteps} timesteps with {algorithm}")
    print(f"Delay type: {delay_type.upper()}")
    if use_stochastic:
        print(f"Experiment config: {experiment_config} ({exp_name})")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"Error: MuJoCo model not found: {model_path}")
        return False
    
    try:
        if use_stochastic:
            # Import stochastic delay training (paper-accurate)
            import rl_training_stochastic
            train_function = rl_training_stochastic.train_paper_accurate_agent
        else:
            # Import original constant delay training
            import rl_training
            train_function = rl_training.train_simple_rl_agent
        
        # Change to parent directory for training
        original_dir = os.getcwd()
        os.chdir(parent_dir)
        
        if use_stochastic:
            # Call stochastic training function
            model = train_function(
                model_path=model_path,
                experiment_config=experiment_config,
                total_timesteps=timesteps,
                algorithm=algorithm
            )
            model_name = f"{algorithm.lower()}_stochastic_delay_exp{experiment_config}"
        else:
            # Call original training function
            model = train_function(
                model_path=model_path,
                total_timesteps=timesteps,
                algorithm=algorithm
            )
            model_name = f"{algorithm.lower()}_simple_adaptive_pd"
        
        # Change back to original directory
        os.chdir(original_dir)
        
        print("="*60)
        print("✓ Training completed successfully!")
        print(f"✓ Best model saved: ./models/{model_name}_best/best_model.zip")
        print(f"✓ Final model saved: ./models/{model_name}_final.zip")
        
        if use_stochastic:
            print(f"✓ Trained with {delay_type} delays: {exp_name}")
            print(f"✓ Experiment config: {experiment_config}")
        
        return True
        
    except ImportError as e:
        print(f"Import failed: {e}")
        print("Make sure you have both rl_training.py and rl_training_stochastic.py in your package")
        return False
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment(model_path, experiment_config=1, use_stochastic=True):
    """Test the environment before training."""
    print("Testing environment...")
    
    try:
        if use_stochastic:
            from rl_training_stochastic import TeleoperationEnvStochastic
            env = TeleoperationEnvStochastic(
                model_path, 
                experiment_config=experiment_config,
                max_episode_steps=10
            )
        else:
            from rl_training import TeleoperationEnv
            env = TeleoperationEnv(model_path, max_episode_steps=10)
        
        obs, info = env.reset()
        print(f"✓ Environment created successfully")
        print(f"✓ Observation space: {env.observation_space.shape}")
        print(f"✓ Action space: {env.action_space.shape}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if use_stochastic:
                print(f"  Step {i}: Reward={reward:.3f}, Sync error={info['sync_error']:.4f}, "
                      f"Obs delay={info['current_obs_delay']}, Total delay={info['total_delay']}")
            else:
                print(f"  Step {i}: Reward={reward:.3f}, Sync error={info['sync_error']:.4f}")
            
            if terminated or truncated:
                obs, info = env.reset()
        
        env.close()
        print("✓ Environment test completed!")
        return True
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='RL Training for Adaptive PD Control with Stochastic Delays')
    parser.add_argument('--model-path', type=str, 
                       default="/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/rl_remote_ws/src/rl_remote_controller/models/franka_fr3/fr3.xml",
                       help='Path to MuJoCo robot model')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Training timesteps')
    parser.add_argument('--algorithm', type=str, default='SAC',
                       choices=['SAC', 'PPO'],
                       help='RL algorithm')
    parser.add_argument('--experiment', type=int, default=1, choices=[1, 2, 3],
                       help='Experiment config: 1(90-130ms), 2(170-210ms), 3(250-290ms)')
    parser.add_argument('--delay-type', type=str, default='stochastic', 
                       choices=['stochastic', 'constant'],
                       help='Use stochastic delays (paper-accurate) or constant delays')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test environment, do not train')
    
    args = parser.parse_args()
    
    use_stochastic = (args.delay_type == 'stochastic')
    
    print("RL Training Launcher")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Delay type: {args.delay_type.upper()}")
    if use_stochastic:
        print(f"Experiment: {args.experiment}")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Test environment first
    env_test_success = test_environment(args.model_path, args.experiment, use_stochastic)
    if not env_test_success:
        print("Environment test failed. Cannot proceed with training.")
        return 1
    
    if args.test_only:
        print("Environment test completed successfully. Exiting (--test-only flag).")
        return 0
    
    # Train
    success = train_rl_model(
        model_path=args.model_path, 
        timesteps=args.timesteps, 
        algorithm=args.algorithm,
        experiment_config=args.experiment,
        use_stochastic=use_stochastic
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())