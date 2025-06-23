"""
Trains RL agent for adaptive PD control in teleoperation with delays.
The remote robot follows the local robot.
"""

import numpy as np
import os

import mujoco
from collections import deque

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

class TeleoperationEnv(gym.Env):
    def __init__(self, 
                 model_path: str,
                 max_episode_steps: int = 500,
                 control_freq: int = 100):
        
        super().__init__()
        
        # Load MuJoCo model
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.current_step = 0
        
        # Joint configuration
        self.n_joints = 7
        
        # Control parameters (from paper)
        self.default_kp = np.array([100.0] * 7)
        self.default_kd = np.array([10.0] * 7)
        self.current_kp = self.default_kp.copy()
        self.current_kd = self.default_kd.copy()
        
        # Force limits
        self.force_limit = np.array([50.0] * 7)
        
        # Joint limits
        self.joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Delay parameters (following paper: up to 290ms)
        self.action_delay = 3   # 30ms action delay (3 timesteps at 100Hz)
        self.observation_delay = 2  # 20ms observation delay (2 timesteps at 100Hz)
        self.max_delay = 10 
       
        # State and action buffers for delay simulation
        self.action_buffer = deque(maxlen=self.max_delay)
        self.local_state_buffer = deque(maxlen=self.max_delay)
        
        # Local robot simulation (generates reference trajectory)
        self.local_positions = np.zeros(7)
        self.local_velocities = np.zeros(7)
        
        # Remote robot state
        self.remote_positions = np.zeros(7)
        self.remote_velocities = np.zeros(7)
        
        # History for observation
        self.error_history = deque(maxlen=10)
        
        # Define action space: [Kp_scale, Kd_scale] (simplified from paper)
        # Paper uses individual joint gains, we use global scaling for simplicity
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.1]),  # Minimum scaling factors
            high=np.array([5.0, 3.0]), # Maximum scaling factors
            dtype=np.float32
        )
        
        # Define observation space:
        # [remote_pos(7), remote_vel(7), local_pos_delayed(7), error(7), action_history(2*max_delay), error_hist(10)]
        obs_dim = 7 + 7 + 7 + 7 + 2*self.max_delay + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize buffers
        for _ in range(self.max_delay):
            self.action_buffer.append(np.array([1.0, 1.0]))  # Default scaling
            self.local_state_buffer.append(np.zeros(7))
        
        for _ in range(10):
            self.error_history.append(0.0)
    
    # This is to reset the environment and initialize the robot state
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset step counter
        self.current_step = 0
        
        # Initialize robot in random but valid configuration
        self.data.qpos[:7] = np.random.uniform(
            self.joint_limits_lower * 0.5,
            self.joint_limits_upper * 0.5
        )
        
        # Initialize local robot (operator) at same position
        self.local_positions = self.data.qpos[:7].copy()
        self.local_velocities = np.zeros(7)
        
        # Generate target trajectory for local robot (operator commands)
        self._generate_local_trajectory()
        
        # Clear buffers
        self.action_buffer.clear()
        self.local_state_buffer.clear()
        self.error_history.clear()
        
        # Initialize buffers
        for _ in range(self.max_delay):
            self.action_buffer.append(np.array([1.0, 1.0]))
            self.local_state_buffer.append(self.local_positions.copy())
        
        for _ in range(10):
            self.error_history.append(0.0)
        
        # Reset control gains
        self.current_kp = self.default_kp.copy()
        self.current_kd = self.default_kd.copy()
        
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment."""
        
        # Update local robot (simulates operator movement)
        self._update_local_robot()
        
        # Apply action delay
        self.action_buffer.append(action.copy())
        delayed_action = list(self.action_buffer)[0]  # Oldest action
        
        # Update PD gains based on (delayed) RL action
        self._update_pd_gains(delayed_action)
        
        # Get delayed local robot state (observation delay)
        delayed_local_pos = list(self.local_state_buffer)[0]  # Oldest state
        
        # Compute PD control for remote robot
        remote_pos = self.data.qpos[:7]
        remote_vel = self.data.qvel[:7]
        
        # Main control objective: minimize synchronization error
        position_error = delayed_local_pos - remote_pos
        
        # Apply adaptive PD control
        torques = self.current_kp * position_error - self.current_kd * remote_vel
        torques = np.clip(torques, -self.force_limit, self.force_limit)
        
        # Apply control and step simulation
        self.data.ctrl[:7] = torques
        mujoco.mj_step(self.model, self.data)
        
        # Update history
        sync_error = np.linalg.norm(position_error)
        self.error_history.append(sync_error)
        
        # Compute reward (main objective: synchronization)
        reward = self._compute_reward(position_error, torques)
        
        # Update step counter
        self.current_step += 1
        
        # Check termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    # Generate a pure trajectory for remote robot to follow
    def _generate_local_trajectory(self):
        """Generate smooth trajectory for local robot (operator)."""
        # Simple sinusoidal trajectory for episode
        self.trajectory_time = 0.0
        self.trajectory_freq = np.random.uniform(0.1, 0.5)  
        self.trajectory_amplitude = np.random.uniform(0.2, 0.6)  
        self.trajectory_center = self.local_positions.copy()
    
    def _update_local_robot(self):
        """Update local robot state (simulates operator movement)."""
        
        self.trajectory_time += self.dt
        
        
        for i in range(7):
            offset = self.trajectory_amplitude * np.sin(2 * np.pi * self.trajectory_freq * self.trajectory_time + i)
            self.local_positions[i] = self.trajectory_center[i] + offset * 0.3  # Scale to safe range
            
            
            self.local_positions[i] = np.clip(
                self.local_positions[i],
                self.joint_limits_lower[i],
                self.joint_limits_upper[i]
            )
        
        # Update local state buffer (for observation delay)
        self.local_state_buffer.append(self.local_positions.copy())
    
    def _update_pd_gains(self, action):
        """Update PD gains based on RL action."""
        kp_scale = np.clip(action[0], 0.1, 5.0)
        kd_scale = np.clip(action[1], 0.1, 3.0)
        
        # Apply uniform scaling (simplified from paper)
        self.current_kp = self.default_kp * kp_scale
        self.current_kd = self.default_kd * kd_scale

    # The closer of the remote robot to the trajectory, the higher the reward
    def _compute_reward(self, position_error, torques):
        """Improved reward function with better shaping."""
        
        sync_error = np.linalg.norm(position_error)
        reward = -sync_error
        
        return reward
    
    def _check_termination(self):
        """Check if episode should terminate early."""
        # Check joint limits
        current_pos = self.data.qpos[:7]
        for i, (pos, lower, upper) in enumerate(zip(current_pos, self.joint_limits_lower, self.joint_limits_upper)):
            if pos <= lower or pos >= upper:
                return True
        
        # Check if synchronization error is too large
        position_error = self.local_positions - current_pos
        if np.linalg.norm(position_error) > 3.0:  # 3 radians is too much
            return True
        
        return False
    
    def _get_observation(self):
        """Get observation following paper's augmented state approach."""
        
        # Current remote robot state
        remote_pos = self.data.qpos[:7]
        remote_vel = self.data.qvel[:7]
        
        # Delayed local robot state (observation delay)
        if len(self.local_state_buffer) >= self.observation_delay:
            delayed_local_pos = list(self.local_state_buffer)[-self.observation_delay]
        else:
            delayed_local_pos = self.local_positions
        
        # Position error
        position_error = delayed_local_pos - remote_pos
        
        # Action history (augmented state from paper)
        action_history = []
        for action in list(self.action_buffer):
            action_history.extend(action)
        
        # Error history
        error_hist = list(self.error_history)
        
        # Combine all observations
        observation = np.concatenate([
            remote_pos,           # Remote robot positions (7)
            remote_vel,           # Remote robot velocities (7)
            delayed_local_pos,    # Local robot positions (delayed) (7)
            position_error,       # Synchronization error (7)
            action_history,       # Action buffer (2 * max_delay)
            error_hist            # Error history (10)
        ])
        
        return observation.astype(np.float32)
    
    def _get_info(self):
        """Get additional information."""
        position_error = self.local_positions - self.data.qpos[:7]
        sync_error = np.linalg.norm(position_error)
        
        return {
            'sync_error': sync_error,
            'action_delay': self.action_delay,
            'observation_delay': self.observation_delay,
            'kp_scale': self.current_kp[0] / self.default_kp[0],
            'kd_scale': self.current_kd[0] / self.default_kd[0],
            'step': self.current_step
        }
    
    def render(self):
        """Render the environment (optional)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass


def train_simple_rl_agent(model_path: str, 
                         total_timesteps: int = 200000,
                         algorithm: str = 'SAC'):
    """
    Train the RL agent for simple adaptive PD control following the paper.
    
    Args:
        model_path: Path to MuJoCo robot model
        total_timesteps: Total training timesteps
        algorithm: RL algorithm ('SAC' recommended as in paper)
    """
    
    # Create output directories
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    print(f"Training Simple RL Agent following McCutcheon & Fallah paper...")
    print(f"Model path: {model_path}")
    print(f"Algorithm: {algorithm}")
    print(f"Total timesteps: {total_timesteps}")
    
    # Create training environment
    def make_env():
        env = TeleoperationEnv(model_path=model_path)
        env = Monitor(env, './logs/')
        return env
    
    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=1)  # Single environment for simplicity
    
    # Create evaluation environment
    eval_env = make_env()
    
    # Configure RL algorithm (SAC recommended from paper)
    if algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            buffer_size=50000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            tensorboard_log='./logs/',
            verbose=1
        )
        model_name = 'sac_simple_adaptive_pd'
    
    elif algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log='./logs/',
            verbose=1
        )
        model_name = 'ppo_simple_adaptive_pd'
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/{model_name}_best/',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f'./models/{model_name}_checkpoints/',
        name_prefix=model_name
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f'./models/{model_name}_final')
    
    print(f"\nTraining completed!")
    print(f"Best model saved at: ./models/{model_name}_best/best_model.zip")
    print(f"Final model saved at: ./models/{model_name}_final.zip")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_model = SAC.load(f'./models/{model_name}_best/best_model.zip')
    
    env = TeleoperationEnv(model_path=model_path)
    obs, info = env.reset()
    
    total_reward = 0
    sync_errors = []
    
    for step in range(100):
        action, _ = test_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        sync_errors.append(info['sync_error'])
        
        if step % 20 == 0:
            print(f"Test step {step}: Sync error = {info['sync_error']:.4f}, "
                  f"Kp_scale = {info['kp_scale']:.3f}, Kd_scale = {info['kd_scale']:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    
    avg_sync_error = np.mean(sync_errors)
    print(f"\nTest results:")
    print(f"Average synchronization error: {avg_sync_error:.4f}")
    print(f"Total reward: {total_reward:.2f}")
    
    return model


def main(args=None):
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple RL Training for Adaptive PD Control')
    parser.add_argument('--model-path', type=str, 
                       default="/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller/models/franka_fr3/fr3.xml",
                       help='Path to MuJoCo robot model')
    parser.add_argument('--timesteps', type=int, default=200000,
                       help='Total training timesteps')
    parser.add_argument('--algorithm', type=str, default='SAC', choices=['SAC', 'PPO'],
                       help='RL algorithm to use')
    
    parsed_args = parser.parse_args(args)
    
    # Check if model file exists
    if not os.path.exists(parsed_args.model_path):
        print(f"Error: MuJoCo model file not found: {parsed_args.model_path}")
        print("Please update the --model-path argument with the correct path.")
        return
    
    # Test environment first
    print("Testing environment...")
    env = TeleoperationEnv(parsed_args.model_path, max_episode_steps=50)
    obs, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation shape: {obs.shape}")
    
    # Run a few test steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Reward={reward:.3f}, Sync error={info['sync_error']:.4f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Environment test completed!\n")
    
    # Start training
    train_simple_rl_agent(
        model_path=parsed_args.model_path,
        total_timesteps=parsed_args.timesteps,
        algorithm=parsed_args.algorithm
    )

if __name__ == "__main__":
    main()