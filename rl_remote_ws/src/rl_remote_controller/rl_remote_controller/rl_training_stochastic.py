"""
Training RL agent with stochastic delays for teleoperation.
Algorithm: SAC

"""

import numpy as np
import os


import mujoco
from collections import deque

# Import gym with 
import gymnasium as gym
from gymnasium import spaces

# SAC is better for continuous control tasks
from stable_baselines3 import SAC

# Implement training environment 
from stable_baselines3.common.env_util import make_vec_env
# Implement evaluation and checkpoint callbacks
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

class TeleoperationEnvStochastic(gym.Env):
    def __init__(self, 
                 model_path: str,
                 experiment_config: int = 1,  # 1: 90-130ms, 2: 170-210ms, 3: 250-290ms
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
       
        # Paper-accurate delay configuration
        self.experiment_config = experiment_config
        self._setup_delay_parameters()
        
        # State and action buffers for delay simulation
        self.action_buffer = deque(maxlen=self.max_total_delay)
        self.local_state_buffer = deque(maxlen=self.max_total_delay)
        
        # Local robot simulation (generates reference trajectory)
        self.local_positions = np.zeros(7)
        self.local_velocities = np.zeros(7)
        
        # Remote robot state
        self.remote_positions = np.zeros(7)
        self.remote_velocities = np.zeros(7)
        
        # History for observation and delay tracking
        self.error_history = deque(maxlen=10)
        self.delay_history = deque(maxlen=20)  # Track delay patterns for your thesis
        
        # Current delays (updated each step)
        self.current_action_delay = self.constant_action_delay
        self.current_obs_delay = self.stochastic_obs_delay_min
        
        # Define action space: [Kp_scale, Kd_scale] (paper's approach)
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.1]),  # Minimum scaling factors
            high=np.array([5.0, 3.0]), # Maximum scaling factors
            dtype=np.float32
        )
        
        # Define observation space (augmented state from paper)
        # [remote_pos(7), remote_vel(7), local_pos_delayed(7), error(7), 
        #  stochastic_action_history(2*stochastic_range), error_hist(10)]
        obs_dim = 7 + 7 + 7 + 7 + 2*self.stochastic_action_range + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize buffers
        self._initialize_buffers()
    
    def _setup_delay_parameters(self):
        """Setup delay parameters following paper's experimental configurations."""
        
        if self.experiment_config == 1:
            # Experiment 1: 90-130ms total delay
            self.constant_action_delay = 8      # 80ms constant action delay
            self.stochastic_obs_delay_min = 1   # 10ms minimum observation delay  
            self.stochastic_obs_delay_max = 5   # 50ms maximum observation delay
            print(f"Experiment 1: 90-130ms delay (80ms action + 10-50ms observation)")
            
        elif self.experiment_config == 2:
            # Experiment 2: 170-210ms total delay
            self.constant_action_delay = 16     # 160ms constant action delay
            self.stochastic_obs_delay_min = 1   # 10ms minimum observation delay
            self.stochastic_obs_delay_max = 5   # 50ms maximum observation delay
            print(f"Experiment 2: 170-210ms delay (160ms action + 10-50ms observation)")
            
        elif self.experiment_config == 3:
            # Experiment 3: 250-290ms total delay
            self.constant_action_delay = 24     # 240ms constant action delay
            self.stochastic_obs_delay_min = 1   # 10ms minimum observation delay
            self.stochastic_obs_delay_max = 5   # 50ms maximum observation delay
            print(f"Experiment 3: 250-290ms delay (240ms action + 10-50ms observation)")
            
        else:
            raise ValueError(f"Unknown experiment config: {self.experiment_config}")
        
        # Calculate derived parameters
        self.stochastic_action_range = self.stochastic_obs_delay_max - self.stochastic_obs_delay_min + 1
        self.max_total_delay = self.constant_action_delay + self.stochastic_obs_delay_max + 5  # Buffer
        
        print(f"Action delay: {self.constant_action_delay} steps (constant)")
        print(f"Observation delay: {self.stochastic_obs_delay_min}-{self.stochastic_obs_delay_max} steps (stochastic)")
        print(f"Stochastic range: {self.stochastic_action_range} steps")
    
    def _initialize_buffers(self):
        """Initialize all delay buffers."""
        # Clear existing buffers
        self.action_buffer.clear()
        self.local_state_buffer.clear()
        self.error_history.clear()
        self.delay_history.clear()
        
        # Initialize action buffer with default actions
        for _ in range(self.max_total_delay):
            self.action_buffer.append(np.array([1.0, 1.0]))  # Default scaling
        
        # Initialize local state buffer
        for _ in range(self.max_total_delay):
            self.local_state_buffer.append(np.zeros(7))
        
        # Initialize error history
        for _ in range(10):
            self.error_history.append(0.0)
        
        # Initialize delay history
        for _ in range(20):
            self.delay_history.append(self.stochastic_obs_delay_min)
    
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
        
        # Re-initialize buffers
        self._initialize_buffers()
        
        # Reset control gains
        self.current_kp = self.default_kp.copy()
        self.current_kd = self.default_kd.copy()
        
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step with paper-accurate stochastic delays."""
        
        # Update local robot (simulates operator movement)
        self._update_local_robot()
        
        # Apply constant action delay (paper's approach)
        self.action_buffer.append(action.copy())
        delayed_action = self._get_delayed_action()
        
        # Update PD gains based on delayed RL action
        self._update_pd_gains(delayed_action)
        
        # Generate stochastic observation delay (paper's key challenge)
        self._generate_stochastic_observation_delay()
        
        # Get delayed local robot state with stochastic delay
        delayed_local_pos = self._get_delayed_local_state()
        
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
        self.delay_history.append(self.current_obs_delay)  # Track delay pattern
        
        # Compute reward (paper's approach: negative synchronization error)
        reward = self._compute_reward(position_error, torques)
        
        # Update step counter
        self.current_step += 1
        
        # Check termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_delayed_action(self):
        """Get action with constant delay (paper's action delay)."""
        if len(self.action_buffer) >= self.constant_action_delay:
            # Get action from exactly constant_action_delay steps ago
            delayed_index = -(self.constant_action_delay + 1)
            return list(self.action_buffer)[delayed_index]
        else:
            # Not enough history, use default action
            return np.array([1.0, 1.0])
    
    def _generate_stochastic_observation_delay(self):
        """Generate stochastic observation delay (paper's key challenge)."""
        # Random observation delay within stochastic range
        self.current_obs_delay = np.random.randint(
            self.stochastic_obs_delay_min, 
            self.stochastic_obs_delay_max + 1
        )
    
    def _get_delayed_local_state(self):
        """Get local robot state with stochastic observation delay."""
        if len(self.local_state_buffer) >= self.current_obs_delay:
            # Get state from current_obs_delay steps ago
            delayed_index = -(self.current_obs_delay + 1)
            return list(self.local_state_buffer)[delayed_index]
        else:
            # Not enough history, use current state
            return self.local_positions
    
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
        
        # Generate smooth sinusoidal motion
        for i in range(7):
            offset = self.trajectory_amplitude * np.sin(2 * np.pi * self.trajectory_freq * self.trajectory_time + i)
            self.local_positions[i] = self.trajectory_center[i] + offset * 0.3
            
            # Keep within joint limits
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
    
    def _compute_reward(self, position_error, torques):
        """Paper-accurate reward function."""
        # Paper's simple approach: negative Euclidean distance
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
        """Get observation with paper-accurate augmented state."""
        
        # Current remote robot state
        remote_pos = self.data.qpos[:7]
        remote_vel = self.data.qvel[:7]
        
        # Delayed local robot state (with current stochastic delay)
        delayed_local_pos = self._get_delayed_local_state()
        
        # Position error
        position_error = delayed_local_pos - remote_pos
        
        # Action history over STOCHASTIC RANGE ONLY (paper's approach)
        stochastic_action_history = []
        for i in range(self.stochastic_action_range):
            if i < len(self.action_buffer):
                action_index = -(i + 1)
                stochastic_action_history.extend(list(self.action_buffer)[action_index])
            else:
                stochastic_action_history.extend([1.0, 1.0])  # Default values
        
        # Error history
        error_hist = list(self.error_history)
        
        # Combine all observations (paper's augmented state)
        observation = np.concatenate([
            remote_pos,                    # Remote robot positions (7)
            remote_vel,                    # Remote robot velocities (7)
            delayed_local_pos,             # Local robot positions (delayed) (7)
            position_error,                # Synchronization error (7)
            stochastic_action_history,     # Action history over stochastic range (2 * stochastic_range)
            error_hist                     # Error history (10)
        ])
        
        return observation.astype(np.float32)
    
    def _get_info(self):
        """Get additional information including delay statistics."""
        position_error = self.local_positions - self.data.qpos[:7]
        sync_error = np.linalg.norm(position_error)
        
        # Calculate delay statistics for your thesis analysis
        recent_delays = list(self.delay_history)[-10:]
        delay_variance = np.var(recent_delays) if len(recent_delays) > 1 else 0.0
        delay_mean = np.mean(recent_delays) if len(recent_delays) > 0 else 0.0
        
        return {
            'sync_error': sync_error,
            'constant_action_delay': self.constant_action_delay,
            'current_obs_delay': self.current_obs_delay,
            'stochastic_obs_delay_range': (self.stochastic_obs_delay_min, self.stochastic_obs_delay_max),
            'delay_variance': delay_variance,      # For your neural network classifier
            'delay_mean': delay_mean,              # For your neural network classifier
            'total_delay': self.constant_action_delay + self.current_obs_delay,
            'kp_scale': self.current_kp[0] / self.default_kp[0],
            'kd_scale': self.current_kd[0] / self.default_kd[0],
            'step': self.current_step,
            'experiment_config': self.experiment_config
        }
    
    def render(self):
        """Render the environment (optional)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass


def train_paper_accurate_agent(model_path: str, 
                              experiment_config: int = 1,
                              total_timesteps: int = 200000,
                              algorithm: str = 'SAC'):
    """
    Train RL agent with paper-accurate stochastic delays.
    
    Args:
        model_path: Path to MuJoCo robot model
        experiment_config: 1 (90-130ms), 2 (170-210ms), 3 (250-290ms)
        total_timesteps: Total training timesteps
        algorithm: RL algorithm ('SAC' recommended as in paper)
    """
    
    # Create output directories
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    experiment_names = {1: "90-130ms", 2: "170-210ms", 3: "250-290ms"}
    exp_name = experiment_names[experiment_config]
    
    print(f"Training Paper-Accurate RL Agent with Stochastic Delays...")
    print(f"Experiment: {exp_name}")
    print(f"Model path: {model_path}")
    print(f"Algorithm: {algorithm}")
    print(f"Total timesteps: {total_timesteps}")
    
    # Create training environment with stochastic delays
    def make_env():
        env = TeleoperationEnvStochastic(
            model_path=model_path, 
            experiment_config=experiment_config
        )
        env = Monitor(env, './logs/')
        return env
    
    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=1)
    
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
        model_name = f'sac_stochastic_delay_exp{experiment_config}'
    
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
        model_name = f'ppo_stochastic_delay_exp{experiment_config}'
    
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
    print("\nTesting trained model with stochastic delays...")
    test_model = SAC.load(f'./models/{model_name}_best/best_model.zip')
    
    env = TeleoperationEnvStochastic(model_path=model_path, experiment_config=experiment_config)
    obs, info = env.reset()
    
    total_reward = 0
    sync_errors = []
    delay_variances = []
    total_delays = []
    
    for step in range(100):
        action, _ = test_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        sync_errors.append(info['sync_error'])
        delay_variances.append(info['delay_variance'])
        total_delays.append(info['total_delay'])
        
        if step % 20 == 0:
            print(f"Test step {step}: Sync error = {info['sync_error']:.4f}, "
                  f"Total delay = {info['total_delay']:.0f}ms, "
                  f"Obs delay = {info['current_obs_delay']:.0f}ms, "
                  f"Kp_scale = {info['kp_scale']:.3f}, Kd_scale = {info['kd_scale']:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    
    avg_sync_error = np.mean(sync_errors)
    avg_delay_variance = np.mean(delay_variances)
    avg_total_delay = np.mean(total_delays)
    
    print(f"\nTest results for {exp_name} experiment:")
    print(f"Average synchronization error: {avg_sync_error:.4f}")
    print(f"Average delay variance: {avg_delay_variance:.4f}")
    print(f"Average total delay: {avg_total_delay:.1f} timesteps ({avg_total_delay*10:.1f}ms)")
    print(f"Total reward: {total_reward:.2f}")
    
    return model


def main(args=None):
    """Main training function with stochastic delays."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper-Accurate RL Training with Stochastic Delays')
    parser.add_argument('--model-path', type=str, 
                       default="/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller/models/franka_fr3/fr3.xml",
                       help='Path to MuJoCo robot model')
    parser.add_argument('--experiment', type=int, default=1, choices=[1, 2, 3],
                       help='Experiment config: 1(90-130ms), 2(170-210ms), 3(250-290ms)')
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
    print("Testing stochastic delay environment...")
    env = TeleoperationEnvStochastic(
        parsed_args.model_path, 
        experiment_config=parsed_args.experiment,
        max_episode_steps=50
    )
    obs, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation shape: {obs.shape}")
    
    # Run a few test steps to show stochastic delays
    print("\nTesting stochastic delay variation:")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Reward={reward:.3f}, Sync error={info['sync_error']:.4f}, "
              f"Obs delay={info['current_obs_delay']}, Total delay={info['total_delay']}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Environment test completed!\n")
    
    
    train_paper_accurate_agent(
        model_path=parsed_args.model_path,
        experiment_config=parsed_args.experiment,
        total_timesteps=parsed_args.timesteps,
        algorithm=parsed_args.algorithm
    )

if __name__ == "__main__":
    main()