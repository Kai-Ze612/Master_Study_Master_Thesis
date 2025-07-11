#!/usr/bin/env python3
"""
Training an RL agent with stochastic delays for teleoperation,
based on the adaptive PD control approach.
Algorithm: SAC
"""

import os
import argparse
import numpy as np
import mujoco
from collections import deque

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

class TeleoperationEnvStochastic(gym.Env):
    """
    A custom Gymnasium environment for simulating the local-remote
    teleoperation task with stochastic delays, as described in the paper.
    """
    def __init__(self,
                 model_path: str,
                 experiment_config: int = 1,
                 max_episode_steps: int = 500,
                 control_freq: int = 100):

        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.dt = 1.0 / control_freq
        self.current_step = 0
        self.n_joints = 7

        # Default PD control parameters
        self.default_kp = np.array([100.0] * 7)
        self.default_kd = np.array([10.0] * 7)
        self.current_kp = self.default_kp.copy()
        self.current_kd = self.default_kd.copy()
        self.force_limit = np.array([50.0] * 7)

        # Joint limits
        self.joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        # Setup delay parameters based on the paper's experiments
        self.experiment_config = experiment_config
        self._setup_delay_parameters()
        self.current_obs_delay = self.stochastic_obs_delay_min
        # State and action buffers for simulating delays
        self.action_buffer = deque(maxlen=self.max_total_delay)
        self.local_state_buffer = deque(maxlen=self.max_total_delay)
        self.error_history = deque(maxlen=10)

        # Define action space: RL agent outputs scaling factors for Kp and Kd
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.1]),   # Min scaling factors
            high=np.array([5.0, 3.0]),  # Max scaling factors
            dtype=np.float32
        )

        # Define observation space (augmented state as per the paper)
        obs_dim = 7 + 7 + 7 + 7 + (2 * self.stochastic_action_range) + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._initialize_buffers()

    def _setup_delay_parameters(self):
        """Setup delay parameters following the paper's experimental configurations."""
        if self.experiment_config == 1:
            self.constant_action_delay = 8      # 80ms
            self.stochastic_obs_delay_min = 1   # 10ms
            self.stochastic_obs_delay_max = 5   # 50ms
        elif self.experiment_config == 2:
            self.constant_action_delay = 16     # 160ms
            self.stochastic_obs_delay_min = 1
            self.stochastic_obs_delay_max = 5
        elif self.experiment_config == 3:
            self.constant_action_delay = 24     # 240ms
            self.stochastic_obs_delay_min = 1
            self.stochastic_obs_delay_max = 5
        else:
            raise ValueError(f"Unknown experiment config: {self.experiment_config}")

        # The number of actions to augment to the state is based on the stochastic range
        self.stochastic_action_range = self.stochastic_obs_delay_max - self.stochastic_obs_delay_min
        self.max_total_delay = self.constant_action_delay + self.stochastic_obs_delay_max + 5

    def _initialize_buffers(self):
        """Initialize all delay buffers with default values."""
        self.action_buffer.clear()
        self.local_state_buffer.clear()
        self.error_history.clear()

        for _ in range(self.max_total_delay):
            self.action_buffer.append(np.array([1.0, 1.0]))  # Default action (no scaling)
            self.local_state_buffer.append(np.zeros(7))
        for _ in range(10):
            self.error_history.append(0.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # Start robot in a random valid configuration
        self.data.qpos[:7] = np.random.uniform(
            self.joint_limits_lower * 0.5, self.joint_limits_upper * 0.5
        )
        self.local_positions = self.data.qpos[:7].copy()
        
        self._generate_local_trajectory()
        self._initialize_buffers()
        
        self.current_kp = self.default_kp.copy()
        self.current_kd = self.default_kd.copy()
        
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        # 1. Update local robot to simulate operator's movement
        self._update_local_robot()

        # 2. Apply constant action delay to the RL agent's action
        self.action_buffer.append(action.copy())
        delayed_action = self._get_delayed_action()

        # 3. Update PD gains based on the delayed action
        self._update_pd_gains(delayed_action)

        # 4. Determine the stochastic observation delay for this step
        self._generate_stochastic_observation_delay()
        
        # 5. Get the local robot's state from the past, using the observation delay
        delayed_local_pos = self._get_delayed_local_state()

        # 6. Compute PD control torque for the remote robot
        remote_pos = self.data.qpos[:7]
        remote_vel = self.data.qvel[:7]
        position_error = delayed_local_pos - remote_pos
        torques = self.current_kp * position_error - self.current_kd * remote_vel
        self.data.ctrl[:7] = np.clip(torques, -self.force_limit, self.force_limit)

        # 7. Step the simulation
        mujoco.mj_step(self.model, self.data)

        # 8. Calculate reward and check for termination
        sync_error = np.linalg.norm(position_error)
        self.error_history.append(sync_error)
        reward = -sync_error  # Reward is negative synchronization error

        self.current_step += 1
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _get_delayed_action(self):
        """Get an action from the past based on the constant action delay."""
        return self.action_buffer[-(self.constant_action_delay + 1)]

    def _generate_stochastic_observation_delay(self):
        """Generate a random observation delay for this step."""
        self.current_obs_delay = np.random.randint(
            self.stochastic_obs_delay_min, self.stochastic_obs_delay_max + 1
        )

    def _get_delayed_local_state(self):
        """Get the local robot's state from the past."""
        return self.local_state_buffer[-(self.current_obs_delay + 1)]

    def _generate_local_trajectory(self):
        """Generate a smooth, random trajectory for the local robot."""
        self.trajectory_time = 0.0
        self.trajectory_freq = np.random.uniform(0.1, 0.5, size=7)
        self.trajectory_amplitude = np.random.uniform(0.1, 0.4, size=7)
        self.trajectory_center = self.local_positions.copy()

    def _update_local_robot(self):
        """Simulate the operator's movement by following a generated trajectory."""
        self.trajectory_time += self.dt
        offsets = self.trajectory_amplitude * np.sin(2 * np.pi * self.trajectory_freq * self.trajectory_time)
        self.local_positions = self.trajectory_center + offsets
        self.local_positions = np.clip(self.local_positions, self.joint_limits_lower, self.joint_limits_upper)
        self.local_state_buffer.append(self.local_positions.copy())

    def _update_pd_gains(self, action):
        """Update Kp and Kd based on the RL agent's scaled output."""
        kp_scale, kd_scale = action[0], action[1]
        self.current_kp = self.default_kp * kp_scale
        self.current_kd = self.default_kd * kd_scale

    def _check_termination(self):
        """Terminate if the robot goes out of bounds or error is too high."""
        if np.any(self.data.qpos[:7] <= self.joint_limits_lower) or \
           np.any(self.data.qpos[:7] >= self.joint_limits_upper):
            return True
        if np.linalg.norm(self.local_positions - self.data.qpos[:7]) > 2.0:
            return True
        return False

    def _get_observation(self):
        """Construct the augmented state for the RL agent."""
        remote_pos = self.data.qpos[:7]
        remote_vel = self.data.qvel[:7]
        delayed_local_pos = self._get_delayed_local_state()
        position_error = delayed_local_pos - remote_pos

        # Get action history over the stochastic range
        action_hist_flat = np.array(list(self.action_buffer)[-self.stochastic_action_range:]).flatten()
        
        observation = np.concatenate([
            remote_pos,
            remote_vel,
            delayed_local_pos,
            position_error,
            action_hist_flat,
            list(self.error_history)
        ])
        return observation.astype(np.float32)

    def _get_info(self):
        return {"sync_error": np.linalg.norm(self.local_positions - self.data.qpos[:7])}

    def close(self):
        pass

def train_agent(model_path: str, experiment_config: int, total_timesteps: int, algorithm: str):
    """Main function to configure and run the RL training."""
    log_dir = "./logs"
    model_dir = "./models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    exp_name = {1: "90-130ms", 2: "170-210ms", 3: "250-290ms"}[experiment_config]
    model_name = f"{algorithm.lower()}_exp{experiment_config}_{exp_name}"

    print(f"--- Training RL Agent for {exp_name} ---")

    env = make_vec_env(
        lambda: Monitor(TeleoperationEnvStochastic(model_path, experiment_config)), 
        n_envs=4 # Use multiple environments for faster training
    )

    if algorithm == 'SAC':
        model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=100_000,
                    batch_size=256, tensorboard_log=log_dir, verbose=0)
    elif algorithm == 'PPO':
        model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=1024,
                    batch_size=64, n_epochs=10, tensorboard_log=log_dir, verbose=0)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    eval_env = Monitor(TeleoperationEnvStochastic(model_path, experiment_config))
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{model_dir}/{model_name}_best/",
                                 log_path=log_dir, eval_freq=10000, deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
    
    model.save(f"{model_dir}/{model_name}_final")
    print(f"\n--- Training Complete ---")
    print(f"Best model saved to: {model_dir}/{model_name}_best/best_model.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for teleoperation with stochastic delays.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the MuJoCo XML model file.")
    parser.add_argument("--experiment", type=int, default=1, choices=[1, 2, 3], help="Experiment config: 1 (90-130ms), 2 (170-210ms), 3 (250-290ms).")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total training timesteps.")
    parser.add_argument("--algo", type=str, default="SAC", choices=["SAC", "PPO"], help="RL algorithm to use.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
    else:
        train_agent(
            model_path=args.model_path,
            experiment_config=args.experiment,
            total_timesteps=args.timesteps,
            algorithm=args.algo
        )