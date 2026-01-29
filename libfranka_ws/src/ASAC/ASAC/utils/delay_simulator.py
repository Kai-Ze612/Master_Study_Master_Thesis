"""
Custom setting up delay patterns for teleoperation experiments.

There are three main delay configurations:
1. Low Delay: Moderate and consistent delays.
2. High Delay: Significant but consistent delays.
3. High Variance Delay: Delays that vary widely to simulate unpredictable network conditions.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Deque
import warnings
import numpy as np

class ExperimentConfig(IntEnum):
    """For mapping config to number"""
    NO_DELAY = 0
    LOW_DELAY = 1
    HIGH_DELAY = 2
    HIGH_VARIANCE = 3
    
@dataclass(frozen=True)
class DelayParameters:
    action_delay: int
    state_delay_min: int
    state_delay_max: int
    name: str
    
    def __post_init__(self) -> None:
        if self.action_delay < 0:
            raise ValueError("Action delay must be non-negative")
        if self.state_delay_min < 0 or self.state_delay_max < 0:
            raise ValueError("State delays must be non-negative")
        if self.state_delay_min > self.state_delay_max:
            raise ValueError("state_delay_min cannot be greater than state_delay_max")
        
class DelaySimulator:
    
    _DELAY_CONFIGS: dict[ExperimentConfig, DelayParameters] = {
        ExperimentConfig.NO_DELAY: DelayParameters(
            action_delay=0,
            state_delay_min=0,
            state_delay_max=0,
            name="No Delay"
        ),
        ExperimentConfig.LOW_DELAY: DelayParameters(
            action_delay=50,
            state_delay_min=120,
            state_delay_max=160,
            name="Low Delay"
        ),
        ExperimentConfig.HIGH_DELAY: DelayParameters(
            action_delay=50,
            state_delay_min=200,
            state_delay_max=240,
            name="High Delay"
        ),
        ExperimentConfig.HIGH_VARIANCE: DelayParameters(
            action_delay=50,
            state_delay_min=40,
            state_delay_max=240,
            name="High Variance Delay"
        ),
    }
    
    def __init__(self,
                 control_freq: int,
                 config: ExperimentConfig,
                 seed: Optional[int] = None) -> None:
        
        # Safety checks
        if config not in self._DELAY_CONFIGS:
            raise ValueError(f"Invalid config: {config}")
        
        self._control_freq = control_freq
        self._config = config
        self._rng = np.random.RandomState(seed)
        
        # setup delay parameters
        self._setup_delay_parameters()
        
    def _setup_delay_parameters(self) -> None:
        
        step_time_ms = 1000.0 / self._control_freq
        
        # Get configuration parameters
        params = self._DELAY_CONFIGS[self._config]
        self._config_name = params.name
        
        # Convert delays from milliseconds to discrete steps
        self._action_delay_steps = int(params.action_delay / step_time_ms)
        self._state_delay_min_steps = int(params.state_delay_min / step_time_ms)
        self._state_delay_max_steps = int(params.state_delay_max / step_time_ms)
        
        # We assume delay is Gaussian distributed
        self._state_delay_mean = (self._state_delay_min_steps + self._state_delay_max_steps) / 2
        self._state_delay_std = (self._state_delay_max_steps - self._state_delay_min_steps) / 4.0

    def reset(self):
        """
        Resets the internal state of the delay simulator.
        Required by TeleoperationEnv.
        """
        # If we had internal stateful noise (like random walk), we would reset it here.
        # Currently, the RNG persists to ensure variety across episodes.
        pass
        
    @property
    def control_freq(self) -> int:
        return self._control_freq
    
    @property
    def config(self) -> ExperimentConfig:
        return self._config

    @config.setter
    def config(self, value: ExperimentConfig) -> None:
        """Allow dynamic switching of config."""
        if value not in self._DELAY_CONFIGS:
            raise ValueError(f"Invalid config: {value}")
        self._config = value
        self._setup_delay_parameters()
    
    @property
    def config_name(self) -> str:
        return self._config_name

    def get_state_delay_steps(self, buffer_length: int) -> int:
        
        if buffer_length < 0:
            raise ValueError(f"buffer_length must be non-negative, got {buffer_length}")
        
        # If buffer too small, return maximum possible delay
        if buffer_length <= self._state_delay_min_steps:
            # Suppress warning for very short initial buffers to avoid log spam
            if buffer_length > 10: 
                warnings.warn(
                    f"Buffer length {buffer_length} insufficient for minimum "
                    f"state delay {self._state_delay_min_steps}. "
                    f"Returning reduced delay: {max(0, buffer_length - 1)}"
                )
            return max(0, buffer_length - 1)
        
        delay_sample = self._rng.normal(self._state_delay_mean, self._state_delay_std)
        delay_steps = int(round(delay_sample))
        
        max_possible_delay = min(self._state_delay_max_steps, buffer_length - 1)
        
        final_delay = np.clip(delay_steps, self._state_delay_min_steps, max_possible_delay)
        
        return int(final_delay)

    def get_delayed_state(self, leader_history: Deque[Tuple[np.ndarray, np.ndarray]], offset_indices: int = 0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Retrieves the leader state (q, qd) from 'delay' seconds ago.
        Required by TeleoperationEnv.
        """
        buffer_len = len(leader_history)
        if buffer_len == 0:
            return np.zeros(7), np.zeros(7), 0.0

        # 1. Calculate how many steps back to look
        delay_steps = self.get_state_delay_steps(buffer_len)
        
        # 2. Calculate current delay in seconds (for logging/network input)
        current_delay_sec = delay_steps * (1.0 / self._control_freq)
        
        # 3. Determine the target index in the deque
        # -1 is the most recent. 
        # offset_indices is used by the RNN builder to look even further back
        target_idx = -1 - delay_steps - offset_indices
        
        # 4. Clamp index to ensure we don't go out of bounds
        if abs(target_idx) > buffer_len:
            target_idx = -buffer_len
            
        try:
            q, qd = leader_history[target_idx]
        except IndexError:
            q, qd = leader_history[0]

        return q, qd, current_delay_sec
    
    def get_action_delay_steps(self) -> int:
        return self._action_delay_steps
    
    def __repr__(self) -> str:
        return (
            f"DelaySimulator(control_freq={self._control_freq}, "
            f"config={self._config.name}, "
            f"name='{self._config_name}')"
        )