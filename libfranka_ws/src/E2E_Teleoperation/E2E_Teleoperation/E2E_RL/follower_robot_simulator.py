"""
MuJoCo-based simulator for the follower robot.

Pipelines:
1. Subscribe to predicted local robot state (for error calculation)
2. Subscribe to true local robot state (for error calculation)
3. Subscribe to RL output tau (RL made decision (action))
4. Step the MuJoCo simulation.
5. Subscribe to remote robot state.
"""

from __future__ import annotations
import logging
from typing import Tuple, Optional, Dict, Any
from collections import deque
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
import E2E_Teleoperation.config.robot_config as cfg

logger = logging.getLogger(__name__)

class FollowerRobotSimulator(gym.Env):
    def __init__(
        self,
        delay_config: ExperimentConfig = ExperimentConfig.HIGH_VARIANCE,
        seed: Optional[int] = 50,
        render: bool = False,
        render_fps: int = 100,
        verbose: bool = True
    ):
        super().__init__()
        
        # Simulator Settings
        self._verbose = verbose
        self._render_enabled = render
        self._render_fps = render_fps
        self._render_interval = int(cfg.CONTROL_FREQ / self._render_fps) if self._render_fps > 0 else 1
        
        # MuJoCo Model and Data
        self.model_path = str(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # Simulation Parameters
        self.n_joints = cfg.N_JOINTS
        self._dt = cfg.DT
        self.model.opt.timestep = self._dt
        
        # Joint Limits
        self.joint_limits_lower = cfg.JOINT_LIMITS_LOWER
        self.joint_limits_upper = cfg.JOINT_LIMITS_UPPER
        self.torque_limits = cfg.TORQUE_LIMITS
        self._delay_config_data = delay_config

        # Action delay
        self._delay_sim = DelaySimulator(cfg.CONTROL_FREQ, config=delay_config, seed=seed)
        self._action_delay_steps = self._delay_sim.get_action_delay_steps()
        
        # Action queue
        self._action_queue = deque()
        
        # Pre-fill with 0
        buffer_size = max(1, self._action_delay_steps)
        for _ in range(buffer_size):
            self._action_queue.append(np.zeros(self.n_joints, dtype=np.float32))
        
        # Viewer
        self._viewer = None
        if self._render_enabled:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
        # Internal State
        self._internal_tick = 0
        
        # Reset to initial state
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[NDArray, Dict]:
        """
        Resets the follower robot to the initial configuration.
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self._delay_sim = DelaySimulator(cfg.CONTROL_FREQ, config=self._delay_config_data, seed=seed)
            self._action_delay_steps = self._delay_sim.get_action_delay_steps()
            
        # Reset Physics State
        self.data.qpos[:self.n_joints] = cfg.INITIAL_JOINT_CONFIG
        self.data.qvel[:self.n_joints] = 0.0
        self.data.qacc[:self.n_joints] = 0.0
        self.data.ctrl[:self.n_joints] = 0.0
        
        # Step once to propagate
        try:
            mujoco.mj_forward(self.model, self.data)
        except Exception as e:
            logger.warning(f"[FollowerSim] Error during reset forward: {e}")
        
        # Internal step counter
        self._internal_tick = 0
        
        self._action_queue.clear()
        
        # Action buffer
        # Max length = action delay steps
        # 1 is to make sure the action delay is at least 1 step
        buffer_size = max(1, self._action_delay_steps)
        
        for _ in range(buffer_size):
            self._action_queue.append(np.zeros(self.n_joints, dtype=np.float32))
        
        if self._render_enabled and self._viewer is not None:
            self._viewer.sync()
            
        return self.data.qpos[:self.n_joints].copy().astype(np.float32), {}

    def step(self, action_tau: NDArray) -> Dict[str, Any]:
        """
        Pipeline:
        1. Safety Check on action
        2. add torque command to queue
        3. apply oldest torque command from queue
        4. Mujoco Control
        5. Get Step feedback
        """
        self._internal_tick += 1
        
        # Check NaN
        if not np.all(np.isfinite(action_tau)):
            logger.warning(f"NaN detected in action at tick {self._internal_tick}. Replacing with zeros.")
            action_tau = np.zeros(self.n_joints, dtype=np.float32)  
        action_tau = np.clip(action_tau, -self.torque_limits, self.torque_limits)
       
        # Add torque command in the end
        self._action_queue.append(action_tau)
       
        # Apply oldest torque
        tau_to_apply = action_tau
        
        # Apply tau Control
        self.data.ctrl[:self.n_joints] = tau_to_apply
        
        # Step Physics
        try:
            mujoco.mj_step(self.model, self.data)
        except Exception as e:
            pass
        
        # Render
        if self._render_enabled:
            self.render()

        # Get State for printout information
        q_current, qd_current  = self.get_joint_state()
       
        # Safety Check
        if not np.all(np.isfinite(q_current)) or not np.all(np.isfinite(qd_current)):
            q_current = cfg.INITIAL_JOINT_CONFIG.copy()
            qd_current = np.zeros(self.n_joints)
        
        # Note: Keeps Dict return type to maintain compatibility with training_env.py
        return {
            "tau_applied": tau_to_apply.astype(np.float32),
            "q_follower": q_current.astype(np.float32),
            "qd_follower": qd_current.astype(np.float32)
        }

    def render(self) -> bool:
        """
        Mujoco rendering step.
        """
        if not self._render_enabled or self._viewer is None: return True
        if not self._viewer.is_running(): return False
        if self._internal_tick % self._render_interval == 0: 
            self._viewer.sync()
        return True

    def get_joint_state(self) -> Tuple[NDArray, NDArray]:
        """
        Returns the remote robot joint state (q, qd) after torque applied
        """
        return (
            self.data.qpos[:self.n_joints].copy().astype(np.float32),
            self.data.qvel[:self.n_joints].copy().astype(np.float32)
        )

    def close(self):
        if self._viewer is not None:
            self._viewer.close()