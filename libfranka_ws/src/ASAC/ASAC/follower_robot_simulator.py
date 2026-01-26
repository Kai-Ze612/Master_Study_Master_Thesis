"""
Follower Robot Simulator
------------------------
MuJoCo-based follower robot with action delay simulation.

CRITICAL: Gravity compensation is applied INSTANTLY (not delayed).
Only the control signal experiences network delay.
"""

from __future__ import annotations
import logging
from typing import Tuple, Optional, Dict, Any
from collections import deque
import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from ASAC.utils.delay_simulator import DelaySimulator, ExperimentConfig
import ASAC.config.robot_config as cfg

logger = logging.getLogger(__name__)


class FollowerRobotSimulator:
    """
    MuJoCo-based follower robot with action delay simulation.
    
    CRITICAL: Gravity compensation is applied INSTANTLY (not delayed).
    Only the control signal (PD torque) experiences network delay.
    """
    
    def __init__(
        self,
        delay_config: ExperimentConfig = ExperimentConfig.HIGH_VARIANCE,
        seed: Optional[int] = 50,
        render: bool = False,
        render_fps: int = 60,
        verbose: bool = True
    ):
        self._verbose = verbose
        self._render_enabled = render
        self._render_fps = render_fps
        self._render_interval = max(1, int(cfg.CONTROL_FREQ / render_fps)) if render_fps > 0 else 1
        
        # Load MuJoCo model
        self.model_path = str(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        self.n_joints = cfg.N_JOINTS
        self.model.opt.timestep = cfg.DT
        self.torque_limits = cfg.TORQUE_LIMITS
        
        # Action delay simulation
        self._delay_sim = DelaySimulator(cfg.CONTROL_FREQ, config=delay_config, seed=seed)
        self._action_delay_steps = self._delay_sim.get_action_delay_steps()
        
        if self._verbose:
            print(f"[Follower] Action delay: {self._action_delay_steps} steps "
                  f"({self._action_delay_steps * cfg.DT * 1000:.1f} ms)")
        
        # Initialize action queue
        self._action_queue = deque()
        buffer_size = self._action_delay_steps
        for _ in range(buffer_size):
            self._action_queue.append(np.zeros(self.n_joints, dtype=np.float32))
        
        # Viewer for visualization
        self._viewer = None
        if self._render_enabled:
            try:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                if self._verbose:
                    print(f"[Follower] MuJoCo viewer launched (FPS: {render_fps})")
            except Exception as e:
                print(f"[Follower] Warning: Could not launch viewer: {e}")
                self._render_enabled = False
        
        self._internal_tick = 0
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[NDArray, Dict]:
        """Reset robot to initial configuration."""
        if seed is not None:
            np.random.seed(seed)
        
        # Set initial joint configuration
        self.data.qpos[:self.n_joints] = cfg.INITIAL_JOINT_CONFIG
        self.data.qvel[:self.n_joints] = 0.0
        
        # Initialize control with gravity compensation
        mujoco.mj_forward(self.model, self.data)
        gravity_comp = self.data.qfrc_bias[:self.n_joints].copy()
        self.data.ctrl[:self.n_joints] = gravity_comp
        
        # Reset internal state
        self._internal_tick = 0
        
        # Reset action queue
        self._action_queue.clear()
        buffer_size = self._action_delay_steps
        for _ in range(buffer_size):
            self._action_queue.append(np.zeros(self.n_joints, dtype=np.float32))
        
        # Sync viewer
        if self._render_enabled and self._viewer is not None:
            try:
                self._viewer.sync()
            except:
                pass
        
        return self.data.qpos[:self.n_joints].copy().astype(np.float32), {}

    def step(self, control_tau: NDArray) -> Dict[str, Any]:
        """
        Step simulation with delayed control but INSTANT gravity compensation.
        
        Args:
            control_tau: Control effort (PD torque), WITHOUT gravity compensation.
                        This signal will be delayed.
        
        Returns:
            Dictionary with robot state after step.
        """
        self._internal_tick += 1
        
        # 1. Sanitize input
        if not np.all(np.isfinite(control_tau)):
            if self._verbose:
                print(f"[Follower] Warning: NaN in control_tau, using zeros")
            control_tau = np.zeros(self.n_joints, dtype=np.float32)
        
        # 2. Clip control signal to torque limits
        control_tau = np.clip(control_tau, -self.torque_limits, self.torque_limits)
        
        # 3. Queue the control signal (DELAYED)
        self._action_queue.append(control_tau.copy())
        delayed_control = self._action_queue.popleft()
        
        # 4. Compute gravity compensation INSTANTLY
        mujoco.mj_forward(self.model, self.data)
        gravity_comp = self.data.qfrc_bias[:self.n_joints].copy()
        
        # 5. Combine: Delayed control + Instant gravity
        final_torque = delayed_control + gravity_comp
        
        # 6. Clip final torque (safety)
        final_torque = np.clip(final_torque, -self.torque_limits, self.torque_limits)
        
        # 7. Apply torque and step simulation
        self.data.ctrl[:self.n_joints] = final_torque
        mujoco.mj_step(self.model, self.data)
        
        # 8. Render if enabled
        if self._render_enabled:
            self._render()
        
        # 9. Return state
        q_curr = self.data.qpos[:self.n_joints].copy()
        qd_curr = self.data.qvel[:self.n_joints].copy()
        
        return {
            "tau_applied": final_torque.astype(np.float32),
            "tau_control": delayed_control.astype(np.float32),
            "tau_gravity": gravity_comp.astype(np.float32),
            "q_follower": q_curr.astype(np.float32),
            "qd_follower": qd_curr.astype(np.float32),
        }

    def _render(self) -> bool:
        """Render at specified FPS."""
        if not self._render_enabled or self._viewer is None:
            return True
        
        if self._internal_tick % self._render_interval == 0:
            try:
                self._viewer.sync()
            except:
                return False
        return True

    def render(self) -> bool:
        """Public render method."""
        return self._render()

    def close(self):
        """Clean up viewer."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except:
                pass
            self._viewer = None