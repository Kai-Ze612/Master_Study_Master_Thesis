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
import heapq
import logging
from typing import Tuple, Optional, List, Dict, Any
import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
import E2E_Teleoperation.config.robot_config as cfg

logger = logging.getLogger(__name__)

class FollowerRobotSimulator:
    def __init__(
        self,
        delay_config: ExperimentConfig = ExperimentConfig.HIGH_VARIANCE,
        seed: Optional[int] = None,
        render: bool = False,
        render_fps: int = 120,
        verbose: bool = True
    ):
        # Simulator Settings
        self._verbose = verbose
        self._render_enabled = render
        self._render_fps = render_fps
        
        # MuJoCo Model and Data
        self.model_path = str(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        
        # Two data structures: one for simulation, one for inverse dynamics calculations
        self.data = mujoco.MjData(self.model)
        self._data_id = mujoco.MjData(self.model) 

        self.control_freq = cfg.CONTROL_FREQ
        sim_timestep = self.model.opt.timestep
        self._n_substeps = int(1.0 / (sim_timestep * self.control_freq))

        self.torque_limits = cfg.TORQUE_LIMITS.copy()
        self.n_joints = cfg.N_JOINTS

        self.delay_simulator = DelaySimulator(self.control_freq, config=delay_config, seed=seed)
        self._action_queue: List[Tuple[int, np.ndarray]] = []
        self._internal_tick = 0
        self._last_executed_torque = np.zeros(self.n_joints)
        self._action_seq_id = 0
        
        total_grace_time = cfg.WARM_UP_DURATION + cfg.NO_DELAY_DURATION
        self._no_delay_steps = int(total_grace_time * self.control_freq)

        self._viewer = None
        self._render_interval = max(1, self.control_freq // self._render_fps)
        if self._render_enabled:
            self._init_viewer()
        self.action_delay_enabled = True
        
    def _init_viewer(self) -> None:
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.cam.azimuth = 135
        self._viewer.cam.elevation = -20
        self._viewer.cam.distance = 2.0
        self._viewer.cam.lookat[:] = [0.4, 0.0, 0.4]

    def _reset_mujoco_data(self, mj_data: mujoco.MjData, q_init: NDArray[np.float64]) -> None:
        mujoco.mj_resetData(self.model, mj_data)
        mj_data.qpos[:self.n_joints] = q_init
        mujoco.mj_forward(self.model, mj_data)

    def reset(self, initial_qpos: NDArray[np.float64]) -> None:
        self._action_queue = []
        heapq.heapify(self._action_queue)
        self._internal_tick = 0
        self._last_executed_torque = np.zeros(self.n_joints)
        self._action_seq_id = 0
        
        self._reset_mujoco_data(self.data, initial_qpos)
        self._reset_mujoco_data(self._data_id, initial_qpos)
        
        if self._render_enabled and self._viewer:
            self._viewer.sync()

    def compute_inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """
        Calculates the torque required to achieve state (q, qd, qdd).
        Used to generate 'Expert Actions' for Behavioral Cloning.
        """
        self._data_id.qpos[:self.n_joints] = q
        self._data_id.qvel[:self.n_joints] = qd
        self._data_id.qacc[:self.n_joints] = qdd
        
        # Calculate inverse dynamics
        mujoco.mj_inverse(self.model, self._data_id)
        
        # Return required torque (gravity + coriolis + inertial)
        return self._data_id.qfrc_inverse[:self.n_joints].copy()

    def step(self, torque_input: np.ndarray) -> Dict[str, Any]:
        """
        Applies the action (torque) to the robot after action delay.
        """
        
        self._internal_tick += 1

        # 1. get action delay steps
        if self._internal_tick < self._no_delay_steps or not self.action_delay_enabled:
            delay_steps = 0
        else:
            delay_steps = int(self.delay_simulator.get_action_delay_steps())

        arrival_time = self._internal_tick + delay_steps
        
        # 2. Push to Queue
        self._action_seq_id += 1
        heapq.heappush(self._action_queue, (arrival_time, self._action_seq_id, torque_input.copy()))  # sequence: (arrival_time, action_id, torque)

        # 3. Retrieve Action step by step (avoiding sudden jumps when turn from no delay period to delay period)
        if self._action_queue and self._action_queue[0][0] <= self._internal_tick:
            _, _, self._last_executed_torque = heapq.heappop(self._action_queue)
        
        # 4. Apply to Physics
        tau_clipped = np.clip(self._last_executed_torque, -self.torque_limits, self.torque_limits)
        self.data.ctrl[:self.n_joints] = tau_clipped

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        if self._render_enabled:
            self.render()

        # 5. Get the current joint positions after applying the torque
        q_current = self.data.qpos[:self.n_joints].copy()
        
        return {
            "tau_applied": self._last_executed_torque,
            "q_follower": q_current
        }

    def render(self) -> bool:
        """ Renders the current simulation state. """
        if not self._render_enabled or self._viewer is None: return True
        if not self._viewer.is_running(): return False
        if self._internal_tick % self._render_interval == 0: self._viewer.sync()
        return True

    def get_joint_state(self) -> Tuple[NDArray, NDArray]:
        """ Returns current joint positions and velocities after applied RL actions. """
        return (
            self.data.qpos[:self.n_joints].copy(),
            self.data.qvel[:self.n_joints].copy()
        )

    def close(self) -> None:
        """ Clean up the viewer."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None