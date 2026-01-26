"""
Leader Robot Simulator (Fixed for Figure-8 Corners)
----------------------------------------------------
Key Fixes:
1. Smoother figure-8 without sharp cusps
2. Better IK failure recovery (interpolate instead of freeze)
3. Velocity limiting to prevent large jumps
4. Debug logging for IK failures
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np
import gymnasium as gym
import mujoco

from ASAC.utils.inverse_kinematics import IKSolver
import ASAC.config.robot_config as cfg


class TrajectoryType(Enum):
    FIGURE_8 = "figure_8"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    LISSAJOUS = "lissajous"


@dataclass
class TrajectoryParams:
    initial_phase: float = 0.0
    center: np.ndarray = None
    scale: np.ndarray = None
    frequency: float = None
    
    def __post_init__(self):
        if self.center is None:
            self.center = cfg.ROBOT.TRAJECTORY_CENTER.copy()
        if self.scale is None:
            self.scale = cfg.ROBOT.TRAJECTORY_SCALE.copy()
        if self.frequency is None:
            self.frequency = cfg.ROBOT.TRAJECTORY_FREQUENCY
    
    @classmethod
    def randomized(cls, actual_start_pos: np.ndarray) -> 'TrajectoryParams':
        center_x = np.random.uniform(*cfg.TRAJ_RANDOM.CENTER_X)
        center_y = np.random.uniform(*cfg.TRAJ_RANDOM.CENTER_Y)
        center_z = cfg.ROBOT.TRAJECTORY_CENTER[2]
        center = np.array([center_x, center_y, center_z], dtype=np.float64)
        
        scale_x = np.random.uniform(*cfg.TRAJ_RANDOM.SCALE_X)
        scale_y = np.random.uniform(*cfg.TRAJ_RANDOM.SCALE_Y)
        scale_z = cfg.ROBOT.TRAJECTORY_SCALE[2]
        scale = np.array([scale_x, scale_y, scale_z], dtype=np.float64)
        
        frequency = np.random.uniform(*cfg.TRAJ_RANDOM.FREQ)
        return cls(center=center, scale=scale, frequency=frequency, initial_phase=0.0)


class TrajectoryGenerator(ABC):
    def __init__(self, params: TrajectoryParams):
        self._params = params
    
    @abstractmethod
    def compute_position(self, t: float) -> np.ndarray:
        pass
    
    def compute_velocity(self, t: float, dt: float = 0.001) -> np.ndarray:
        """Numerical velocity computation."""
        p1 = self.compute_position(t - dt/2)
        p2 = self.compute_position(t + dt/2)
        return (p2 - p1) / dt
   
    def _compute_phase(self, t: float) -> float:
        return t * self._params.frequency * 2 * np.pi + self._params.initial_phase


class Figure8TrajectoryGenerator(TrajectoryGenerator):
    """
    Smooth figure-8 (lemniscate) trajectory.
    
    Parametric form that avoids sharp cusps:
        x(t) = A * sin(2*phase)
        y(t) = B * sin(phase)
        z(t) = C * sin(2*phase)  (small vertical motion)
    
    This creates a smooth infinity symbol shape.
    """
    def compute_position(self, t: float) -> np.ndarray:
        phase = self._compute_phase(t)
        
        # Standard figure-8 (lemniscate)
        dx = self._params.scale[0] * np.sin(2 * phase)
        dy = self._params.scale[1] * np.sin(phase)
        dz = self._params.scale[2] * np.sin(2 * phase)
        
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)


class CircleTrajectoryGenerator(TrajectoryGenerator):
    """Simple circle in XY plane."""
    def compute_position(self, t: float) -> np.ndarray:
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.cos(phase)
        dy = self._params.scale[1] * np.sin(phase)
        dz = 0.0
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)


class EllipseTrajectoryGenerator(TrajectoryGenerator):
    """Tilted ellipse for more interesting motion."""
    def compute_position(self, t: float) -> np.ndarray:
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.cos(phase)
        dy = self._params.scale[1] * np.sin(phase)
        dz = self._params.scale[2] * np.sin(phase)  # Tilt
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)


class LissajousTrajectoryGenerator(TrajectoryGenerator):
    """Lissajous curve - very smooth, no cusps."""
    def compute_position(self, t: float) -> np.ndarray:
        phase = self._compute_phase(t)
        # 3:2 Lissajous - smooth without cusps
        dx = self._params.scale[0] * np.sin(3 * phase)
        dy = self._params.scale[1] * np.sin(2 * phase)
        dz = self._params.scale[2] * np.sin(phase)
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)


class LeaderRobotSimulator(gym.Env):
    """
    Leader robot that generates joint-space trajectories via IK.
    
    Key improvements:
    1. Better IK failure handling (smooth interpolation)
    2. Joint velocity limiting
    3. Debug logging
    """
    
    # Maximum joint velocity (rad/s) - prevents large jumps
    MAX_JOINT_VEL = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 3.0], dtype=np.float64)
    
    def __init__(
        self,
        model_path=None,
        control_freq=None,
        trajectory_type="figure_8",
        randomize_params=False,
        verbose=False,
        **kwargs
    ):
        super().__init__()
        
        if model_path is None:
            model_path = cfg.DEFAULT_MUJOCO_MODEL_PATH
        if control_freq is None:
            control_freq = cfg.CONTROL_FREQ
       
        self.n_joints = cfg.N_JOINTS
        self._dt = 1.0 / control_freq
        self._randomize_params = randomize_params
        self._verbose = verbose
        
        # Load MuJoCo model
        self.model_path = str(model_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # IK solver
        self.ik_solver = IKSolver(
            self.model, 
            cfg.JOINT_LIMITS_LOWER, 
            cfg.JOINT_LIMITS_UPPER
        )
        
        # Parse trajectory type
        if isinstance(trajectory_type, str):
            try:
                self._trajectory_type = TrajectoryType(trajectory_type)
            except ValueError:
                print(f"[Leader] Unknown trajectory type '{trajectory_type}', using figure_8")
                self._trajectory_type = TrajectoryType.FIGURE_8
        else:
            self._trajectory_type = trajectory_type

        # Initialize physics to get actual EE position
        self.data.qpos[:self.n_joints] = cfg.INITIAL_JOINT_CONFIG
        mujoco.mj_forward(self.model, self.data)
        
        # Get end-effector site
        try:
            ee_site_id = self.model.site('panda_ee_site').id
        except KeyError:
            # Fallback: try other common names
            for name in ['ee_site', 'end_effector', 'tool_site', 'gripper_site']:
                try:
                    ee_site_id = self.model.site(name).id
                    break
                except KeyError:
                    continue
            else:
                # Use last body position as fallback
                print("[Leader] Warning: No EE site found, using last body")
                ee_site_id = 0
        
        self.actual_spawn_pos = self.data.site_xpos[ee_site_id].copy()
        
        # Setup trajectory
        if self._randomize_params:
            self._params = TrajectoryParams.randomized(self.actual_spawn_pos)
        else:
            self._params = TrajectoryParams()
        
        # Create trajectory generator
        generators = {
            TrajectoryType.FIGURE_8: Figure8TrajectoryGenerator,
            TrajectoryType.CIRCLE: CircleTrajectoryGenerator,
            TrajectoryType.ELLIPSE: EllipseTrajectoryGenerator,
            TrajectoryType.LISSAJOUS: LissajousTrajectoryGenerator,
        }
        self._generator = generators.get(self._trajectory_type, Figure8TrajectoryGenerator)(self._params)
        self.traj_start_pos = self._generator.compute_position(0.0)
        
        # State
        self._q_start = cfg.INITIAL_JOINT_CONFIG.copy().astype(np.float64)
        self._q_current = self._q_start.copy()
        self._q_prev = self._q_start.copy()
        self._trajectory_time = 0.0
        
        # IK failure tracking
        self._consecutive_ik_failures = 0
        self._total_ik_failures = 0
        self._last_valid_q = self._q_start.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._trajectory_time = 0.0
        self._q_current = self._q_start.copy()
        self._q_prev = self._q_start.copy()
        self._last_valid_q = self._q_start.copy()
        self._consecutive_ik_failures = 0
        self._total_ik_failures = 0
        
        # Reset IK solver
        self.ik_solver.reset_trajectory(q_start=self._q_start)
        
        # Randomize trajectory if enabled
        if self._randomize_params:
            self._params = TrajectoryParams.randomized(self.actual_spawn_pos)
            generators = {
                TrajectoryType.FIGURE_8: Figure8TrajectoryGenerator,
                TrajectoryType.CIRCLE: CircleTrajectoryGenerator,
                TrajectoryType.ELLIPSE: EllipseTrajectoryGenerator,
                TrajectoryType.LISSAJOUS: LissajousTrajectoryGenerator,
            }
            self._generator = generators.get(self._trajectory_type, Figure8TrajectoryGenerator)(self._params)
            self.traj_start_pos = self._generator.compute_position(0.0)
        
        return self._q_current.astype(np.float32), {}

    def step(self):
        self._trajectory_time += self._dt
        t = self._trajectory_time

        # Compute target Cartesian position
        if t < cfg.ROBOT.WARM_UP_DURATION:
            # Smooth interpolation to trajectory start
            progress = t / cfg.ROBOT.WARM_UP_DURATION
            # Use smooth step for interpolation
            smooth_progress = progress * progress * (3 - 2 * progress)  # Smoothstep
            current_target_pos = (1 - smooth_progress) * self.actual_spawn_pos + smooth_progress * self.traj_start_pos
        else:
            movement_time = t - cfg.ROBOT.WARM_UP_DURATION
            current_target_pos = self._generator.compute_position(movement_time)
        
        # Solve IK
        q_target_raw, ik_success, ik_info = self.ik_solver.solve(current_target_pos, self._q_current)
        
        # Handle IK result
        if ik_success and q_target_raw is not None:
            # IK succeeded
            self._consecutive_ik_failures = 0
            self._last_valid_q = q_target_raw.copy()
            q_target = q_target_raw
        else:
            # IK failed
            self._consecutive_ik_failures += 1
            self._total_ik_failures += 1
            
            if self._verbose and self._consecutive_ik_failures == 1:
                print(f"[Leader] IK failed at t={t:.2f}s, target={current_target_pos}")
            
            # Recovery strategy: interpolate towards last valid position
            # This prevents sudden jumps and gives IK a chance to recover
            if self._consecutive_ik_failures < 10:
                # Small interpolation towards last valid
                alpha = 0.1
                q_target = (1 - alpha) * self._q_current + alpha * self._last_valid_q
            else:
                # After many failures, just hold position
                q_target = self._q_current.copy()
        
        # Apply velocity limiting to prevent large jumps
        q_delta = q_target - self._q_current
        max_delta = self.MAX_JOINT_VEL * self._dt
        q_delta_clipped = np.clip(q_delta, -max_delta, max_delta)
        q_target_limited = self._q_current + q_delta_clipped
        
        # Update state
        self._q_prev = self._q_current.copy()
        self._q_current = q_target_limited.copy()
        
        # Compute velocity
        qd_raw = (self._q_current - self._q_prev) / self._dt
        
        return (
            self._q_current.astype(np.float32),
            qd_raw.astype(np.float32),
            None, 0.0, False, False,
            {'ik_success': ik_success, 'ik_failures': self._total_ik_failures}
        )
    
    def get_trajectory_info(self) -> dict:
        """Return trajectory information for debugging."""
        return {
            'type': self._trajectory_type.value,
            'center': self._params.center.copy(),
            'scale': self._params.scale.copy(),
            'frequency': self._params.frequency,
            'total_ik_failures': self._total_ik_failures,
        }