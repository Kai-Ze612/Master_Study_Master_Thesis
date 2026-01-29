"""
Hyperparameters for State-Augmented PD Gain Tuning
"""

import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

######################################
# 1. SYSTEM & PATHS
######################################
CONFIG_FILE_PATH = Path(__file__).resolve()

# Inner Python Package: .../src/ASAC/ASAC
PYTHON_PACKAGE_ROOT = CONFIG_FILE_PATH.parent.parent 

# Workspace Src: .../src
WORKSPACE_SRC = PYTHON_PACKAGE_ROOT.parent.parent

# --- [UPDATED] Save models in ASAC/ASAC/trained_RL ---
CHECKPOINT_DIR = PYTHON_PACKAGE_ROOT / "trained_RL"
LOG_DIR = PYTHON_PACKAGE_ROOT / "logs" # General logs if needed

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Path to MuJoCo scene
DEFAULT_MUJOCO_MODEL_PATH = (
    WORKSPACE_SRC / "multipanda_ros2" / "franka_description" / "mujoco" / "franka" / "scene.xml"
)

######################################
# 2. GLOBAL CONSTANTS
######################################
N_JOINTS = 7
CONTROL_FREQ = 250
DT = 1.0 / CONTROL_FREQ

# Physics Limits
JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.5445, -3.0159], dtype=np.float32)
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 4.5169, 3.0159], dtype=np.float32)
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)
INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.5708, 0.785], dtype=np.float32)

# Normalization Stats
Q_MEAN = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.5708, 0.785], dtype=np.float32)
Q_STD  = np.ones(7, dtype=np.float32)
QD_MEAN = np.zeros(7, dtype=np.float32)
QD_STD  = np.ones(7, dtype=np.float32) * 2.0

@dataclass(frozen=True)
class PDGainConfig:
    # Tunable ranges for the RL agent
    KP_MIN: np.ndarray = field(default_factory=lambda: np.array([10.0]*7, dtype=np.float32))
    KP_MAX: np.ndarray = field(default_factory=lambda: np.array([300.0]*7, dtype=np.float32))
    KD_MIN: np.ndarray = field(default_factory=lambda: np.array([1.0]*7, dtype=np.float32))
    KD_MAX: np.ndarray = field(default_factory=lambda: np.array([50.0]*7, dtype=np.float32))

@dataclass(frozen=True)
class RobotConfig:
    N_JOINTS: int = N_JOINTS
    CONTROL_FREQ: int = CONTROL_FREQ
    DT: float = DT
    
    TORQUE_LIMITS: np.ndarray = field(default_factory=lambda: TORQUE_LIMITS)
    JOINT_LIMITS_LOWER: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_LOWER)
    JOINT_LIMITS_UPPER: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_UPPER)
    INITIAL_JOINT_CONFIG: np.ndarray = field(default_factory=lambda: INITIAL_JOINT_CONFIG)
    
    # Normalization
    Q_MEAN: np.ndarray = field(default_factory=lambda: Q_MEAN)
    Q_STD: np.ndarray = field(default_factory=lambda: Q_STD)
    QD_MEAN: np.ndarray = field(default_factory=lambda: QD_MEAN)
    QD_STD: np.ndarray = field(default_factory=lambda: QD_STD)

    MAX_EPISODE_STEPS: int = 5200
    WARM_UP_DURATION: float = 0.5
    
    TRAJECTORY_CENTER: np.ndarray = field(default_factory=lambda: np.array([0.3, 0, 0.5], dtype=np.float32))
    TRAJECTORY_SCALE: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.02], dtype=np.float32))
    TRAJECTORY_FREQUENCY: float = 0.1
    
    # --- State Augmentation ---
    FRAME_STACK: int = 5
    LEADER_HISTORY_BUFFER_LEN: int = 200  # <--- [ADDED] This was missing
    
    # Input per frame: Leader(15) + Follower(14) + PrevAction(14) = 43
    # Total Input = 43 * 5 = 215
    RL_OBS_DIM: int = 43 * FRAME_STACK
    
    # Action Dim: 7 Kp + 7 Kd = 14
    ACTION_DIM: int = 14
    
    # Network Architecture
    ACTOR_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [512, 256, 256])
    CRITIC_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [512, 256, 256])
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float = 0.5

    CHECKPOINT_DIR: Path = CHECKPOINT_DIR
    LOG_DIR: Path = LOG_DIR
    DEFAULT_MUJOCO_MODEL_PATH: Path = DEFAULT_MUJOCO_MODEL_PATH

@dataclass
class TrainConfig:
    SEED: int = 42
    TOTAL_TIMESTEPS: int = 2_000_000
    BATCH_SIZE: int = 1024       
    BUFFER_SIZE: int = 1_000_000
    GAMMA: float = 0.99
    
    WARMUP_STEPS: int = 20_000
    EVAL_INTERVAL: int = 10_000
    LOG_FREQ: int = 1000
    
    # Training Speed
    TRAIN_FREQUENCY: int = 1  # Train every X steps
    UTD_RATIO: float = 1.0    # Updates per data step
    
    ACTOR_LR: float = 3e-4    
    CRITIC_LR: float = 3e-4
    ALPHA_LR: float = 3e-4
    
    GRAD_CLIP: float = 1.0

@dataclass
class RewardConfig:
    W_POS: float = 4.0
    W_VEL: float = 0.5
    W_ENERGY: float = 0.005 # Low penalty to allow stiff gains
    W_SMOOTH: float = 0.1
    
    SCALE_POS: float = 5.0 
    SCALE_VEL: float = 1.0
    
    PENALTY_DIVERGENCE: float = -20.0

@dataclass
class TrajRandomConfig:
    CENTER_X: Tuple[float, float] = (0.25, 0.35)
    CENTER_Y: Tuple[float, float] = (-0.1, 0.1)
    SCALE_X: Tuple[float, float] = (0.15, 0.25)
    SCALE_Y: Tuple[float, float] = (0.15, 0.25)
    FREQ: Tuple[float, float] = (0.05, 0.15)

@dataclass
class SACConfig:
    TARGET_TAU: float = 0.005
    REWARD_SCALE: float = 1.0 
    TARGET_ENTROPY_RATIO: float = 0.9 
    INITIAL_ALPHA: float = 0.2 

##################################################################
ROBOT = RobotConfig()
PD_GAINS = PDGainConfig()
TRAIN = TrainConfig()
REWARD = RewardConfig()
TRAJ_RANDOM = TrajRandomConfig()
SAC = SACConfig()
##################################################################