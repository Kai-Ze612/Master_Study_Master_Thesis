"""
Shared training Config
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

######################################
# 1. SYSTEM & PATHS
######################################
CONFIG_FILE_PATH = Path(__file__).resolve()
PACKAGE_ROOT = CONFIG_FILE_PATH.parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
WORKSPACE_SRC = PROJECT_ROOT.parent

# Output Directories
CHECKPOINT_DIR = PACKAGE_ROOT / "trained_RL"
LOG_DIR = PACKAGE_ROOT / "logs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# MuJoCo Scene
DEFAULT_MUJOCO_MODEL_PATH = (
    WORKSPACE_SRC / "multipanda_ros2" / "franka_description" / "mujoco" / "franka" / "scene.xml"
)

# Model Weights Paths
PRETRAINED_ACTOR_PATH = CHECKPOINT_DIR / "pretrained_actor.pth"

######################################
# 2. GLOBAL CONSTANTS (Physics & Hardware)
######################################
N_JOINTS = 7
CONTROL_FREQ = 250
DT = 1.0 / CONTROL_FREQ

# Hardware Limits (Franka Emika Panda)
JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.5445, -3.0159], dtype=np.float32)
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 4.5169, 3.0159], dtype=np.float32)
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)

# Safety / Action Space Limits
MAX_ACTION_TORQUE = np.array([30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 10.0], dtype=np.float32)
INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.5708, 0.785], dtype=np.float32)

# Normalization Stats (Z-Score Normalization)
Q_MEAN = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.5708, 0.785], dtype=np.float32)
Q_STD  = np.array([2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0], dtype=np.float32)
QD_MEAN = np.zeros(7, dtype=np.float32)
QD_STD  = np.ones(7, dtype=np.float32) * 2.0
DELAY_INPUT_NORM_FACTOR = 100.0

@dataclass(frozen=True)
class RobotConfig:
    # --- Physical Constants ---
    N_JOINTS: int = N_JOINTS
    CONTROL_FREQ: int = CONTROL_FREQ
    DT: float = DT
    TORQUE_LIMITS: np.ndarray = field(default_factory=lambda: TORQUE_LIMITS)
    MAX_ACTION_TORQUE: np.ndarray = field(default_factory=lambda: MAX_ACTION_TORQUE)
    JOINT_LIMITS_LOWER: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_LOWER)
    JOINT_LIMITS_UPPER: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_UPPER)
    INITIAL_JOINT_CONFIG: np.ndarray = field(default_factory=lambda: INITIAL_JOINT_CONFIG)
    
    # --- Normalization ---
    Q_MEAN: np.ndarray = field(default_factory=lambda: Q_MEAN)
    Q_STD: np.ndarray = field(default_factory=lambda: Q_STD)
    QD_MEAN: np.ndarray = field(default_factory=lambda: QD_MEAN)
    QD_STD: np.ndarray = field(default_factory=lambda: QD_STD)
    DELAY_INPUT_NORM_FACTOR: float = DELAY_INPUT_NORM_FACTOR

    # --- Inverse Kinematics (Leader Sim) ---
    IK_POSITION_TOLERANCE: float = 0.01
    IK_JACOBIAN_MAX_ITER: int = 300
    IK_JACOBIAN_STEP_SIZE: float = 0.01
    IK_JACOBIAN_DAMPING: float = 0.1
    IK_NULL_SPACE_GAIN: float = 0.5
    
    # --- Simulation & Termination ---
    MAX_EPISODE_STEPS: int = 2500
    MAX_JOINT_ERROR_TERMINATION: float = 0.5
    WARM_UP_DURATION: float = 0.5
    NO_DELAY_DURATION: float = 0.5
    
    # --- Trajectory Generator ---
    TRAJECTORY_CENTER: np.ndarray = field(default_factory=lambda: np.array([0.3, 0, 0.5], dtype=np.float32))
    TRAJECTORY_SCALE: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.02], dtype=np.float32))
    TRAJECTORY_FREQUENCY: float = 0.1
    
    # --- Architecture Dimensions ---
    RNN_SEQ_LEN: int = 50
    RNN_HIDDEN_DIM: int = 256
    RNN_NUM_LAYERS: int = 3
    LSTM_PRED_HEAD_DIM: int = 128
    
    # Prediction Rollout Cap (prevents infinite loops in LSTM)
    MAX_PREDICTION_ROLLOUT_STEPS: int = 60  
    
    ACTOR_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [512, 256])
    CRITIC_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [512, 256])
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float = 0.5
    
    # State Dimensions
    ROBOT_STATE_DIM: int = 14       
    
    # Estimator: Leader(15) -> Leader(14)
    # Input: 7 (q) + 7 (qd) + 1 (delay) = 15
    # Output: 7 (q) + 7 (qd) = 14
    ESTIMATOR_INPUT_DIM: int = 15   
    ESTIMATOR_OUTPUT_DIM: int = 14  
    
    # History Buffers
    ACTION_HISTORY_LEN: int = 50
    LEADER_HISTORY_BUFFER_LEN: int = 200 # For the deque in training_env
    
    # Calculated Observation Dimensions
    TARGET_HISTORY_DIM: int = RNN_SEQ_LEN * 36 
    RL_OBS_DIM: int = 14 + (RNN_SEQ_LEN * 36) + (ACTION_HISTORY_LEN * 7) + 7
    
    # ACTOR INPUT: 
    # RealFollower(14) + Latent(256) + ActionHistory(350) + PrevAction(7) = 627
    ACTOR_INPUT_DIM: int = 14 + RNN_HIDDEN_DIM + (ACTION_HISTORY_LEN * 7) + 7  

    # CRITIC INPUT:
    # FullObs(2171) + PredLeader(14) + Action(7)
    CRITIC_INPUT_DIM: int = (14 + (RNN_SEQ_LEN * 36) + (ACTION_HISTORY_LEN * 7) + 7) + 14 + 7  

    # --- System Paths ---
    CHECKPOINT_DIR: Path = CHECKPOINT_DIR
    LOG_DIR: Path = LOG_DIR
    DEFAULT_MUJOCO_MODEL_PATH: Path = DEFAULT_MUJOCO_MODEL_PATH
    PRETRAINED_ACTOR_PATH: Path = PRETRAINED_ACTOR_PATH

@dataclass
class TrainConfig:
    SEED: int = 42
    TOTAL_TIMESTEPS: int = 1_000_000
    BATCH_SIZE: int = 4096 
    BUFFER_SIZE: int = 1_000_000
    GAMMA: float = 0.99
    
    WARMUP_STEPS: int = 50_000
    EVAL_INTERVAL: int = 10_000
    LOG_FREQ: int = 1024
    TRAIN_FREQUENCY: int = 32
    
    # Weighted Loss for Physics-Aware Extrapolation
    # This was hardcoded as 0.5 in e2e_algorithm.py
    AUX_LOSS_WEIGHT: float = 0.5 

    # Learning Rates
    ENCODER_LR: float = 3e-5
    ACTOR_LR: float = 3e-5    
    CRITIC_LR: float = 3e-5
    ALPHA_LR: float = 3e-5
    
    GRAD_CLIP_CRITIC: float = 10.0
    GRAD_CLIP_ACTOR: float = 10.0

    ENABLE_EARLY_STOP: bool = True
    EARLY_STOP_PATIENCE: int = 30
    EARLY_STOP_MIN_DELTA: float = 1.0

@dataclass
class BCConfig:
    """Hyperparameters for Behavior Cloning Pre-training"""
    STEPS_TO_COLLECT: int = 50_000
    BATCH_SIZE: int = 1024
    EPOCHS: int = 30
    LR: float = 3e-4
    SAVE_PATH: Path = PRETRAINED_ACTOR_PATH
    
@dataclass
class SACConfig:
    TARGET_TAU: float = 0.01
    REWARD_SCALE: float = 1.0 
    TARGET_ENTROPY_RATIO: float = 0.5 
    INITIAL_ALPHA: float = 0.5 

######################################
# 4. INSTANTIATE CONFIGS
######################################

ROBOT = RobotConfig()
TRAIN = TrainConfig()
BC = BCConfig()
SAC = SACConfig()