import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

######################################
# 1. SYSTEM & PATHS
######################################

CONFIG_FILE_PATH = Path(__file__).resolve()
CONFIG_DIR = CONFIG_FILE_PATH.parent
PACKAGE_ROOT = CONFIG_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
WORKSPACE_SRC = PROJECT_ROOT.parent

CHECKPOINT_DIR = PACKAGE_ROOT / "trained_RL"
LOG_DIR = PACKAGE_ROOT / "logs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MUJOCO_MODEL_PATH = (
    WORKSPACE_SRC / "multipanda_ros2" / "franka_description" / "mujoco" / "franka" / "scene.xml"
)

######################################
# 2. ROBOT PHYSICAL PARAMETERS
######################################
N_JOINTS = 7
EE_BODY_NAME = "panda_hand"
TCP_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float32)
JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.5445, -3.0159], dtype=np.float32)
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 4.5169, 3.0159], dtype=np.float32)
JOINT_LIMIT_MARGIN = 0.05 # margin before limits

TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)
MAX_ACTION_TORQUE = np.array([20.0, 20.0, 20.0, 20.0, 5.0, 5.0, 5.0], dtype=np.float32)

# Franka intial joint pose
INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.5708, 0.785], dtype=np.float32)

######################################
# 3. SIMULATION & ENVIRONMENT
######################################
CONTROL_FREQ = 250
DT = 1.0 / CONTROL_FREQ

WARM_UP_DURATION = 1
NO_DELAY_DURATION = 1

MAX_EPISODE_STEPS = 5500
MAX_JOINT_ERROR_TERMINATION = 1.0

######################################
# 4. IK parameters
######################################
IK_POSITION_TOLERANCE = 0.01
IK_JACOBIAN_MAX_ITER = 300
IK_OPTIMIZATION_MAX_ITER = 100
IK_JACOBIAN_STEP_SIZE = 0.01
IK_JACOBIAN_DAMPING = 0.1
IK_NULL_SPACE_GAIN = 0.5

######################################
# 5. TRAJECTORY GENERATION
######################################
TRAJECTORY_CENTER = np.array([0.3, 0, 0.5], dtype=np.float32)
TRAJECTORY_SCALE = np.array([0.2, 0.2, 0.02], dtype=np.float32)
TRAJECTORY_FREQUENCY = 0.1

TRAJ_RANDOM_CENTER_X = (0.25, 0.35)   # Center X range
TRAJ_RANDOM_CENTER_Y = (-0.1, 0.1)    # Center Y range
TRAJ_RANDOM_SCALE_X = (0.15, 0.25)    # Scale X range
TRAJ_RANDOM_SCALE_Y = (0.15, 0.25)    # Scale Y range
TRAJ_RANDOM_FREQ = (0.05, 0.15)       # Frequency range

######################################
# 6. DATA PROCESSING & NORMALIZATION
######################################

# --- State Normalization ---
Q_MEAN = np.array([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78], dtype=np.float32)
Q_STD  = np.array([1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32) 
QD_MEAN = np.zeros(7, dtype=np.float32)
QD_STD  = np.ones(7, dtype=np.float32) * 2.0

# --- Input Scaling ---
DELAY_INPUT_NORM_FACTOR = 100.0

######################################
# 7. NETWORK ARCHITECTURE
######################################

# LSTM Parameters
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 3
RNN_SEQUENCE_LENGTH = 80

LSTM_PRED_HEAD_DIM = 128     # Hidden layer for prediction head
LSTM_AR_PROJ_DIM = 64        # Hidden layer for AR projection

ACTOR_HIDDEN_DIMS = [512, 256]      # Actor Layers
CRITIC_HIDDEN_DIMS = [512, 256]     # Critic Layers
LOG_STD_MIN = -20.0                 # Regularization bounds for actor network, this will prevent too small std e^-20 = 2e-9
LOG_STD_MAX = 2.0                   # Regularization bounds for actor network, this will prevent too large std e^2 = 7.39

# --- Dimensions ---
ROBOT_STATE_DIM = 14         # 7 Pos + 7 Vel (true q + qd)
ESTIMATOR_INPUT_DIM = 15     # 7 Pos + 7 Vel + 1 delay
ESTIMATOR_OUTPUT_DIM = 14    # 7 Pos + 7 Vel (predicted q + qd)

# LSTM input
TARGET_HISTORY_DIM = RNN_SEQUENCE_LENGTH * ESTIMATOR_INPUT_DIM  # 80 * 15 = 1200

# RL OBSERVATION DIMENSION (For Replay Buffer)
# Structure: [RemoteState(14) | TargetHistory(1200) | PrevAction(7)]
# For E2E model, the LSTM is also part of the policy, so the entire history is included in the observation.
RL_OBS_DIM = ROBOT_STATE_DIM + TARGET_HISTORY_DIM + N_JOINTS    # 14 + 1200 + 7 = 1221

# ACTOR NETWORK DIMENSION
# Structure: [RemoteState(14) | PredState(14) | PrevAction(7)]
ACTOR_INPUT_DIM = ROBOT_STATE_DIM + ESTIMATOR_OUTPUT_DIM + N_JOINTS # 14 + 14 + 7 = 35

# CRITIC NETWORK DIMENSION
# Structure: [PredState(14) | CurrentAction(7)]
CRITIC_INPUT_DIM = ESTIMATOR_OUTPUT_DIM + N_JOINTS              # 14 + 7 = 21

######################################
# 8. TRAINING HYPERPARAMETERS
######################################

# --- General ---
SEED = 42
BATCH_SIZE = 4096
BUFFER_SIZE = 1_000_000
GAMMA = 0.99
TAU = 0.005

# --- Schedule & Logging ---
STAGE1_STEPS = 20_000
STAGE2_STEPS = 10_00_000
LOG_FREQ = 500
VAL_FREQ = 5000
TOTAL_TIMESTEPS = 1000000    # Total RL steps (Phase 2)
EVAL_INTERVAL = 5000         # How often to run the evaluation episodes
BUFFER_SIZE = 1000000

# --- Learning Rates ---
ENCODER_LR = 1e-4
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
ALPHA_LR = 1e-4  # Global default

# Early stops
EARLY_STOPPING_PATIENCE = 50

######################################
# 9. DATACLASSES
######################################

@dataclass(frozen=True)
class RobotConfig:
    # --- Physical Params ---
    N_JOINTS: int = N_JOINTS
    CONTROL_FREQ: int = CONTROL_FREQ
    DT: float = DT
    TORQUE_LIMITS: np.ndarray = field(default_factory=lambda: TORQUE_LIMITS)
    MAX_ACTION_TORQUE: np.ndarray = field(default_factory=lambda: MAX_ACTION_TORQUE)
    JOINT_LIMITS_LOWER: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_LOWER)
    JOINT_LIMITS_UPPER: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_UPPER)
    
    # --- Normalization Stats ---
    Q_MEAN: np.ndarray = field(default_factory=lambda: Q_MEAN)
    Q_STD: np.ndarray = field(default_factory=lambda: Q_STD)
    QD_MEAN: np.ndarray = field(default_factory=lambda: QD_MEAN)
    QD_STD: np.ndarray = field(default_factory=lambda: QD_STD)
    
    # --- Trajectory Gen ---
    TRAJECTORY_CENTER: np.ndarray = field(default_factory=lambda: TRAJECTORY_CENTER)
    TRAJECTORY_SCALE: np.ndarray = field(default_factory=lambda: TRAJECTORY_SCALE)
    TRAJECTORY_FREQUENCY: float = TRAJECTORY_FREQUENCY
    
    # --- Network Architecture ---
    RNN_SEQ_LEN: int = RNN_SEQUENCE_LENGTH
    RNN_HIDDEN_DIM: int = RNN_HIDDEN_DIM
    RNN_NUM_LAYERS: int = RNN_NUM_LAYERS
    ACTOR_HIDDEN_DIMS: List[int] = field(default_factory=lambda: ACTOR_HIDDEN_DIMS)
    CRITIC_HIDDEN_DIMS: List[int] = field(default_factory=lambda: CRITIC_HIDDEN_DIMS)
    LSTM_PRED_HEAD_DIM: int = LSTM_PRED_HEAD_DIM
    LOG_STD_MIN: float = LOG_STD_MIN
    LOG_STD_MAX: float = LOG_STD_MAX

    # --- CRITICAL DIMENSIONS (The Refactor) ---
    ROBOT_STATE_DIM: int = ROBOT_STATE_DIM           # 14
    ESTIMATOR_INPUT_DIM: int = ESTIMATOR_INPUT_DIM   # 15
    ESTIMATOR_OUTPUT_DIM: int = ESTIMATOR_OUTPUT_DIM # 14
    
    # 1. Raw Data for LSTM (1200)
    TARGET_HISTORY_DIM: int = TARGET_HISTORY_DIM
    
    # 2. Total Buffer Size (1221) - RENAMED from OBS_DIM
    RL_OBS_DIM: int = RL_OBS_DIM
    
    # 3. Network Inputs (Calculated above)
    ACTOR_INPUT_DIM: int = ACTOR_INPUT_DIM           # 35
    CRITIC_INPUT_DIM: int = CRITIC_INPUT_DIM         # 21

    # --- Paths & Sim ---
    PROJECT_ROOT: Path = PROJECT_ROOT
    CHECKPOINT_DIR: Path = CHECKPOINT_DIR
    LOG_DIR: Path = LOG_DIR
    DEFAULT_MUJOCO_MODEL_PATH: Path = DEFAULT_MUJOCO_MODEL_PATH
    MAX_EPISODE_STEPS: int = MAX_EPISODE_STEPS
    MAX_JOINT_ERROR_TERMINATION: float = MAX_JOINT_ERROR_TERMINATION
    INITIAL_JOINT_CONFIG: np.ndarray = field(default_factory=lambda: INITIAL_JOINT_CONFIG)
    WARM_UP_DURATION: float = WARM_UP_DURATION
    NO_DELAY_DURATION: float = NO_DELAY_DURATION

# (TrainConfig and SACConfig remain unchanged)
@dataclass
class TrainConfig:
    SEED: int = SEED
    BATCH_SIZE: int = BATCH_SIZE
    BUFFER_SIZE: int = BUFFER_SIZE
    GAMMA: float = GAMMA
    STAGE1_STEPS: int = STAGE1_STEPS
    STAGE2_STEPS: int = STAGE2_STEPS
    ENCODER_LR: float = ENCODER_LR
    ACTOR_LR: float = ACTOR_LR
    CRITIC_LR: float = CRITIC_LR
    ALPHA_LR: float = ALPHA_LR
    LOG_FREQ: int = LOG_FREQ
    VAL_FREQ: int = VAL_FREQ

@dataclass
class SACConfig:
    WARMUP_STEPS: int = 10000
    REWARD_SCALE: float = 1.0
    TARGET_TAU: float = 0.001
    POLICY_DELAY: int = 1 
    Q_CLIP_MAX: float = 100.0
    GRAD_CLIP_CRITIC: float = 1.0
    GRAD_CLIP_ACTOR: float = 1.0
    TARGET_ENTROPY_RATIO: float = 0.05
    ALPHA_LR: float = 1e-4

######################################
# 10. INSTANTIATE CONFIGS
######################################

ROBOT = RobotConfig()
TRAIN = TrainConfig()
SAC = SACConfig()