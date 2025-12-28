import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

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

# --- RNN / LSTM Estimator ---
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 3
RNN_SEQUENCE_LENGTH = 80
LSTM_PRED_HEAD_DIM = 128     # Hidden layer for prediction head
LSTM_AR_PROJ_DIM = 64        # Hidden layer for AR projection

# --- MLP Actor/Critic ---
MLP_HIDDEN_DIMS = [512, 256]
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# --- Dimensions ---
ROBOT_STATE_DIM = 14         # 7 Pos + 7 Vel
ESTIMATOR_STATE_DIM = 15     # 7 Pos + 7 Vel + 1 Delay
ESTIMATOR_OUTPUT_DIM = 14
ROBOT_HISTORY_DIM = RNN_SEQUENCE_LENGTH * ROBOT_STATE_DIM
TARGET_HISTORY_DIM = RNN_SEQUENCE_LENGTH * ESTIMATOR_STATE_DIM
ESTIMATOR_INPUT_DIM = 15

OBS_DIM = (
    ROBOT_STATE_DIM + 
    (RNN_SEQUENCE_LENGTH * ROBOT_STATE_DIM) + 
    (RNN_SEQUENCE_LENGTH * ESTIMATOR_INPUT_DIM) + 
    N_JOINTS
)

######################################
# 8. TRAINING HYPERPARAMETERS
######################################

# --- General ---
SEED = 42
BATCH_SIZE = 1024
BUFFER_SIZE = 1_000_000
GAMMA = 0.99
TAU = 0.005

# --- Schedule & Logging ---
STAGE1_STEPS = 10_000
STAGE2_STEPS = 10_00_000
LOG_FREQ = 500
VAL_FREQ = 5000

# --- Learning Rates ---
ENCODER_LR = 1e-4
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4  # Global default

######################################
# 9. DATACLASSES
######################################

@dataclass(frozen=True)
class RobotConfig:
    """Encapsulates Robot, Simulation, and Network Constants"""
    N_JOINTS: int = N_JOINTS
    CONTROL_FREQ: int = CONTROL_FREQ
    DT: float = DT
    TORQUE_LIMITS: np.ndarray = field(default_factory=lambda: TORQUE_LIMITS)
    MAX_ACTION_TORQUE: np.ndarray = field(default_factory=lambda: MAX_ACTION_TORQUE)
    JOINT_LIMITS_LOWER: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_LOWER)
    JOINT_LIMITS_UPPER: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_UPPER)
    Q_MEAN: np.ndarray = field(default_factory=lambda: Q_MEAN)
    Q_STD: np.ndarray = field(default_factory=lambda: Q_STD)
    QD_MEAN: np.ndarray = field(default_factory=lambda: QD_MEAN)
    QD_STD: np.ndarray = field(default_factory=lambda: QD_STD)
    TRAJECTORY_CENTER: np.ndarray = field(default_factory=lambda: TRAJECTORY_CENTER)
    TRAJECTORY_SCALE: np.ndarray = field(default_factory=lambda: TRAJECTORY_SCALE)
    TRAJECTORY_FREQUENCY: float = TRAJECTORY_FREQUENCY
    RNN_SEQ_LEN: int = RNN_SEQUENCE_LENGTH
    RNN_HIDDEN_DIM: int = RNN_HIDDEN_DIM
    RNN_NUM_LAYERS: int = RNN_NUM_LAYERS
    ROBOT_STATE_DIM: int = ROBOT_STATE_DIM
    ESTIMATOR_INPUT_DIM: int = ESTIMATOR_STATE_DIM
    ESTIMATOR_OUTPUT_DIM: int = ESTIMATOR_OUTPUT_DIM
    ESTIMATOR_STATE_DIM: int = ESTIMATOR_STATE_DIM
    ROBOT_HISTORY_DIM: int = ROBOT_HISTORY_DIM
    TARGET_HISTORY_DIM: int = TARGET_HISTORY_DIM
    OBS_DIM: int = OBS_DIM
    PROJECT_ROOT: Path = PROJECT_ROOT
    CHECKPOINT_DIR: Path = CHECKPOINT_DIR
    LOG_DIR: Path = LOG_DIR
    DEFAULT_MUJOCO_MODEL_PATH: Path = DEFAULT_MUJOCO_MODEL_PATH
    MAX_EPISODE_STEPS: int = MAX_EPISODE_STEPS
    MAX_JOINT_ERROR_TERMINATION: float = MAX_JOINT_ERROR_TERMINATION
    INITIAL_JOINT_CONFIG: np.ndarray = field(default_factory=lambda: INITIAL_JOINT_CONFIG)
    WARM_UP_DURATION: float = WARM_UP_DURATION
    NO_DELAY_DURATION: float = NO_DELAY_DURATION


@dataclass
class TrainConfig:
    """Encapsulates General Training Hyperparameters"""
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
    """
    Stage 3 SAC Fine-tuning Hyperparameters
    """
    # Warmup
    WARMUP_STEPS: int = 10000
    
    # Reward
    REWARD_SCALE: float = 1.0
    
    # Target Network
    TARGET_TAU: float = 0.001
    
    # Policy Updates
    POLICY_DELAY: int = 1 
    
    # Stability / Clipping
    Q_CLIP_MAX: float = 100.0
    GRAD_CLIP_CRITIC: float = 1.0
    GRAD_CLIP_ACTOR: float = 1.0
    
    # Entropy Tuning
    TARGET_ENTROPY_RATIO: float = 0.05
    ALPHA_LR: float = 1e-4  # Specific LR for Alpha tuning in SAC

######################################
# 10. INSTANTIATE CONFIGS
######################################

ROBOT = RobotConfig()
TRAIN = TrainConfig()
SAC = SACConfig()