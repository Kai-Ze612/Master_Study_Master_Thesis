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

CHECKPOINT_DIR = PACKAGE_ROOT / "trained_RL"
LOG_DIR = PACKAGE_ROOT / "logs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MUJOCO_MODEL_PATH = (
    WORKSPACE_SRC / "multipanda_ros2" / "franka_description" / "mujoco" / "franka" / "scene.xml"
)

######################################
# 2. GLOBAL CONSTANTS (Physics & Hardware)
######################################
N_JOINTS = 7
CONTROL_FREQ = 250
DT = 1.0 / CONTROL_FREQ

# Joint Limits & Torques
JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.5445, -3.0159], dtype=np.float32)
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 4.5169, 3.0159], dtype=np.float32)
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)
MAX_ACTION_TORQUE = np.array([20.0, 20.0, 20.0, 20.0, 5.0, 5.0, 5.0], dtype=np.float32)
INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.5708, 0.785], dtype=np.float32)

# Normalization Stats
Q_MEAN = np.array([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78], dtype=np.float32)
Q_STD  = np.array([1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32) 
QD_MEAN = np.zeros(7, dtype=np.float32)
QD_STD  = np.ones(7, dtype=np.float32) * 2.0
DELAY_INPUT_NORM_FACTOR = 100.0

######################################
# 3. CONFIGURATION DATACLASSES
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
    INITIAL_JOINT_CONFIG: np.ndarray = field(default_factory=lambda: INITIAL_JOINT_CONFIG)
    
    # --- Normalization ---
    Q_MEAN: np.ndarray = field(default_factory=lambda: Q_MEAN)
    Q_STD: np.ndarray = field(default_factory=lambda: Q_STD)
    QD_MEAN: np.ndarray = field(default_factory=lambda: QD_MEAN)
    QD_STD: np.ndarray = field(default_factory=lambda: QD_STD)
    DELAY_INPUT_NORM_FACTOR: float = DELAY_INPUT_NORM_FACTOR

    # --- IK Parameters (RESTORED) ---
    IK_POSITION_TOLERANCE: float = 0.01
    IK_JACOBIAN_MAX_ITER: int = 300
    IK_JACOBIAN_STEP_SIZE: float = 0.01
    IK_JACOBIAN_DAMPING: float = 0.1
    IK_NULL_SPACE_GAIN: float = 0.5
    
    # --- Sim Params ---
    MAX_EPISODE_STEPS: int = 5500
    MAX_JOINT_ERROR_TERMINATION: float = 1.0
    WARM_UP_DURATION: float = 1.0
    NO_DELAY_DURATION: float = 1.0
    
    # --- Trajectory Gen ---
    TRAJECTORY_CENTER: np.ndarray = field(default_factory=lambda: np.array([0.3, 0, 0.5], dtype=np.float32))
    TRAJECTORY_SCALE: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.02], dtype=np.float32))
    TRAJECTORY_FREQUENCY: float = 0.1
    
    # --- Network Architecture ---
    # LSTM
    RNN_SEQ_LEN: int = 80
    RNN_HIDDEN_DIM: int = 256
    RNN_NUM_LAYERS: int = 3
    LSTM_PRED_HEAD_DIM: int = 128
    
    # MLP
    ACTOR_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [512, 256])
    CRITIC_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [512, 256])
    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    # Dimensions
    ROBOT_STATE_DIM: int = 14       # 7 Pos + 7 Vel
    ESTIMATOR_INPUT_DIM: int = 15   # 7 Pos + 7 Vel + 1 Delay
    ESTIMATOR_OUTPUT_DIM: int = 14  # Predicted State
    
    # 1. Raw Data for LSTM (80 * 15 = 1200)
    TARGET_HISTORY_DIM: int = 80 * 15 
    
    # 2. Total Observation Dim (14 + 1200 + 7 = 1221)
    # [RemoteState(14) | TargetHistory(1200) | PrevAction(7)]
    RL_OBS_DIM: int = 14 + (80 * 15) + 7
   
    # 3. Network Inputs
    # Actor: [Remote(14) | Pred(14) | PrevAction(7)] = 35
    ACTOR_INPUT_DIM: int = 14 + 14 + 7  
    # Critic: [Pred(14) | Action(7)] = 21
    CRITIC_INPUT_DIM: int = 14 + 7      

    # Paths
    CHECKPOINT_DIR: Path = CHECKPOINT_DIR
    LOG_DIR: Path = LOG_DIR
    DEFAULT_MUJOCO_MODEL_PATH: Path = DEFAULT_MUJOCO_MODEL_PATH

@dataclass
class TrainConfig:
    SEED: int = 42
    BATCH_SIZE: int = 4096  # Large batch for stable gradients
    BUFFER_SIZE: int = 1_000_000
    GAMMA: float = 0.99
    
    # Steps
    TOTAL_TIMESTEPS: int = 1_000_000  # Renamed from STAGE2_STEPS
    WARMUP_STEPS: int = 10_000        # For Action Scaling
    LSTM_BURNIN_STEPS: int = 2_000    # For LSTM Burn-in
    
    # Optimization (Differential Learning Rates)
    ENCODER_LR: float = 1e-5          # Slow learning for physics (1e-5)
    ACTOR_LR: float = 3e-4            # Fast learning for policy (3e-4)
    CRITIC_LR: float = 3e-4
    ALPHA_LR: float = 3e-4
    
    # Logging
    LOG_FREQ: int = 500
    EVAL_INTERVAL: int = 5000
    VAL_FREQ: int = 5000

@dataclass
class SACConfig:
    TARGET_TAU: float = 0.005
    REWARD_SCALE: float = 1.0
    GRAD_CLIP_CRITIC: float = 1.0
    GRAD_CLIP_ACTOR: float = 1.0
    TARGET_ENTROPY_RATIO: float = 0.5  # 50% of max entropy

######################################
# 4. INSTANTIATE CONFIGS
######################################

ROBOT = RobotConfig()
TRAIN = TrainConfig()
SAC = SACConfig()