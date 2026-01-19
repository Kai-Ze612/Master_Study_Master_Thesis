"""
Hyperparameters of training
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

PRETRAINED_DIR = CHECKPOINT_DIR / "pre_trained_BC"
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
PRETRAINED_CHECKPOINT_PATH = PRETRAINED_DIR / "best_checkpoint.pth"

PRETRAINED_ACTOR_PATH = PRETRAINED_CHECKPOINT_PATH
NORMALIZATION_FILE_PATH = CHECKPOINT_DIR / "normalization.npz" 

######################################
# 2. GLOBAL CONSTANTS
######################################
N_JOINTS = 7
CONTROL_FREQ = 200
DT = 1.0 / CONTROL_FREQ

JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.5445, -3.0159], dtype=np.float32)
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 4.5169, 3.0159], dtype=np.float32)
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)
MAX_ACTION_TORQUE = np.array([30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 10.0], dtype=np.float32)
INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.5708, 0.785], dtype=np.float32)

# Normalization Logic
_Q_MEAN_DEFAULT = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.5708, 0.785], dtype=np.float32)
_Q_STD_DEFAULT  = np.array([1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32)
_QD_MEAN_DEFAULT = np.zeros(7, dtype=np.float32)
_QD_STD_DEFAULT  = np.ones(7, dtype=np.float32) * 2.0

# Initialize Global Variables
Q_MEAN = _Q_MEAN_DEFAULT
Q_STD  = _Q_STD_DEFAULT
QD_MEAN = _QD_MEAN_DEFAULT
QD_STD  = _QD_STD_DEFAULT

def load_combined_normalization(path=PRETRAINED_CHECKPOINT_PATH):
    """
    Loads normalization stats from the combined .pth file directly into config memory.
    """
    global Q_MEAN, Q_STD, QD_MEAN, QD_STD
    
    if not path.exists():
        print(f"[Config] No checkpoint found at {path}. Using Defaults.")
        return

    try:
        print(f"[Config] Loading combined checkpoint: {path}")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        # Check for 'norm' key (New Format)
        if isinstance(checkpoint, dict) and 'norm' in checkpoint:
            stats = checkpoint['norm']
            Q_MEAN = stats['q_mean'].astype(np.float32)
            Q_STD  = stats['q_std'].astype(np.float32)
            QD_MEAN = stats['qd_mean'].astype(np.float32)
            QD_STD  = stats['qd_std'].astype(np.float32)
            print("[Config] Normalization stats loaded successfully (Combined).")
        # Check for legacy npz file (Fallback)
        elif NORMALIZATION_FILE_PATH.exists():
             data = np.load(NORMALIZATION_FILE_PATH)
             Q_MEAN = data['q_mean'].astype(np.float32)
             Q_STD  = data['q_std'].astype(np.float32)
             QD_MEAN = data['qd_mean'].astype(np.float32)
             QD_STD  = data['qd_std'].astype(np.float32)
             print("[Config] Normalization stats loaded from legacy .npz file.")
            
    except Exception as e:
        print(f"[Config] Error loading normalization: {e}. Using Defaults.")

# Attempt Load
load_combined_normalization()

DELAY_INPUT_NORM_FACTOR = 100.0

@dataclass(frozen=True)
class RobotConfig:
    N_JOINTS: int = N_JOINTS
    CONTROL_FREQ: int = CONTROL_FREQ
    DT: float = DT
    TORQUE_LIMITS: np.ndarray = field(default_factory=lambda: TORQUE_LIMITS)
    MAX_ACTION_TORQUE: np.ndarray = field(default_factory=lambda: MAX_ACTION_TORQUE)
    JOINT_LIMITS_LOWER: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_LOWER)
    JOINT_LIMITS_UPPER: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_UPPER)
    INITIAL_JOINT_CONFIG: np.ndarray = field(default_factory=lambda: INITIAL_JOINT_CONFIG)
    
    # Normalization (Using Global Variables updated by loader)
    Q_MEAN: np.ndarray = field(default_factory=lambda: Q_MEAN)
    Q_STD: np.ndarray = field(default_factory=lambda: Q_STD)
    QD_MEAN: np.ndarray = field(default_factory=lambda: QD_MEAN)
    QD_STD: np.ndarray = field(default_factory=lambda: QD_STD)
    DELAY_INPUT_NORM_FACTOR: float = DELAY_INPUT_NORM_FACTOR

    # IK parameters
    IK_POSITION_TOLERANCE: float = 0.01
    IK_JACOBIAN_MAX_ITER: int = 300
    IK_JACOBIAN_STEP_SIZE: float = 0.01
    IK_JACOBIAN_DAMPING: float = 0.1
    IK_NULL_SPACE_GAIN: float = 0.5
    
    # Env steps
    MAX_EPISODE_STEPS: int = 2000
    MAX_JOINT_ERROR_TERMINATION: float = 0.5
    
    WARM_UP_DURATION: float = 0.5
    NO_DELAY_DURATION: float = 0.5
    
    TRAJECTORY_CENTER: np.ndarray = field(default_factory=lambda: np.array([0.3, 0, 0.5], dtype=np.float32))
    TRAJECTORY_SCALE: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.02], dtype=np.float32))
    TRAJECTORY_FREQUENCY: float = 0.1
    
    RNN_SEQ_LEN: int = 80
    RNN_HIDDEN_DIM: int = 256
    RNN_NUM_LAYERS: int = 3
    LSTM_PRED_HEAD_DIM: int = 128
    MAX_PREDICTION_ROLLOUT_STEPS: int = 50
    
    ACTOR_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [512, 256])
    CRITIC_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [512, 256])
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float = 0.5
    
    ROBOT_STATE_DIM: int = 14       
    ESTIMATOR_INPUT_DIM: int = 15   
    ESTIMATOR_OUTPUT_DIM: int = 14  
    
    ACTION_HISTORY_LEN: int = 50
    LEADER_HISTORY_BUFFER_LEN: int = 200
    
    TARGET_HISTORY_DIM: int = RNN_SEQ_LEN * 36 
    RL_OBS_DIM: int = 14 + (RNN_SEQ_LEN * 36) + (ACTION_HISTORY_LEN * 7) + 7
    ACTOR_INPUT_DIM: int = 14 + RNN_HIDDEN_DIM + (ACTION_HISTORY_LEN * 7) + 7  
    CRITIC_INPUT_DIM: int = (14 + (RNN_SEQ_LEN * 36) + (ACTION_HISTORY_LEN * 7) + 7) + 14 + 7  

    CHECKPOINT_DIR: Path = CHECKPOINT_DIR
    LOG_DIR: Path = LOG_DIR
    DEFAULT_MUJOCO_MODEL_PATH: Path = DEFAULT_MUJOCO_MODEL_PATH
    
    # [MODIFIED] Point to Combined Checkpoint
    PRETRAINED_ACTOR_PATH: Path = PRETRAINED_CHECKPOINT_PATH
    NORMALIZATION_FILE_PATH: Path = NORMALIZATION_FILE_PATH

@dataclass
class TrainConfig:
    SEED: int = 42
    TOTAL_TIMESTEPS: int = 2_000_000
    
    BATCH_SIZE: int = 2048       
    BUFFER_SIZE: int = 1_000_000
    GAMMA: float = 0.99
    
    WARMUP_STEPS: int = 25_000
    EVAL_INTERVAL: int = 10_000
    LOG_FREQ: int = 1000
    TRAIN_FREQUENCY: int = 64
    
    JOINT_OPTIMIZATION: bool = False   
    
    DEBUG_MODE: bool = True
    DEBUG_LOG_INTERVAL_TRAIN: int = 500
    
    AUX_LOSS_GRADIENT_SCALE: float = 0.5
    
    ENCODER_LR: float = 3e-5 
    ACTOR_LR: float = 3e-5    
    CRITIC_LR: float = 1e-4
    ALPHA_LR: float = 1e-4
    
    GRAD_CLIP: float = 0.5
    ENABLE_EARLY_STOP: bool = True
    EARLY_STOP_PATIENCE: int = 30
    EARLY_STOP_MIN_DELTA: float = 1.0

@dataclass
class BCConfig:
    STEPS_TO_COLLECT: int = 50_000
    BATCH_SIZE: int = 2048
    EPOCHS: int = 100
    LR: float = 6e-4
    SAVE_PATH: Path = PRETRAINED_CHECKPOINT_PATH

@dataclass
class BCExpertConfig:
    KP: float = 100.0
    KD: float = 20.0  

@dataclass
class RewardConfig:
    W_POS: float = 2.0
    W_VEL: float = 0.5
    W_ENERGY: float = 0.1
    W_SMOOTH: float = 0.2
    
    SCALE_POS: float = 2.0 
    SCALE_VEL: float = 1.0
    
    MIN_CLIP: float = -5.0
    MAX_CLIP: float = 3.0
    
    PENALTY_DIVERGENCE: float = -10.0
    PENALTY_LIMIT: float = -0.1
    PENALTY_MAX_ERROR: float = -0.1

@dataclass
class TrajRandomConfig:
    CENTER_X: Tuple[float, float] = (0.25, 0.35)
    CENTER_Y: Tuple[float, float] = (-0.1, 0.1)
    SCALE_X: Tuple[float, float] = (0.15, 0.25)
    SCALE_Y: Tuple[float, float] = (0.15, 0.25)
    FREQ: Tuple[float, float] = (0.05, 0.15)

@dataclass
class EvalConfig:
    NUM_EPISODES: int = 5
    DEBUG_LOG_INTERVAL: int = 50
    DEBUG_PRINT_INTERVAL: int = 500

@dataclass
class SACConfig:
    TARGET_TAU: float = 0.01
    REWARD_SCALE: float = 1.0 
    TARGET_ENTROPY_RATIO: float = 0.5 
    INITIAL_ALPHA: float = 0.5 


##################################################################
ROBOT = RobotConfig()
TRAIN = TrainConfig()
BC = BCConfig()
BC_EXPERT = BCExpertConfig()
REWARD = RewardConfig()
TRAJ_RANDOM = TrajRandomConfig()
EVAL = EvalConfig()
SAC = SACConfig()
##################################################################