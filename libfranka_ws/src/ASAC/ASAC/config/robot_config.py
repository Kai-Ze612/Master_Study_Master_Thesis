"""
Robot and Training Configuration
--------------------------------
Centralized configuration for A-SAC PD Gain Tuning.
"""

import numpy as np
from pathlib import Path

# =============================================================================
# 1. SYSTEM & PATHS (Dynamic Resolution)
# =============================================================================
CONFIG_FILE_PATH = Path(__file__).resolve()
PACKAGE_ROOT = CONFIG_FILE_PATH.parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
WORKSPACE_SRC = PROJECT_ROOT.parent

# Global Directories
CHECKPOINT_DIR = PACKAGE_ROOT / "trained_RL"
LOG_DIR = PACKAGE_ROOT / "logs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Pre-trained paths
PRETRAINED_DIR = CHECKPOINT_DIR / "pre_trained_BC"
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
PRETRAINED_CHECKPOINT_PATH = PRETRAINED_DIR / "best_checkpoint.pth"
PRETRAINED_ACTOR_PATH = PRETRAINED_CHECKPOINT_PATH
NORMALIZATION_FILE_PATH = CHECKPOINT_DIR / "normalization.npz"

# Simulation Asset Path
DEFAULT_MUJOCO_MODEL_PATH = (
    WORKSPACE_SRC / "multipanda_ros2" / "franka_description" / "mujoco" / "franka" / "scene.xml"
)

# Simulation Timing
CONTROL_FREQ = 1000  # Hz
DT = 1.0 / CONTROL_FREQ

# =============================================================================
# Robot Physical Constants (Global)
# =============================================================================
N_JOINTS = 7

# Joint Limits (Franka Emika Panda)
JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_LIMITS_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
TORQUE_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12], dtype=np.float32)

# Initial Configuration (Rest Pose)
INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

# =============================================================================
# PD Gain Ranges (The Action Space)
# =============================================================================
class PD_GAINS:
    # Base gains (used for initialization and regularization)
    KP_BASE = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], dtype=np.float32)
    KD_BASE = np.array([30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0], dtype=np.float32)

    # Allowable ranges for the RL agent
    KP_MIN = np.array([100.0] * 7, dtype=np.float32)
    KP_MAX = np.array([1000.0, 1000.0, 1000.0, 1000.0, 500.0, 300.0, 100.0], dtype=np.float32)
    
    KD_MIN = np.array([1.0] * 7, dtype=np.float32)
    KD_MAX = np.array([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 20.0], dtype=np.float32)

    # Pre-calculated ranges for normalization
    KP_RANGE = KP_MAX - KP_MIN
    KD_RANGE = KD_MAX - KD_MIN

# =============================================================================
# Architecture & Observation Dimensions
# =============================================================================
class ROBOT:
    # Point to the global variable
    CHECKPOINT_DIR = CHECKPOINT_DIR
    
    # Dimensions
    N_JOINTS = 7
    
    # --- Sequence Lengths ---
    RNN_SEQ_LEN = 80  
    LEADER_HISTORY_BUFFER_LEN = 300  
    ACTION_HISTORY_LEN = 80
    
    # --- Dynamic Dimension Calculation ---
    # Feature count per step: Leader (14) + Delay (1) + Follower (14) + Prev Gains (14) = 43
    _FEATS_PER_STEP = (N_JOINTS * 2) + 1 + (N_JOINTS * 2) + (N_JOINTS * 2)
    
    # Total Obs: Current (14) + Sequence (43*80) + Prev Gains (14)
    RL_OBS_DIM = (N_JOINTS * 2) + (_FEATS_PER_STEP * RNN_SEQ_LEN) + (N_JOINTS * 2)

    # RL Network
    ASAC_HIDDEN_DIMS = [256, 256, 256]
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    # Environment Settings
    MAX_EPISODE_STEPS = 2000
    MAX_ACTION_TORQUE = 87.0  # Clipping for safety
    MAX_JOINT_ERROR_TERMINATION = 2.0  # Rad
    WARM_UP_DURATION = 1.0  # Seconds
    
    # Normalization Statistics (Approximate)
    Q_MEAN = np.zeros(N_JOINTS, dtype=np.float32)
    Q_STD = np.ones(N_JOINTS, dtype=np.float32)
    QD_MEAN = np.zeros(N_JOINTS, dtype=np.float32)
    QD_STD = np.ones(N_JOINTS, dtype=np.float32)
    
    # IK Settings
    TRAJECTORY_CENTER = np.array([0.5, 0.0, 0.5])
    TRAJECTORY_SCALE = np.array([0.2, 0.2, 0.1])
    TRAJECTORY_FREQUENCY = 0.5
    
    IK_JACOBIAN_MAX_ITER = 50
    IK_JACOBIAN_DAMPING = 0.05
    IK_JACOBIAN_STEP_SIZE = 0.5
    IK_POSITION_TOLERANCE = 0.001
    IK_NULL_SPACE_GAIN = 0.1

# =============================================================================
# Training Hyperparameters
# =============================================================================
class TRAIN:
    GAMMA = 0.99
    BATCH_SIZE = 256
    BUFFER_SIZE = 1_000_000
    ACTOR_LR = 3e-4
    CRITIC_LR = 3e-4
    ALPHA_LR = 3e-4
    
    TOTAL_TIMESTEPS = 1_000_000
    WARMUP_STEPS = 10_000
    LOG_FREQ = 1000
    TRAIN_FREQUENCY = 1
    UTD_RATIO = 1  # Updates per Data
    GRAD_CLIP = 1.0
    EVAL_INTERVAL = 10_000

class SAC:
    INITIAL_ALPHA = 1.0
    TARGET_TAU = 0.005
    POLICY_DELAY = 2
    TARGET_NOISE = 0.2
    NOISE_CLIP = 0.5
    ALPHA_MIN = 0.05
    ALPHA_MAX = 20.0
    Q_CLIP = 1000.0

# =============================================================================
# Reward Function
# =============================================================================
class REWARD:
    # Weights
    W_POS = 1.0
    W_VEL = 0.1
    W_SMOOTH = 0.05
    W_GAIN_REG = 0.01
    
    # Scaling (Sensitivity)
    SCALE_POS = 10.0
    SCALE_VEL = 1.0
    
    # Clipping
    MIN_CLIP = -10.0
    MAX_CLIP = 10.0
    
    # Penalties
    PENALTY_DIVERGENCE = -10.0

# =============================================================================
# Trajectory Randomization
# =============================================================================
class TRAJ_RANDOM:
    CENTER_X = (0.4, 0.6)
    CENTER_Y = (-0.1, 0.1)
    SCALE_X = (0.1, 0.3)
    SCALE_Y = (0.1, 0.3)
    FREQ = (0.2, 0.8)