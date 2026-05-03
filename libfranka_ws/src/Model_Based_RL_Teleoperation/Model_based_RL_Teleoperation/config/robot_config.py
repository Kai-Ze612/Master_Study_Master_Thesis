"""
Shared Robot configuration settings
"""

import numpy as np

######################################
# File Paths
######################################
# DEFAULT_MUJOCO_MODEL_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/multipanda_ros2/franka_description/mujoco/franka/scene.xml"
RL_MODEL_PATH = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent_autoregressive/rl_training_output/ModelBasedSAC_FULL_RANGE_COVER_figure_8_20251203_220547/best_policy.pth"
# LSTM_MODEL_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent_autoregressive/lstm_training_output/LSTM_AR_FULL_RANGE_COVER_20251202_215350/best_model.pth"
######################################
DEFAULT_MUJOCO_MODEL_PATH = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/multipanda_ros2/franka_description/mujoco/franka/scene.xml"
LSTM_MODEL_PATH = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent_autoregressive/lstm_training_output/LSTM_AR_FULL_RANGE_COVER_20251202_215350/best_model.pth"
######################################
CHECKPOINT_DIR_RL = "./rl_training_output"
CHECKPOINT_DIR_LSTM = "./lstm_training_output"

######################################
# Franka Panda Robot Parameters
######################################
N_JOINTS = 7
EE_BODY_NAME = "panda_hand"
TCP_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float32)

JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.5445, -3.0159], dtype=np.float32)
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 4.5169, 3.0159], dtype=np.float32)
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)

MAX_TORQUE_COMPENSATION = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)

INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.5708, 0.785], dtype=np.float32)
JOINT_LIMIT_MARGIN = 0.05  # Margin to avoid hitting joint limits

WARM_UP_DURATION = 1 # sec (before starting moving)
NO_DELAY_DURATION = 1 # sec (before starting delay simulation)

MAX_JOINT_CHANGE_PER_STEP = 0.01

Q_MEAN = np.array([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78], dtype=np.float32)
Q_STD  = np.array([1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32) 
QD_MEAN = np.zeros(7, dtype=np.float32)
QD_STD  = np.ones(7, dtype=np.float32) * 2.0 

# Smoothing factor for LSTM output (0.0 = frozen, 1.0 = raw output)
PREDICTION_EMA_ALPHA = 0.4

######################################
# Control Parameters
######################################
DEFAULT_CONTROL_FREQ = 250
DEFAULT_PUBLISH_FREQ = 250

DEFAULT_KP_REMOTE = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
DEFAULT_KD_REMOTE = np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0])

######################################
# IK Solver Parameterss
######################################
# 1. Tolerances & Iterations
IK_POSITION_TOLERANCE = 0.01 # meters
IK_JACOBIAN_MAX_ITER = 200
IK_OPTIMIZATION_MAX_ITER = 100

# 2. Damping & Step
IK_JACOBIAN_STEP_SIZE = 0.01 # Larger = Faster but less stable
IK_JACOBIAN_DAMPING = 0.1    # Larger = More stable but less responsive

IK_NULL_SPACE_GAIN = 0.5

######################################
# Trajectory Generation Parameters
######################################
TRAJECTORY_CENTER = np.array([0.3, 0, 0.5], dtype=np.float32)
TRAJECTORY_SCALE = np.array([0.2, 0.2, 0.02], dtype=np.float32)
TRAJECTORY_FREQUENCY = 0.1  # Hz

######################################
# pre-trained LSTM hyperparameters
######################################
ESTIMATOR_LEARNING_RATE = 5e-4
ESTIMATOR_BATCH_SIZE = 256
ESTIMATOR_BUFFER_SIZE = 200000
ESTIMATOR_TOTAL_UPDATES = 500000
ESTIMATOR_VAL_STEPS = 5000
ESTIMATOR_VAL_FREQ = 5000
ESTIMATOR_PATIENCE = 10
ESTIMATOR_LR_PATIENCE = 5

ESTIMATOR_PREDICTION_HORIZON = 10
MAX_ESTIMATOR_PREDICTION_HORIZON = 50

RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 3
RNN_SEQUENCE_LENGTH = 150 # Input sequence for LSTM

DELAY_INPUT_NORM_FACTOR = 100.0
TARGET_DELTA_SCALE = 10.0

AUGMENTATION_STRIDE = 4
DT = 1.0 / DEFAULT_CONTROL_FREQ

ESTIMATOR_STATE_DIM = 15  # 7 q + 7 qd + 1 delay
ESTIMATOR_OUTPUT_DIM = 14  # 7 predicted q + 7 predicted qd

MAX_AR_STEPS = int(240 / (1/DEFAULT_CONTROL_FREQ * 1000) + 5)

######################################
# RL Environment Parameters
######################################
MAX_JOINT_ERROR_TERMINATION = 1.0  # radians (legacy, retained for compatibility)

REMOTE_HISTORY_LEN = 5

OBS_DIM = (
    N_JOINTS +                              # remote_q: 7
    N_JOINTS +                              # remote_qd: 7
    REMOTE_HISTORY_LEN * (N_JOINTS) +       # remote_q_history: 5 * 7
    REMOTE_HISTORY_LEN * (N_JOINTS) +       # remote_qd_history: 5 * 7
    N_JOINTS +                              # predicted_q : 7
    N_JOINTS +                              # predicted_qd: 7
    N_JOINTS +                              # error_q: 7
    N_JOINTS +                                # error_qd: 7
    1                                        # current_delay: 1
)
# Total dim: 113

######################################
# SAC Hyperparameters
######################################
# SAC MLP architecture
SAC_MLP_HIDDEN_DIMS = [512, 256]
SAC_ACTIVATION = 'relu'

# Learning Rates
SAC_LEARNING_RATE = 3e-5        # LR for Actor and Critic
ALPHA_LEARNING_RATE = 3e-4      # LR for temperature auto-tuning

# SAC Parameters
SAC_GAMMA = 0.99
SAC_TAU = 0.005
SAC_TARGET_ENTROPY = 'auto'

# Action distribution numerical stability
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Training Schedule
SAC_BUFFER_SIZE = 1_000_000     # Max size of replay buffer
# SAC_BATCH_SIZE = 256            # batch size of gradient updates

# Training Schedule
# SAC_START_STEPS = 5000          # Number of random exploration steps (before learning)
SAC_UPDATES_PER_STEP = 1.0       # Number of SAC updates per env step
SAC_TOTAL_TIMESTEPS = 3_000_000  # Total training timesteps

SAC_START_STEPS = 5000        # Was 5000
SAC_VAL_FREQ = 5000           # Was 5000
SAC_BATCH_SIZE = 128

# Validation and Early Stopping
# SAC_VAL_FREQ = 5000
SAC_VAL_EPISODES = 3
SAC_EARLY_STOPPING_PATIENCE = 30

######################################
# Reward Function Configuration
######################################
TRACKING_ERROR_SCALE = 10       # Position tracking weight (lambda_p)
VELOCITY_ERROR_SCALE = 5        # Velocity tracking weight (lambda_v)
ACTION_PENALTY_WEIGHT = 0.01    # Action regularization weight (lambda_a)

# === Safety penalty constants (Modified: per-step r_safe per paper Eq. 18) ===
SAFETY_PENALTY_PER_STEP = -5.0       # Per-violation penalty (rho); scaled from paper -100 for stability
SAFETY_TRACK_THRESHOLD = 0.5         # rad, tracking error trigger threshold (epsilon_track)
SAFETY_PRED_THRESHOLD = 0.3          # rad, prediction divergence trigger threshold (epsilon_est)

# Reward clipping bounds (Modified: prevent variance explosion from r_safe)
REWARD_CLIP_MIN = -20.0              # Clip floor for stable critic learning
REWARD_CLIP_MAX = 5.0                # Clip ceiling (rewards typically negative)
# === End modification ===

######################################
# Environment Settings
######################################
NUM_ENVIRONMENTS = 1
MAX_EPISODE_STEPS = 4200


######################################
# Logging and Checkpointing
######################################
LOG_FREQ = 100   # Log metrics every N *env steps*
SAVE_FREQ = 1000  # Save checkpoint every N *env steps*

######################################
# Deployment Parameters
######################################

MAX_INFERENCE_TIME = 0.9 * (1.0 / DEFAULT_CONTROL_FREQ)  # 90% of control cycle time for safety

DEPLOYMENT_HISTORY_BUFFER_SIZE = 200  # Must be > max_delay_steps + RNN sequence length
