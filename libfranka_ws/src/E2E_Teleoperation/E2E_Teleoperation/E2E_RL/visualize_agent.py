import torch
import numpy as np
import collections
from collections import deque
from pathlib import Path

# Project Imports
import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.e2e_network import JointActor
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = cfg.PRETRAINED_DIR / "best_checkpoint_best.pth"

def verify_fixed():
    print(f"[Info] Loading Model: {CHECKPOINT_PATH}")
    
    if not CHECKPOINT_PATH.exists():
        print("[Error] Checkpoint not found.")
        return

    # 1. Load Checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if 'norm' in checkpoint:
        stats = checkpoint['norm']
        object.__setattr__(cfg.ROBOT, 'Q_MEAN', stats['q_mean'])
        object.__setattr__(cfg.ROBOT, 'Q_STD', stats['q_std'])
        object.__setattr__(cfg.ROBOT, 'QD_MEAN', stats['qd_mean'])
        object.__setattr__(cfg.ROBOT, 'QD_STD', stats['qd_std'])

    actor = JointActor().to(DEVICE)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    
    # 2. Env
    env = TeleoperationEnv(
        delay_config=ExperimentConfig.HIGH_VARIANCE,
        trajectory_type=TrajectoryType.FIGURE_8,
        randomize_trajectory=False, 
        render_mode=None
    )
    obs, info = env.reset()
    
    # 3. Buffers
    FIXED_DELAY_STEPS = 15 # ~150ms
    delay_buffer = deque(maxlen=FIXED_DELAY_STEPS + 1)
    lstm_history = deque(maxlen=cfg.ROBOT.RNN_SEQ_LEN)
    
    # Pre-fill
    for _ in range(cfg.ROBOT.RNN_SEQ_LEN):
        lstm_history.append(np.zeros(15, dtype=np.float32))

    Kp = np.array([150.0]*4 + [50.0]*2 + [20.0])
    Kd = np.array([10.0]*4 + [5.0]*2 + [2.0])

    print("\n" + "="*90)
    print(f"{'Step':<6} | {'Input Delay':<12} | {'J6 True':<10} | {'J6 Pred':<10} | {'Error':<10} | {'Status'}")
    print("="*90)

    for i in range(1000):
        # --- A. Data Prep ---
        true_q = info['leader_q']
        true_qd = info['leader_qd']
        
        delay_buffer.append((true_q, true_qd))
        
        if len(delay_buffer) < FIXED_DELAY_STEPS:
            obs_q, obs_qd = delay_buffer[0]
            curr_delay_sec = 0.0
        else:
            obs_q, obs_qd = delay_buffer.popleft()
            curr_delay_sec = FIXED_DELAY_STEPS * cfg.DT
        
        # --- B. Normalization (THE FIX) ---
        q_norm = (obs_q - cfg.ROBOT.Q_MEAN) / cfg.ROBOT.Q_STD
        qd_norm = (obs_qd - cfg.ROBOT.QD_MEAN) / cfg.ROBOT.QD_STD
        
        # [CRITICAL FIX] 
        # We REMOVE the multiplication by cfg.ROBOT.DELAY_INPUT_NORM_FACTOR
        # We suspect the factor was making the input huge (e.g. 150.0).
        # We pass Seconds directly (0.15), which is a safe neural net input.
        delay_input_val = curr_delay_sec 
        delay_norm = np.array([delay_input_val], dtype=np.float32)
        
        input_vec = np.concatenate([q_norm, qd_norm, delay_norm]).astype(np.float32)
        lstm_history.append(input_vec)
        
        # --- C. Prediction ---
        history_tensor = torch.tensor(np.array(lstm_history), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            _, pred_state_norm, _ = actor.base_encoder(history_tensor)
            pred_state = pred_state_norm.cpu().numpy()[0]
            pred_q = (pred_state[:7] * cfg.ROBOT.Q_STD) + cfg.ROBOT.Q_MEAN
            
        # --- D. Logging ---
        if i % 10 == 0:
            j_idx = 5 
            err = true_q[j_idx] - pred_q[j_idx]
            
            status = "OK"
            color = "\033[92m" 
            if abs(err) > 0.05: status, color = "DRIFT", "\033[93m"
            if abs(err) > 0.2: status, color = "BAD", "\033[91m"
            
            # We print 'delay_input_val' to verify it is small (0.15) and not huge (150)
            print(f"{i:<6} | {delay_input_val:10.4f}   | {true_q[j_idx]:8.4f}   | {pred_q[j_idx]:8.4f}   | {color}{err:8.4f}\033[0m   | {status}")

        # --- E. Physics ---
        curr_q = info['follower_q']
        curr_qd = info['follower_qd']
        tau = Kp * (true_q - curr_q) + Kd * (true_qd - curr_qd)
        tau = np.clip(tau, -cfg.ROBOT.MAX_ACTION_TORQUE, cfg.ROBOT.MAX_ACTION_TORQUE)
        
        obs, _, term, trunc, info = env.step(tau)
        
        if term or trunc:
            obs, info = env.reset()
            lstm_history.clear()
            for _ in range(cfg.ROBOT.RNN_SEQ_LEN):
                lstm_history.append(np.zeros(15, dtype=np.float32))
            delay_buffer.clear()

if __name__ == "__main__":
    verify_fixed()