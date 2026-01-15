"""
Behavior Cloning (BC) Pre-training Script - ENHANCED
Includes:
1. Noise Injection (to fix Covariate Shift)
2. Domain Randomization (Trajectories & Delays)
3. Data Augmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random

from E2E_Teleoperation.E2E_RL.e2e_network import JointActor
import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
from E2E_Teleoperation.E2E_RL.leader_robot_simulator import TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_undelayed_expert_action(f_q, f_qd, target_q, target_qd):
    """
    Computes a PD control action based on the undelayed target state.
    Used as the 'expert' for behavioral cloning.
    """
    Kp = cfg.BC_EXPERT.KP
    Kd = cfg.BC_EXPERT.KD
    pos_err = target_q - f_q
    vel_err = target_qd - f_qd
    tau = (Kp * pos_err) + (Kd * vel_err)
    return np.clip(tau, -cfg.ROBOT.MAX_ACTION_TORQUE, cfg.ROBOT.MAX_ACTION_TORQUE)

def collect_data(steps_per_config, noise_level=0.0):
    """
    Collects data with domain randomization and noise injection.
    
    Args:
        noise_level (float): Std dev of Gaussian noise added to action to force drift.
                             The network learns to RECOVER from this drift.
    """
    obs_data = []
    act_data = []
    
    # 1. Randomize Configs to ensure robustness
    configs_to_run = [
        (ExperimentConfig.LOW_DELAY, TrajectoryType.FIGURE_8),
        (ExperimentConfig.HIGH_VARIANCE, TrajectoryType.FIGURE_8),
        (ExperimentConfig.LOW_DELAY, TrajectoryType.SQUARE),
        (ExperimentConfig.HIGH_VARIANCE, TrajectoryType.LISSAJOUS_COMPLEX),
    ]
    
    steps_per_sub_config = steps_per_config // len(configs_to_run)
    
    print(f">>> Collecting Data: Total Steps={steps_per_config} | Noise={noise_level}")

    for delay_conf, traj_type in configs_to_run:
        print(f"   -> Config: {delay_conf.name} | Traj: {traj_type.name}")
        
        # Initialize env with Random Trajectories enabled
        env = TeleoperationEnv(
            delay_config=delay_conf,
            trajectory_type=traj_type,
            randomize_trajectory=True, # CRITICAL: Randomize scales/centers
            render_mode=None
        )
        
        obs, info = env.reset()
        current_steps = 0
        
        pbar = tqdm(total=steps_per_sub_config, desc="Collecting", leave=False)
        
        while current_steps < steps_per_sub_config:
            # 1. Get Expert Action (What SHOULD we do?)
            # The expert looks at the TRUE target, not the delayed one
            true_leader_q = info['leader_q']
            # We estimate target velocity as 0 or use finite diff if available, 
            # but expert PD usually tracks current pos
            true_leader_qd = np.zeros(7) 
            
            expert_action = get_undelayed_expert_action(
                info['follower_q'], 
                np.zeros(7), # follower velocity approx
                true_leader_q, 
                true_leader_qd
            )
            
            # 2. Record (State, Expert_Action)
            # We record the state BEFORE we mess it up with noise
            obs_data.append(obs)
            act_data.append(expert_action)
            
            # 3. Inject Noise into the environment step
            # This causes the robot to drift, so the NEXT state will be "off-track"
            # The NEXT expert action will therefore be a "recovery action"
            noise = np.random.normal(0, noise_level, size=expert_action.shape)
            noisy_action = expert_action + noise
            noisy_action = np.clip(noisy_action, -cfg.ROBOT.MAX_ACTION_TORQUE, cfg.ROBOT.MAX_ACTION_TORQUE)
            
            next_obs, _, terminated, truncated, info = env.step(noisy_action)
            
            obs = next_obs
            current_steps += 1
            pbar.update(1)
            
            if terminated or truncated:
                obs, info = env.reset()
                
        pbar.close()
        env.close()
            
    return np.array(obs_data, dtype=np.float32), np.array(act_data, dtype=np.float32)

def train_bc():
    # 1. Collect Data with Noise (DART / Noise Injection)
    # We collect 80% with noise (to learn recovery) and 20% clean (for precision)
    
    print("--- Phase 1: Collecting Noisy Data (Learning Recovery) ---")
    X_noisy, Y_noisy = collect_data(steps_per_config=40_000, noise_level=2.0) # 2.0 Nm noise
    
    print("--- Phase 2: Collecting Clean Data (Learning Precision) ---")
    X_clean, Y_clean = collect_data(steps_per_config=10_000, noise_level=0.0)
    
    # Concatenate
    X_all = np.concatenate([X_noisy, X_clean], axis=0)
    Y_all = np.concatenate([Y_noisy, Y_clean], axis=0)
    
    print(f">>> Total Dataset Size: {X_all.shape[0]}")
    
    # 2. Setup Training
    actor = JointActor().to(DEVICE)
    optimizer = optim.Adam(actor.parameters(), lr=cfg.BC.LR)
    loss_fn = nn.MSELoss()
    
    dataset = TensorDataset(torch.from_numpy(X_all).to(DEVICE), torch.from_numpy(Y_all).to(DEVICE))
    loader = DataLoader(dataset, batch_size=cfg.BC.BATCH_SIZE, shuffle=True)
    
    print(f">>> Starting BC Training on {DEVICE}...")
    actor.train()
    
    # 3. Train Loop
    best_loss = float('inf')
    
    for epoch in range(cfg.BC.EPOCHS):
        total_loss = 0
        for batch_obs, batch_act in loader:
            optimizer.zero_grad()
            
            # Forward (we only care about the Actor head, not the prediction head for BC usually, 
            # but your network does both. We ignore pred output here or add auxiliary loss?)
            # For pure BC of action, we just minimize action error.
            mu, _, _, _ = actor(batch_obs)
            
            # Deterministic action for BC
            pred_action = torch.tanh(mu) * actor.action_scale
            
            loss = loss_fn(pred_action, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:02d} | MSE Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(actor.state_dict(), cfg.ROBOT.PRETRAINED_ACTOR_PATH)
            print(f"    >>> Model Saved!")

if __name__ == "__main__":
    train_bc()