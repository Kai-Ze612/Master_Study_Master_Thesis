"""
Behavior Cloning (BC) Pre-training Script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from pathlib import Path

from E2E_Teleoperation.E2E_RL.e2e_network import JointActor
import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_undelayed_expert_action(f_q, f_qd, target_q, target_qd):
    """
    Computes a PD control action based on the undelayed target state.
    Used as the 'expert' for behavioral cloning.
    """
    Kp = 40.0 
    Kd = 5.0
    pos_err = target_q - f_q
    vel_err = target_qd - f_qd
    tau = (Kp * pos_err) + (Kd * vel_err)
    # Use config for max torque limits
    return np.clip(tau, -cfg.ROBOT.MAX_ACTION_TORQUE, cfg.ROBOT.MAX_ACTION_TORQUE)

def collect_data(env):
    """
    Collects state-action pairs from the environment using the expert controller.
    """
    print(f">>> Collecting {cfg.BC.STEPS_TO_COLLECT} samples from Undelayed Expert...")
    obs_data = []
    act_data = []
    
    obs, info = env.reset()
    
    # Use configured collection steps
    for _ in tqdm(range(cfg.BC.STEPS_TO_COLLECT)):
        l_q = info['leader_q']
        f_q = env.follower.get_joint_state()[0] 
        f_qd = env.follower.get_joint_state()[1]
        
        # Leader velocity is assumed 0 for the static target approximation in simple BC,
        # or you can extract it from the simulator if available.
        l_qd_raw = np.zeros(7) 

        expert_tau = get_undelayed_expert_action(f_q, f_qd, l_q, l_qd_raw)
        
        obs_data.append(obs)
        act_data.append(expert_tau)
        
        obs, _, terminated, truncated, info = env.step(expert_tau)
        
        if terminated or truncated:
            obs, info = env.reset()
            
    return np.array(obs_data, dtype=np.float32), np.array(act_data, dtype=np.float32)

def train():
    env = TeleoperationEnv() 
    actor = JointActor().to(DEVICE)
    
    # Use configured Learning Rate
    optimizer = optim.Adam(actor.parameters(), lr=cfg.BC.LR)
    loss_fn = nn.MSELoss()
    
    X, Y = collect_data(env)
    
    dataset = TensorDataset(torch.from_numpy(X).to(DEVICE), torch.from_numpy(Y).to(DEVICE))
    
    # Use configured Batch Size
    loader = DataLoader(dataset, batch_size=cfg.BC.BATCH_SIZE, shuffle=True)
    
    print(f">>> Starting Training on {DEVICE}...")
    actor.train()
    
    # Use configured Epochs
    for epoch in range(cfg.BC.EPOCHS):
        total_loss = 0
        for batch_obs, batch_act in loader:
            optimizer.zero_grad()
            mu, _, _, _ = actor(batch_obs)
            pred_action = torch.tanh(mu) * actor.action_scale
            loss = loss_fn(pred_action, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:02d} | MSE Loss: {avg_loss:.4f}")
    
    # Use configured Save Path
    save_path = cfg.BC.SAVE_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), save_path)
    print(f">>> Pre-training Complete. Saved to {save_path}")
    env.close()

if __name__ == "__main__":
    train()