# End-to-End Reinforcement Learning for Robust Teleoperation under Stochastic Delays

This repository contains the implementation, simulation environment, and experimental results of the Master's Thesis **"End-to-End Reinforcement Learning for Robust Teleoperation under Stochastic Delays"** conducted at the **Technical University of Munich (TUM)** *Munich Institute of Robotics and Machine Intelligence (MIRMI)*.

## 📖 Overview
This repository contains the implementation, simulation environment, and experimental results of the Master's Thesis **"End-to-End Reinforcement Learning for Robust Teleoperation under Stochastic Delays"** conducted at **TUM MIRMI**.

Unlike traditional **Two-Stage** approaches that decouple state prediction from control, this framework jointly optimizes a **Delay-Adaptive LSTM Encoder** and a **Soft Actor-Critic (SAC)** policy. By explicitly conditioning on the instantaneous delay magnitude and outputting direct torque commands, the system eliminates the performance ceiling caused by cascaded error propagation.

**Author:** **Kai-Ze Deng** (M.Sc. Robotics, Cognition and Intelligence)  
**Supervisor:** **Dr. Zewen Yang** (MIRMI, TUM)

## Repository Structure

The repository tracks the progressive development of the solution, containing the baselines and the final proposed framework.

## Repository Structure

The repository is organized as a ROS 2 workspace (`libfranka_ws`). Each major framework is contained within its own package, following a modular structure (Config, Nodes, Utils, and Core Algorithms).

```text
libfranka_ws/
├── src/
│   ├── E2E_Teleoperation/                 # [Proposed Method] Novel End-to-End LSTM-based Policy
│   │   ├── config/                        # Hyperparameter configurations
│   │   ├── E2E_RL/                        # Core Reinforcement Learning implementation
│   │   │   ├── sac_policy_network.py      # Network architecture (Actor-Critic + LSTM)
│   │   │   ├── sac_training_algorithm.py  # SAC algorithm logic
│   │   │   ├── training_env.py            # Gymnasium environment wrapper
│   │   │   ├── train_agent.py             # Main training entry point
│   │   │   ├── local_robot_simulator.py   # Leader robot physics/simulation
│   │   │   └── remote_robot_simulator.py  # Follower robot physics/simulation
│   │   ├── nodes/                         # ROS 2 Nodes for deployment
│   │   └── utils/                         # Shared utilities (delay simulator, IK solver)
│   │
│   ├── Model_Based_RL_Teleoperation/      # [Previous Iteration] Dynamics-aware LSTM + residual RL framework
│   │
│   ├── Hierarchical_RL_Teleoperation/     # [Previous Iteration] Multi-level RL architecture
│   │
│   ├── SBSP/                              # [Baseline] SOTA Model-Based RL Framework
│   │
│   ├── ASAC/                              # [Baseline] SOTA Model-Free RL Framework
│   │
│   ├── mujoco_ros_pkgs/                   # [Simulation] MuJoCo physics engine interface for ROS2
│   │
│   └── multipanda_ros2/                   # [Simulation] Franka Panda robot descriptions & scenes
│   |                                      # (Adapted from: [github.com/tenfoldpaper/multipanda_ros2]
```

## Installation and Usage
### Prerequisites
* Operating System: Ubuntu 22.04 (Jammy)
* ROS 2 Humble
* Python: 3.10+

### Install Python Libraries
```bash
pip3 install -r requirements.txt
```

### Install Simulation Environment (Multipanda)
Please follow the official installation guide from the Multipanda repository:
[https://github.com/tenfoldpaper/multipanda_ros2](https://github.com/tenfoldpaper/multipanda_ros2)

Ensure thay you can launch the default simulation scene before proceeding:
```bash
ros2 launch multipanda_bringup multipanda.launch.py
```

### Setup this repository
```bash
cd ~/libfranka_ws

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build workspace
colcon build

source install/setup.bash
```
