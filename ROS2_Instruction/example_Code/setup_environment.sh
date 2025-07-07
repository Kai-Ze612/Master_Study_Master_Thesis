# This script serves as a setup environment for running ROS2 project.
# Run every time you want to work on the project.

#!/bin/bash

# Master Thesis ROS2 Environment Setup
export MASTER_THESIS_PATH="//media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis"
export FRANKA_WS_PATH="$MASTER_THESIS_PATH/rl_remote_ws"

## Activate conda environment
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate master_thesis

# Source ROS2
source /opt/ros/humble/setup.bash

# Source workspace if built
source "$FRANKA_WS_PATH/install/setup.bash"

# MuJoCo environment
export MUJOCO_PATH=/opt/mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/mujoco/lib

# Change to workspace directory
cd "$FRANKA_WS_PATH"
echo "Environment Ready!"
echo "Workspace: $FRANKA_WS_PATH"