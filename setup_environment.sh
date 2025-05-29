## This script serves as a setup for the Master Thesis ROS2 environment.
## Run every time you want to work on the project.

#!/bin/bash

# Master Thesis ROS2 Environment Setup
export MASTER_THESIS_PATH="/media/kai/Kai_Backup/Master_Study/Practical_Project/Practical Courses/Master_Thesis/Master_Study_Master_Thesis"
export FRANKA_WS_PATH="$MASTER_THESIS_PATH/fr3_mujoco_ws"

## Activate conda environment
conda activate master_thesis

# Source ROS2
source /opt/ros/humble/setup.bash

# Source workspace if built
if [ -f "$FRANKA_WS_PATH/install/setup.bash" ]; then
    source $FRANKA_WS_PATH/install/setup.bash
    echo " Franka MuJoCo workspace sourced"
else
    echo "⚠️  Workspace not built yet. Run 'colcon build' first."
fi

# MuJoCo environment
export MUJOCO_PATH=/opt/mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/mujoco/lib

# Change to workspace directory
cd $FRANKA_WS_PATH

echo " Master Thesis Environment Ready!"
echo " Workspace: $FRANKA_WS_PATH"
echo " Environment: franka_ros2"