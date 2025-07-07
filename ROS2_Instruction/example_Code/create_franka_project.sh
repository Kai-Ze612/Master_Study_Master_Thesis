#!/bin/bash

set -e  # Exit on any error

echo "Setting up Franka MuJoCo ROS2 Project for Master Thesis..."

# Define paths
MASTER_THESIS_PATH="/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis"
FRANKA_WS_PATH="$MASTER_THESIS_PATH/fr3_mujoco_ws"

# Create workspace structure
echo "Creating workspace structure..."
mkdir -p $FRANKA_WS_PATH/src
cd $FRANKA_WS_PATH/src

# Source ROS2
source /opt/ros/humble/setup.bash

# Create package
## Modify here if want to use CPP
echo "Creating ROS2 package..."
ros2 pkg create --build-type ament_python franka_mujoco_controller \
  --dependencies rclpy sensor_msgs std_msgs geometry_msgs

# Create additional directories
cd franka_mujoco_controller
mkdir -p launch config models/meshes

# Make Python file executable
chmod +x franka_mujoco_controller/__init__.py

echo " Project structure created successfully!"