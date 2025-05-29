#!/bin/bash

set -e  # Exit on any error

echo "Setting up Franka MuJoCo ROS2 Project for Master Thesis..."

# Define paths
MASTER_THESIS_PATH="/media/kai/Kai_Backup/Master_Study/Practical_Project/Practical Courses/Master_Thesis/Master_Study_Master_Thesis"
WS_PATH="$MASTER_THESIS_PATH/fr3_mujoco_ws"

# Create workspace structure
echo "Creating workspace structure..."
mkdir -p $WS_PATH/src
cd $WS_PATH/src

# Source ROS2
source /opt/ros/humble/setup.bash

# Create package
echo "Creating ROS2 package..."
ros2 pkg create --build-type ament_python franka_mujoco_controller \
  --dependencies rclpy sensor_msgs std_msgs geometry_msgs

# Create additional directories
cd franka_mujoco_controller
mkdir -p launch config models/meshes

# Make Python file executable
chmod +x franka_mujoco_controller/__init__.py

echo " Project structure created successfully!"
echo " Location: $WS_PATH"
echo ""
echo "Next steps:"
echo "1. Add your Python controller code to: franka_mujoco_controller/mujoco_controller.py"
echo "2. Add your MuJoCo model to: models/franka_fr3.xml"
echo "3. Run: source $MASTER_THESIS_PATH/setup_environment.sh"
echo "4. Build: colcon build"
echo "5. Run: ros2 run franka_mujoco_controller mujoco_controller"