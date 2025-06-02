# fr3_mujoco_ws

## How to Run

Set up the environment
```bash
source /media/kai/Kai_Backup/Study/Master_Thesis/My_Master_Thesis/Master_Study_Master_Thesis/setup_environment.sh
```

Direct to the folder

```bash
cd /media/kai/Kai_Backup/Master_Study/Practical_Project/Practical_Courses/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws
```

Build the project  
This will build the package name: franka_mujoco_controller

```bash
colcon build --packages-select franka_mujoco_controller
```

ros2 run <package_name> <executable_name> [arguments]

```bash
ros2 run franka_mujoco_controller mujoco_controller
```
Start
```bash
ros2 topic pub /start_push std_msgs/msg/String "{data: 'start'}"
```

## Build the project

1. setup_environment.sh is used to build the necessary environment for the project
2. clone FR3 ROS2 and MuJoCo repo
3. create package structure

```bash
cd /media/kai/Kai_Backup/Master_Study/Practical_Project/Practical_Courses/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws
```

```bash
ros2 pkg create --build-type ament_python franka_mujoco_controller \
  --dependencies rclpy sensor_msgs std_msgs geometry_msgs
```

1. update the setup.py under the path

```bash
cd /media/kai/Kai_Backup/Master_Study/Practical_Project/Practical_Courses/Master_Thesis/Master_Study_Master_Thesis/fr3_mujoco_ws/src/franka_mujoco_controller/setup.py
```

1. create the main python controller.py file