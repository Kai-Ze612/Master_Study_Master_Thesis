## ROS launch
ROS launch is a Python script that defines how to start and coordinate multiple nodes, processes, and parameters in a robotics system. Using ROS launch can avoid using multiple terminal windows and manually starting each node individually.

ROS2 launch file is typically written in Python file and use the `launch` package API to define the system configuration.

```bash
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction

def launch():
    """
    Launch file that explicitly uses bash (not sh) and proper syntax
    """
   
    return LaunchDescription([
        # Start controller with explicit bash shell
        TimerAction(
            ## Add a time delay before starting the controller
            period=2.0,
            
            ## add series of actions to be excuted
            actions=[
                ExecuteProcess(
                    cmd=[
                        '/bin/bash',
                        '-c',
                        'cd /media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis && '
                        'source setup_environment.sh && '
                        'echo "Environment loaded successfully" && '
                        'ros2 run franka_mujoco_controller mujoco_controller'
                    ],
                    ## Give process name
                    name='mujoco_controller',
				    ## Give the output on the terminal
                    output='screen'
                    # Remove shell=True to avoid shell interpretation issues
                )
            ]
        ),
        
        # Send start command
        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'topic', 'pub', '--once',
                        '/start_push', 'std_msgs/msg/String',
                        '{data: start}'  # Fixed quotes
                    ],
                    name='start_command',
                    output='screen'
                )
            ]
        )
    ])
```

## Cmake.txt
After launch.py, we need an cmake.txt file to find launch file by package name

```bash
# Install launch files so they can be found by package name
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME} )
```