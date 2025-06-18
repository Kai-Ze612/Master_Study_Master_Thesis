from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction

def generate_launch_description():
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
                        'ros2 run franka_mujoco_controller teleoperation_2_local'
                    ],
                    name='teleoperation',
                    output='screen'
                    # Remove shell=True to avoid shell interpretation issues
                )
            ]
        ),

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
                        'ros2 run franka_mujoco_controller teleoperation_2_remote'
                    ],
                    name='teleoperation',
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
                        
                        ## This will start ROS2
                        '{data: start}'  # Fixed quotes
                    ],
                    name='start_command',
                    output='screen'
                )
            ]
        )
    ])