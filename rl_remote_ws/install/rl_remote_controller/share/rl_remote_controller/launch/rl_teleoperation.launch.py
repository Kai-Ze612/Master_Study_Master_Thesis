#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    """
    Simple launch file for teleoperation system
    """
    
    return LaunchDescription([
        # Local Robot Node
        Node(
            package='rl_remote_controller',
            executable='local',
            name='local_robot_controller',
            output='screen',
            parameters=[{
                'use_sim_time': False,
            }]
        ),
        
        # Remote Robot Node (with small delay to avoid conflicts)
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='rl_remote_controller',
                    executable='rl_remote',
                    name='remote_robot_controller',
                    output='screen',
                    parameters=[{
                        'use_sim_time': False,
                    }]
                )
            ]
        ),
    ])