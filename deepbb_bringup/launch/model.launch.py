import os
import xacro

from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

def generate_launch_description():
    deepbb_description_path = os.path.join(get_package_share_path('deepbb_description'))
    xacro_path = os.path.join(deepbb_description_path, 'urdf', 'deep_ballbot.xacro')
    default_rviz_config_path = os.path.join(deepbb_description_path, 'rviz', 'urdf.rviz')

    return LaunchDescription([

        # Args
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            choices=['true', 'false']
        ),
        DeclareLaunchArgument(
            'rvizconfig',
            default_value=str(default_rviz_config_path),
        ),

        # Nodes
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': xacro.process_file(xacro_path).toxml(),
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rvizconfig')],
        )
    ])
