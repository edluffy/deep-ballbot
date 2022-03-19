import os

from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

def generate_launch_description():
    deepbb_bringup_path = os.path.join(get_package_share_path('deepbb_bringup'))
    gazebo_path = os.path.join(get_package_share_path('gazebo_ros'))

    return LaunchDescription([

        # Launch files
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                deepbb_bringup_path, 'launch', 'model.launch.py')]),
            launch_arguments={'use_sim_time': 'true'}.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                gazebo_path, 'launch', 'gazebo.launch.py')]),
        ),

        # Args
        DeclareLaunchArgument(
            'use_gui',
            default_value='true',
            choices=['true', 'false']
        ),

        # Nodes
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            condition=UnlessCondition(LaunchConfiguration('use_gui'))
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            condition=IfCondition(LaunchConfiguration('use_gui'))
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'bot'],
            output='screen'
        )
    ])
