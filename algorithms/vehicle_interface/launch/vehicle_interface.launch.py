from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    default_params_file = os.path.join(
        get_package_share_directory('vehicle_interface'),
        'config',
        'params.yaml'
    )

    params_file = LaunchConfiguration('params_file')
    drive_mode = LaunchConfiguration('drive_mode')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to vehicle_interface params.yaml'
        ),

        DeclareLaunchArgument(
            'drive_mode',
            default_value='sim',
            description='Output mode: sim or real'
        ),

        Node(
            package='vehicle_interface',
            executable='vehicle_interface_node',
            name='vehicle_interface_node',
            output='screen',
            parameters=[
                params_file,
                {'drive_mode': drive_mode}
            ]
        )
    ])
