from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    default_params_file = os.path.join(
        get_package_share_directory('planning'),
        'config',
        'params.yaml'
    )

    params_file = LaunchConfiguration('params_file')
    waypoint_csv = LaunchConfiguration('waypoint_csv')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to planning params.yaml'
        ),
        DeclareLaunchArgument(
            'waypoint_csv',
            default_value='/sim_ws/src/planning/waypoints/track02_raceline_safe.csv',
            description='Optional waypoint/raceline CSV override'
        ),

        Node(
            package='planning',
            executable='waypoint_planner_node',
            name='waypoint_planner_node',
            output='screen',
            parameters=[
                params_file,
                {'waypoint_csv': waypoint_csv}
            ]
        )
    ])
