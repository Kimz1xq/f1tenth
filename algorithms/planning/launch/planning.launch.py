from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def _as_bool(value):
    return str(value).lower() in ('1', 'true', 'yes', 'on')


def _launch_setup(context):
    params_file = LaunchConfiguration('params_file').perform(context)
    local_planner = _as_bool(
        LaunchConfiguration('local_planner').perform(context))
    common_overrides = {
        'odom_topic': LaunchConfiguration('odom_topic').perform(context),
        'base_frame_id': LaunchConfiguration('base_frame_id').perform(context),
    }
    waypoint_overrides = {
        'waypoint_csv': LaunchConfiguration('waypoint_csv').perform(context),
    }
    if not local_planner:
        waypoint_overrides['path_topic'] = '/planning/path'

    actions = [Node(
        package='planning',
        executable='waypoint_planner_node',
        name='waypoint_planner_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file').perform(context),
            waypoint_overrides,
        ],
    )]
    if local_planner:
        actions.append(Node(
            package='planning',
            executable='local_obstacle_planner_node',
            name='local_obstacle_planner_node',
            output='screen',
            parameters=[params_file, common_overrides],
        ))
    return actions


def generate_launch_description():
    package_share = get_package_share_directory('planning')
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(package_share, 'config', 'params.yaml'),
            description='Path to planning params.yaml'
        ),
        DeclareLaunchArgument(
            'waypoint_csv',
            default_value=os.path.join(
                package_share, 'waypoints', 'track03_raceline.csv'),
            description='Optional waypoint/raceline CSV override'
        ),
        DeclareLaunchArgument('local_planner', default_value='true'),
        DeclareLaunchArgument(
            'odom_topic', default_value='/ego_racecar/odom'),
        DeclareLaunchArgument(
            'base_frame_id', default_value='ego_racecar/base_link'),
        OpaqueFunction(function=_launch_setup),
    ])
