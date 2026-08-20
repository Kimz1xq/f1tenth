"""Shared map-server and AMCL launch for the physical F1TENTH car."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory('f1tenth_bringup')
    amcl_config = os.path.join(
        bringup_share, 'config', 'amcl_common.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('map_yaml'),
        DeclareLaunchArgument('base_frame_id', default_value='base_link'),
        DeclareLaunchArgument('odom_frame_id', default_value='odom'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': LaunchConfiguration('map_yaml'),
                'topic': 'map',
                'frame_id': 'map',
                'use_sim_time': False,
            }],
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[
                amcl_config,
                {
                    'base_frame_id': LaunchConfiguration('base_frame_id'),
                    'odom_frame_id': LaunchConfiguration('odom_frame_id'),
                    'scan_topic': LaunchConfiguration('scan_topic'),
                    'use_sim_time': False,
                    # The physical start pose is supplied from RViz, not a
                    # simulator-specific or persisted pose.
                    'always_reset_initial_pose': True,
                    'save_pose_rate': -1.0,
                },
            ],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'bond_timeout': 0.0,
                'node_names': ['map_server', 'amcl'],
            }],
        ),
    ])
