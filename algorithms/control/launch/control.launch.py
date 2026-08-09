from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    default_params_file = os.path.join(
        get_package_share_directory('control'),
        'config',
        'params.yaml'
    )

    params_file = LaunchConfiguration('params_file')
    mpc_params_file = LaunchConfiguration('mpc_params_file')
    drive_mode = LaunchConfiguration('drive_mode')
    controller = LaunchConfiguration('controller')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to control params.yaml'
        ),

        DeclareLaunchArgument(
            'drive_mode',
            default_value='sim',
            description='Drive output mode: sim or real'
        ),
        DeclareLaunchArgument(
            'controller',
            default_value='pure_pursuit',
            description='Controller: pure_pursuit or mpc'
        ),
        DeclareLaunchArgument(
            'mpc_params_file',
            default_value=os.path.join(
                get_package_share_directory('control'), 'config',
                'mpc_params.yaml'),
            description='Path to Linear MPC params.yaml'
        ),

        Node(
            package='control',
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            output='screen',
            parameters=[
                params_file,
                {'drive_mode': drive_mode}
            ],
            condition=IfCondition(PythonExpression([
                "'", controller, "' == 'pure_pursuit'"
            ]))
        ),

        Node(
            package='control',
            executable='linear_mpc_node',
            name='linear_mpc_node',
            output='screen',
            parameters=[mpc_params_file],
            condition=IfCondition(PythonExpression([
                "'", controller, "' == 'mpc'"
            ]))
        )
    ])
