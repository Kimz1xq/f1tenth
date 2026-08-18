import os
import math
import re

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _parse_dynamic_speed(profile_name):
    """Return m/s encoded by speed_<value>, or None for a named profile."""
    if not profile_name.startswith('speed_'):
        return None

    encoded = profile_name[len('speed_'):]
    if re.fullmatch(r'\d+(?:\.\d+)?', encoded):
        numeric = encoded
    elif re.fullmatch(r'\d+_\d+', encoded):
        # Keep compatibility with shell-friendly names such as speed_0_85.
        numeric = encoded.replace('_', '.', 1)
    else:
        raise RuntimeError(
            f'Invalid dynamic MPC speed {profile_name!r}; '
            'use speed_0.85, speed_1.2, or speed_2.')

    speed = float(numeric)
    if not math.isfinite(speed) or speed <= 0.0 or speed > 5.5:
        raise RuntimeError(
            f'Controller speed must be greater than 0 and at most 5.5 m/s; '
            f'got {speed!r}.')
    return speed


def _as_bool(value):
    return str(value).lower() in ('1', 'true', 'yes', 'on')


def _launch_setup(context):
    controller = LaunchConfiguration('controller').perform(context)
    if controller == 'none':
        return [LogInfo(msg='Controller disabled (controller:=none)')]

    if controller == 'pure_pursuit':
        return [Node(
            package='control',
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file').perform(context),
                {'drive_mode': LaunchConfiguration('drive_mode').perform(context)},
            ],
        )]

    if controller in ('unicorn_l1', 'unicorn_l1_dynamic'):
        use_dynamic_speed_limit = controller == 'unicorn_l1_dynamic'
        profile_name = LaunchConfiguration('mpc_profile').perform(context)
        requested_speed = _parse_dynamic_speed(profile_name)
        if requested_speed is None:
            raise RuntimeError(
                'UNICORN L1 requires mpc_profile:=speed_<m/s> '
                '(for example speed_1.0).')
        min_reference_speed = min(
            requested_speed,
            max(0.20, min(0.45, requested_speed * 0.40)),
        )
        avoidance_value = LaunchConfiguration(
            'avoidance_speed_limit').perform(context)
        avoidance_speed_limit = (
            requested_speed if avoidance_value == 'auto'
            else float(avoidance_value))
        if avoidance_speed_limit <= 0.0:
            raise RuntimeError(
                'avoidance_speed_limit must be auto or a positive m/s value')
        parameters = {
            'enabled': _as_bool(
                LaunchConfiguration('enabled').perform(context)),
            'global_frame_id': LaunchConfiguration(
                'global_frame_id').perform(context),
            'base_frame_id': LaunchConfiguration(
                'base_frame_id').perform(context),
            'odom_topic': LaunchConfiguration('odom_topic').perform(context),
            'drive_topic': LaunchConfiguration('drive_topic').perform(context),
            'collision_topic': LaunchConfiguration(
                'collision_topic').perform(context),
            'emergency_stop_topic': LaunchConfiguration(
                'emergency_stop_topic').perform(context),
            'target_speed': requested_speed,
            'max_speed': requested_speed,
            'max_lateral_acceleration': float(LaunchConfiguration(
                'max_lateral_acceleration').perform(context)),
            'max_longitudinal_acceleration': float(LaunchConfiguration(
                'max_longitudinal_acceleration').perform(context)),
            'max_longitudinal_deceleration': float(LaunchConfiguration(
                'max_longitudinal_deceleration').perform(context)),
            'avoidance_speed_limit': avoidance_speed_limit,
            'use_dynamic_speed_limit': use_dynamic_speed_limit,
            'min_reference_speed': min_reference_speed,
            'min_command_speed': float(LaunchConfiguration(
                'min_command_speed').perform(context)),
        }
        return [
            LogInfo(msg=(
                f'Controller={controller} speed={requested_speed:.2f}m/s '
                f'corner_min={min_reference_speed:.2f}m/s')),
            Node(
                package='control',
                executable='unicorn_l1_node',
                name='unicorn_l1_node',
                output='screen',
                parameters=[parameters],
            ),
        ]

    if controller != 'mpc':
        raise RuntimeError(
            f'Unknown controller {controller!r}; use none, pure_pursuit, '
            'unicorn_l1, unicorn_l1_dynamic, or mpc.')

    config_path = LaunchConfiguration('mpc_params_file').perform(context)
    profile_name = LaunchConfiguration('mpc_profile').perform(context)
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    profiles = config.get('profiles', {})
    requested_speed = _parse_dynamic_speed(profile_name)
    if requested_speed is not None:
        parameters = dict(config.get('common', {}))
        parameters.update(config.get('speed_template', {}))
        parameters['target_speed'] = requested_speed
        parameters['max_speed'] = requested_speed
        parameters['min_reference_speed'] = min(
            requested_speed,
            max(0.20, min(0.45, requested_speed * 0.40)),
        )
        selection_log = (
            f'Controller=mpc dynamic_speed={requested_speed:.2f}m/s '
            f'corner_min={parameters["min_reference_speed"]:.2f}m/s')
    elif profile_name in profiles:
        parameters = dict(config.get('common', {}))
        parameters.update(profiles[profile_name])
        selection_log = f'Controller=mpc profile={profile_name}'
    else:
        available = ', '.join(sorted(profiles))
        raise RuntimeError(
            f'Unknown MPC profile {profile_name!r}; use speed_<m/s> '
            f'(for example speed_0.85 or speed_2), or: {available}')

    parameters.update({
        'enabled': _as_bool(LaunchConfiguration('enabled').perform(context)),
        'global_frame_id': LaunchConfiguration(
            'global_frame_id').perform(context),
        'base_frame_id': LaunchConfiguration(
            'base_frame_id').perform(context),
        'odom_topic': LaunchConfiguration('odom_topic').perform(context),
        'drive_topic': LaunchConfiguration('drive_topic').perform(context),
        'min_command_speed': float(LaunchConfiguration(
            'min_command_speed').perform(context)),
        'collision_topic': LaunchConfiguration(
            'collision_topic').perform(context),
        'emergency_stop_topic': LaunchConfiguration(
            'emergency_stop_topic').perform(context),
    })

    return [
        LogInfo(msg=selection_log),
        Node(
            package='control',
            executable='linear_mpc_node',
            name='linear_mpc_node',
            output='screen',
            parameters=[parameters],
        ),
    ]


def generate_launch_description():
    package_share = get_package_share_directory('control')
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(package_share, 'config', 'params.yaml'),
        ),
        DeclareLaunchArgument('drive_mode', default_value='sim'),
        DeclareLaunchArgument(
            'controller',
            default_value='pure_pursuit',
            description=(
                'none, pure_pursuit, unicorn_l1, unicorn_l1_dynamic, or mpc'),
        ),
        DeclareLaunchArgument(
            'mpc_profile',
            default_value='speed_0.55',
            description='speed_<m/s> or a named profile in mpc_params.yaml',
        ),
        DeclareLaunchArgument(
            'mpc_params_file',
            default_value=os.path.join(
                package_share, 'config', 'mpc_params.yaml'),
        ),
        DeclareLaunchArgument('enabled', default_value='false'),
        DeclareLaunchArgument('global_frame_id', default_value='map'),
        DeclareLaunchArgument(
            'base_frame_id', default_value='ego_racecar/base_link'),
        DeclareLaunchArgument(
            'odom_topic', default_value='/ego_racecar/odom'),
        DeclareLaunchArgument('drive_topic', default_value='/drive'),
        DeclareLaunchArgument(
            'min_command_speed',
            default_value='0.0',
            description='Minimum non-zero speed command for actuator deadband'),
        DeclareLaunchArgument(
            'max_lateral_acceleration',
            default_value='1.50',
            description='UNICORN L1 cornering limit in m/s^2'),
        DeclareLaunchArgument(
            'max_longitudinal_acceleration',
            default_value='2.0',
            description='UNICORN L1 acceleration command limit in m/s^2'),
        DeclareLaunchArgument(
            'max_longitudinal_deceleration',
            default_value='4.0',
            description='UNICORN L1 deceleration command limit in m/s^2'),
        DeclareLaunchArgument(
            'avoidance_speed_limit',
            default_value='auto',
            description=(
                'Hard UNICORN L1 obstacle speed cap; auto lets the planner '
                'publish a curvature-derived limit')),
        DeclareLaunchArgument(
            'collision_topic', default_value='/ego_racecar/collision'),
        DeclareLaunchArgument(
            'emergency_stop_topic',
            default_value='/safety/emergency_stop'),
        OpaqueFunction(function=_launch_setup),
    ])
