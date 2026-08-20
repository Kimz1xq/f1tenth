import os
import math
import re

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


SOFTWARE_SPEED_CEILING = 20.0


def _parse_dynamic_speed(profile_name, maximum_speed=SOFTWARE_SPEED_CEILING):
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
    if (not math.isfinite(speed)
            or speed <= 0.0
            or speed > float(maximum_speed)):
        raise RuntimeError(
            'Controller speed must be greater than 0 and at most '
            f'{float(maximum_speed):g} m/s; '
            f'got {speed!r}.')
    return speed


def _as_bool(value):
    return str(value).lower() in ('1', 'true', 'yes', 'on')


def _launch_setup(context):
    package_share = get_package_share_directory('control')
    controller = LaunchConfiguration('controller').perform(context)
    drive_mode = LaunchConfiguration('drive_mode').perform(context)
    if drive_mode not in ('sim', 'real'):
        raise RuntimeError("drive_mode must be 'sim' or 'real'")
    maximum_speed = float(
        LaunchConfiguration('maximum_speed').perform(context))
    if (not math.isfinite(maximum_speed)
            or maximum_speed <= 0.0
            or maximum_speed > SOFTWARE_SPEED_CEILING):
        raise RuntimeError(
            'maximum_speed must be greater than 0 and at most '
            f'{SOFTWARE_SPEED_CEILING:g} m/s; got {maximum_speed!r}.')
    if controller == 'none':
        return [LogInfo(msg='Controller disabled (controller:=none)')]

    if controller == 'pure_pursuit':
        profile_name = LaunchConfiguration('mpc_profile').perform(context)
        requested_speed = _parse_dynamic_speed(profile_name, maximum_speed)
        if requested_speed is None:
            raise RuntimeError(
                'Pure Pursuit requires mpc_profile:=speed_<m/s> '
                '(for example speed_1.0).')
        return [
            LogInfo(msg=(
                f'Controller=pure_pursuit speed={requested_speed:.2f}m/s')),
            Node(
                package='control',
                executable='pure_pursuit_node',
                name='pure_pursuit_node',
                output='screen',
                parameters=[
                    LaunchConfiguration('params_file').perform(context),
                    {
                        'drive_mode': drive_mode,
                        'global_frame_id': LaunchConfiguration(
                            'global_frame_id').perform(context),
                        'base_frame_id': LaunchConfiguration(
                            'base_frame_id').perform(context),
                        'odom_topic': LaunchConfiguration(
                            'odom_topic').perform(context),
                        'drive_topic': LaunchConfiguration(
                            'drive_topic').perform(context),
                        'wheelbase': float(LaunchConfiguration(
                            'wheelbase').perform(context)),
                        'max_steering_angle': float(LaunchConfiguration(
                            'max_steering_angle').perform(context)),
                        'target_speed': requested_speed,
                        'max_speed': requested_speed,
                        'min_speed': min(0.25, requested_speed),
                    },
                ],
            ),
        ]

    if controller == 'forza_map':
        profile_name = LaunchConfiguration('mpc_profile').perform(context)
        requested_speed = _parse_dynamic_speed(profile_name, maximum_speed)
        if requested_speed is None:
            raise RuntimeError(
                'ForzaETH MAP requires mpc_profile:=speed_<m/s> '
                '(for example speed_3.0).')
        table_argument = LaunchConfiguration(
            'steering_lookup_table').perform(context)
        if table_argument == 'auto':
            table_path = os.path.join(
                package_share, 'config',
                'forzaeth_linear_bicycle_lookup_table.csv')
        else:
            table_path = table_argument
        if not os.path.isfile(table_path):
            raise RuntimeError(
                'ForzaETH MAP steering lookup table not found: '
                + table_path)

        min_reference_speed = min(
            requested_speed,
            max(0.20, min(0.45, requested_speed * 0.40)),
        )
        avoidance_value = LaunchConfiguration(
            'avoidance_speed_limit').perform(context)
        avoidance_speed_limit = (
            requested_speed if avoidance_value == 'auto'
            else float(avoidance_value))
        # One controller parameter set is used in both environments.  The
        # simulator is calibrated to the physical vehicle instead of carrying
        # a second set of mode-specific gains.
        map_parameters = {
            't_clip_min': 0.9,
            't_clip_max': 5.0,
            'm_l1': 0.55,
            'q_l1': -0.03,
            'speed_lookahead': 0.25,
            'lat_err_coeff': 1.0,
            'acc_scaler_for_steer': 1.2,
            'dec_scaler_for_steer': 0.9,
            'start_scale_speed': 7.0,
            'end_scale_speed': 8.0,
            'downscale_factor': 0.20,
            'speed_lookahead_for_steer': 0.0,
        }
        return [
            LogInfo(msg=(
                'Controller=forza_map (ForzaETH MAP) '
                f'speed={requested_speed:.2f}m/s model={table_path}')),
            Node(
                package='control',
                executable='forza_map_node',
                name='forza_map_node',
                output='screen',
                parameters=[{
                    'enabled': _as_bool(
                        LaunchConfiguration('enabled').perform(context)),
                    'global_frame_id': LaunchConfiguration(
                        'global_frame_id').perform(context),
                    'odom_frame_id': LaunchConfiguration(
                        'odom_frame_id').perform(context),
                    'base_frame_id': LaunchConfiguration(
                        'base_frame_id').perform(context),
                    'odom_topic': LaunchConfiguration(
                        'odom_topic').perform(context),
                    'drive_topic': LaunchConfiguration(
                        'drive_topic').perform(context),
                    'collision_topic': LaunchConfiguration(
                        'collision_topic').perform(context),
                    'emergency_stop_topic': LaunchConfiguration(
                        'emergency_stop_topic').perform(context),
                    'target_speed': requested_speed,
                    'max_speed': requested_speed,
                    'min_reference_speed': min_reference_speed,
                    'min_command_speed': float(LaunchConfiguration(
                        'min_command_speed').perform(context)),
                    'max_lateral_acceleration': float(LaunchConfiguration(
                        'max_lateral_acceleration').perform(context)),
                    'max_longitudinal_acceleration': float(
                        LaunchConfiguration(
                            'max_longitudinal_acceleration').perform(context)),
                    'max_longitudinal_deceleration': float(
                        LaunchConfiguration(
                            'max_longitudinal_deceleration').perform(context)),
                    'wheelbase': float(LaunchConfiguration(
                        'wheelbase').perform(context)),
                    'max_steering_angle': float(LaunchConfiguration(
                        'max_steering_angle').perform(context)),
                    'max_steering_rate': float(LaunchConfiguration(
                        'max_steering_rate').perform(context)),
                    'transform_fault_grace': float(LaunchConfiguration(
                        'transform_fault_grace').perform(context)),
                    'avoidance_speed_limit': avoidance_speed_limit,
                    'use_dynamic_speed_limit': True,
                    'steering_lookup_table': table_path,
                    'stop_on_collision': True,
                    'stop_on_emergency_stop': True,
                }, map_parameters],
            ),
        ]

    if controller == 'unicorn_l1':
        # UNICORN builds a spatial speed profile from every received path.  The
        # local planner still owns AEB, but must not impose one scalar speed on
        # an entire avoidance manoeuvre.
        # The local planner publishes a_y = v^2*kappa based limits only while
        # an avoidance path is active. This preserves straight-line speed and
        # prevents a high top-speed request from overrunning a tight detour.
        use_dynamic_speed_limit = True
        profile_name = LaunchConfiguration('mpc_profile').perform(context)
        requested_speed = _parse_dynamic_speed(profile_name, maximum_speed)
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
            'odom_frame_id': LaunchConfiguration(
                'odom_frame_id').perform(context),
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
            'wheelbase': float(LaunchConfiguration(
                'wheelbase').perform(context)),
            'max_steering_angle': float(LaunchConfiguration(
                'max_steering_angle').perform(context)),
            'max_steering_rate': float(LaunchConfiguration(
                'max_steering_rate').perform(context)),
            'transform_fault_grace': float(LaunchConfiguration(
                'transform_fault_grace').perform(context)),
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

    if controller not in ('mpc', 'mpcc'):
        raise RuntimeError(
            f'Unknown controller {controller!r}; use none, pure_pursuit, '
            'unicorn_l1, forza_map, mpc, or mpcc.')

    config_path = LaunchConfiguration('mpc_params_file').perform(context)
    profile_name = LaunchConfiguration('mpc_profile').perform(context)
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    profiles = config.get('profiles', {})
    requested_speed = _parse_dynamic_speed(profile_name, maximum_speed)
    if requested_speed is not None:
        parameters = dict(config.get('common', {}))
        parameters.update(config.get('speed_template', {}))
        if controller == 'mpcc':
            parameters.update(config.get('mpcc_template', {}))
        parameters['target_speed'] = requested_speed
        parameters['max_speed'] = requested_speed
        parameters['min_reference_speed'] = min(
            requested_speed,
            max(0.20, min(0.45, requested_speed * 0.40)),
        )
        selection_log = (
            f'Controller={controller} dynamic_speed={requested_speed:.2f}m/s '
            f'corner_min={parameters["min_reference_speed"]:.2f}m/s')
    elif profile_name in profiles:
        parameters = dict(config.get('common', {}))
        parameters.update(profiles[profile_name])
        if controller == 'mpcc':
            parameters.update(config.get('mpcc_template', {}))
        selection_log = f'Controller={controller} profile={profile_name}'
    else:
        available = ', '.join(sorted(profiles))
        raise RuntimeError(
            f'Unknown MPC profile {profile_name!r}; use speed_<m/s> '
            f'(for example speed_0.85 or speed_2), or: {available}')

    parameters.update({
        'enabled': _as_bool(LaunchConfiguration('enabled').perform(context)),
        'global_frame_id': LaunchConfiguration(
            'global_frame_id').perform(context),
        'odom_frame_id': LaunchConfiguration(
            'odom_frame_id').perform(context),
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
        'wheelbase': float(LaunchConfiguration(
            'wheelbase').perform(context)),
        'max_steering_angle': float(LaunchConfiguration(
            'max_steering_angle').perform(context)),
        'max_steering_rate': float(LaunchConfiguration(
            'max_steering_rate').perform(context)),
    })

    return [
        LogInfo(msg=selection_log),
        Node(
            package='control',
            executable=(
                'nonlinear_mpcc_node'
                if controller == 'mpcc' else 'linear_mpc_node'),
            name=(
                'nonlinear_mpcc_node'
                if controller == 'mpcc' else 'linear_mpc_node'),
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
            'maximum_speed',
            default_value=str(SOFTWARE_SPEED_CEILING),
            description=(
                'Software command ceiling in m/s. The selected speed remains '
                'a separate target and may be reduced by dynamics or safety.')),
        DeclareLaunchArgument(
            'controller',
            default_value='pure_pursuit',
            description=(
                'none, pure_pursuit, unicorn_l1, forza_map, mpc, or mpcc'),
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
        DeclareLaunchArgument(
            'steering_lookup_table',
            default_value='auto',
            description=(
                'ForzaETH MAP steering CSV; auto selects the official SIM '
                'model and is intentionally rejected in real mode'),
        ),
        DeclareLaunchArgument('enabled', default_value='false'),
        DeclareLaunchArgument('global_frame_id', default_value='map'),
        DeclareLaunchArgument('odom_frame_id', default_value='odom'),
        DeclareLaunchArgument(
            'base_frame_id', default_value='ego_racecar/base_link'),
        DeclareLaunchArgument(
            'odom_topic', default_value='/ego_racecar/odom'),
        DeclareLaunchArgument('drive_topic', default_value='/drive'),
        DeclareLaunchArgument('wheelbase', default_value='0.324'),
        DeclareLaunchArgument('max_steering_angle', default_value='0.4189'),
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
            'max_steering_rate',
            default_value='3.2',
            description='Physical steering slew limit in rad/s'),
        DeclareLaunchArgument(
            'transform_fault_grace',
            default_value='0.10',
            description=(
                'Seconds to hold the last safe command for a transient TF '
                'fault; collision and AEB still stop immediately')),
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
