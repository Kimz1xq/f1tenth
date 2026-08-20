import csv
import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


SOFTWARE_SPEED_CEILING = 20.0


def _include(package, launch_file, arguments):
    source = PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory(package), 'launch', launch_file))
    return IncludeLaunchDescription(
        source,
        launch_arguments={key: str(value) for key, value in arguments.items()}.items(),
    )


def _raceline_start_pose(csv_path):
    """
    Derive a simulator start pose from the selected raceline.

    The simulator must start on the same path and tangent that the controller
    will track.  Keeping a copied xyz/yaw tuple in the track catalog allows it
    to become stale whenever a raceline is regenerated.
    """
    with open(csv_path, newline='') as stream:
        rows = [
            row for row in csv.reader(stream)
            if row and not row[0].strip().startswith('#')
        ]
    if len(rows) < 3:
        raise RuntimeError(
            f'Raceline needs at least two points: {csv_path}')

    header = [cell.strip().lower() for cell in rows[0]]
    try:
        x_index = next(header.index(name) for name in ('x', 'x_m')
                       if name in header)
        y_index = next(header.index(name) for name in ('y', 'y_m')
                       if name in header)
    except StopIteration as error:
        raise RuntimeError(
            f'Raceline has no x/y columns: {csv_path}') from error

    points = [
        (float(row[x_index]), float(row[y_index]))
        for row in rows[1:]
    ]
    if len(points) < 8:
        raise RuntimeError(
            f'Raceline needs at least eight points: {csv_path}')

    count = len(points)
    segment_yaw = [
        math.atan2(
            points[(index + 1) % count][1] - points[index][1],
            points[(index + 1) % count][0] - points[index][0],
        )
        for index in range(count)
    ]
    curvature = []
    for index in range(count):
        yaw_delta = (
            segment_yaw[index] - segment_yaw[index - 1] + math.pi
        ) % (2.0 * math.pi) - math.pi
        distance = math.hypot(
            points[index][0] - points[index - 1][0],
            points[index][1] - points[index - 1][1],
        )
        curvature.append(abs(yaw_delta) / max(distance, 1e-6))

    # Start in the calmest local straight, not at an arbitrary CSV boundary.
    # This avoids launching from rest in a tight bend and applies identically
    # to every selected map/raceline.
    radius = 3
    local_peak_curvature = [
        max(curvature[(index + offset) % count]
            for offset in range(-radius, radius + 1))
        for index in range(count)
    ]
    start_index = min(
        range(count), key=local_peak_curvature.__getitem__)
    x, y = points[start_index]
    return x, y, segment_yaw[start_index]


def _launch_setup(context, catalog_path, vehicle_path):
    mode = LaunchConfiguration('mode').perform(context)
    maximum_speed = float(
        LaunchConfiguration('maximum_speed').perform(context))
    if (not math.isfinite(maximum_speed)
            or maximum_speed <= 0.0
            or maximum_speed > SOFTWARE_SPEED_CEILING):
        raise RuntimeError(
            'maximum_speed must be greater than 0 and at most '
            f'{SOFTWARE_SPEED_CEILING:g} m/s; got {maximum_speed!r}.')
    requested_speed = float(LaunchConfiguration('speed').perform(context))
    if (not math.isfinite(requested_speed)
            or not 0.0 < requested_speed <= maximum_speed):
        raise RuntimeError(
            'speed must be greater than 0 and at most maximum_speed '
            f'({maximum_speed:g} m/s); got {requested_speed!r}.')
    speed_profile = 'speed_%g' % requested_speed

    with open(catalog_path, 'r') as stream:
        catalog = yaml.safe_load(stream)
    with open(vehicle_path, 'r') as stream:
        vehicle_catalog = yaml.safe_load(stream)
    vehicle = vehicle_catalog['vehicle']
    simulation_model = vehicle_catalog['simulation']

    track_name = LaunchConfiguration('track').perform(context)
    tracks = catalog.get('tracks', {})
    if track_name not in tracks:
        available = ', '.join(sorted(tracks))
        raise RuntimeError(
            f'Unknown track {track_name!r}; available tracks: {available}')
    track = tracks[track_name]
    planning_share = get_package_share_directory('planning')
    catalog_raceline = os.path.join(
        planning_share, 'waypoints', track['raceline'])
    waypoint_argument = LaunchConfiguration('waypoint_csv').perform(context)
    waypoint_csv = (
        catalog_raceline if waypoint_argument == 'auto'
        else waypoint_argument)

    if mode == 'real':
        controller = LaunchConfiguration('controller').perform(context)
        return [
            LogInfo(msg=(
                f'mode=real controller={controller} '
                f'speed={requested_speed:.2f}m/s '
                'obstacle_policy=automatic '
                'output=/auto enabled=false')),
            _include('planning', 'planning.launch.py', {
                'waypoint_csv': waypoint_csv,
                # The planner stays active for Q1/Q2/Q3. With no detected
                # obstacle it republishes the global path; otherwise it
                # selects a collision-free local path automatically.
                'local_planner': 'true',
                'odom_topic': '/odom',
                'base_frame_id': 'base_link',
                'maximum_planning_speed': requested_speed,
                'max_lateral_acceleration': LaunchConfiguration(
                    'max_lateral_acceleration').perform(context),
                'planning_deceleration': LaunchConfiguration(
                    'max_longitudinal_deceleration').perform(context),
                'vehicle_length': vehicle['length'],
                'vehicle_width': vehicle['width'],
                'wheelbase': vehicle['wheelbase'],
                'max_steering_angle': vehicle['max_steering_angle'],
            }),
            _include('control', 'control.launch.py', {
                'controller': controller,
                'mpc_profile': speed_profile,
                'drive_mode': 'real',
                'maximum_speed': maximum_speed,
                'enabled': 'false',
                'global_frame_id': 'map',
                'base_frame_id': 'base_link',
                'odom_topic': '/odom',
                'drive_topic': '/auto',
                'wheelbase': vehicle['wheelbase'],
                'max_steering_angle': vehicle['max_steering_angle'],
                'max_steering_rate': vehicle['max_steering_rate'],
                'transform_fault_grace': '0.10',
                'min_command_speed': LaunchConfiguration(
                    'min_command_speed').perform(context),
                'max_lateral_acceleration': LaunchConfiguration(
                    'max_lateral_acceleration').perform(context),
                'max_longitudinal_acceleration': LaunchConfiguration(
                    'max_longitudinal_acceleration').perform(context),
                'max_longitudinal_deceleration': LaunchConfiguration(
                    'max_longitudinal_deceleration').perform(context),
                'avoidance_speed_limit': LaunchConfiguration(
                    'avoidance_speed_limit').perform(context),
                'steering_lookup_table': LaunchConfiguration(
                    'steering_lookup_table').perform(context),
                'collision_topic': '/control/collision',
                'emergency_stop_topic': '/safety/emergency_stop',
            }),
        ]
    if mode != 'sim':
        raise RuntimeError("mode must be 'sim' or 'real'")

    controller = LaunchConfiguration('controller').perform(context)
    friction_arg = LaunchConfiguration('friction').perform(context)
    friction = (
        simulation_model['friction_mu']
        if friction_arg == 'auto' else friction_arg)
    obstacle_mode = LaunchConfiguration('obstacles').perform(context)

    if track.get('start') == 'raceline':
        start_x, start_y, start_yaw = _raceline_start_pose(
            waypoint_csv)
    else:
        start_x, start_y, start_yaw = track['start']
    common = {
        'map_path': os.path.join(
            get_package_share_directory('f1tenth_gym_ros'),
            'maps', track['map_name']),
        'map_ext': track['map_ext'],
        'start_x': start_x,
        'start_y': start_y,
        'start_yaw': start_yaw,
        # Place and validate simulated obstacles against the same generic
        # reference that the selected controller tracks. The planner still
        # discovers every obstacle from LaserScan only.
        'centerline': waypoint_csv,
        'friction': friction,
        'obstacles': obstacle_mode,
        'obstacle_seed': LaunchConfiguration(
            'obstacle_seed').perform(context),
        'amcl_odom_noise': LaunchConfiguration(
            'amcl_odom_noise').perform(context),
        'rviz': LaunchConfiguration('rviz').perform(context),
        'wheelbase': vehicle['wheelbase'],
        'vehicle_length': vehicle['length'],
        'vehicle_width': vehicle['width'],
        'scan_distance_to_base_link': vehicle['laser_offset_x'],
        'max_steering_angle': vehicle['max_steering_angle'],
        'max_steering_rate': vehicle['max_steering_rate'],
        'steering_command_delay': vehicle['steering_command_delay'],
        'max_acceleration': vehicle['max_acceleration'],
    }

    return [
        LogInfo(msg=(
            f'track={track_name} controller={controller} '
            f'speed={requested_speed:.2f}m/s friction={friction}')),
        _include('f1tenth_gym_ros', 'gym_bridge_launch.py', common),
        _include('planning', 'planning.launch.py', {
            'waypoint_csv': waypoint_csv,
            # The same planner and AEB chain runs in both modes. `obstacles`
            # only controls simulator fixture spawning.
            'local_planner': 'true',
            # Use the same requested speed in planning and control. This is
            # independent of the selected map and keeps sim/real behavior
            # comparable without track-specific tuning.
            'maximum_planning_speed': requested_speed,
            'max_lateral_acceleration': LaunchConfiguration(
                'max_lateral_acceleration').perform(context),
            'planning_deceleration': LaunchConfiguration(
                'max_longitudinal_deceleration').perform(context),
            'vehicle_length': vehicle['length'],
            'vehicle_width': vehicle['width'],
            'wheelbase': vehicle['wheelbase'],
            'max_steering_angle': vehicle['max_steering_angle'],
        }),
        _include('control', 'control.launch.py', {
            'controller': controller,
            'mpc_profile': speed_profile,
            'drive_mode': 'sim',
            'maximum_speed': maximum_speed,
            'wheelbase': vehicle['wheelbase'],
            'max_steering_angle': vehicle['max_steering_angle'],
            'max_steering_rate': vehicle['max_steering_rate'],
            'transform_fault_grace': '0.10',
            'max_lateral_acceleration': LaunchConfiguration(
                'max_lateral_acceleration').perform(context),
            'max_longitudinal_acceleration': LaunchConfiguration(
                'max_longitudinal_acceleration').perform(context),
            'max_longitudinal_deceleration': LaunchConfiguration(
                'max_longitudinal_deceleration').perform(context),
            'avoidance_speed_limit': LaunchConfiguration(
                'avoidance_speed_limit').perform(context),
            'steering_lookup_table': LaunchConfiguration(
                'steering_lookup_table').perform(context),
        }),
    ]


def generate_launch_description():
    bringup_share = get_package_share_directory('f1tenth_bringup')
    catalog_path = os.path.join(bringup_share, 'config', 'tracks.yaml')
    vehicle_path = os.path.join(
        bringup_share, 'config', 'vehicle_model.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='sim'),
        DeclareLaunchArgument('track', default_value='track03'),
        DeclareLaunchArgument(
            'waypoint_csv', default_value='auto',
            description='auto selects the raceline for track'),
        DeclareLaunchArgument(
            'controller',
            default_value='unicorn_l1',
            description=(
                'none, pure_pursuit, unicorn_l1, forza_map, mpc, or mpcc'),
        ),
        DeclareLaunchArgument(
            'speed',
            default_value='1.0',
            description='Common sim/real target speed in m/s',
        ),
        DeclareLaunchArgument(
            'maximum_speed',
            default_value=str(SOFTWARE_SPEED_CEILING),
            description=(
                'Software command ceiling only; requested speed and dynamic '
                'path limits remain active.')),
        DeclareLaunchArgument(
            'min_command_speed',
            default_value='0.30',
            description='Real vehicle non-zero command floor in m/s',
        ),
        DeclareLaunchArgument(
            'max_lateral_acceleration',
            default_value='1.50',
            description='UNICORN L1 cornering limit in m/s^2',
        ),
        DeclareLaunchArgument(
            'max_longitudinal_acceleration',
            default_value='2.0',
            description='UNICORN L1 acceleration limit in m/s^2',
        ),
        DeclareLaunchArgument(
            'max_longitudinal_deceleration',
            default_value='4.0',
            description='UNICORN L1 deceleration limit in m/s^2',
        ),
        DeclareLaunchArgument(
            'avoidance_speed_limit',
            default_value='auto',
            description=(
                'Hard obstacle speed cap; auto uses the local planner '
                'curvature-derived speed limit'),
        ),
        DeclareLaunchArgument(
            'steering_lookup_table',
            default_value='auto',
            description=(
                'ForzaETH MAP steering CSV; auto uses the linear bicycle '
                'model as the initial sim/real model'),
        ),
        DeclareLaunchArgument(
            'friction',
            default_value='auto',
            description='auto uses the selected track friction_mu',
        ),
        DeclareLaunchArgument('obstacles', default_value='false'),
        DeclareLaunchArgument(
            'amcl_odom_noise',
            default_value='auto',
            description=(
                'Simulation AMCL odometry-noise coefficient; auto uses the '
                'same 0.2 motion model as real AMCL'),
        ),
        DeclareLaunchArgument(
            'obstacle_seed',
            default_value='-1',
            description=(
                'Simulation obstacle seed; -1 randomizes each fresh run'),
        ),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(
            function=_launch_setup,
            kwargs={
                'catalog_path': catalog_path,
                'vehicle_path': vehicle_path,
            },
        ),
    ])
