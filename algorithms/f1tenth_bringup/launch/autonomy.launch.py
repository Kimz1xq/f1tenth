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


def _include(package, launch_file, arguments):
    source = PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory(package), 'launch', launch_file))
    return IncludeLaunchDescription(
        source,
        launch_arguments={key: str(value) for key, value in arguments.items()}.items(),
    )


def _launch_setup(context, catalog_path):
    mode = LaunchConfiguration('mode').perform(context)
    if mode == 'real':
        controller = LaunchConfiguration('controller').perform(context)
        mpc_profile = LaunchConfiguration('mpc_profile').perform(context)
        waypoint_csv = LaunchConfiguration('waypoint_csv').perform(context)
        obstacle_mode = LaunchConfiguration('obstacles').perform(context)
        return [
            LogInfo(msg=(
                f'mode=real controller={controller} '
                f'mpc_profile={mpc_profile} obstacles={obstacle_mode} '
                'output=/auto enabled=false')),
            _include('planning', 'planning.launch.py', {
                'waypoint_csv': waypoint_csv,
                'local_planner': obstacle_mode,
                'odom_topic': '/odom',
                'base_frame_id': 'base_link',
            }),
            _include('control', 'control.launch.py', {
                'controller': controller,
                'mpc_profile': mpc_profile,
                'drive_mode': 'real',
                'enabled': 'false',
                'global_frame_id': 'map',
                'base_frame_id': 'base_link',
                'odom_topic': '/odom',
                'drive_topic': '/auto',
                'min_command_speed': LaunchConfiguration(
                    'min_command_speed').perform(context),
                'collision_topic': '/control/collision',
                'emergency_stop_topic': '/safety/emergency_stop',
            }),
        ]
    if mode != 'sim':
        raise RuntimeError("mode must be 'sim' or 'real'")

    with open(catalog_path, 'r') as stream:
        catalog = yaml.safe_load(stream)

    track_name = LaunchConfiguration('track').perform(context)
    tracks = catalog.get('tracks', {})
    if track_name not in tracks:
        available = ', '.join(sorted(tracks))
        raise RuntimeError(
            f'Unknown track {track_name!r}; available tracks: {available}')

    track = tracks[track_name]
    controller = LaunchConfiguration('controller').perform(context)
    mpc_profile = LaunchConfiguration('mpc_profile').perform(context)
    friction_arg = LaunchConfiguration('friction').perform(context)
    friction = track['friction_mu'] if friction_arg == 'auto' else friction_arg

    start_x, start_y, start_yaw = track['start']
    common = {
        'map_path': track['map_path'],
        'map_ext': track['map_ext'],
        'start_x': start_x,
        'start_y': start_y,
        'start_yaw': start_yaw,
        'centerline': track['centerline'],
        'friction': friction,
        'obstacles': LaunchConfiguration('obstacles').perform(context),
        'rviz': LaunchConfiguration('rviz').perform(context),
    }

    return [
        LogInfo(msg=(
            f'track={track_name} controller={controller} '
            f'mpc_profile={mpc_profile} friction={friction}')),
        _include('f1tenth_gym_ros', 'gym_bridge_launch.py', common),
        _include('planning', 'planning.launch.py', {
            'waypoint_csv': track['raceline'],
        }),
        _include('control', 'control.launch.py', {
            'controller': controller,
            'mpc_profile': mpc_profile,
            'drive_mode': 'sim',
        }),
    ]


def generate_launch_description():
    catalog_path = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'config',
        'tracks.yaml',
    )
    default_waypoint = os.path.join(
        get_package_share_directory('planning'),
        'waypoints',
        'track03_raceline.csv',
    )

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='sim'),
        DeclareLaunchArgument('track', default_value='track03'),
        DeclareLaunchArgument('waypoint_csv', default_value=default_waypoint),
        DeclareLaunchArgument(
            'controller',
            default_value='mpc',
            description='none, pure_pursuit, or mpc',
        ),
        DeclareLaunchArgument(
            'mpc_profile',
            default_value='speed_0.55',
            description='Dynamic speed_<m/s> (e.g. speed_0.85 or speed_2)',
        ),
        DeclareLaunchArgument(
            'min_command_speed',
            default_value='0.30',
            description='Real vehicle non-zero command floor in m/s',
        ),
        DeclareLaunchArgument(
            'friction',
            default_value='auto',
            description='auto uses the selected track friction_mu',
        ),
        DeclareLaunchArgument('obstacles', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(
            function=_launch_setup,
            kwargs={'catalog_path': catalog_path},
        ),
    ])
