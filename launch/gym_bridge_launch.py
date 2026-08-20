# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import math
import os
import yaml


def _as_bool(value):
    return str(value).lower() in ('1', 'true', 'yes', 'on')


def _launch_setup(context, config, config_dict):
    parameters = config_dict['bridge']['ros__parameters']
    map_path = LaunchConfiguration('map_path').perform(context)
    map_ext = LaunchConfiguration('map_ext').perform(context)
    centerline = LaunchConfiguration('centerline').perform(context)
    start_x = float(LaunchConfiguration('start_x').perform(context))
    start_y = float(LaunchConfiguration('start_y').perform(context))
    start_yaw = float(LaunchConfiguration('start_yaw').perform(context))
    amcl_odom_noise_argument = LaunchConfiguration(
        'amcl_odom_noise').perform(context)
    if amcl_odom_noise_argument == 'auto':
        # Gym publishes its simulated state directly as odometry. A small
        # non-zero value retains particle diversity without pretending that
        # this ideal odometry has the much larger error of a physical encoder.
        amcl_odom_noise = 0.01
    else:
        amcl_odom_noise = float(amcl_odom_noise_argument)
        if (not math.isfinite(amcl_odom_noise)
                or not 0.0 <= amcl_odom_noise <= 1.0):
            raise RuntimeError(
                'amcl_odom_noise must be auto or a value from 0.0 to 1.0')
    bridge_overrides = {
        'map_path': map_path,
        'map_img_ext': map_ext,
        'sx': start_x,
        'sy': start_y,
        'stheta': start_yaw,
        'friction_mu': float(LaunchConfiguration('friction').perform(context)),
        'obstacle_path_csv': centerline,
        'random_obstacles_enabled': _as_bool(
            LaunchConfiguration('obstacles').perform(context)),
        'random_obstacle_seed': int(
            LaunchConfiguration('obstacle_seed').perform(context)),
    }
    has_opp = parameters['num_agent'] > 1

    actions = []
    if _as_bool(LaunchConfiguration('rviz').perform(context)):
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=['-d', os.path.join(
                get_package_share_directory('f1tenth_gym_ros'),
                'launch', 'gym_bridge.rviz')]
        ))

    actions.extend([
        Node(
            package='f1tenth_gym_ros',
            executable='gym_bridge',
            name='bridge',
            output='screen',
            parameters=[config, bridge_overrides]
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                # In simulation every launch owns its map server and AMCL.
                # Disabling bond timeout avoids an unnecessary lifecycle reset
                # when RViz briefly stalls the container during rendering.
                'bond_timeout': 0.0,
                'node_names': ['map_server', 'amcl'],
            }]
        ),
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': map_path + '.yaml',
                'topic': 'map',
                'frame_id': 'map',
                'use_sim_time': False,
            }]
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[
                os.path.join(
                    get_package_share_directory('f1tenth_gym_ros'),
                    'config', 'amcl.yaml'),
                {
                    # A new simulation run must use the selected track start,
                    # never the pose AMCL estimated immediately before a crash.
                    'always_reset_initial_pose': True,
                    'save_pose_rate': -1.0,
                    'initial_pose.x': start_x,
                    'initial_pose.y': start_y,
                    'initial_pose.yaw': start_yaw,
                    # AMCL alpha values describe odometry error, not a map.
                    # A numeric override supports measured noisy-odom models
                    # without creating track-specific localization files.
                    'alpha1': amcl_odom_noise,
                    'alpha2': amcl_odom_noise,
                    'alpha3': amcl_odom_noise,
                    'alpha4': amcl_odom_noise,
                    'alpha5': amcl_odom_noise,
                },
            ]
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='ego_robot_state_publisher',
            parameters=[{'robot_description': Command([
                'xacro ', os.path.join(
                    get_package_share_directory('f1tenth_gym_ros'),
                    'launch', 'ego_racecar.xacro')
            ])}],
            remappings=[('/robot_description', 'ego_robot_description')]
        ),
    ])

    if has_opp:
        actions.append(Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='opp_robot_state_publisher',
            parameters=[{'robot_description': Command([
                'xacro ', os.path.join(
                    get_package_share_directory('f1tenth_gym_ros'),
                    'launch', 'opp_racecar.xacro')
            ])}],
            remappings=[('/robot_description', 'opp_robot_description')]
        ))

    return actions


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'config',
        'sim.yaml'
        )
    with open(config, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    parameters = config_dict['bridge']['ros__parameters']

    arguments = [
        DeclareLaunchArgument('map_path', default_value=parameters['map_path']),
        DeclareLaunchArgument('map_ext', default_value=parameters['map_img_ext']),
        DeclareLaunchArgument('start_x', default_value=str(parameters['sx'])),
        DeclareLaunchArgument('start_y', default_value=str(parameters['sy'])),
        DeclareLaunchArgument('start_yaw', default_value=str(parameters['stheta'])),
        DeclareLaunchArgument(
            'centerline', default_value=parameters['obstacle_path_csv']),
        DeclareLaunchArgument(
            'friction', default_value=str(parameters['friction_mu'])),
        DeclareLaunchArgument('obstacles', default_value='false'),
        DeclareLaunchArgument(
            'amcl_odom_noise',
            default_value='auto',
            description=(
                'AMCL odometry motion-noise coefficient. auto uses the '
                'low-noise model appropriate for Gym state odometry; use a '
                'measured numeric value for another odometry source'),
        ),
        DeclareLaunchArgument(
            'obstacle_seed',
            default_value='-1',
            description=(
                '-1 chooses a fresh random layout; a non-negative value '
                'replays the same obstacle layout for regression tests'),
        ),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(
            function=_launch_setup,
            kwargs={'config': config, 'config_dict': config_dict},
        ),
    ]
    return LaunchDescription(arguments)
