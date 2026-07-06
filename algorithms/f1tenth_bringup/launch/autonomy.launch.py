from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    drive_mode = LaunchConfiguration('drive_mode')
    waypoint_csv = LaunchConfiguration('waypoint_csv')

    return LaunchDescription([
        DeclareLaunchArgument(
            'drive_mode',
            default_value='sim',
            description='Vehicle output mode: sim or real'
        ),

        DeclareLaunchArgument(
            'waypoint_csv',
            default_value='/sim_ws/src/planning/waypoints/waypoints.csv',
            description='Path to waypoint CSV file'
        ),

        Node(
            package='localization',
            executable='localization_node',
            name='localization_node',
            output='screen',
            parameters=[{
                'input_odom_topic': '/ego_racecar/odom',
                'output_odom_topic': '/localization/odom',
                'output_pose_topic': '/localization/pose',
            }]
        ),

        Node(
            package='planning',
            executable='waypoint_planner_node',
            name='waypoint_planner_node',
            output='screen',
            parameters=[{
                'waypoint_csv': waypoint_csv,
                'path_topic': '/planning/path',
                'marker_topic': '/planning/markers',
                'frame_id': 'map',
                'publish_rate': 2.0,
            }]
        ),

        Node(
            package='control',
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            output='screen',
            parameters=[{
                # Important:
                # Pure Pursuit always publishes common command to /control/drive_cmd.
                # vehicle_interface_node converts it to sim or real output.
                'drive_mode': 'sim',

                'odom_topic': '/localization/odom',
                'path_topic': '/planning/path',

                'sim_drive_topic': '/control/drive_cmd',

                'real_speed_topic': '/unused/commands/motor/speed',
                'real_servo_topic': '/unused/commands/servo/position',

                'wheelbase': 0.33,
                'lookahead_distance': 1.0,
                'max_steering_angle': 0.4189,

                'target_speed': 1.0,
                'min_speed': 0.0,
                'max_speed': 1.0,
                'corner_slowdown_gain': 0.5,

                'kp': 1.0,
                'ki': 0.0,
                'kd': 0.05,

                'speed_to_erpm_gain': 3000.0,
                'speed_to_erpm_offset': 0.0,

                'servo_center': 0.5,
                'servo_gain': 1.0,
                'servo_min': 0.0,
                'servo_max': 1.0,

                'control_rate': 30.0,
            }]
        ),

        Node(
            package='vehicle_interface',
            executable='vehicle_interface_node',
            name='vehicle_interface_node',
            output='screen',
            parameters=[{
                'drive_mode': drive_mode,

                'input_drive_topic': '/control/drive_cmd',

                'sim_drive_topic': '/drive',

                'real_speed_topic': '/commands/motor/speed',
                'real_servo_topic': '/commands/servo/position',

                'safety_stop_topic': '/safety/stop_required',
                'use_safety_stop': True,

                'watchdog_timeout': 0.5,
                'publish_rate': 50.0,

                'speed_to_erpm_gain': 3000.0,
                'speed_to_erpm_offset': 0.0,

                'servo_center': 0.5,
                'servo_gain': 1.0,
                'servo_min': 0.0,
                'servo_max': 1.0,
            }]
        ),
    ])
