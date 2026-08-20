"""
ROS 2 Humble adapter for the ForzaETH MAP controller.

The lateral-control equations and steering-table lookup follow the
MIT-licensed ForzaETH Race Stack (``controller/map.py`` and
``steering_lookup/lookup_steer_angle.py``):
https://github.com/ForzaETH/race_stack/tree/ros2-humble

Only the I/O layer is adapted: this project supplies ``nav_msgs/Path``, TF and
odometry and expects ``AckermannDriveStamped``.  Path-speed fields from the
full ForzaETH stack are replaced by this project's dynamics-limited spatial
speed profile.
"""

import math

import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from .unicorn_l1_node import UnicornL1Node


class SteeringLookup:
    """ForzaETH speed/lateral-acceleration to steering-angle lookup."""

    def __init__(self, csv_path):
        table = np.loadtxt(csv_path, delimiter=',')
        if table.ndim != 2 or table.shape[0] < 3 or table.shape[1] < 3:
            raise RuntimeError('invalid MAP steering lookup table: ' + csv_path)
        self.speeds = table[0, 1:]
        self.steering = table[1:, 0]
        self.acceleration = table[1:, 1:]
        if np.any(np.diff(self.speeds) <= 0.0):
            raise RuntimeError('MAP lookup speeds must be increasing')

    @property
    def maximum_speed(self):
        return float(self.speeds[-1])

    def lookup(self, lateral_acceleration, speed):
        sign = 1.0 if lateral_acceleration >= 0.0 else -1.0
        demand = abs(float(lateral_acceleration))
        column = int(np.argmin(np.abs(self.speeds - float(speed))))
        acceleration = self.acceleration[:, column]
        valid = np.isfinite(acceleration)
        acceleration = acceleration[valid]
        steering = self.steering[valid]
        if len(acceleration) == 0:
            raise RuntimeError('MAP lookup column has no valid samples')
        order = np.argsort(acceleration)
        angle = np.interp(
            demand, acceleration[order], steering[order],
            left=steering[order][0], right=steering[order][-1])
        return sign * float(angle)


class ForzaMapNode(UnicornL1Node):
    """Model- and Acceleration-based Pursuit using a steering model LUT."""

    controller_label = 'ForzaETH MAP'
    topic_prefix = '/forza_map'

    def __init__(self):
        self.stop_on_collision = True
        self.stop_on_emergency_stop = True
        self.filtered_longitudinal_acceleration = 0.0
        self.previous_measured_speed = None
        self.previous_measurement_time = None
        super().__init__()

        self.declare_parameter('steering_lookup_table', '')
        self.declare_parameter('acc_scaler_for_steer', 1.0)
        self.declare_parameter('dec_scaler_for_steer', 1.0)
        self.declare_parameter('start_scale_speed', 7.0)
        self.declare_parameter('end_scale_speed', 8.0)
        self.declare_parameter('downscale_factor', 0.20)
        self.declare_parameter('speed_lookahead_for_steer', 0.0)
        self.declare_parameter('acceleration_filter_alpha', 0.15)
        self.declare_parameter('stop_on_collision', True)
        self.declare_parameter('stop_on_emergency_stop', True)

        table_path = str(self.get_parameter('steering_lookup_table').value)
        if not table_path:
            raise RuntimeError(
                'ForzaETH MAP requires steering_lookup_table')
        self.steering_lookup = SteeringLookup(table_path)
        self.stop_on_collision = bool(
            self.get_parameter('stop_on_collision').value)
        self.stop_on_emergency_stop = bool(
            self.get_parameter('stop_on_emergency_stop').value)
        for name in (
                'acc_scaler_for_steer', 'dec_scaler_for_steer',
                'start_scale_speed', 'end_scale_speed', 'downscale_factor',
                'speed_lookahead_for_steer', 'acceleration_filter_alpha'):
            setattr(self, name, float(self.get_parameter(name).value))
        if self.end_scale_speed <= self.start_scale_speed:
            raise RuntimeError(
                'end_scale_speed must be greater than start_scale_speed')
        if not 0.0 <= self.downscale_factor <= 1.0:
            raise RuntimeError('downscale_factor must be in [0, 1]')
        if not 0.0 <= self.acceleration_filter_alpha <= 1.0:
            raise RuntimeError('acceleration_filter_alpha must be in [0, 1]')
        if self.max_speed > self.steering_lookup.maximum_speed + 1e-6:
            raise RuntimeError(
                'requested speed %.2f m/s exceeds steering model limit '
                '%.2f m/s' % (
                    self.max_speed, self.steering_lookup.maximum_speed))
        self.get_logger().info(
            'ForzaETH steering model: %s (0..%.2f m/s)'
            % (table_path, self.steering_lookup.maximum_speed))

    def collision_callback(self, message):
        """Ignore Gym collision flags only when launch explicitly allows it."""
        if not self.stop_on_collision:
            self.collision = False
            return
        super().collision_callback(message)

    def emergency_stop_callback(self, message):
        """Retain AEB on the car while permitting crash tests in Gym."""
        if not self.stop_on_emergency_stop:
            self.emergency_stop = False
            return
        super().emergency_stop_callback(message)

    def odom_callback(self, message):
        speed = max(0.0, float(message.twist.twist.linear.x))
        now = self.get_clock().now()
        if (self.previous_measured_speed is not None
                and self.previous_measurement_time is not None):
            dt = (now - self.previous_measurement_time).nanoseconds * 1e-9
            if 1e-3 < dt < 0.25:
                measured = (speed - self.previous_measured_speed) / dt
                alpha = self.acceleration_filter_alpha
                self.filtered_longitudinal_acceleration = (
                    (1.0 - alpha) * self.filtered_longitudinal_acceleration
                    + alpha * measured)
        self.previous_measured_speed = speed
        self.previous_measurement_time = now
        super().odom_callback(message)

    def compute_command(self, x, y, yaw, speed):
        _, distance, path_heading, path_s, lateral_error = (
            self.nearest_path_state(
                x, y, update_index=True, reference_yaw=yaw))
        heading_error = self.angle_difference(path_heading, yaw)
        if distance > self.max_path_distance:
            raise RuntimeError(
                'vehicle is %.2f m from path (limit %.2f m)'
                % (distance, self.max_path_distance))
        if abs(heading_error) > self.max_heading_error:
            raise RuntimeError(
                'heading error is %.1f deg (limit %.1f deg)'
                % (math.degrees(abs(heading_error)),
                   math.degrees(self.max_heading_error)))

        # The upstream MAP speed lookahead compensates computation and
        # actuation delay.  The source SIM configuration sets it to zero.
        speed_s = path_s + max(0.0, speed) * self.speed_lookahead
        command_speed = float(self.interpolate_path(
            self.path_speed_profile, speed_s)[0])

        lateral_norm = self.clamp(abs(lateral_error) / 0.5, 0.0, 1.0)
        curvature_s = np.linspace(path_s, path_s + 1.0, num=8)
        mean_curvature = float(np.mean(np.abs(self.interpolate_path(
            self.path_curvature, curvature_s))))
        curvature_norm = self.clamp(
            2.0 * (mean_curvature / 0.8) - 2.0, 0.0, 1.0)
        command_speed *= (
            1.0 - self.lat_err_coeff
            + self.lat_err_coeff
            * math.exp(-lateral_norm * curvature_norm))
        if abs(heading_error) >= math.pi / 9.0:
            command_speed *= (
                1.0 - 0.5 * min(abs(heading_error), math.pi / 2.0)
                / (math.pi / 2.0))
        command_speed = self.clamp(
            command_speed, self.min_reference_speed, self.max_speed)
        if command_speed > 0.0:
            command_speed = max(command_speed, self.min_command_speed)
        if self.avoidance_active:
            avoidance_limit = self.avoidance_speed_limit
            if (self.use_dynamic_speed_limit
                    and self.dynamic_speed_limit is not None
                    and self.age_seconds(self.last_speed_limit_time)
                    <= self.speed_limit_timeout):
                avoidance_limit = min(
                    avoidance_limit, self.dynamic_speed_limit)
            command_speed = min(command_speed, avoidance_limit)

        l1_distance = self.clamp(
            self.q_l1 + speed * self.m_l1,
            max(self.t_clip_min, math.sqrt(2.0) * abs(lateral_error)),
            self.t_clip_max)
        target_s = path_s + l1_distance
        target_x = float(self.interpolate_path(
            self.path_points[:, 0], target_s)[0])
        target_y = float(self.interpolate_path(
            self.path_points[:, 1], target_s)[0])
        vector_x = target_x - x
        vector_y = target_y - y
        vector_norm = max(math.hypot(vector_x, vector_y), 1e-6)
        eta = math.asin(self.clamp(
            (-math.sin(yaw) * vector_x + math.cos(yaw) * vector_y)
            / vector_norm, -1.0, 1.0))

        lookup_s = (
            path_s + max(0.0, speed) * self.speed_lookahead_for_steer)
        lookup_speed = float(self.interpolate_path(
            self.path_speed_profile, lookup_s)[0])
        lookup_speed = max(float(self.steering_lookup.speeds[0]), lookup_speed)
        lateral_acceleration = (
            2.0 * lookup_speed * lookup_speed
            / max(l1_distance, 1e-6) * math.sin(eta))
        steering = self.steering_lookup.lookup(
            lateral_acceleration, lookup_speed)

        if self.filtered_longitudinal_acceleration >= 1.0:
            steering *= self.acc_scaler_for_steer
        elif self.filtered_longitudinal_acceleration <= -1.0:
            steering *= self.dec_scaler_for_steer
        scale_progress = self.clamp(
            (lookup_speed - self.start_scale_speed)
            / (self.end_scale_speed - self.start_scale_speed), 0.0, 1.0)
        steering *= 1.0 - scale_progress * self.downscale_factor
        steering *= self.clamp(1.0 + speed / 10.0, 1.0, 1.25)
        steering = self.clamp(
            steering,
            self.previous_steering - self.max_steering_delta,
            self.previous_steering + self.max_steering_delta)
        steering = self.clamp(
            steering, -self.max_steering_angle, self.max_steering_angle)

        return command_speed, steering, (
            x, y, target_x, target_y, l1_distance,
            mean_curvature, distance)


def main(args=None):
    rclpy.init(args=args)
    node = ForzaMapNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        try:
            executor.shutdown()
            if rclpy.ok():
                node.publish_stop()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RCLError):
            pass


if __name__ == '__main__':
    main()
