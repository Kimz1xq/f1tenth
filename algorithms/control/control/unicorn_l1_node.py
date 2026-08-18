"""
Humble adapter for the UNICORN Racing Stack L1 controller strategy.

The controller structure is adapted from the MIT-licensed HMCL-UNIST
UNICORN Racing Stack (ROS 2 Jazzy):
https://github.com/HMCL-UNIST/unicorn-racing-stack

This adapter intentionally uses this project's existing nav_msgs/Path, TF,
odometry, safety, and Ackermann interfaces.  It does not pretend to be MPC:
UNICORN's current controller is an L1/Pure-Pursuit controller.
"""

import math

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float32
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class UnicornL1Node(Node):
    """L1/Pure-Pursuit tracker using UNICORN's adaptive guidance strategy."""

    def __init__(self):
        super().__init__('unicorn_l1_node')

        self.declare_parameter('enabled', False)
        self.declare_parameter('solve_when_disabled', True)
        self.declare_parameter('global_frame_id', 'map')
        self.declare_parameter('base_frame_id', 'ego_racecar/base_link')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('path_topic', '/planning/path')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('collision_topic', '/ego_racecar/collision')
        self.declare_parameter(
            'emergency_stop_topic', '/safety/emergency_stop')
        self.declare_parameter(
            'avoidance_active_topic', '/planning/avoidance_active')
        self.declare_parameter('speed_limit_topic', '/planning/speed_limit')
        self.declare_parameter('speed_limit_timeout', 0.50)
        self.declare_parameter('use_dynamic_speed_limit', False)

        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('target_speed', 1.0)
        self.declare_parameter('min_reference_speed', 0.30)
        self.declare_parameter('min_command_speed', 0.0)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_lateral_acceleration', 1.50)
        self.declare_parameter('max_longitudinal_acceleration', 2.0)
        self.declare_parameter('max_longitudinal_deceleration', 4.0)
        self.declare_parameter('avoidance_speed_limit', 1.50)
        self.declare_parameter('max_steering_angle', 0.4189)
        self.declare_parameter('max_steering_delta', 0.40)

        # UNICORN controller.yaml defaults. Distance-window curvature sampling
        # replaces its waypoint-index window so behavior is path-resolution
        # independent.
        self.declare_parameter('t_clip_min', 0.70)
        self.declare_parameter('t_clip_max', 8.00)
        self.declare_parameter('m_l1', 0.47)
        self.declare_parameter('q_l1', -0.20)
        self.declare_parameter('curvature_factor', 0.145)
        self.declare_parameter('future_constant', 0.05)
        self.declare_parameter('curvature_window_start', 0.50)
        self.declare_parameter('curvature_window_end', 1.50)
        self.declare_parameter('lat_err_coeff', 1.0)
        self.declare_parameter('speed_factor_for_lat_err', 1.0)
        self.declare_parameter('speed_factor_for_curvature', 1.0)
        self.declare_parameter('heading_kp', 0.8)
        self.declare_parameter('heading_kd', 0.05)
        self.declare_parameter('heading_filter_alpha', 0.10)
        self.declare_parameter('heading_gain_speed', 15.0)
        self.declare_parameter('heading_slowdown_threshold_deg', 10.0)

        self.declare_parameter('max_path_distance', 0.80)
        self.declare_parameter('max_heading_error', 1.0472)
        self.declare_parameter('odom_timeout', 0.50)
        self.declare_parameter('path_timeout', 2.00)
        self.declare_parameter('search_back_points', 5)
        self.declare_parameter('search_forward_points', 30)

        for name in (
                'enabled', 'solve_when_disabled', 'global_frame_id',
                'base_frame_id', 'odom_topic', 'path_topic', 'drive_topic',
                'collision_topic', 'emergency_stop_topic',
                'avoidance_active_topic', 'speed_limit_topic',
                'use_dynamic_speed_limit'):
            setattr(self, name, self.get_parameter(name).value)
        for name in (
                'wheelbase', 'target_speed', 'min_reference_speed',
                'min_command_speed', 'max_speed',
                'max_lateral_acceleration',
                'max_longitudinal_acceleration',
                'max_longitudinal_deceleration', 'avoidance_speed_limit',
                'max_steering_angle',
                'max_steering_delta', 't_clip_min', 't_clip_max', 'm_l1',
                'q_l1', 'curvature_factor', 'future_constant',
                'curvature_window_start', 'curvature_window_end',
                'lat_err_coeff', 'speed_factor_for_lat_err',
                'speed_factor_for_curvature', 'heading_kp', 'heading_kd',
                'heading_filter_alpha', 'heading_gain_speed',
                'heading_slowdown_threshold_deg', 'max_path_distance',
                'max_heading_error', 'odom_timeout', 'path_timeout',
                'speed_limit_timeout'):
            setattr(self, name, float(self.get_parameter(name).value))
        self.search_back_points = int(
            self.get_parameter('search_back_points').value)
        self.search_forward_points = int(
            self.get_parameter('search_forward_points').value)
        control_rate = float(self.get_parameter('control_rate').value)
        self.control_dt = 1.0 / max(control_rate, 1.0)

        if self.t_clip_min <= 0.0 or self.t_clip_max < self.t_clip_min:
            raise RuntimeError('invalid L1 distance limits')
        if self.curvature_window_end < self.curvature_window_start:
            raise RuntimeError('invalid curvature sampling window')
        if not 0.0 <= self.min_command_speed <= self.max_speed:
            raise RuntimeError(
                'min_command_speed must be between 0 and max_speed')
        if self.max_lateral_acceleration <= 0.0:
            raise RuntimeError('max_lateral_acceleration must be positive')
        if (self.max_longitudinal_acceleration <= 0.0
                or self.max_longitudinal_deceleration <= 0.0):
            raise RuntimeError(
                'longitudinal acceleration limits must be positive')
        if self.avoidance_speed_limit <= 0.0:
            raise RuntimeError('avoidance_speed_limit must be positive')

        self.current_odom = None
        self.last_odom_time = None
        self.last_path_time = None
        self.collision = False
        self.emergency_stop = False
        self.avoidance_active = False
        self.dynamic_speed_limit = None
        self.last_speed_limit_time = None
        self.path_points = None
        self.path_yaw = None
        self.path_curvature = None
        self.path_segment_lengths = None
        self.path_cumulative = None
        self.path_length = None
        self.yaw_lap_change = None
        self.nearest_index = None
        self.previous_steering = 0.0
        self.previous_command_speed = 0.0
        self.filtered_heading_error = None
        self.previous_heading_error = None
        self.last_solution_ok = False
        self.last_status_message = None
        self.last_status_time = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(Path, self.path_topic, self.path_callback, 10)
        self.create_subscription(
            Bool, self.collision_topic, self.collision_callback, 10)
        self.create_subscription(
            Bool, self.emergency_stop_topic,
            self.emergency_stop_callback, 10)
        self.create_subscription(
            Bool, self.avoidance_active_topic,
            self.avoidance_active_callback, 10)
        self.create_subscription(
            Float32, self.speed_limit_topic,
            self.speed_limit_callback, 10)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10)
        self.proposed_drive_pub = self.create_publisher(
            AckermannDriveStamped, '/unicorn_l1/proposed_drive', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/unicorn_l1/markers', 10)
        self.enable_service = self.create_service(
            SetBool, '/control/enable', self.enable_callback)
        self.timer = self.create_timer(
            self.control_dt, self.control_loop)

        self.get_logger().info(
            'UNICORN L1 ready (enabled=%s, rate=%.1fHz, target=%.2fm/s)'
            % (self.enabled, control_rate, self.target_speed))
        self.get_logger().info(
            'Humble adapter: nav_msgs/Path + TF -> Ackermann /drive '
            '(dynamic_speed_limit=%s)' % self.use_dynamic_speed_limit)

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(minimum, min(value, maximum))

    @staticmethod
    def angle_difference(target, source):
        return math.atan2(
            math.sin(target - source), math.cos(target - source))

    @staticmethod
    def quaternion_to_yaw(quaternion):
        return math.atan2(
            2.0 * (quaternion.w * quaternion.z
                   + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y * quaternion.y
                         + quaternion.z * quaternion.z))

    def odom_callback(self, message):
        self.current_odom = message
        self.last_odom_time = self.get_clock().now()

    def collision_callback(self, message):
        self.collision = bool(message.data)
        if self.collision and self.enabled:
            self.enabled = False
            self.previous_command_speed = 0.0
            self.publish_stop()
            self.get_logger().error(
                'UNICORN L1 disabled: collision input is active')

    def emergency_stop_callback(self, message):
        self.emergency_stop = bool(message.data)
        if self.emergency_stop:
            self.previous_command_speed = 0.0
            self.publish_stop()

    def avoidance_active_callback(self, message):
        self.avoidance_active = bool(message.data)

    def speed_limit_callback(self, message):
        value = float(message.data)
        if math.isfinite(value) and value >= 0.0:
            self.dynamic_speed_limit = value
            self.last_speed_limit_time = self.get_clock().now()

    def path_callback(self, message):
        if len(message.poses) < 4:
            return
        points = np.asarray([
            [pose.pose.position.x, pose.pose.position.y]
            for pose in message.poses
        ], dtype=float)
        if np.linalg.norm(points[0] - points[-1]) < 1e-4:
            points = points[:-1]
        if len(points) < 4:
            return
        changed = (
            self.path_points is None
            or len(points) != len(self.path_points)
            or np.max(np.abs(points - self.path_points)) > 1e-6)
        if changed:
            self.set_closed_path(points)
            self.nearest_index = None
            self.get_logger().info(
                'UNICORN L1 received closed path: %d points, %.2f m'
                % (len(points), self.path_length),
                throttle_duration_sec=2.0)
        self.last_path_time = self.get_clock().now()

    def set_closed_path(self, points):
        next_points = np.roll(points, -1, axis=0)
        segments = next_points - points
        segment_lengths = np.linalg.norm(segments, axis=1)
        if np.any(segment_lengths < 1e-4):
            raise RuntimeError('path contains duplicate adjacent points')

        yaw = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
        direction = 1.0 if np.sum(np.diff(yaw)) >= 0.0 else -1.0
        closed_yaw = yaw[0] + direction * 2.0 * math.pi
        while closed_yaw - yaw[-1] > math.pi:
            closed_yaw -= 2.0 * math.pi
        while closed_yaw - yaw[-1] < -math.pi:
            closed_yaw += 2.0 * math.pi
        yaw_lap_change = closed_yaw - yaw[0]
        previous_yaw = np.roll(yaw, 1)
        previous_yaw[0] = yaw[-1] - yaw_lap_change
        next_yaw = np.roll(yaw, -1)
        next_yaw[-1] = yaw[0] + yaw_lap_change
        arc_span = np.roll(segment_lengths, 1) + segment_lengths

        self.path_points = points
        self.path_yaw = yaw
        self.path_curvature = (
            (next_yaw - previous_yaw) / np.maximum(arc_span, 1e-6))
        self.path_segment_lengths = segment_lengths
        self.path_cumulative = np.concatenate(
            ([0.0], np.cumsum(segment_lengths)))
        self.path_length = float(self.path_cumulative[-1])
        self.yaw_lap_change = float(yaw_lap_change)

    def age_seconds(self, stamp):
        if stamp is None:
            return float('inf')
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def readiness_problem(self):
        if self.path_points is None:
            return 'no path'
        if self.age_seconds(self.last_path_time) > self.path_timeout:
            return 'path is stale'
        if self.current_odom is None:
            return 'no odometry'
        if self.age_seconds(self.last_odom_time) > self.odom_timeout:
            return 'odometry is stale'
        if self.collision:
            return 'collision input is active; clear it before starting'
        if self.emergency_stop:
            return 'local planner emergency stop is active'
        return None

    def lookup_vehicle_pose(self):
        transform = self.tf_buffer.lookup_transform(
            self.global_frame_id, self.base_frame_id, Time(),
            timeout=Duration(seconds=0.03))
        translation = transform.transform.translation
        return (
            translation.x,
            translation.y,
            self.quaternion_to_yaw(transform.transform.rotation),
        )

    def candidate_indices(self):
        count = len(self.path_points)
        if self.nearest_index is None:
            return range(count)
        return [
            (self.nearest_index + offset) % count
            for offset in range(-self.search_back_points,
                                self.search_forward_points + 1)
        ]

    def nearest_path_state(self, x, y, update_index=False):
        position = np.asarray([x, y])
        best = None
        count = len(self.path_points)
        indices = self.candidate_indices() if update_index else range(count)
        for index in indices:
            segment = (
                self.path_points[(index + 1) % count]
                - self.path_points[index])
            relative = position - self.path_points[index]
            fraction = self.clamp(
                float(np.dot(relative, segment) / np.dot(segment, segment)),
                0.0, 1.0)
            projection = self.path_points[index] + fraction * segment
            delta = position - projection
            distance = float(np.linalg.norm(delta))
            candidate = (distance, index, fraction, projection)
            if best is None or distance < best[0]:
                best = candidate

        distance, index, fraction, projection = best
        path_s = (
            self.path_cumulative[index]
            + fraction * self.path_segment_lengths[index])
        heading = float(self.path_yaw[index])
        normal = np.asarray([-math.sin(heading), math.cos(heading)])
        signed_error = float(np.dot(position - projection, normal))
        if update_index:
            self.nearest_index = index
        return index, distance, heading, path_s, signed_error

    def interpolate_path(self, values, sample_s, lap_change=0.0):
        samples = np.atleast_1d(sample_s).astype(float)
        laps = np.floor(samples / self.path_length).astype(int)
        wrapped = np.mod(samples, self.path_length)
        closed_values = np.concatenate(
            (values, [values[0] + lap_change]))
        result = np.interp(wrapped, self.path_cumulative, closed_values)
        return result + laps * lap_change

    def compute_command(self, x, y, yaw, speed):
        _, distance, path_heading, path_s, _ = self.nearest_path_state(
            x, y, update_index=True)
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

        # UNICORN predicts a short future vehicle pose before selecting L1.
        beta = math.atan(0.48 * math.tan(self.previous_steering))
        future_x = x + speed * math.cos(yaw + beta) * self.future_constant
        future_y = y + speed * math.sin(yaw + beta) * self.future_constant
        future_yaw = yaw + (
            speed / self.wheelbase * math.sin(beta) * self.future_constant)
        _, _, _, future_s, future_lateral_error = self.nearest_path_state(
            future_x, future_y)

        curvature_s = np.linspace(
            future_s + self.curvature_window_start,
            future_s + self.curvature_window_end,
            num=8)
        mean_curvature = float(np.mean(np.abs(self.interpolate_path(
            self.path_curvature, curvature_s))))

        speed_for_l1 = self.clamp(
            self.target_speed, max(0.0, speed - 1.0), speed + 1.0)
        curvature_scaler = (
            self.curvature_factor * mean_curvature * speed * speed)
        raw_l1 = self.m_l1 * speed_for_l1 - curvature_scaler + self.q_l1
        lower_l1 = max(
            self.t_clip_min,
            math.sqrt(2.0) * abs(future_lateral_error))
        l1_distance = self.clamp(raw_l1, lower_l1, self.t_clip_max)

        target_s = future_s + l1_distance
        target_x = float(self.interpolate_path(
            self.path_points[:, 0], target_s)[0])
        target_y = float(self.interpolate_path(
            self.path_points[:, 1], target_s)[0])
        vector_x = target_x - future_x
        vector_y = target_y - future_y
        vector_norm = max(math.hypot(vector_x, vector_y), 1e-6)
        eta = self.angle_difference(
            math.atan2(vector_y, vector_x), future_yaw)
        steering = math.atan(
            2.0 * self.wheelbase * math.sin(eta) / vector_norm)

        # UNICORN's filtered, speed-scaled heading correction.
        if self.filtered_heading_error is None:
            self.filtered_heading_error = eta
        alpha = self.clamp(self.heading_filter_alpha, 0.0, 1.0)
        self.filtered_heading_error = (
            alpha * eta + (1.0 - alpha) * self.filtered_heading_error)
        gain = self.heading_kp * self.clamp(
            speed / max(self.heading_gain_speed, 1e-3), 0.0, 1.0)
        derivative = 0.0
        if self.previous_heading_error is not None:
            derivative = (
                self.filtered_heading_error - self.previous_heading_error)
            derivative /= self.control_dt
        self.previous_heading_error = self.filtered_heading_error
        steering += (
            gain * self.filtered_heading_error
            + self.heading_kd * derivative)

        # Match UNICORN's lateral-error steering scaling and steering slew cap.
        steering *= math.exp(math.log(2.0) * abs(future_lateral_error))
        steering = self.clamp(
            steering,
            self.previous_steering - self.max_steering_delta,
            self.previous_steering + self.max_steering_delta)
        steering = self.clamp(
            steering, -self.max_steering_angle, self.max_steering_angle)

        # The upstream stack gets base speed from its optimized raceline.  Here
        # the launch speed is the cap, then the same heading/lateral-curvature
        # reductions are applied to remain compatible with nav_msgs/Path.
        curvature_norm = self.clamp(
            2.0 * (mean_curvature / 0.8) - 2.0, 0.0, 1.0)
        curvature_norm *= self.speed_factor_for_curvature
        lateral_norm = self.clamp(
            abs(future_lateral_error), 0.0, 1.0)
        lateral_norm *= self.speed_factor_for_lat_err
        command_speed = self.target_speed * (
            1.0 - self.lat_err_coeff
            + self.lat_err_coeff
            * math.exp(-lateral_norm * curvature_norm))
        threshold = math.radians(self.heading_slowdown_threshold_deg)
        if abs(heading_error) >= threshold:
            if abs(heading_error) < math.pi / 2.0:
                command_speed *= (
                    1.0 - 0.5 * abs(heading_error) / (math.pi / 2.0))
            else:
                command_speed *= 0.5
        # Generic curvature speed cap from lateral acceleration a_y=v^2*kappa.
        # This reaches target_speed on straights and anticipates turns using
        # the same forward curvature window as the L1-distance calculation.
        curve_speed = math.sqrt(
            self.max_lateral_acceleration
            / max(mean_curvature, 1e-3))
        command_speed = min(command_speed, curve_speed)
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

        return command_speed, steering, (
            future_x, future_y, target_x, target_y, l1_distance,
            mean_curvature, distance)

    def publish_ackermann(self, publisher, speed, steering):
        message = AckermannDriveStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.base_frame_id
        message.drive.speed = float(speed)
        message.drive.steering_angle = float(steering)
        publisher.publish(message)

    def publish_stop(self):
        self.publish_ackermann(self.drive_pub, 0.0, 0.0)

    def publish_markers(self, geometry):
        future_x, future_y, target_x, target_y, l1_distance, _, _ = geometry
        message = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for marker_id, namespace, x, y, red, green, blue in (
                (0, 'future_pose', future_x, future_y, 0.2, 0.6, 1.0),
                (1, 'l1_target', target_x, target_y, 1.0, 0.8, 0.0)):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.global_frame_id
            marker.ns = namespace
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.08
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.12
            marker.scale.y = 0.12
            marker.scale.z = 0.12
            marker.color.a = 0.9
            marker.color.r = red
            marker.color.g = green
            marker.color.b = blue
            message.markers.append(marker)

        line = Marker()
        line.header.stamp = stamp
        line.header.frame_id = self.global_frame_id
        line.ns = 'l1_guidance'
        line.id = 2
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.035
        line.color.a = 0.9
        line.color.r = 1.0
        line.color.g = 0.8
        line.points = [
            Point(x=future_x, y=future_y, z=0.08),
            Point(x=target_x, y=target_y, z=0.08),
        ]
        message.markers.append(line)
        self.marker_pub.publish(message)

    def warn_throttled(self, message):
        now = self.get_clock().now()
        if (message != self.last_status_message
                or self.last_status_time is None
                or (now - self.last_status_time).nanoseconds
                > 2_000_000_000):
            self.get_logger().warn(message)
            self.last_status_message = message
            self.last_status_time = now

    def enable_callback(self, request, response):
        if not request.data:
            self.enabled = False
            self.previous_steering = 0.0
            self.previous_command_speed = 0.0
            self.publish_stop()
            response.success = True
            response.message = 'UNICORN L1 stopped'
            self.get_logger().info(response.message)
            return response

        problem = self.readiness_problem()
        if problem is not None:
            response.success = False
            response.message = 'Cannot start UNICORN L1: ' + problem
            self.get_logger().error(response.message)
            return response
        if not self.last_solution_ok:
            response.success = False
            response.message = (
                'Cannot start UNICORN L1: no valid dry-run command yet')
            self.get_logger().error(response.message)
            return response

        self.previous_command_speed = max(
            0.0, float(self.current_odom.twist.twist.linear.x))
        self.enabled = True
        response.success = True
        response.message = 'UNICORN L1 enabled'
        self.get_logger().info(response.message)
        return response

    def control_loop(self):
        problem = self.readiness_problem()
        if problem is not None:
            self.last_solution_ok = False
            self.publish_stop()
            if self.enabled:
                self.warn_throttled('UNICORN L1 safety stop: ' + problem)
            return
        if not self.enabled and not self.solve_when_disabled:
            self.publish_stop()
            return

        try:
            x, y, yaw = self.lookup_vehicle_pose()
            speed = max(
                0.0, float(self.current_odom.twist.twist.linear.x))
            command_speed, steering, geometry = self.compute_command(
                x, y, yaw, speed)
            self.publish_markers(geometry)
            self.last_solution_ok = True
            if not self.enabled:
                self.publish_ackermann(
                    self.proposed_drive_pub, command_speed, steering)
                self.publish_stop()
                return
            command_speed = self.clamp(
                command_speed,
                max(0.0, self.previous_command_speed
                    - self.max_longitudinal_deceleration * self.control_dt),
                self.previous_command_speed
                + self.max_longitudinal_acceleration * self.control_dt)
            self.previous_command_speed = command_speed
            self.publish_ackermann(
                self.proposed_drive_pub, command_speed, steering)
            self.previous_steering = steering
            self.publish_ackermann(
                self.drive_pub, command_speed, steering)
        except (TransformException, RuntimeError, ValueError) as error:
            self.last_solution_ok = False
            self.previous_command_speed = 0.0
            self.publish_stop()
            self.warn_throttled('UNICORN L1 safety stop: ' + str(error))


def main(args=None):
    rclpy.init(args=args)
    node = UnicornL1Node()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if rclpy.ok():
            node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
