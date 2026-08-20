import math

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float32
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameter('drive_mode', 'sim')
        self.declare_parameter('enabled', False)

        self.declare_parameter('global_frame_id', 'map')
        self.declare_parameter('base_frame_id', 'ego_racecar/base_link')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('path_topic', '/planning/path')
        self.declare_parameter(
            'emergency_stop_topic', '/safety/emergency_stop')
        self.declare_parameter(
            'avoidance_active_topic', '/planning/avoidance_active')
        self.declare_parameter('speed_limit_topic', '/planning/speed_limit')
        # Controllers publish one platform-neutral Ackermann command in both
        # modes. The simulator bridge or the real ackermann_mux/VESC adapter
        # owns the final actuator conversion.
        self.declare_parameter('drive_topic', '/drive')

        self.declare_parameter('wheelbase', 0.324)
        self.declare_parameter('lookahead_distance', 0.70)
        # Velocity-scaled lookahead is the standard Adaptive/Regulated Pure
        # Pursuit mechanism.  Distance limits, rather than track coordinates,
        # keep the behavior portable across maps and waypoint resolutions.
        self.declare_parameter('lookahead_time', 0.20)
        self.declare_parameter('minimum_lookahead_distance', 0.55)
        self.declare_parameter('maximum_lookahead_distance', 2.50)
        self.declare_parameter('max_steering_angle', 0.4189)
        self.declare_parameter('max_path_distance', 1.00)
        self.declare_parameter('max_heading_error', 1.0472)
        self.declare_parameter('search_back_points', 8)
        self.declare_parameter('search_forward_points', 30)

        self.declare_parameter('target_speed', 0.60)
        self.declare_parameter('min_speed', 0.25)
        self.declare_parameter('max_speed', 0.80)
        self.declare_parameter('corner_slowdown_gain', 0.55)
        self.declare_parameter('use_dynamic_speed_limit', True)
        self.declare_parameter('speed_limit_timeout', 0.50)
        self.declare_parameter('max_steering_rate', 3.2)

        self.declare_parameter('control_rate', 30.0)
        self.declare_parameter('odom_timeout', 0.50)
        self.declare_parameter('path_timeout', 2.00)

        self.drive_mode = self.get_parameter('drive_mode').value
        self.enabled = bool(self.get_parameter('enabled').value)

        self.global_frame_id = self.get_parameter('global_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.path_topic = self.get_parameter('path_topic').value
        self.emergency_stop_topic = self.get_parameter(
            'emergency_stop_topic').value
        self.avoidance_active_topic = self.get_parameter(
            'avoidance_active_topic').value
        self.speed_limit_topic = self.get_parameter('speed_limit_topic').value
        self.drive_topic = self.get_parameter('drive_topic').value

        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.lookahead_distance = float(
            self.get_parameter('lookahead_distance').value)
        self.lookahead_time = float(
            self.get_parameter('lookahead_time').value)
        self.minimum_lookahead_distance = float(self.get_parameter(
            'minimum_lookahead_distance').value)
        self.maximum_lookahead_distance = float(self.get_parameter(
            'maximum_lookahead_distance').value)
        self.max_steering_angle = float(
            self.get_parameter('max_steering_angle').value)
        self.max_path_distance = float(
            self.get_parameter('max_path_distance').value)
        self.max_heading_error = float(
            self.get_parameter('max_heading_error').value)
        self.search_back_points = int(
            self.get_parameter('search_back_points').value)
        self.search_forward_points = int(
            self.get_parameter('search_forward_points').value)

        self.target_speed = float(self.get_parameter('target_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.corner_slowdown_gain = float(
            self.get_parameter('corner_slowdown_gain').value)
        self.use_dynamic_speed_limit = bool(
            self.get_parameter('use_dynamic_speed_limit').value)
        self.speed_limit_timeout = float(
            self.get_parameter('speed_limit_timeout').value)
        self.max_steering_rate = float(
            self.get_parameter('max_steering_rate').value)

        self.odom_timeout = float(self.get_parameter('odom_timeout').value)
        self.path_timeout = float(self.get_parameter('path_timeout').value)
        control_rate = float(self.get_parameter('control_rate').value)

        if self.drive_mode not in ('sim', 'real'):
            raise RuntimeError("drive_mode must be 'sim' or 'real'")
        if (self.minimum_lookahead_distance <= 0.0
                or self.maximum_lookahead_distance
                < self.minimum_lookahead_distance):
            raise RuntimeError('invalid lookahead distance limits')
        if self.lookahead_time < 0.0:
            raise RuntimeError('lookahead_time must be non-negative')
        if self.max_steering_rate <= 0.0:
            raise RuntimeError('max_steering_rate must be positive')

        self.current_odom = None
        self.current_path = None
        self.last_odom_time = None
        self.last_path_time = None
        self.nearest_index = None
        self.emergency_stop = False
        self.avoidance_active = False
        self.dynamic_speed_limit = None
        self.last_speed_limit_time = None
        self.previous_steering = 0.0
        self.control_dt = 1.0 / max(control_rate, 1.0)
        self.last_status_message = None
        self.last_status_time = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(
            Path, self.path_topic, self.path_callback, 10)
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

        self.enable_service = self.create_service(
            SetBool, '/control/enable', self.enable_callback)
        self.timer = self.create_timer(
            1.0 / max(control_rate, 1.0), self.control_loop)

        self.get_logger().info(
            'Pure Pursuit ready (enabled=%s, pose=%s -> %s, path=%s, '
            'drive=%s)' % (
                self.enabled,
                self.global_frame_id,
                self.base_frame_id,
                self.path_topic,
                self.drive_topic,
            ))
        self.get_logger().info(
            'Start/stop: ros2 service call /control/enable '
            'std_srvs/srv/SetBool "{data: true|false}"')

    def odom_callback(self, msg):
        self.current_odom = msg
        self.last_odom_time = self.get_clock().now()

    def path_callback(self, msg):
        if not msg.poses:
            return
        if self.current_path is None or len(self.current_path.poses) != len(msg.poses):
            self.nearest_index = None
        self.current_path = msg
        self.last_path_time = self.get_clock().now()

    def emergency_stop_callback(self, msg):
        self.emergency_stop = bool(msg.data)
        if self.emergency_stop:
            self.publish_stop()

    def avoidance_active_callback(self, msg):
        self.avoidance_active = bool(msg.data)

    def speed_limit_callback(self, msg):
        value = float(msg.data)
        if math.isfinite(value) and value >= 0.0:
            self.dynamic_speed_limit = value
            self.last_speed_limit_time = self.get_clock().now()

    def enable_callback(self, request, response):
        if not request.data:
            self.enabled = False
            self.nearest_index = None
            self.previous_steering = 0.0
            self.publish_stop()
            response.success = True
            response.message = 'Pure Pursuit stopped'
            self.get_logger().info(response.message)
            return response

        problem = self.readiness_problem()
        if problem is not None:
            self.enabled = False
            self.publish_stop()
            response.success = False
            response.message = 'Cannot start: ' + problem
            self.get_logger().error(response.message)
            return response
        if self.emergency_stop:
            self.enabled = False
            self.publish_stop()
            response.success = False
            response.message = 'Cannot start: emergency stop is active'
            self.get_logger().error(response.message)
            return response

        try:
            x, y, yaw = self.lookup_vehicle_pose()
        except TransformException as error:
            self.enabled = False
            self.publish_stop()
            response.success = False
            response.message = 'Cannot start: TF unavailable: ' + str(error)
            self.get_logger().error(response.message)
            return response

        self.nearest_index = None
        _, path_distance, path_heading = self.nearest_path_state(x, y)
        heading_error = math.atan2(
            math.sin(path_heading - yaw), math.cos(path_heading - yaw))
        if path_distance > self.max_path_distance:
            self.enabled = False
            self.publish_stop()
            response.success = False
            response.message = (
                'Cannot start: vehicle is %.2f m from path (limit %.2f m)'
                % (path_distance, self.max_path_distance))
            self.get_logger().error(response.message)
            return response
        if abs(heading_error) > self.max_heading_error:
            self.enabled = False
            self.publish_stop()
            response.success = False
            response.message = (
                'Cannot start: heading error is %.1f deg (limit %.1f deg)'
                % (math.degrees(abs(heading_error)),
                   math.degrees(self.max_heading_error)))
            self.get_logger().error(response.message)
            return response

        self.enabled = True
        self.previous_steering = 0.0
        response.success = True
        response.message = 'Pure Pursuit enabled'
        self.get_logger().info(response.message)
        return response

    @staticmethod
    def quaternion_to_yaw(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(minimum, min(value, maximum))

    def lookup_vehicle_pose(self):
        transform = self.tf_buffer.lookup_transform(
            self.global_frame_id,
            self.base_frame_id,
            Time(),
            timeout=Duration(seconds=0.03),
        )
        translation = transform.transform.translation
        yaw = self.quaternion_to_yaw(transform.transform.rotation)
        return translation.x, translation.y, yaw

    def age_seconds(self, stamp):
        if stamp is None:
            return float('inf')
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def readiness_problem(self):
        if self.current_path is None or not self.current_path.poses:
            return 'no global path'
        if self.age_seconds(self.last_path_time) > self.path_timeout:
            return 'global path is stale'
        if self.current_odom is None:
            return 'no odometry'
        if self.age_seconds(self.last_odom_time) > self.odom_timeout:
            return 'odometry is stale'
        return None

    def measured_speed(self):
        if self.current_odom is None:
            return 0.0
        return max(0.0, float(self.current_odom.twist.twist.linear.x))

    def active_lookahead_distance(self):
        scaled = self.lookahead_distance + self.lookahead_time * self.measured_speed()
        return self.clamp(
            scaled,
            self.minimum_lookahead_distance,
            self.maximum_lookahead_distance,
        )

    def candidate_indices(self, count):
        if self.nearest_index is None:
            return range(count)
        return [
            (self.nearest_index + offset) % count
            for offset in range(-self.search_back_points,
                                self.search_forward_points + 1)
        ]

    def nearest_path_state(self, x, y):
        poses = self.current_path.poses
        count = len(poses)
        nearest_idx = min(
            self.candidate_indices(count),
            key=lambda idx: math.hypot(
                poses[idx].pose.position.x - x,
                poses[idx].pose.position.y - y,
            ),
        )
        nearest_dist = math.hypot(
            poses[nearest_idx].pose.position.x - x,
            poses[nearest_idx].pose.position.y - y,
        )
        previous = poses[(nearest_idx - 1) % count].pose.position
        following = poses[(nearest_idx + 1) % count].pose.position
        path_heading = math.atan2(
            following.y - previous.y, following.x - previous.x)
        return nearest_idx, nearest_dist, path_heading

    def find_lookahead_point(self, x, y, yaw):
        poses = self.current_path.poses
        count = len(poses)
        if count < 2:
            return None

        nearest_idx, nearest_dist, path_heading = self.nearest_path_state(x, y)
        self.nearest_index = nearest_idx

        if nearest_dist > self.max_path_distance:
            return None
        heading_error = math.atan2(
            math.sin(path_heading - yaw), math.cos(path_heading - yaw))
        if abs(heading_error) > self.max_heading_error:
            return None

        travelled = 0.0
        previous = poses[nearest_idx].pose.position
        for offset in range(1, count + 1):
            idx = (nearest_idx + offset) % count
            point = poses[idx].pose.position
            travelled += math.hypot(point.x - previous.x, point.y - previous.y)
            previous = point

            if travelled < self.active_lookahead_distance():
                continue

            dx = point.x - x
            dy = point.y - y
            x_car = math.cos(yaw) * dx + math.sin(yaw) * dy
            y_car = -math.sin(yaw) * dx + math.cos(yaw) * dy
            if x_car > 0.0:
                return x_car, y_car, math.hypot(dx, dy), nearest_dist

        return None

    def compute_steering(self, x_car, y_car, lookahead_dist):
        if lookahead_dist < 1e-6:
            return 0.0
        curvature = 2.0 * y_car / (lookahead_dist ** 2)
        steering = math.atan(self.wheelbase * curvature)
        return self.clamp(
            steering, -self.max_steering_angle, self.max_steering_angle)

    def compute_speed(self, steering):
        steer_ratio = abs(steering) / max(self.max_steering_angle, 1e-6)
        speed = self.target_speed * (
            1.0 - self.corner_slowdown_gain * steer_ratio)
        speed = self.clamp(speed, self.min_speed, self.max_speed)
        if (self.avoidance_active
                and self.use_dynamic_speed_limit
                and self.dynamic_speed_limit is not None
                and self.age_seconds(self.last_speed_limit_time)
                <= self.speed_limit_timeout):
            speed = min(speed, self.dynamic_speed_limit)
        return max(0.0, speed)

    def publish_drive(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame_id
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.drive_pub.publish(msg)

    def publish_stop(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame_id
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_pub.publish(msg)

    def rate_limit_steering(self, requested):
        maximum_delta = self.max_steering_rate * self.control_dt
        steering = self.clamp(
            requested,
            self.previous_steering - maximum_delta,
            self.previous_steering + maximum_delta,
        )
        self.previous_steering = steering
        return steering

    def warn_throttled(self, message):
        now = self.get_clock().now()
        if (message != self.last_status_message or
                self.last_status_time is None or
                (now - self.last_status_time).nanoseconds > 2_000_000_000):
            self.get_logger().warn(message)
            self.last_status_message = message
            self.last_status_time = now

    def control_loop(self):
        if not self.enabled:
            self.publish_stop()
            return

        if self.emergency_stop:
            self.publish_stop()
            self.warn_throttled('Safety stop: emergency stop is active')
            return

        problem = self.readiness_problem()
        if problem is not None:
            self.publish_stop()
            self.warn_throttled('Safety stop: ' + problem)
            return

        try:
            x, y, yaw = self.lookup_vehicle_pose()
        except TransformException as error:
            self.publish_stop()
            self.warn_throttled('Safety stop: TF unavailable: ' + str(error))
            return

        lookahead = self.find_lookahead_point(x, y, yaw)
        if lookahead is None:
            self.publish_stop()
            self.warn_throttled(
                'Safety stop: no valid lookahead point or vehicle too far from path')
            return

        x_car, y_car, lookahead_dist, _ = lookahead
        steering = self.rate_limit_steering(
            self.compute_steering(x_car, y_car, lookahead_dist))
        self.publish_drive(self.compute_speed(steering), steering)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_stop()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
