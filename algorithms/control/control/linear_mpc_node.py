"""
Linear MPC raceline tracker for the F1TENTH Gym bridge.

The kinematic bicycle model and quadratic-program structure are adapted from
the MIT-licensed F1TENTH Lab 7 MPC example:
https://github.com/jasonf27/f1tenth_autonomous_anonymous/tree/main/
lab-7-model-predictive-control-autonomous-anonymous-main
"""

import math
import time

import cvxpy as cp
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float64
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener


class LinearMpcNode(Node):
    NODE_NAME = 'linear_mpc_node'
    DISPLAY_NAME = 'Linear MPC'
    STATE_SIZE = 4  # x, y, speed, yaw
    INPUT_SIZE = 2  # acceleration, steering angle

    def __init__(self):
        super().__init__(self.NODE_NAME)

        self.declare_parameter('enabled', False)
        self.declare_parameter('solve_when_disabled', True)
        self.declare_parameter('global_frame_id', 'map')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'ego_racecar/base_link')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('path_topic', '/planning/path')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('collision_topic', '/ego_racecar/collision')
        self.declare_parameter(
            'emergency_stop_topic', '/safety/emergency_stop')

        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('horizon_steps', 12)
        self.declare_parameter('dt', 0.10)
        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('target_speed', 0.55)
        self.declare_parameter('min_reference_speed', 0.30)
        self.declare_parameter('min_command_speed', 0.0)
        self.declare_parameter('max_speed', 0.80)
        self.declare_parameter('max_acceleration', 1.50)
        self.declare_parameter('max_steering_angle', 0.4189)
        self.declare_parameter('max_steering_rate', 1.20)
        self.declare_parameter('corner_slowdown_gain', 0.35)

        self.declare_parameter('q_x', 20.0)
        self.declare_parameter('q_y', 20.0)
        self.declare_parameter('q_speed', 3.0)
        self.declare_parameter('q_yaw', 8.0)
        self.declare_parameter('qf_scale', 1.5)
        self.declare_parameter('r_acceleration', 0.20)
        self.declare_parameter('r_steering', 1.00)
        self.declare_parameter('rd_acceleration', 0.50)
        self.declare_parameter('rd_steering', 8.00)

        self.declare_parameter('max_path_distance', 0.80)
        self.declare_parameter('max_heading_error', 1.0472)
        self.declare_parameter('odom_timeout', 0.50)
        self.declare_parameter('path_timeout', 2.00)
        self.declare_parameter('search_back_points', 5)
        self.declare_parameter('search_forward_points', 30)

        self.enabled = bool(self.get_parameter('enabled').value)
        self.solve_when_disabled = bool(
            self.get_parameter('solve_when_disabled').value)
        self.global_frame_id = self.get_parameter('global_frame_id').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.path_topic = self.get_parameter('path_topic').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.collision_topic = self.get_parameter('collision_topic').value
        self.emergency_stop_topic = self.get_parameter(
            'emergency_stop_topic').value

        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.horizon = int(self.get_parameter('horizon_steps').value)
        self.dt = float(self.get_parameter('dt').value)
        control_rate = float(self.get_parameter('control_rate').value)
        self.control_period_ms = 1000.0 / max(control_rate, 1.0)
        self.target_speed = float(self.get_parameter('target_speed').value)
        self.min_reference_speed = float(
            self.get_parameter('min_reference_speed').value)
        self.min_command_speed = float(
            self.get_parameter('min_command_speed').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.max_acceleration = float(
            self.get_parameter('max_acceleration').value)
        self.max_steering_angle = float(
            self.get_parameter('max_steering_angle').value)
        self.max_steering_rate = float(
            self.get_parameter('max_steering_rate').value)
        self.corner_slowdown_gain = float(
            self.get_parameter('corner_slowdown_gain').value)

        q = np.diag([
            float(self.get_parameter('q_x').value),
            float(self.get_parameter('q_y').value),
            float(self.get_parameter('q_speed').value),
            float(self.get_parameter('q_yaw').value),
        ])
        self.q = q
        self.qf = q * float(self.get_parameter('qf_scale').value)
        self.r = np.diag([
            float(self.get_parameter('r_acceleration').value),
            float(self.get_parameter('r_steering').value),
        ])
        self.rd = np.diag([
            float(self.get_parameter('rd_acceleration').value),
            float(self.get_parameter('rd_steering').value),
        ])

        self.max_path_distance = float(
            self.get_parameter('max_path_distance').value)
        self.max_heading_error = float(
            self.get_parameter('max_heading_error').value)
        self.odom_timeout = float(self.get_parameter('odom_timeout').value)
        self.path_timeout = float(self.get_parameter('path_timeout').value)
        self.search_back_points = int(
            self.get_parameter('search_back_points').value)
        self.search_forward_points = int(
            self.get_parameter('search_forward_points').value)

        if self.horizon < 2:
            raise RuntimeError('horizon_steps must be at least 2')
        if self.dt <= 0.0:
            raise RuntimeError('dt must be positive')
        if not 0.0 <= self.min_command_speed <= self.max_speed:
            raise RuntimeError(
                'min_command_speed must be between 0 and max_speed')

        self.current_odom = None
        self.last_odom_time = None
        self.last_path_time = None
        self.collision = False
        self.emergency_stop = False
        self.path_points = None
        self.path_yaw = None
        self.path_curvature = None
        self.path_segment_lengths = None
        self.path_cumulative = None
        self.path_length = None
        self.yaw_lap_change = None
        self.nearest_index = None
        self.previous_steering = 0.0
        self.last_solution_ok = False
        self.consecutive_solver_failures = 0
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
            Bool,
            self.emergency_stop_topic,
            self.emergency_stop_callback,
            10)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10)
        self.proposed_drive_pub = self.create_publisher(
            AckermannDriveStamped, '/mpc/proposed_drive', 10)
        self.predicted_path_pub = self.create_publisher(
            Path, '/mpc/predicted_path', 10)
        self.reference_path_pub = self.create_publisher(
            Path, '/mpc/reference_path', 10)
        self.solve_time_pub = self.create_publisher(
            Float64, '/mpc/solve_time_ms', 10)
        self.enable_service = self.create_service(
            SetBool, '/control/enable', self.enable_callback)

        self.build_mpc_problem()
        self.timer = self.create_timer(
            1.0 / max(control_rate, 1.0), self.control_loop)

        self.get_logger().info(
            '%s ready (enabled=%s, N=%d, dt=%.2fs, target=%.2fm/s)'
            % (self.DISPLAY_NAME, self.enabled, self.horizon, self.dt,
               self.target_speed))
        self.get_logger().info(
            'Disabled mode solves and visualizes predictions, but publishes stop')

    def odom_callback(self, msg):
        self.current_odom = msg
        self.last_odom_time = self.get_clock().now()

    def collision_callback(self, msg):
        self.collision = bool(msg.data)
        if self.collision and self.enabled:
            self.enabled = False
            self.publish_stop()
            self.get_logger().error('MPC disabled: simulator collision reported')

    def emergency_stop_callback(self, msg):
        self.emergency_stop = bool(msg.data)
        if self.emergency_stop:
            self.publish_stop()

    def path_callback(self, msg):
        if len(msg.poses) < 4:
            return

        points = np.asarray([
            [pose.pose.position.x, pose.pose.position.y]
            for pose in msg.poses
        ], dtype=float)
        if np.linalg.norm(points[0] - points[-1]) < 1e-4:
            points = points[:-1]
        if len(points) < 4:
            return

        changed = (
            self.path_points is None
            or len(points) != len(self.path_points)
            or np.max(np.abs(points - self.path_points)) > 1e-6
        )
        if changed:
            self.set_closed_path(points)
            self.nearest_index = None
            self.get_logger().info(
                'MPC received closed path: %d points, %.2f m'
                % (len(points), self.path_length),
                throttle_duration_sec=2.0)
        self.last_path_time = self.get_clock().now()

    def set_closed_path(self, points):
        next_points = np.roll(points, -1, axis=0)
        segments = next_points - points
        segment_lengths = np.linalg.norm(segments, axis=1)
        if np.any(segment_lengths < 1e-4):
            raise RuntimeError('Global path contains duplicate adjacent points')

        yaw = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
        direction = 1.0 if np.sum(np.diff(yaw)) >= 0.0 else -1.0
        yaw_lap_change = direction * 2.0 * math.pi
        closed_yaw = yaw[0] + yaw_lap_change
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
        curvature = (next_yaw - previous_yaw) / np.maximum(arc_span, 1e-6)

        self.path_points = points
        self.path_yaw = yaw
        self.path_curvature = curvature
        self.path_segment_lengths = segment_lengths
        self.path_cumulative = np.concatenate(
            ([0.0], np.cumsum(segment_lengths)))
        self.path_length = float(self.path_cumulative[-1])
        self.yaw_lap_change = float(yaw_lap_change)

    def build_mpc_problem(self):
        n = self.horizon
        self.state_variable = cp.Variable((self.STATE_SIZE, n + 1))
        self.input_variable = cp.Variable((self.INPUT_SIZE, n))
        self.initial_state_parameter = cp.Parameter(self.STATE_SIZE)
        self.reference_state_parameter = cp.Parameter((self.STATE_SIZE, n + 1))
        self.reference_input_parameter = cp.Parameter((self.INPUT_SIZE, n))
        self.previous_steering_parameter = cp.Parameter()
        self.a_parameters = [
            cp.Parameter((self.STATE_SIZE, self.STATE_SIZE)) for _ in range(n)
        ]
        self.b_parameters = [
            cp.Parameter((self.STATE_SIZE, self.INPUT_SIZE)) for _ in range(n)
        ]
        self.c_parameters = [cp.Parameter(self.STATE_SIZE) for _ in range(n)]

        objective = 0.0
        constraints = [
            self.state_variable[:, 0] == self.initial_state_parameter,
            self.state_variable[2, :] >= 0.0,
        ]

        for step in range(n):
            state_error = (
                self.state_variable[:, step]
                - self.reference_state_parameter[:, step])
            input_error = (
                self.input_variable[:, step]
                - self.reference_input_parameter[:, step])
            objective += cp.quad_form(state_error, self.q)
            objective += cp.quad_form(input_error, self.r)
            constraints += [
                self.state_variable[:, step + 1]
                == self.a_parameters[step] @ self.state_variable[:, step]
                + self.b_parameters[step] @ self.input_variable[:, step]
                + self.c_parameters[step],
                cp.abs(self.input_variable[0, step])
                <= self.max_acceleration,
                cp.abs(self.input_variable[1, step])
                <= self.max_steering_angle,
            ]
            if step == 0:
                constraints.append(
                    cp.abs(self.input_variable[1, 0]
                           - self.previous_steering_parameter)
                    <= self.max_steering_rate * self.dt)
            else:
                input_change = (
                    self.input_variable[:, step]
                    - self.input_variable[:, step - 1])
                objective += cp.quad_form(input_change, self.rd)
                constraints.append(
                    cp.abs(input_change[1])
                    <= self.max_steering_rate * self.dt)

        terminal_error = (
            self.state_variable[:, n]
            - self.reference_state_parameter[:, n])
        objective += cp.quad_form(terminal_error, self.qf)
        self.problem = cp.Problem(cp.Minimize(objective), constraints)

    @staticmethod
    def quaternion_to_yaw(q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @staticmethod
    def angle_difference(target, source):
        return math.atan2(
            math.sin(target - source), math.cos(target - source))

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(minimum, min(value, maximum))

    def age_seconds(self, stamp):
        if stamp is None:
            return float('inf')
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def lookup_vehicle_pose(self):
        # AMCL's map->odom correction and high-rate odometry do not always
        # share a recent timestamp while the car is stationary. Asking TF for
        # map->base_link as one chain can therefore fail or return a delayed
        # pose. Compose the newest localization correction with the newest
        # odometry pose, which is also the controller's velocity source.
        map_from_odom = self.tf_buffer.lookup_transform(
            self.global_frame_id,
            self.odom_frame_id,
            Time(),
            timeout=Duration(seconds=0.03),
        )
        map_translation = map_from_odom.transform.translation
        odom_pose = self.current_odom.pose.pose
        map_yaw = self.quaternion_to_yaw(
            map_from_odom.transform.rotation)
        odom_yaw = self.quaternion_to_yaw(odom_pose.orientation)
        cosine = math.cos(map_yaw)
        sine = math.sin(map_yaw)
        return (
            map_translation.x
            + cosine * odom_pose.position.x
            - sine * odom_pose.position.y,
            map_translation.y
            + sine * odom_pose.position.x
            + cosine * odom_pose.position.y,
            self.angle_difference(map_yaw + odom_yaw, 0.0),
        )

    def readiness_problem(self):
        if self.path_points is None:
            return 'no global path'
        if self.age_seconds(self.last_path_time) > self.path_timeout:
            return 'global path is stale'
        if self.current_odom is None:
            return 'no odometry'
        if self.age_seconds(self.last_odom_time) > self.odom_timeout:
            return 'odometry is stale'
        if self.collision:
            return 'collision is active; reset simulator pose first'
        if self.emergency_stop:
            return 'local planner emergency stop is active'
        return None

    def candidate_indices(self):
        count = len(self.path_points)
        if self.nearest_index is None:
            return range(count)
        return [
            (self.nearest_index + offset) % count
            for offset in range(-self.search_back_points,
                                self.search_forward_points + 1)
        ]

    def nearest_path_state(self, x, y):
        position = np.array([x, y])
        best = None
        count = len(self.path_points)
        for index in self.candidate_indices():
            segment = (
                self.path_points[(index + 1) % count]
                - self.path_points[index])
            relative = position - self.path_points[index]
            fraction = self.clamp(
                float(np.dot(relative, segment) / np.dot(segment, segment)),
                0.0,
                1.0,
            )
            projection = self.path_points[index] + fraction * segment
            distance = float(np.linalg.norm(projection - position))
            candidate = (distance, index, fraction)
            if best is None or candidate[0] < best[0]:
                best = candidate

        distance, nearest, fraction = best
        path_s = (
            self.path_cumulative[nearest]
            + fraction * self.path_segment_lengths[nearest])
        return nearest, distance, float(self.path_yaw[nearest]), path_s

    def interpolate_path(self, values, sample_s, lap_change=0.0):
        laps = np.floor(sample_s / self.path_length).astype(int)
        wrapped = np.mod(sample_s, self.path_length)
        closed_values = np.concatenate(([values[0]], values[1:],
                                        [values[0] + lap_change]))
        return np.interp(wrapped, self.path_cumulative, closed_values) + (
            laps * lap_change)

    def build_reference(self, x, y, yaw):
        nearest, distance, path_heading, start_s = self.nearest_path_state(x, y)
        self.nearest_index = nearest
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

        # Advance spatial references using the curvature-dependent speed.
        # Keeping a fixed target-speed spacing here would ask the position
        # states to move quickly while simultaneously asking the speed state
        # to slow down, which produces corner cutting.
        sample_s = np.empty(self.horizon + 1)
        sample_s[0] = start_s
        for step in range(self.horizon):
            step_curvature = float(self.interpolate_path(
                self.path_curvature, np.asarray([sample_s[step]]))[0])
            step_speed = self.clamp(
                self.target_speed
                / (1.0 + self.corner_slowdown_gain * abs(step_curvature)),
                self.min_reference_speed,
                self.max_speed,
            )
            sample_s[step + 1] = sample_s[step] + step_speed * self.dt
        reference = np.zeros((self.STATE_SIZE, self.horizon + 1))
        reference[0] = self.interpolate_path(
            self.path_points[:, 0], sample_s)
        reference[1] = self.interpolate_path(
            self.path_points[:, 1], sample_s)
        reference[3] = self.interpolate_path(
            self.path_yaw, sample_s, self.yaw_lap_change)
        reference[3] += 2.0 * math.pi * round(
            (yaw - reference[3, 0]) / (2.0 * math.pi))

        curvature = self.interpolate_path(self.path_curvature, sample_s)
        reference[2] = np.clip(
            self.target_speed
            / (1.0 + self.corner_slowdown_gain * np.abs(curvature)),
            self.min_reference_speed,
            self.max_speed,
        )
        reference_input = np.zeros((self.INPUT_SIZE, self.horizon))
        reference_input[1] = np.clip(
            np.arctan(self.wheelbase * curvature[:-1]),
            -self.max_steering_angle,
            self.max_steering_angle,
        )
        return reference, reference_input, distance

    def model_matrices(self, state, control):
        _, _, speed, yaw = state
        acceleration, steering = control
        dt = self.dt
        wheelbase = self.wheelbase

        a_matrix = np.eye(self.STATE_SIZE)
        a_matrix[0, 2] = dt * math.cos(yaw)
        a_matrix[0, 3] = -dt * speed * math.sin(yaw)
        a_matrix[1, 2] = dt * math.sin(yaw)
        a_matrix[1, 3] = dt * speed * math.cos(yaw)
        a_matrix[3, 2] = dt * math.tan(steering) / wheelbase

        b_matrix = np.zeros((self.STATE_SIZE, self.INPUT_SIZE))
        b_matrix[2, 0] = dt
        b_matrix[3, 1] = (
            dt * speed / (wheelbase * math.cos(steering) ** 2))

        nonlinear_next = np.array([
            state[0] + dt * speed * math.cos(yaw),
            state[1] + dt * speed * math.sin(yaw),
            speed + dt * acceleration,
            yaw + dt * speed * math.tan(steering) / wheelbase,
        ])
        c_vector = nonlinear_next - a_matrix @ state - b_matrix @ control
        return a_matrix, b_matrix, c_vector

    def solve_mpc(self, current_state, reference, reference_input):
        self.initial_state_parameter.value = current_state
        self.reference_state_parameter.value = reference
        self.reference_input_parameter.value = reference_input
        self.previous_steering_parameter.value = self.previous_steering

        for step in range(self.horizon):
            matrices = self.model_matrices(
                reference[:, step], reference_input[:, step])
            self.a_parameters[step].value = matrices[0]
            self.b_parameters[step].value = matrices[1]
            self.c_parameters[step].value = matrices[2]

        started = time.perf_counter()
        self.problem.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            eps_abs=1e-3,
            eps_rel=1e-3,
            max_iter=4000,
        )
        solve_ms = (time.perf_counter() - started) * 1000.0

        solve_msg = Float64()
        solve_msg.data = solve_ms
        self.solve_time_pub.publish(solve_msg)

        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError('solver status: ' + str(self.problem.status))
        if self.input_variable.value is None or self.state_variable.value is None:
            raise RuntimeError('solver returned no solution')

        return (
            np.asarray(self.input_variable.value),
            np.asarray(self.state_variable.value),
            solve_ms,
        )

    def publish_path(self, publisher, states):
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.global_frame_id
        for index in range(states.shape[1]):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(states[0, index])
            pose.pose.position.y = float(states[1, index])
            yaw = float(states[3, index])
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            pose.pose.orientation.w = math.cos(yaw * 0.5)
            message.poses.append(pose)
        publisher.publish(message)

    def publish_drive(self, speed, steering):
        self.publish_ackermann(self.drive_pub, speed, steering)

    def publish_proposed_drive(self, speed, steering):
        self.publish_ackermann(self.proposed_drive_pub, speed, steering)

    def publish_ackermann(self, publisher, speed, steering):
        message = AckermannDriveStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.base_frame_id
        message.drive.speed = float(speed)
        message.drive.steering_angle = float(steering)
        publisher.publish(message)

    def publish_stop(self):
        self.publish_drive(0.0, 0.0)

    def warn_throttled(self, message):
        now = self.get_clock().now()
        if (message != self.last_status_message
                or self.last_status_time is None
                or (now - self.last_status_time).nanoseconds > 2_000_000_000):
            self.get_logger().warn(message)
            self.last_status_message = message
            self.last_status_time = now

    def enable_callback(self, request, response):
        if not request.data:
            self.enabled = False
            self.previous_steering = 0.0
            self.publish_stop()
            response.success = True
            response.message = self.DISPLAY_NAME + ' stopped'
            self.get_logger().info(response.message)
            return response

        problem = self.readiness_problem()
        if problem is not None:
            response.success = False
            response.message = 'Cannot start MPC: ' + problem
            self.get_logger().error(response.message)
            return response
        if not self.last_solution_ok:
            response.success = False
            response.message = 'Cannot start MPC: no valid dry-run solution yet'
            self.get_logger().error(response.message)
            return response

        try:
            x, y, yaw = self.lookup_vehicle_pose()
            self.nearest_index = None
            self.build_reference(x, y, yaw)
        except (TransformException, RuntimeError) as error:
            response.success = False
            response.message = 'Cannot start MPC: ' + str(error)
            self.get_logger().error(response.message)
            return response

        self.enabled = True
        self.consecutive_solver_failures = 0
        response.success = True
        response.message = self.DISPLAY_NAME + ' enabled'
        self.get_logger().info(response.message)
        return response

    def control_loop(self):
        problem = self.readiness_problem()
        if problem is not None:
            self.last_solution_ok = False
            self.publish_stop()
            if self.enabled:
                self.warn_throttled('MPC safety stop: ' + problem)
            return
        if not self.enabled and not self.solve_when_disabled:
            self.publish_stop()
            return

        try:
            x, y, yaw = self.lookup_vehicle_pose()
            speed = max(0.0, float(self.current_odom.twist.twist.linear.x))
            reference, reference_input, _ = self.build_reference(x, y, yaw)
            yaw = reference[3, 0] + self.angle_difference(yaw, reference[3, 0])
            current_state = np.array([x, y, speed, yaw])
            control, predicted, solve_ms = self.solve_mpc(
                current_state, reference, reference_input)
            self.publish_path(self.reference_path_pub, reference)
            self.publish_path(self.predicted_path_pub, predicted)
            self.last_solution_ok = True
            self.consecutive_solver_failures = 0

            acceleration = float(control[0, 0])
            steering = self.clamp(
                float(control[1, 0]),
                -self.max_steering_angle,
                self.max_steering_angle,
            )
            command_speed = self.clamp(
                speed + acceleration * self.dt, 0.0, self.max_speed)
            if command_speed > 0.0:
                command_speed = max(command_speed, self.min_command_speed)
            self.publish_proposed_drive(command_speed, steering)
            if not self.enabled:
                self.publish_stop()
                return
            # Track the last command that was actually sent to the vehicle.
            # Dry-run predictions must not pre-load the steering-rate state,
            # otherwise enabling MPC could jump directly to a large angle.
            self.previous_steering = steering
            self.publish_drive(command_speed, steering)

            if solve_ms > self.control_period_ms:
                self.warn_throttled(
                    'MPC solve time %.1f ms exceeds %.1f ms control period'
                    % (solve_ms, self.control_period_ms))
        except (TransformException, RuntimeError, cp.error.SolverError) as error:
            self.last_solution_ok = False
            self.consecutive_solver_failures += 1
            self.publish_stop()
            self.warn_throttled('MPC safety stop: ' + str(error))
            if self.enabled and self.consecutive_solver_failures >= 3:
                self.enabled = False
                self.get_logger().error(
                    'MPC disabled after 3 consecutive solver/control failures')


def main(args=None):
    rclpy.init(args=args)
    node = LinearMpcNode()
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
