#!/usr/bin/env python3
"""Run a time-limited controller test and report path-tracking error."""

import argparse
import csv
import math
import os
import statistics
import sys
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float64, String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener


class ClosedLoopTest(Node):
    def __init__(self):
        super().__init__('closed_loop_test')
        self.path = None
        self.global_path = None
        self.collision = False
        self.emergency_stop = False
        self.avoidance_active = False
        self.local_status = ''
        self.ground_truth_xy = None
        self.estimated_xy = None
        self.actual_speed = 0.0
        self.command_speed = 0.0
        self.command_steering = 0.0
        self.solve_time_ms = math.nan
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Path, '/planning/path', self.path_callback, 10)
        self.create_subscription(
            Path, '/planning/global_path', self.global_path_callback, 10)
        self.create_subscription(
            Bool, '/ego_racecar/collision', self.collision_callback, 10)
        self.create_subscription(
            Odometry, '/ground_truth/odom', self.ground_truth_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose',
            self.amcl_pose_callback, 10)
        self.create_subscription(
            AckermannDriveStamped, '/drive', self.drive_callback, 10)
        self.create_subscription(
            Bool, '/safety/emergency_stop', self.emergency_stop_callback, 10)
        self.create_subscription(
            Bool, '/planning/avoidance_active', self.avoidance_callback, 10)
        self.create_subscription(
            String, '/planning/local_status', self.status_callback, 10)
        self.create_subscription(
            Float64, '/mpc/solve_time_ms', self.solve_time_callback, 10)
        self.enable_client = self.create_client(SetBool, '/control/enable')

    def path_callback(self, msg):
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(points) > 2 and math.dist(points[0], points[-1]) < 1e-4:
            points.pop()
        self.path = points
        # With the local planner disabled, the waypoint planner publishes
        # directly to /planning/path and /planning/global_path does not exist.
        # Use the same immutable path as the global progress reference.
        if self.global_path is None:
            self.global_path = list(points)

    def global_path_callback(self, msg):
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(points) > 2 and math.dist(points[0], points[-1]) < 1e-4:
            points.pop()
        self.global_path = points

    def collision_callback(self, msg):
        self.collision = msg.data

    def ground_truth_callback(self, msg):
        position = msg.pose.pose.position
        self.ground_truth_xy = (position.x, position.y)
        self.actual_speed = float(msg.twist.twist.linear.x)

    def amcl_pose_callback(self, msg):
        position = msg.pose.pose.position
        self.estimated_xy = (position.x, position.y)

    def drive_callback(self, msg):
        self.command_speed = float(msg.drive.speed)
        self.command_steering = float(msg.drive.steering_angle)

    def emergency_stop_callback(self, msg):
        self.emergency_stop = bool(msg.data)

    def avoidance_callback(self, msg):
        self.avoidance_active = bool(msg.data)

    def status_callback(self, msg):
        self.local_status = msg.data

    def solve_time_callback(self, msg):
        self.solve_time_ms = float(msg.data)

    def vehicle_xy(self):
        if self.estimated_xy is not None:
            return self.estimated_xy
        # AMCL publishes only after the robot moves.  During the stationary
        # pre-start phase use simulator truth for test-harness readiness; the
        # controller itself must still establish its configured TF before its
        # enable service succeeds.
        if self.ground_truth_xy is not None:
            return self.ground_truth_xy
        transform = self.tf_buffer.lookup_transform(
            'map', 'ego_racecar/base_link', Time(),
            timeout=Duration(seconds=0.05))
        t = transform.transform.translation
        return t.x, t.y

    def set_enabled(self, enabled):
        request = SetBool.Request()
        request.data = enabled
        future = self.enable_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result() is None:
            raise RuntimeError('No response from /control/enable')
        result = future.result()
        if not result.success:
            raise RuntimeError(result.message)
        return result.message


def nearest_index_and_distance(points, x, y, previous_index=None):
    if previous_index is None:
        indices = range(len(points))
    else:
        indices = [
            (previous_index + offset) % len(points)
            for offset in range(-4, 21)
        ]
    return min(
        ((index, math.hypot(points[index][0] - x, points[index][1] - y))
         for index in indices),
        key=lambda item: item[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=45.0)
    parser.add_argument('--laps', type=float, default=0.0,
                        help='Stop after this many laps; 0 disables lap stop')
    parser.add_argument('--max-error', type=float, default=0.75)
    parser.add_argument(
        '--stall-timeout', type=float, default=5.0,
        help='Fail when the vehicle makes no meaningful progress for N seconds')
    parser.add_argument('--output', default='/tmp/track02_closed_loop.csv')
    parser.add_argument(
        '--already-enabled', action='store_true',
        help='Record an already-running controller, then stop it on exit')
    args = parser.parse_args()

    rclpy.init()
    node = ClosedLoopTest()
    samples = []
    distance_travelled = 0.0
    progress_points = 0
    previous_truth_xy = None
    previous_index = None
    previous_elapsed = 0.0
    next_report = 0.0
    enabled = False
    failure_reason = None
    avoidance_time = 0.0
    emergency_stop_time = 0.0
    lap_times = []
    completed_laps = 0
    previous_lap_elapsed = 0.0

    try:
        deadline = time.monotonic() + 8.0
        while (node.path is None or
               not node.enable_client.wait_for_service(timeout_sec=0.1)):
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() > deadline:
                raise RuntimeError('Path or /control/enable service unavailable')

        # Wait for the simulator measurement stream. AMCL does not necessarily
        # emit a new pose while the vehicle is stationary.
        deadline = time.monotonic() + 8.0
        while node.ground_truth_xy is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() > deadline:
                raise RuntimeError('ground-truth odometry unavailable')

        if not args.already_enabled:
            print(node.set_enabled(True), flush=True)
        enabled = True
        started = time.monotonic()
        last_motion_time = started
        next_report = started

        while time.monotonic() - started < args.duration:
            rclpy.spin_once(node, timeout_sec=0.04)
            now = time.monotonic()
            try:
                estimated_x, estimated_y = node.vehicle_xy()
            except TransformException:
                continue

            _, estimated_error = nearest_index_and_distance(
                node.path, estimated_x, estimated_y, previous_index)
            truth_x, truth_y = (
                node.ground_truth_xy
                if node.ground_truth_xy is not None
                else (estimated_x, estimated_y))
            nearest_index, global_truth_error = nearest_index_and_distance(
                node.global_path, truth_x, truth_y, previous_index)
            _, local_truth_error = nearest_index_and_distance(
                node.path, truth_x, truth_y, None)
            localization_error = math.hypot(
                estimated_x - truth_x, estimated_y - truth_y)
            if previous_truth_xy is not None:
                step_distance = math.dist(
                    previous_truth_xy, (truth_x, truth_y))
                distance_travelled += step_distance
                if step_distance > 1e-3 or abs(node.actual_speed) > 0.10:
                    last_motion_time = now
            previous_truth_xy = (truth_x, truth_y)

            if previous_index is not None:
                delta = (
                    (nearest_index - previous_index) % len(node.global_path))
                if delta > len(node.global_path) // 2:
                    delta -= len(node.global_path)
                progress_points += delta
            previous_index = nearest_index

            elapsed = now - started
            sample_dt = max(0.0, elapsed - previous_elapsed)
            previous_elapsed = elapsed
            if node.avoidance_active:
                avoidance_time += sample_dt
            if node.emergency_stop:
                emergency_stop_time += sample_dt
            samples.append((
                elapsed, estimated_x, estimated_y, truth_x, truth_y,
                nearest_index, estimated_error, local_truth_error,
                global_truth_error, localization_error,
                node.actual_speed, node.command_speed,
                node.command_steering, int(node.avoidance_active),
                int(node.emergency_stop), node.local_status,
                node.solve_time_ms))
            if node.collision:
                failure_reason = 'Collision reported by F1TENTH Gym'
                break
            if now - last_motion_time > max(0.1, args.stall_timeout):
                failure_reason = (
                    'Vehicle stalled for %.1f s (%s)'
                    % (args.stall_timeout, node.local_status or 'no status'))
                break
            # During obstacle avoidance the selected local path intentionally
            # leaves the global raceline. Safety is tracking that selected
            # path without a simulator collision, not minimizing global CTE.
            if local_truth_error > args.max_error:
                failure_reason = (
                    'Safety limit exceeded: selected-path error %.3f m'
                    % local_truth_error)
                break

            current_laps = max(
                0, progress_points // max(len(node.global_path), 1))
            while completed_laps < current_laps:
                completed_laps += 1
                lap_times.append(elapsed - previous_lap_elapsed)
                previous_lap_elapsed = elapsed

            if now >= next_report:
                print(
                    't=%5.1fs  truth=(%6.2f,%6.2f)  true_cte=%.3fm  '
                    'amcl_error=%.3fm  progress=%.2f laps' % (
                        elapsed, truth_x, truth_y, global_truth_error,
                        localization_error,
                        progress_points / max(len(node.global_path), 1)),
                    flush=True)
                next_report = now + 2.0
            if (args.laps > 0.0 and
                    progress_points / max(
                        len(node.global_path), 1) >= args.laps):
                break
    finally:
        if enabled:
            try:
                print(node.set_enabled(False), flush=True)
            except Exception as error:
                print('STOP SERVICE FAILED:', error, flush=True)
        node.destroy_node()
        rclpy.shutdown()

    if not samples:
        raise RuntimeError('No tracking samples collected')

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            'time_s', 'x_m', 'y_m', 'truth_x_m', 'truth_y_m',
            'nearest_index', 'estimated_local_cte_m', 'truth_local_cte_m',
            'truth_global_cte_m', 'localization_error_m',
            'actual_speed_mps', 'command_speed_mps',
            'command_steering_rad', 'avoidance_active', 'emergency_stop',
            'local_status', 'solve_time_ms'])
        writer.writerows(samples)

    estimated_errors = [sample[6] for sample in samples]
    local_truth_errors = [sample[7] for sample in samples]
    global_truth_errors = [sample[8] for sample in samples]
    localization_errors = [sample[9] for sample in samples]
    actual_speeds = [sample[10] for sample in samples]
    command_speeds = [sample[11] for sample in samples]
    solve_times = [sample[16] for sample in samples
                   if math.isfinite(sample[16])]

    def p95(values):
        ordered = sorted(values)
        return ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)]

    print('RESULT', flush=True)
    print('  duration_s       : %.2f' % samples[-1][0], flush=True)
    print('  distance_m       : %.2f' % distance_travelled, flush=True)
    print('  progress_laps    : %.3f' % (
        progress_points / max(len(node.global_path or []), 1)), flush=True)
    print('  lap_times_s      : %s' % (
        ', '.join('%.2f' % value for value in lap_times) or 'none'),
        flush=True)
    print('  actual_speed_mean_mps : %.3f' % statistics.fmean(
        actual_speeds), flush=True)
    print('  actual_speed_p95_mps  : %.3f' % p95(actual_speeds), flush=True)
    print('  actual_speed_max_mps  : %.3f' % max(actual_speeds), flush=True)
    print('  command_speed_mean_mps: %.3f' % statistics.fmean(
        command_speeds), flush=True)
    print('  command_speed_max_mps : %.3f' % max(command_speeds), flush=True)
    if solve_times:
        print('  solve_time_mean_ms    : %.3f' % statistics.fmean(
            solve_times), flush=True)
        print('  solve_time_p95_ms     : %.3f' % p95(solve_times), flush=True)
        print('  solve_time_max_ms     : %.3f' % max(solve_times), flush=True)
    print('  avoidance_time_s      : %.3f' % avoidance_time, flush=True)
    print('  emergency_stop_time_s : %.3f' % emergency_stop_time, flush=True)
    print('  estimated_cte_mean_m : %.3f' % statistics.fmean(
        estimated_errors), flush=True)
    print('  estimated_cte_p95_m  : %.3f' % p95(estimated_errors), flush=True)
    print('  estimated_cte_max_m  : %.3f' % max(estimated_errors), flush=True)
    print('  local_cte_mean_m     : %.3f' % statistics.fmean(
        local_truth_errors), flush=True)
    print('  local_cte_p95_m      : %.3f' % p95(local_truth_errors), flush=True)
    print('  local_cte_max_m      : %.3f' % max(local_truth_errors), flush=True)
    print('  global_cte_mean_m    : %.3f' % statistics.fmean(
        global_truth_errors), flush=True)
    print('  global_cte_p95_m     : %.3f' % p95(global_truth_errors), flush=True)
    print('  global_cte_max_m     : %.3f' % max(global_truth_errors), flush=True)
    print('  localization_mean_m  : %.3f' % statistics.fmean(
        localization_errors), flush=True)
    print('  localization_p95_m   : %.3f' % p95(
        localization_errors), flush=True)
    print('  samples          : %d' % len(samples), flush=True)
    print('  csv              : %s' % args.output, flush=True)
    print('  outcome          : %s' % (
        'PASS' if failure_reason is None else 'FAIL: ' + failure_reason),
        flush=True)

    if failure_reason is not None:
        print(failure_reason, file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
