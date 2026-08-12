#!/usr/bin/env python3
"""Run a time-limited controller test and report path-tracking error."""

import argparse
import csv
import math
import os
import statistics
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener


class ClosedLoopTest(Node):
    def __init__(self):
        super().__init__('closed_loop_test')
        self.path = None
        self.collision = False
        self.ground_truth_xy = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Path, '/planning/path', self.path_callback, 10)
        self.create_subscription(
            Bool, '/ego_racecar/collision', self.collision_callback, 10)
        self.create_subscription(
            Odometry, '/ground_truth/odom', self.ground_truth_callback, 10)
        self.enable_client = self.create_client(SetBool, '/control/enable')

    def path_callback(self, msg):
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(points) > 2 and math.dist(points[0], points[-1]) < 1e-4:
            points.pop()
        self.path = points

    def collision_callback(self, msg):
        self.collision = msg.data

    def ground_truth_callback(self, msg):
        position = msg.pose.pose.position
        self.ground_truth_xy = (position.x, position.y)

    def vehicle_xy(self):
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
    parser.add_argument('--output', default='/tmp/track02_closed_loop.csv')
    args = parser.parse_args()

    rclpy.init()
    node = ClosedLoopTest()
    samples = []
    distance_travelled = 0.0
    progress_points = 0
    previous_truth_xy = None
    previous_index = None
    next_report = 0.0
    enabled = False

    try:
        deadline = time.monotonic() + 8.0
        while (node.path is None or
               not node.enable_client.wait_for_service(timeout_sec=0.1)):
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() > deadline:
                raise RuntimeError('Path or /control/enable service unavailable')

        while True:
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                node.vehicle_xy()
                break
            except TransformException:
                if time.monotonic() > deadline:
                    raise RuntimeError('map -> ego_racecar/base_link unavailable')

        print(node.set_enabled(True), flush=True)
        enabled = True
        started = time.monotonic()
        next_report = started

        while time.monotonic() - started < args.duration:
            rclpy.spin_once(node, timeout_sec=0.04)
            now = time.monotonic()
            try:
                estimated_x, estimated_y = node.vehicle_xy()
            except TransformException:
                continue

            nearest_index, estimated_error = nearest_index_and_distance(
                node.path, estimated_x, estimated_y, previous_index)
            truth_x, truth_y = (
                node.ground_truth_xy
                if node.ground_truth_xy is not None
                else (estimated_x, estimated_y))
            _, truth_error = nearest_index_and_distance(
                node.path, truth_x, truth_y, nearest_index)
            localization_error = math.hypot(
                estimated_x - truth_x, estimated_y - truth_y)
            if previous_truth_xy is not None:
                distance_travelled += math.dist(
                    previous_truth_xy, (truth_x, truth_y))
            previous_truth_xy = (truth_x, truth_y)

            if previous_index is not None:
                delta = (nearest_index - previous_index) % len(node.path)
                if delta > len(node.path) // 2:
                    delta -= len(node.path)
                progress_points += delta
            previous_index = nearest_index

            elapsed = now - started
            samples.append((
                elapsed, estimated_x, estimated_y, truth_x, truth_y,
                nearest_index, estimated_error, truth_error,
                localization_error))
            if node.collision:
                raise RuntimeError('Collision reported by F1TENTH Gym')
            if truth_error > args.max_error:
                raise RuntimeError(
                    'Safety limit exceeded: ground-truth path error %.3f m'
                    % truth_error)

            if now >= next_report:
                print(
                    't=%5.1fs  truth=(%6.2f,%6.2f)  true_cte=%.3fm  '
                    'amcl_error=%.3fm  progress=%.2f laps' % (
                        elapsed, truth_x, truth_y, truth_error,
                        localization_error,
                        progress_points / max(len(node.path), 1)),
                    flush=True)
                next_report = now + 2.0
            if (args.laps > 0.0 and
                    progress_points / max(len(node.path), 1) >= args.laps):
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
            'nearest_index', 'error_m', 'truth_error_m',
            'localization_error_m'])
        writer.writerows(samples)

    estimated_errors = [sample[6] for sample in samples]
    truth_errors = [sample[7] for sample in samples]
    localization_errors = [sample[8] for sample in samples]

    def p95(values):
        ordered = sorted(values)
        return ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)]

    print('RESULT', flush=True)
    print('  duration_s       : %.2f' % samples[-1][0], flush=True)
    print('  distance_m       : %.2f' % distance_travelled, flush=True)
    print('  progress_laps    : %.3f' % (
        progress_points / max(len(node.path or []), 1)), flush=True)
    print('  estimated_cte_mean_m : %.3f' % statistics.fmean(
        estimated_errors), flush=True)
    print('  estimated_cte_p95_m  : %.3f' % p95(estimated_errors), flush=True)
    print('  estimated_cte_max_m  : %.3f' % max(estimated_errors), flush=True)
    print('  truth_cte_mean_m     : %.3f' % statistics.fmean(
        truth_errors), flush=True)
    print('  truth_cte_p95_m      : %.3f' % p95(truth_errors), flush=True)
    print('  truth_cte_max_m      : %.3f' % max(truth_errors), flush=True)
    print('  localization_mean_m  : %.3f' % statistics.fmean(
        localization_errors), flush=True)
    print('  localization_p95_m   : %.3f' % p95(
        localization_errors), flush=True)
    print('  samples          : %d' % len(samples), flush=True)
    print('  csv              : %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
