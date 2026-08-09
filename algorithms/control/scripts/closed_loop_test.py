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

from nav_msgs.msg import Path
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener


class ClosedLoopTest(Node):
    def __init__(self):
        super().__init__('closed_loop_test')
        self.path = None
        self.collision = False
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Path, '/planning/path', self.path_callback, 10)
        self.create_subscription(
            Bool, '/ego_racecar/collision', self.collision_callback, 10)
        self.enable_client = self.create_client(SetBool, '/control/enable')

    def path_callback(self, msg):
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(points) > 2 and math.dist(points[0], points[-1]) < 1e-4:
            points.pop()
        self.path = points

    def collision_callback(self, msg):
        self.collision = msg.data

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
    previous_xy = None
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
                x, y = node.vehicle_xy()
            except TransformException:
                continue

            nearest_index, error = nearest_index_and_distance(
                node.path, x, y, previous_index)
            if previous_xy is not None:
                distance_travelled += math.dist(previous_xy, (x, y))
            previous_xy = (x, y)

            if previous_index is not None:
                delta = (nearest_index - previous_index) % len(node.path)
                if delta > len(node.path) // 2:
                    delta -= len(node.path)
                progress_points += delta
            previous_index = nearest_index

            elapsed = now - started
            samples.append((elapsed, x, y, nearest_index, error))
            if node.collision:
                raise RuntimeError('Collision reported by F1TENTH Gym')
            if error > args.max_error:
                raise RuntimeError(
                    'Safety limit exceeded: path error %.3f m' % error)

            if now >= next_report:
                print(
                    't=%5.1fs  pose=(%6.2f,%6.2f)  path_error=%.3fm  '
                    'progress=%.2f laps' % (
                        elapsed, x, y, error,
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
        writer.writerow(['time_s', 'x_m', 'y_m', 'nearest_index', 'error_m'])
        writer.writerows(samples)

    errors = [sample[-1] for sample in samples]
    errors_sorted = sorted(errors)
    p95 = errors_sorted[min(int(0.95 * len(errors_sorted)), len(errors_sorted) - 1)]
    print('RESULT', flush=True)
    print('  duration_s       : %.2f' % samples[-1][0], flush=True)
    print('  distance_m       : %.2f' % distance_travelled, flush=True)
    print('  progress_laps    : %.3f' % (
        progress_points / max(len(node.path or []), 1)), flush=True)
    print('  mean_error_m     : %.3f' % statistics.fmean(errors), flush=True)
    print('  p95_error_m      : %.3f' % p95, flush=True)
    print('  max_error_m      : %.3f' % max(errors), flush=True)
    print('  samples          : %d' % len(samples), flush=True)
    print('  csv              : %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
