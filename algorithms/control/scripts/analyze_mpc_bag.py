#!/usr/bin/env python3
"""Summarize an F1TENTH MPC rosbag recorded by this package."""

import argparse
import bisect
import math
import os
import sqlite3
import statistics

import numpy as np
from rclpy.serialization import deserialize_message

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float64


MESSAGE_TYPES = {
    '/mpc/solve_time_ms': Float64,
    '/drive': AckermannDriveStamped,
    '/ego_racecar/collision': Bool,
    '/amcl_pose': PoseWithCovarianceStamped,
    '/ego_racecar/odom': Odometry,
    '/ground_truth/odom': Odometry,
    '/planning/path': Path,
    '/mpc/predicted_path': Path,
}


def percentile(values, percentile_value):
    ordered = sorted(values)
    if not ordered:
        return float('nan')
    index = min(int(percentile_value * len(ordered)), len(ordered) - 1)
    return ordered[index]


def locate_database(path):
    if os.path.isfile(path):
        return path
    databases = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.endswith('.db3'))
    if len(databases) != 1:
        raise RuntimeError(
            'Expected exactly one .db3 file in %s, found %d'
            % (path, len(databases)))
    return databases[0]


def read_topic(connection, topic_name):
    topic = connection.execute(
        'SELECT id, type FROM topics WHERE name = ?', (topic_name,)
    ).fetchone()
    if topic is None:
        raise RuntimeError('Topic is missing from bag: ' + topic_name)
    expected_type = MESSAGE_TYPES[topic_name]
    rows = connection.execute(
        'SELECT timestamp, data FROM messages WHERE topic_id = ? '
        'ORDER BY timestamp', (topic[0],))
    return [
        (timestamp, deserialize_message(data, expected_type))
        for timestamp, data in rows
    ]


def read_optional_topic(connection, topic_name):
    topic = connection.execute(
        'SELECT id FROM topics WHERE name = ?', (topic_name,)
    ).fetchone()
    if topic is None:
        return []
    return read_topic(connection, topic_name)


def pose_xy(message):
    if isinstance(message, Odometry):
        position = message.pose.pose.position
    else:
        position = message.pose.pose.position
    return position.x, position.y


def pose_yaw(message):
    orientation = message.pose.pose.orientation
    return math.atan2(
        2.0 * (orientation.w * orientation.z
               + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y
                     + orientation.z * orientation.z))


def angle_difference(first, second):
    return math.atan2(math.sin(first - second), math.cos(first - second))


def nearest_rows(source_rows, target_rows, active_start, active_end):
    target_times = [timestamp for timestamp, _ in target_rows]
    for timestamp, source_message in source_rows:
        if not active_start <= timestamp <= active_end:
            continue
        insertion = bisect.bisect_left(target_times, timestamp)
        candidates = [
            index for index in (insertion - 1, insertion)
            if 0 <= index < len(target_rows)
        ]
        if not candidates:
            continue
        nearest = min(
            candidates, key=lambda index: abs(target_times[index] - timestamp))
        yield timestamp, source_message, target_rows[nearest][1], abs(
            target_times[nearest] - timestamp) * 1e-6


def path_points(path_rows):
    if not path_rows:
        return None
    poses = path_rows[0][1].poses
    points = np.asarray([
        [pose.pose.position.x, pose.pose.position.y] for pose in poses
    ], dtype=float)
    if len(points) > 2 and np.linalg.norm(points[0] - points[-1]) < 1e-6:
        points = points[:-1]
    return points


def point_to_closed_path_distance(point, points):
    starts = points
    ends = np.roll(points, -1, axis=0)
    segments = ends - starts
    squared_lengths = np.sum(segments * segments, axis=1)
    relative = np.asarray(point) - starts
    fractions = np.divide(
        np.sum(relative * segments, axis=1), squared_lengths,
        out=np.zeros_like(squared_lengths), where=squared_lengths > 1e-12)
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = starts + fractions[:, None] * segments
    return float(np.min(np.linalg.norm(projections - point, axis=1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', help='Rosbag directory or sqlite3 .db3 file')
    parser.add_argument(
        '--mpc-dt', type=float, default=0.10,
        help='Seconds between poses in /mpc/predicted_path')
    args = parser.parse_args()

    database = locate_database(args.bag)
    connection = sqlite3.connect('file:%s?mode=ro' % database, uri=True)
    try:
        solve_rows = read_topic(connection, '/mpc/solve_time_ms')
        drive_rows = read_topic(connection, '/drive')
        collision_rows = read_topic(connection, '/ego_racecar/collision')
        amcl_rows = read_topic(connection, '/amcl_pose')
        raw_odom_rows = read_topic(connection, '/ego_racecar/odom')
        ground_truth_rows = read_optional_topic(
            connection, '/ground_truth/odom')
        planning_path_rows = read_optional_topic(connection, '/planning/path')
        predicted_path_rows = read_optional_topic(
            connection, '/mpc/predicted_path')
    finally:
        connection.close()

    active_drive = [
        (timestamp, message) for timestamp, message in drive_rows
        if abs(message.drive.speed) > 1e-4
    ]
    if not active_drive:
        raise RuntimeError('Bag contains no non-zero drive commands')
    active_start = active_drive[0][0]
    active_end = active_drive[-1][0]

    solve_times = [
        message.data for timestamp, message in solve_rows
        if active_start <= timestamp <= active_end
    ]
    speeds = [message.drive.speed for _, message in active_drive]
    steerings = [message.drive.steering_angle for _, message in active_drive]
    collision_true = sum(
        1 for timestamp, message in collision_rows
        if active_start <= timestamp <= active_end and message.data)

    truth_source = '/ground_truth/odom'
    if not ground_truth_rows:
        # Older bags predate the explicit topic. In this simulator the bridge's
        # raw odometry pose is copied directly from the Gym state, so it is a
        # valid fallback even though its frame is named odom.
        ground_truth_rows = raw_odom_rows
        truth_source = '/ego_racecar/odom (Gym-state fallback)'
    localization_errors = []
    localization_yaw_errors = []
    localization_match_gaps = []
    for _, amcl_message, truth_message, match_gap_ms in nearest_rows(
            amcl_rows, ground_truth_rows, active_start, active_end):
        amcl_x, amcl_y = pose_xy(amcl_message)
        truth_x, truth_y = pose_xy(truth_message)
        localization_errors.append(
            math.hypot(amcl_x - truth_x, amcl_y - truth_y))
        localization_yaw_errors.append(abs(angle_difference(
            pose_yaw(amcl_message), pose_yaw(truth_message))))
        localization_match_gaps.append(match_gap_ms)

    reference_points = path_points(planning_path_rows)
    truth_cte = []
    if reference_points is not None:
        for timestamp, truth_message in ground_truth_rows:
            if active_start <= timestamp <= active_end:
                truth_cte.append(point_to_closed_path_distance(
                    pose_xy(truth_message), reference_points))

    prediction_errors = {0.5: [], 1.0: [], 1.5: []}
    truth_times = [timestamp for timestamp, _ in ground_truth_rows]
    for timestamp, predicted_message in predicted_path_rows:
        if not active_start <= timestamp <= active_end:
            continue
        for horizon_seconds, values in prediction_errors.items():
            pose_index = int(round(horizon_seconds / args.mpc_dt))
            if pose_index >= len(predicted_message.poses):
                continue
            target_time = timestamp + int(horizon_seconds * 1e9)
            insertion = bisect.bisect_left(truth_times, target_time)
            candidates = [
                index for index in (insertion - 1, insertion)
                if 0 <= index < len(ground_truth_rows)
            ]
            if not candidates:
                continue
            nearest = min(
                candidates,
                key=lambda index: abs(truth_times[index] - target_time))
            # Ignore truncated bag tails and large timing mismatches.
            if abs(truth_times[nearest] - target_time) > 0.05 * 1e9:
                continue
            predicted_position = predicted_message.poses[
                pose_index].pose.position
            truth_x, truth_y = pose_xy(ground_truth_rows[nearest][1])
            values.append(math.hypot(
                predicted_position.x - truth_x,
                predicted_position.y - truth_y))

    active_duration = (active_end - active_start) * 1e-9
    print('MPC BAG SUMMARY')
    print('  database                  : %s' % database)
    print('  active_duration_s         : %.3f' % active_duration)
    print('  active_drive_messages     : %d' % len(active_drive))
    print('  command_rate_hz           : %.2f' % (
        (len(active_drive) - 1) / max(active_duration, 1e-9)))
    print('  speed_mean_mps            : %.3f' % statistics.fmean(speeds))
    print('  speed_max_mps             : %.3f' % max(speeds))
    print('  abs_steering_mean_rad     : %.3f' % statistics.fmean(map(abs, steerings)))
    print('  abs_steering_max_rad      : %.3f' % max(map(abs, steerings)))
    print('  solver_mean_ms            : %.3f' % statistics.fmean(solve_times))
    print('  solver_p95_ms             : %.3f' % percentile(solve_times, 0.95))
    print('  solver_max_ms             : %.3f' % max(solve_times))
    print('  collision_true_messages   : %d' % collision_true)
    print('  localization_truth_source : %s' % truth_source)
    if localization_errors:
        print('  amcl_position_mean_m      : %.3f' % (
            statistics.fmean(localization_errors)))
        print('  amcl_position_p95_m       : %.3f' % (
            percentile(localization_errors, 0.95)))
        print('  amcl_position_max_m       : %.3f' % max(localization_errors))
        print('  amcl_yaw_mean_deg         : %.2f' % math.degrees(
            statistics.fmean(localization_yaw_errors)))
        print('  amcl_yaw_p95_deg          : %.2f' % math.degrees(
            percentile(localization_yaw_errors, 0.95)))
        print('  amcl_yaw_max_deg          : %.2f' % math.degrees(
            max(localization_yaw_errors)))
        print('  localization_match_p95_ms : %.3f' % percentile(
            localization_match_gaps, 0.95))
    if truth_cte:
        print('  ground_truth_cte_mean_m   : %.3f' % statistics.fmean(truth_cte))
        print('  ground_truth_cte_p95_m    : %.3f' % percentile(truth_cte, 0.95))
        print('  ground_truth_cte_max_m    : %.3f' % max(truth_cte))
    for horizon_seconds, values in prediction_errors.items():
        if values:
            label = 'prediction_%.1fs' % horizon_seconds
            print('  %-27s: %.3f / %.3f / %.3f' % (
                label + '_mean/p95/max_m',
                statistics.fmean(values),
                percentile(values, 0.95),
                max(values)))


if __name__ == '__main__':
    main()
