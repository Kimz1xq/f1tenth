#!/usr/bin/env python3
"""Summarize an F1TENTH MPC rosbag recorded by this package."""

import argparse
import bisect
import math
import os
import sqlite3
import statistics

from rclpy.serialization import deserialize_message

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64


MESSAGE_TYPES = {
    '/mpc/solve_time_ms': Float64,
    '/drive': AckermannDriveStamped,
    '/ego_racecar/collision': Bool,
    '/amcl_pose': PoseWithCovarianceStamped,
    '/ego_racecar/odom': Odometry,
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


def pose_xy(message):
    if isinstance(message, Odometry):
        position = message.pose.pose.position
    else:
        position = message.pose.pose.position
    return position.x, position.y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', help='Rosbag directory or sqlite3 .db3 file')
    args = parser.parse_args()

    database = locate_database(args.bag)
    connection = sqlite3.connect('file:%s?mode=ro' % database, uri=True)
    try:
        solve_rows = read_topic(connection, '/mpc/solve_time_ms')
        drive_rows = read_topic(connection, '/drive')
        collision_rows = read_topic(connection, '/ego_racecar/collision')
        amcl_rows = read_topic(connection, '/amcl_pose')
        odom_rows = read_topic(connection, '/ego_racecar/odom')
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

    odom_times = [timestamp for timestamp, _ in odom_rows]
    localization_errors = []
    for timestamp, amcl_message in amcl_rows:
        if not active_start <= timestamp <= active_end:
            continue
        insertion = bisect.bisect_left(odom_times, timestamp)
        candidates = [
            index for index in (insertion - 1, insertion)
            if 0 <= index < len(odom_rows)
        ]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda i: abs(odom_times[i] - timestamp))
        amcl_x, amcl_y = pose_xy(amcl_message)
        odom_x, odom_y = pose_xy(odom_rows[nearest][1])
        localization_errors.append(math.hypot(amcl_x - odom_x, amcl_y - odom_y))

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
    if localization_errors:
        print('  amcl_position_mean_m      : %.3f' % (
            statistics.fmean(localization_errors)))
        print('  amcl_position_p95_m       : %.3f' % (
            percentile(localization_errors, 0.95)))
        print('  amcl_position_max_m       : %.3f' % max(localization_errors))


if __name__ == '__main__':
    main()
