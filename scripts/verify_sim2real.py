#!/usr/bin/env python3
"""Check invariants required by the shared sim/real autonomy stack."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with path.open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def main():
    failures = []
    vehicle = load_yaml(
        ROOT / 'algorithms/f1tenth_bringup/config/vehicle_model.yaml'
    )['vehicle']
    sim = load_yaml(ROOT / 'config/sim.yaml')['bridge']['ros__parameters']
    planning = load_yaml(
        ROOT / 'algorithms/planning/config/params.yaml'
    )['local_obstacle_planner_node']['ros__parameters']
    control = load_yaml(
        ROOT / 'algorithms/control/config/params.yaml'
    )['pure_pursuit_node']['ros__parameters']

    comparisons = {
        'wheelbase': (vehicle['wheelbase'], sim['wheelbase'],
                      planning['wheelbase'], control['wheelbase']),
        'vehicle_length': (vehicle['length'], sim['vehicle_length'],
                           planning['vehicle_length']),
        'vehicle_width': (vehicle['width'], sim['vehicle_width'],
                          planning['vehicle_width']),
        'max_steering_angle': (
            vehicle['max_steering_angle'], sim['max_steering_angle'],
            planning['max_steering_angle'], control['max_steering_angle']),
        'laser_offset_x': (
            vehicle['laser_offset_x'], sim['scan_distance_to_base_link']),
    }
    for name, values in comparisons.items():
        require(
            max(values) - min(values) < 1e-9,
            f'{name} differs: {values}', failures)

    pure_pursuit_source = (
        ROOT / 'algorithms/control/control/pure_pursuit_node.py'
    ).read_text(encoding='utf-8')
    require(
        'real_speed_topic' not in pure_pursuit_source
        and 'real_servo_topic' not in pure_pursuit_source,
        'Pure Pursuit still bypasses the common Ackermann adapter', failures)
    for topic in (
            '/safety/emergency_stop', '/planning/avoidance_active',
            '/planning/speed_limit'):
        require(
            topic in pure_pursuit_source,
            f'Pure Pursuit does not use common safety input {topic}',
            failures)
    require(
        "'drive_topic': '/auto'" in (
            ROOT / 'algorithms/f1tenth_bringup/launch/autonomy.launch.py'
        ).read_text(encoding='utf-8'),
        'real mode does not route the common Ackermann command to /auto',
        failures)
    require(
        bool(sim['stop_vehicle_on_collision']),
        'sim collision behavior does not match a stopped physical vehicle',
        failures)
    gym_launch = (ROOT / 'launch/gym_bridge_launch.py').read_text(
        encoding='utf-8')
    require(
        "'config', 'amcl_common.yaml'" in gym_launch,
        'simulation does not load the shared AMCL model', failures)
    real_localization = (
        ROOT / 'algorithms/f1tenth_bringup/launch/localization.launch.py'
    ).read_text(encoding='utf-8')
    require(
        "'config', 'amcl_common.yaml'" in real_localization,
        'real mode does not load the shared AMCL model', failures)

    if failures:
        print('SIM2REAL CHECK: FAIL')
        for failure in failures:
            print(f'- {failure}')
        return 1

    print('SIM2REAL CHECK: PASS')
    print('- controllers/planner/raceline: one shared source tree')
    print('- outputs: Ackermann in both modes; only platform adapter differs')
    print('- vehicle geometry, steering limits, LiDAR offset: consistent')
    print('- remaining calibration inputs: measured tire friction, latency, '
          'odometry/AMCL error, and actuator response')
    return 0


if __name__ == '__main__':
    sys.exit(main())
