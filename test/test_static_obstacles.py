import math

import numpy as np
import pytest

from f1tenth_gym_ros.static_obstacles import (
    StaticObstacle,
    generate_obstacles,
    inject_obstacles_into_scan,
    lap_count_transition,
    rectangle_corners,
    rectangles_overlap,
    resolve_obstacle_seed,
    vehicle_hits_obstacle,
)


def test_lap_count_transition_handles_completion_and_reset():
    assert lap_count_transition(0, 0) == (0, 0)
    assert lap_count_transition(0, 1) == (1, 1)
    assert lap_count_transition(1, 3) == (3, 2)
    assert lap_count_transition(3, 0) == (0, 0)


def test_obstacle_seed_supports_random_and_reproducible_modes():
    assert resolve_obstacle_seed(42, 3) == 45
    assert resolve_obstacle_seed(-1, 3, entropy=123456) == 123456


def circle_path(count=100, radius=3.0):
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    yaws = angles + math.pi * 0.5
    return points, yaws


def test_seeded_generation_is_reproducible_and_spaced():
    points, yaws = circle_path()
    arguments = dict(
        count=2, seed=17, length=0.2, width=0.12, height=0.2,
        lateral_offset=0.2, start_xy=(3.0, 0.0), start_clearance=1.0,
        min_spacing=2.0, passage_offset=0.2, passage_radius=0.2,
    )
    first = generate_obstacles(points, yaws, **arguments)
    second = generate_obstacles(points, yaws, **arguments)
    assert first == second
    assert len(first) == 2
    assert math.hypot(first[0].x - first[1].x,
                      first[0].y - first[1].y) >= 2.0


def test_scan_is_shortened_by_box_ahead():
    obstacle = StaticObstacle(0, 2.0, 0.0, 0.0, 0.4, 0.4, 0.2)
    ranges = inject_obstacles_into_scan(
        [10.0, 10.0, 10.0], (0.0, 0.0, 0.0),
        -0.1, 0.1, 0.0, [obstacle], 30.0)
    assert ranges[1] == pytest.approx(1.8)
    assert ranges[0] > ranges[1]
    assert ranges[2] > ranges[1]


def test_oriented_rectangle_collision():
    first = rectangle_corners(0.0, 0.0, 0.0, 0.58, 0.31)
    overlapping = rectangle_corners(0.2, 0.0, math.pi / 4.0, 0.2, 0.2)
    separate = rectangle_corners(2.0, 0.0, 0.0, 0.2, 0.2)
    assert rectangles_overlap(first, overlapping)
    assert not rectangles_overlap(first, separate)
    obstacle = StaticObstacle(0, 0.2, 0.0, math.pi / 4.0, 0.2, 0.2, 0.2)
    assert vehicle_hits_obstacle((0.0, 0.0, 0.0), 0.58, 0.31, obstacle)
