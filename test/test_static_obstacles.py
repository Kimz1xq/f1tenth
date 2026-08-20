import math

import numpy as np
import pytest

from f1tenth_gym_ros.static_obstacles import (
    ClosedPathLapTracker,
    StaticObstacle,
    generate_obstacles,
    inject_obstacles_into_scan,
    lap_count_transition,
    passage_candidate_is_feasible,
    point_before_path_index,
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


def test_closed_path_lap_tracker_counts_only_a_complete_circuit():
    points, _ = circle_path(count=120, radius=3.0)
    tracker = ClosedPathLapTracker(points, points[0])
    for point in points[1:]:
        assert tracker.update(point) == 0
    assert tracker.update(points[0]) == 1
    assert tracker.lap_count == 1


def test_closed_path_lap_tracker_rejects_nearby_branch_jump():
    # Two long, spatially close sections are far apart in path order.
    points = np.asarray([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
        [3.0, 2.0], [2.0, 0.1], [1.0, 0.1], [0.0, 0.1],
    ])
    tracker = ClosedPathLapTracker(points, points[0], search_distance=0.6)
    assert tracker.update([0.9, 0.0]) == 0
    # This position lies exactly on the remote return branch. A global nearest
    # lookup could jump there; continuity must keep progress on the first leg.
    assert tracker.update([1.0, 0.1]) == 0
    assert tracker.lap_count == 0
    assert tracker.progress < 2.0


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


def test_start_clearance_uses_forward_path_distance_not_euclidean_distance():
    # The final branch returns close to the start in XY but is almost one lap
    # ahead in path order and must remain a valid candidate.
    points = np.asarray([
        [0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, 2.0],
        [2.0, 2.0], [0.1, 0.1],
    ])
    following = np.roll(points, -1, axis=0)
    yaws = np.arctan2(
        following[:, 1] - points[:, 1],
        following[:, 0] - points[:, 0])
    obstacles = generate_obstacles(
        points, yaws, count=1, seed=4,
        length=0.2, width=0.12, height=0.2,
        lateral_offset=0.0, start_xy=(0.0, 0.0),
        start_clearance=6.0, min_spacing=1.0,
        passage_offset=0.2, passage_radius=0.2,
        validator=lambda *args: int(args[4]) == 5)
    assert obstacles[0].x == pytest.approx(0.1)
    assert obstacles[0].y == pytest.approx(0.1)


def test_point_before_path_index_wraps_and_interpolates():
    points = np.asarray([
        [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0],
    ])
    assert np.allclose(point_before_path_index(points, 1, 0.5), [1.5, 0.0])
    assert np.allclose(point_before_path_index(points, 0, 0.5), [0.0, 0.5])


class FreeMap:
    def rectangle_is_free(self, obstacle):
        del obstacle
        return True


def test_passage_validator_enforces_ackermann_curvature():
    points, yaws = circle_path(radius=3.0)
    normal = np.asarray([-math.sin(yaws[0]), math.cos(yaws[0])])
    center = points[0] + 0.16 * normal
    obstacle = StaticObstacle(
        0, center[0], center[1], yaws[0], 0.20, 0.12, 0.20)
    assert passage_candidate_is_feasible(
        FreeMap(), points, yaws, 0, obstacle, 0.18,
        0.58, 0.31, 0.33, 0.4189, obstacle_side=1.0)

    tight_points, tight_yaws = circle_path(radius=0.40)
    tight_normal = np.asarray([
        -math.sin(tight_yaws[0]), math.cos(tight_yaws[0])])
    tight_center = tight_points[0] + 0.16 * tight_normal
    tight_obstacle = StaticObstacle(
        0, tight_center[0], tight_center[1], tight_yaws[0],
        0.20, 0.12, 0.20)
    assert not passage_candidate_is_feasible(
        FreeMap(), tight_points, tight_yaws, 0, tight_obstacle, 0.18,
        0.58, 0.31, 0.33, 0.4189, obstacle_side=1.0)


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
