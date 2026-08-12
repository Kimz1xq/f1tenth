"""Unit tests for local obstacle planning geometry."""

import numpy as np

from planning.local_planner_core import (
    ClosedPathGeometry,
    cluster_ordered_points,
    ordered_candidate_offsets,
    sample_closed_path,
    sample_path_window,
    update_tracked_obstacles,
)


def test_locked_avoidance_side_is_preferred_but_not_exclusive():
    """A blocked locked side must not hide feasible opposite-side candidates."""
    offsets = ordered_candidate_offsets(
        [0.0, -0.2, 0.2, -0.3, 0.3],
        locked_offset=-0.2,
        obstacle_lateral=0.1)
    assert offsets[:2] == [-0.2, -0.3]
    assert set(offsets[2:]) == {0.2, 0.3}
    assert 0.0 not in offsets


def square_path():
    """Return a counter-clockwise square with enough samples per side."""
    return np.asarray([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [2.0, 1.0], [2.0, 2.0], [1.0, 2.0],
        [0.0, 2.0], [0.0, 1.0],
    ])


def test_projection_and_forward_distance():
    """Projection returns consistent arc length and signed lateral offset."""
    geometry = ClosedPathGeometry(square_path())
    s_value, lateral, distance = geometry.project([0.8, 0.2])
    assert abs(s_value - 0.8) < 1e-6
    assert abs(lateral - 0.2) < 1e-6
    assert abs(distance - 0.2) < 1e-6
    assert abs(geometry.forward_distance(7.5, 0.5) - 1.0) < 1e-6


def test_offset_bump_is_smooth_and_local():
    """A lateral avoidance bump peaks at the obstacle and returns to path."""
    geometry = ClosedPathGeometry(square_path())
    shifted = geometry.offset_bump(1.0, 0.3, 1.0, 1.0)
    assert np.allclose(shifted[0], geometry.points[0])
    assert np.allclose(shifted[1], [1.0, 0.3])
    assert np.allclose(shifted[2], geometry.points[2])


def test_scan_clusters_break_on_missing_beam():
    """Non-consecutive beam indices cannot form one obstacle cluster."""
    points = [
        (1, 0.0, 0.0), (2, 0.05, 0.0),
        (5, 0.10, 0.0), (6, 0.15, 0.0),
    ]
    clusters = cluster_ordered_points(
        points, max_gap=0.2, min_points=2, max_diameter=0.3)
    assert len(clusters) == 2


def test_dense_closed_path_sampling():
    """Dense sampling includes every side and respects maximum spacing."""
    samples = sample_closed_path(square_path(), spacing=0.1)
    gaps = np.linalg.norm(np.roll(samples, -1, axis=0) - samples, axis=1)
    assert len(samples) >= 80
    assert float(np.max(gaps)) <= 0.101


def test_forward_window_wraps_lap_boundary():
    """A local sample window remains continuous across the lap boundary."""
    geometry = ClosedPathGeometry(square_path())
    samples = sample_path_window(
        geometry.points, geometry, 7.8, 0.4, spacing=0.1)
    gaps = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    assert len(samples) == 5
    assert float(np.max(gaps)) <= 0.101


def test_obstacle_requires_repeated_spatially_consistent_hits():
    """A one-frame cluster stays tentative until repeated observations."""
    tracks = []
    next_id = 0
    for index in range(3):
        observation = [(np.asarray([1.0 + 0.01 * index, 0.2]),
                        0.09, 1.0, 0.2)]
        tracks, next_id = update_tracked_obstacles(
            tracks, observation, index * 0.1, next_id,
            match_distance=0.20, memory_seconds=0.8)
        assert len(tracks) == 1
        assert tracks[0]['hits'] == index + 1
    assert next_id == 1


def test_obstacle_tracks_expire_and_match_once_per_scan():
    """Stale tracks disappear and two clusters cannot add two hits to one track."""
    tracks, next_id = update_tracked_obstacles(
        [], [(np.asarray([1.0, 0.0]), 0.09, 1.0, 0.0)],
        0.0, 0, match_distance=0.20, memory_seconds=0.8)
    tracks, next_id = update_tracked_obstacles(
        tracks,
        [(np.asarray([1.01, 0.0]), 0.09, 1.0, 0.0),
         (np.asarray([1.02, 0.0]), 0.09, 1.0, 0.0)],
        0.1, next_id, match_distance=0.20, memory_seconds=0.8)
    assert len(tracks) == 2
    assert sorted(track['hits'] for track in tracks) == [1, 2]

    tracks, next_id = update_tracked_obstacles(
        tracks, [], 1.0, next_id,
        match_distance=0.20, memory_seconds=0.8)
    assert tracks == []
