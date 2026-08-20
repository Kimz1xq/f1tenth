"""Unit tests for local obstacle planning geometry."""

import numpy as np

from planning.local_planner_core import (
    ClosedPathGeometry,
    adaptive_candidate_offsets,
    adaptive_map_endpoint_threshold,
    cluster_ordered_points,
    closed_spline_curvature_percentile,
    minimum_quintic_transition_length,
    minimum_surface_clearance,
    minimum_surface_footprint_clearance,
    minimum_clustered_path_clearance,
    nearest_clustered_corridor_distance,
    ordered_candidate_offsets,
    path_curvature_percentile,
    sample_closed_path,
    sample_path_window,
    speed_dependent_horizon,
    spline_path_curvature_percentile,
    swept_rectangle_samples,
    update_tracked_obstacles,
)


def test_map_endpoint_threshold_tracks_bounded_registration_residual():
    """Most wall returns may relax the filter, but never without a bound."""
    clearances = np.asarray([0.18] * 80 + [0.65] * 20)
    threshold = adaptive_map_endpoint_threshold(
        clearances, 0.30, 60.0, 0.08, 0.25)
    assert np.isclose(threshold, 0.30)

    shifted = clearances + 0.20
    shifted_threshold = adaptive_map_endpoint_threshold(
        shifted, 0.30, 60.0, 0.08, 0.25)
    assert np.isclose(shifted_threshold, 0.46)
    assert shifted_threshold <= 0.55


def test_swept_rectangle_samples_include_buffered_corners():
    """Footprint samples cover longitudinal and buffered lateral extremes."""
    path = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    samples = swept_rectangle_samples(
        path, length=0.58, width=0.31, lateral_buffer=0.05)
    assert np.isclose(np.min(samples[:, 0]), -0.29)
    assert np.isclose(np.max(samples[:, 0]), 1.29)
    assert np.isclose(np.min(samples[:, 1]), -0.205)
    assert np.isclose(np.max(samples[:, 1]), 0.205)


def test_adaptive_offsets_clear_both_sides_without_track_coordinates():
    offsets = adaptive_candidate_offsets(
        obstacle_lateral=0.10,
        required_clearance=0.30,
        spacing=0.05,
        count=3,
        maximum_offset=0.70)
    assert offsets[0] == 0.0
    assert any(value >= 0.40 for value in offsets)
    assert any(value <= -0.20 for value in offsets)
    assert len(offsets) == len(set(round(value, 6) for value in offsets))


def test_quintic_transition_length_grows_with_offset_and_speed_constraint():
    short = minimum_quintic_transition_length(0.20, 0.50)
    wide = minimum_quintic_transition_length(0.40, 0.50)
    high_speed_limit = minimum_quintic_transition_length(0.20, 0.10)
    assert wide > short
    assert high_speed_limit > short


def test_locked_avoidance_side_never_reverses_during_manoeuvre():
    """A committed path must stop rather than cross through an obstacle."""
    offsets = ordered_candidate_offsets(
        [0.0, -0.2, 0.2, -0.3, 0.3],
        locked_offset=-0.2,
        obstacle_lateral=0.1)
    assert offsets[:2] == [-0.2, -0.3]
    assert offsets == [-0.2, -0.3]


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


def test_reachable_maneuver_never_starts_behind_vehicle():
    """Late detection starts a new transition at the vehicle, not behind it."""
    geometry = ClosedPathGeometry(square_path())
    shifted = geometry.reachable_offset_maneuver(
        vehicle_s=0.5,
        center_s=1.0,
        offset=0.3,
        preferred_before_distance=2.0,
        after_distance=1.0)
    # The vertex behind the vehicle remains on the reference path. The
    # obstacle-centred legacy bump would already offset this point.
    assert np.allclose(shifted[0], geometry.points[0])
    assert np.allclose(shifted[1], [1.0, 0.3])
    assert np.allclose(shifted[2], geometry.points[2])


def test_reachable_maneuver_retains_preferred_start_when_visible_early():
    """Early detection preserves the configured smooth transition length."""
    geometry = ClosedPathGeometry(square_path())
    shifted = geometry.reachable_offset_maneuver(
        vehicle_s=7.0,
        center_s=2.0,
        offset=0.3,
        preferred_before_distance=1.0,
        after_distance=1.0)
    assert np.allclose(shifted[0], geometry.points[0])
    assert np.allclose(shifted[1], geometry.points[1])
    assert np.allclose(
        shifted[2], geometry.points[2] + 0.3 * geometry.normals[2])
    assert np.allclose(shifted[3], geometry.points[3])


def test_reachable_plateau_holds_offset_for_finite_obstacle():
    """A finite obstacle gets a constant-offset footprint-clear section."""
    geometry = ClosedPathGeometry(square_path())
    shifted = geometry.reachable_offset_plateau(
        vehicle_s=7.0,
        center_s=2.0,
        offset=0.3,
        preferred_before_distance=1.0,
        after_distance=1.0,
        hold_before=1.0,
        hold_after=1.0)
    assert shifted is not None
    assert np.allclose(shifted[0], geometry.points[0])
    for index in (1, 2, 3):
        assert np.allclose(
            shifted[index],
            geometry.points[index] + 0.3 * geometry.normals[index])
    assert np.allclose(shifted[4], geometry.points[4])


def test_reachable_plateau_rejects_obstacle_inside_vehicle_hold():
    """A path transition cannot begin after the footprint reaches object."""
    geometry = ClosedPathGeometry(square_path())
    assert geometry.reachable_offset_plateau(
        vehicle_s=1.8,
        center_s=2.0,
        offset=0.3,
        preferred_before_distance=1.0,
        after_distance=1.0,
        hold_before=0.3,
        hold_after=0.3) is None


def test_speed_horizon_grows_and_is_bounded():
    """Planning distance grows with speed but respects sensor bounds."""
    slow = speed_dependent_horizon(1.0, 0.25, 4.0, 0.5, 3.2, 6.0)
    fast = speed_dependent_horizon(5.0, 0.25, 4.0, 0.5, 3.2, 6.0)
    extreme = speed_dependent_horizon(20.0, 0.25, 4.0, 0.5, 3.2, 6.0)
    assert slow == 3.2
    assert 4.8 < fast < 5.0
    assert extreme == 6.0


def test_curvature_metric_distinguishes_straight_and_turn():
    """The robust metric is near zero on a line and positive on a turn."""
    straight = np.column_stack((np.linspace(0.0, 2.0, 21), np.zeros(21)))
    theta = np.linspace(0.0, np.pi / 2.0, 21)
    turn = np.column_stack((np.cos(theta), np.sin(theta)))
    assert path_curvature_percentile(straight) < 1e-6
    assert 0.8 < path_curvature_percentile(turn) < 1.2


def test_spline_curvature_metric_is_stable_on_sampled_arc():
    theta = np.linspace(0.0, np.pi / 2.0, 24)
    turn = np.column_stack((np.cos(theta), np.sin(theta)))
    straight = np.column_stack((np.linspace(0.0, 2.0, 24), np.zeros(24)))
    assert spline_path_curvature_percentile(straight) < 1e-6
    assert 0.95 < spline_path_curvature_percentile(turn) < 1.05


def test_closed_spline_curvature_matches_circle():
    """Periodic spline curvature is resolution-independent on a circle."""
    theta = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    circle = np.column_stack((2.0 * np.cos(theta), 2.0 * np.sin(theta)))
    curvature = closed_spline_curvature_percentile(circle, percentile=99.0)
    assert 0.48 < curvature < 0.52


def test_scan_clusters_break_on_missing_beam():
    """Non-consecutive beam indices cannot form one obstacle cluster."""
    points = [
        (1, 0.0, 0.0), (2, 0.05, 0.0),
        (5, 0.10, 0.0), (6, 0.15, 0.0),
    ]
    clusters = cluster_ordered_points(
        points, max_gap=0.2, min_points=2, max_diameter=0.3)
    assert len(clusters) == 2


def test_aeb_corridor_ignores_isolated_points_rejected_by_clustering():
    """A single noisy return cannot become an AEB obstacle by itself."""
    indexed = [(1, 0.12, 0.0)]
    clusters = cluster_ordered_points(
        indexed, max_gap=0.2, min_points=3, max_diameter=0.3)
    assert np.isinf(nearest_clustered_corridor_distance(clusters, 0.2))


def test_aeb_corridor_accepts_a_dense_forward_cluster():
    """A geometrically valid cluster in front still reaches AEB."""
    indexed = [
        (1, 0.40, -0.02),
        (2, 0.38, 0.00),
        (3, 0.41, 0.02),
    ]
    clusters = cluster_ordered_points(
        indexed, max_gap=0.2, min_points=3, max_diameter=0.3)
    distance = nearest_clustered_corridor_distance(clusters, 0.2)
    assert abs(distance - 0.38) < 1e-9


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


def test_obstacle_track_retains_observed_surface_geometry():
    """A track exposes the latest measured boundary, not a fake centre."""
    first_surface = np.asarray([[1.0, 0.1], [1.0, 0.2]])
    second_surface = np.asarray([[0.98, 0.08], [1.01, 0.22]])
    tracks, next_id = update_tracked_obstacles(
        [], [(np.mean(first_surface, axis=0), 0.11, 1.0, 0.15,
              first_surface, 0.10, 0.20)],
        0.0, 0, match_distance=0.20, memory_seconds=0.8)
    tracks, _ = update_tracked_obstacles(
        tracks, [(np.mean(second_surface, axis=0), 0.11, 1.0, 0.15,
                  second_surface, 0.08, 0.22)],
        0.1, next_id, match_distance=0.20, memory_seconds=0.8)
    assert np.allclose(tracks[0]['surface_points'], second_surface)
    assert np.isclose(tracks[0]['lateral_min'], 0.08)
    assert np.isclose(tracks[0]['lateral_max'], 0.22)


def test_surface_clearance_uses_observed_boundary_once():
    """Observed obstacle depth is not added again as a fake radius."""
    path = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    surface = np.asarray([[0.5, 0.25], [0.6, 0.25]])
    clearance = minimum_surface_clearance(
        path, surface, required_distance=0.20)
    assert np.isclose(clearance, 0.05)


def test_surface_footprint_clearance_accounts_for_vehicle_length():
    """A front-face return cannot be cleared using lateral radius alone."""
    path = np.asarray([[0.0, 0.0], [0.1, 0.0]])
    surface = np.asarray([[0.25, 0.0]])
    clearance = minimum_surface_footprint_clearance(
        path, surface, vehicle_length=0.58, vehicle_width=0.31)
    assert clearance < 0.0


def test_surface_footprint_clearance_is_signed_outside_buffered_box():
    """The geometric margin expands both axes without a fake object radius."""
    path = np.asarray([[0.0, 0.0], [0.1, 0.0]])
    surface = np.asarray([[0.0, 0.225]])
    clearance = minimum_surface_footprint_clearance(
        path, surface, vehicle_length=0.58, vehicle_width=0.31,
        safety_margin=0.05)
    assert np.isclose(clearance, 0.02)


def test_path_aligned_aeb_clears_bypassed_obstacle():
    """A bypassed object must not stop a path whose footprint clears it."""
    obstacle = [np.asarray([
        [0.90, -0.04], [0.90, 0.00], [0.90, 0.04],
    ])]
    collision_path = np.asarray([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0],
    ])
    bypass_path = np.asarray([
        [0.0, 0.0], [0.3, 0.25], [0.6, 0.50],
        [1.0, 0.50], [1.5, 0.50],
    ])
    collision = minimum_clustered_path_clearance(
        collision_path, obstacle, 0.58, 0.31, 0.01)
    bypass = minimum_clustered_path_clearance(
        bypass_path, obstacle, 0.58, 0.31, 0.01)
    assert collision < 0.0
    assert bypass > 0.0


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
