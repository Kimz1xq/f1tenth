"""Geometry helpers for the scan-based F1TENTH local planner."""

import math

import numpy as np
from scipy.interpolate import CubicSpline


QUINTIC_SMOOTHSTEP_MAX_SECOND_DERIVATIVE = 10.0 * math.sqrt(3.0) / 3.0


def adaptive_map_endpoint_threshold(
        clearances, base_threshold, registration_percentile,
        registration_margin, maximum_extra):
    """Bound the scan/map wall threshold using current registration error."""
    values = np.asarray(clearances, dtype=float)
    values = values[np.isfinite(values) & (values >= 0.0)]
    base_threshold = max(0.0, float(base_threshold))
    maximum_extra = max(0.0, float(maximum_extra))
    if len(values) == 0:
        return base_threshold
    residual = float(np.percentile(
        values, np.clip(registration_percentile, 0.0, 100.0)))
    adaptive = residual + max(0.0, float(registration_margin))
    return min(base_threshold + maximum_extra,
               max(base_threshold, adaptive))


def angle_difference(target, source):
    """Return the wrapped signed angle from source to target."""
    return math.atan2(math.sin(target - source), math.cos(target - source))


def ordered_candidate_offsets(offsets, locked_offset, obstacle_lateral):
    """
    Prefer a free side initially and commit to it during the manoeuvre.

    Reversing sides after lateral motion has begun creates a new path through
    the obstacle. Conservative initial object extents keep that first choice
    valid as more of a static obstacle becomes visible; if it is no longer
    safe, stopping is preferable to crossing through the object.
    """
    offsets = list(offsets)
    if locked_offset is not None:
        locked_sign = math.copysign(1.0, locked_offset)
        offsets = [
            value for value in offsets
            if abs(value) > 1e-3
            and math.copysign(1.0, value) == locked_sign
        ]
        offsets.sort(key=lambda value: (
            abs(value - locked_offset),
            abs(value),
        ))
    elif obstacle_lateral >= 0.0:
        offsets.sort(key=lambda value: (value > 0.0, abs(value)))
    else:
        offsets.sort(key=lambda value: (value < 0.0, abs(value)))
    return offsets


def adaptive_candidate_offsets(
        obstacle_lateral, required_clearance, spacing, count,
        maximum_offset):
    """Generate map-independent Frenet candidates around an obstacle."""
    required_clearance = max(0.0, float(required_clearance))
    spacing = max(1e-3, float(spacing))
    count = max(1, int(count))
    maximum_offset = max(0.0, float(maximum_offset))
    obstacle_lateral = float(obstacle_lateral)

    values = [0.0]
    left_start = obstacle_lateral + required_clearance
    right_start = obstacle_lateral - required_clearance
    for index in range(count):
        left = left_start + index * spacing
        right = right_start - index * spacing
        if abs(left) <= maximum_offset + 1e-9:
            values.append(left)
        if abs(right) <= maximum_offset + 1e-9:
            values.append(right)

    unique = []
    for value in values:
        if not any(abs(value - other) < 1e-6 for other in unique):
            unique.append(float(value))
    return unique


def minimum_quintic_transition_length(offset, maximum_curvature):
    """Return a generic curvature-bounded smoothstep transition length."""
    offset = abs(float(offset))
    maximum_curvature = max(1e-3, float(maximum_curvature))
    return math.sqrt(
        QUINTIC_SMOOTHSTEP_MAX_SECOND_DERIVATIVE
        * offset / maximum_curvature)


def speed_dependent_horizon(
        speed, reaction_time, deceleration, margin,
        minimum, maximum):
    """
    Return a bounded perception/planning horizon for the current speed.

    The distance is the reaction distance plus a constant-deceleration
    stopping distance and a geometric planning margin.  It therefore scales
    with vehicle dynamics instead of a particular map or obstacle position.
    """
    speed = max(0.0, float(speed))
    deceleration = max(0.1, float(deceleration))
    distance = (
        speed * max(0.0, float(reaction_time))
        + speed * speed / (2.0 * deceleration)
        + max(0.0, float(margin)))
    return max(float(minimum), min(distance, float(maximum)))


def path_curvature_percentile(points, percentile=90.0):
    """Estimate a robust absolute-curvature percentile of an XY polyline."""
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        return 0.0
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-5
    if np.count_nonzero(valid) < 3:
        return 0.0
    headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    heading_delta = np.abs(np.diff(headings))
    arc_span = 0.5 * (lengths[:-1] + lengths[1:])
    valid_curvature = arc_span > 1e-5
    curvature = heading_delta[valid_curvature] / arc_span[valid_curvature]
    if len(curvature) == 0:
        return 0.0
    return float(np.percentile(curvature, np.clip(percentile, 0.0, 100.0)))


def spline_path_curvature_percentile(
        points, percentile=90.0, sample_spacing=0.03):
    """Estimate local curvature without polyline waypoint impulses."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        return path_curvature_percentile(points, percentile)
    keep = np.concatenate((
        [True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5))
    points = points[keep]
    if len(points) < 4:
        return path_curvature_percentile(points, percentile)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    spline_x = CubicSpline(cumulative, points[:, 0], bc_type='natural')
    spline_y = CubicSpline(cumulative, points[:, 1], bc_type='natural')
    count = max(
        len(points) * 3,
        int(math.ceil(cumulative[-1] / max(sample_spacing, 1e-3))) + 1)
    samples = np.linspace(0.0, cumulative[-1], count)
    dx = spline_x(samples, 1)
    dy = spline_y(samples, 1)
    ddx = spline_x(samples, 2)
    ddy = spline_y(samples, 2)
    denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    curvature = np.abs(dx * ddy - dy * ddx) / denominator
    return float(np.percentile(
        curvature, np.clip(percentile, 0.0, 100.0)))


def closed_spline_curvature_percentile(
        points, percentile=99.0, sample_spacing=0.03):
    """Estimate closed-path curvature without polyline-corner impulses."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError('Closed spline requires at least four XY points')
    if np.linalg.norm(points[0] - points[-1]) < 1e-6:
        points = points[:-1]
    segments = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths < 1e-5):
        raise ValueError('Closed spline contains duplicate adjacent points')
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    closed = np.vstack((points, points[0]))
    spline_x = CubicSpline(
        cumulative, closed[:, 0], bc_type='periodic')
    spline_y = CubicSpline(
        cumulative, closed[:, 1], bc_type='periodic')
    count = max(
        len(points) * 4,
        int(math.ceil(cumulative[-1] / max(sample_spacing, 1e-3))))
    samples = np.linspace(
        0.0, cumulative[-1], count, endpoint=False)
    dx = spline_x(samples, 1)
    dy = spline_y(samples, 1)
    ddx = spline_x(samples, 2)
    ddy = spline_y(samples, 2)
    denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    curvature = np.abs(dx * ddy - dy * ddx) / denominator
    return float(np.percentile(
        curvature, np.clip(percentile, 0.0, 100.0)))


def swept_rectangle_samples(points, length, width, lateral_buffer=0.0):
    """Sample the swept rectangular vehicle footprint along a path."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError('Footprint path requires at least two XY points')
    tangent = np.gradient(points, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm < 1e-6):
        raise ValueError('Footprint path contains a degenerate tangent')
    tangent /= norm[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    half_length = 0.5 * max(0.0, float(length))
    half_width = (
        0.5 * max(0.0, float(width))
        + max(0.0, float(lateral_buffer)))
    samples = []
    for longitudinal in (-half_length, 0.0, half_length):
        for lateral in (-half_width, 0.0, half_width):
            samples.append(
                points
                + longitudinal * tangent
                + lateral * normal)
    return np.concatenate(samples, axis=0)


class ClosedPathGeometry:
    """Arc-length representation of a non-self-intersecting closed path."""

    def __init__(self, points):
        points = np.asarray(points, dtype=float)
        if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) < 1e-6:
            points = points[:-1]
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
            raise ValueError('A closed path requires at least four XY points')

        next_points = np.roll(points, -1, axis=0)
        segments = next_points - points
        lengths = np.linalg.norm(segments, axis=1)
        if np.any(lengths < 1e-5):
            raise ValueError('Closed path contains duplicate adjacent points')

        self.points = points
        self.segments = segments
        self.segment_lengths = lengths
        self.cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        self.length = float(self.cumulative[-1])
        self.yaw = np.arctan2(segments[:, 1], segments[:, 0])
        self.segment_normals = np.column_stack(
            (-np.sin(self.yaw), np.cos(self.yaw)))
        unit_tangents = segments / lengths[:, None]
        vertex_tangents = unit_tangents + np.roll(unit_tangents, 1, axis=0)
        vertex_norms = np.linalg.norm(vertex_tangents, axis=1)
        degenerate = vertex_norms < 1e-6
        vertex_tangents[~degenerate] /= vertex_norms[~degenerate, None]
        vertex_tangents[degenerate] = unit_tangents[degenerate]
        self.normals = np.column_stack(
            (-vertex_tangents[:, 1], vertex_tangents[:, 0]))

    def project(self, point):
        """Project an XY point and return s, signed lateral offset and distance."""
        point = np.asarray(point, dtype=float)
        relative = point - self.points
        fractions = np.sum(relative * self.segments, axis=1) / (
            self.segment_lengths ** 2)
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = self.points + fractions[:, None] * self.segments
        distances = np.linalg.norm(projections - point, axis=1)
        index = int(np.argmin(distances))
        projection = projections[index]
        lateral = float(np.dot(
            point - projection, self.segment_normals[index]))
        s_value = (
            self.cumulative[index]
            + float(fractions[index]) * self.segment_lengths[index])
        return float(s_value), lateral, float(distances[index])

    def circular_delta(self, values, center_s):
        """Return signed shortest arc distance from center_s to each s value."""
        values = np.asarray(values, dtype=float)
        return (values - center_s + 0.5 * self.length) % self.length - (
            0.5 * self.length)

    def offset_bump(self, center_s, offset, before_distance, after_distance):
        """
        Create a minimum-jerk lateral path offset around center_s.

        Quintic smoothstep has zero first and second derivative at both ends,
        avoiding the steering-rate discontinuity of a piecewise linear shift.
        """
        point_s = self.cumulative[:-1]
        delta = self.circular_delta(point_s, center_s)
        weights = np.zeros(len(self.points), dtype=float)

        before = (delta >= -before_distance) & (delta <= 0.0)
        after = (delta > 0.0) & (delta <= after_distance)
        before_progress = np.clip(
            (delta[before] + before_distance) / before_distance, 0.0, 1.0)
        after_progress = np.clip(
            delta[after] / after_distance, 0.0, 1.0)
        weights[before] = (
            10.0 * before_progress ** 3
            - 15.0 * before_progress ** 4
            + 6.0 * before_progress ** 5)
        weights[after] = 1.0 - (
            10.0 * after_progress ** 3
            - 15.0 * after_progress ** 4
            + 6.0 * after_progress ** 5)
        return self.points + self.normals * (offset * weights[:, None])

    def reachable_offset_maneuver(
            self, vehicle_s, center_s, offset,
            preferred_before_distance, after_distance):
        """
        Create an avoidance path whose forward transition is reachable.

        A conventional obstacle-centred bump starts ``before_distance`` ahead
        of the obstacle.  If an occluded obstacle is first observed after that
        start point, publishing the old bump asks the vehicle to jump onto the
        middle of a lateral transition.  Instead, retain the reference path
        up to the preferred start when it is still ahead, or start at the
        current vehicle projection when detection is late.  Curvature checks
        can then reject a genuinely too-short manoeuvre and trigger AEB.
        """
        vehicle_s = float(vehicle_s) % self.length
        center_s = float(center_s) % self.length
        offset = float(offset)
        preferred_before_distance = max(
            1e-3, float(preferred_before_distance))
        after_distance = max(1e-3, float(after_distance))

        distance_to_center = self.forward_distance(vehicle_s, center_s)
        transition_distance = min(
            preferred_before_distance, distance_to_center)
        start_s = (
            center_s - transition_distance) % self.length

        point_s = self.cumulative[:-1]
        from_start = np.mod(point_s - start_s, self.length)
        weights = np.zeros(len(self.points), dtype=float)

        before = from_start <= transition_distance
        before_progress = np.clip(
            from_start[before] / transition_distance, 0.0, 1.0)
        weights[before] = (
            10.0 * before_progress ** 3
            - 15.0 * before_progress ** 4
            + 6.0 * before_progress ** 5)

        from_center = np.mod(point_s - center_s, self.length)
        after = from_center <= after_distance
        after_progress = np.clip(
            from_center[after] / after_distance, 0.0, 1.0)
        weights[after] = np.maximum(
            weights[after],
            1.0 - (
                10.0 * after_progress ** 3
                - 15.0 * after_progress ** 4
                + 6.0 * after_progress ** 5))
        return self.points + self.normals * (offset * weights[:, None])

    def reachable_offset_plateau(
            self, vehicle_s, center_s, offset,
            preferred_before_distance, after_distance,
            hold_before=0.0, hold_after=0.0):
        """
        Create a reachable offset with a full-clearance obstacle plateau.

        The original obstacle-centred bump reaches its requested offset at one
        point and immediately starts returning.  A finite vehicle passing a
        finite obstacle instead needs to retain the offset until its complete
        footprint has cleared the object.  This primitive uses two independent
        minimum-jerk transitions joined by that constant-offset section.

        All distances are Frenet arc lengths.  The transition starts at the
        vehicle projection when the preferred start is already behind, so a
        newly detected obstacle never causes a discontinuous path jump.
        """
        vehicle_s = float(vehicle_s) % self.length
        center_s = float(center_s) % self.length
        offset = float(offset)
        preferred_before_distance = max(
            1e-3, float(preferred_before_distance))
        after_distance = max(1e-3, float(after_distance))
        hold_before = max(0.0, float(hold_before))
        hold_after = max(0.0, float(hold_after))

        distance_to_center = self.forward_distance(vehicle_s, center_s)
        distance_to_plateau = distance_to_center - hold_before
        if distance_to_plateau <= 1e-3:
            return None
        before_distance = min(
            preferred_before_distance, distance_to_plateau)
        start_s = (
            center_s - hold_before - before_distance) % self.length
        plateau_end_s = (center_s + hold_after) % self.length

        point_s = self.cumulative[:-1]
        from_start = np.mod(point_s - start_s, self.length)
        plateau_start = before_distance
        plateau_end = before_distance + hold_before + hold_after
        maneuver_end = plateau_end + after_distance
        weights = np.zeros(len(self.points), dtype=float)

        transition_in = from_start <= plateau_start
        progress_in = np.clip(
            from_start[transition_in] / before_distance, 0.0, 1.0)
        weights[transition_in] = (
            10.0 * progress_in ** 3
            - 15.0 * progress_in ** 4
            + 6.0 * progress_in ** 5)

        plateau = (
            (from_start > plateau_start)
            & (from_start <= plateau_end))
        weights[plateau] = 1.0

        transition_out = (
            (from_start > plateau_end)
            & (from_start <= maneuver_end))
        progress_out = np.clip(
            (from_start[transition_out] - plateau_end) / after_distance,
            0.0, 1.0)
        weights[transition_out] = 1.0 - (
            10.0 * progress_out ** 3
            - 15.0 * progress_out ** 4
            + 6.0 * progress_out ** 5)

        # ``plateau_end_s`` documents the geometric anchor and guards against
        # accidentally accepting a manoeuvre that wraps over an entire lap.
        if self.forward_distance(start_s, plateau_end_s) >= self.length - 1e-3:
            return None
        return self.points + self.normals * (offset * weights[:, None])

    def forward_distance(self, start_s, target_s):
        """Return positive wrapped distance from start_s to target_s."""
        return float((target_s - start_s) % self.length)


def sample_closed_path(points, spacing=0.05):
    """Densely sample every segment of a closed XY polyline."""
    points = np.asarray(points, dtype=float)
    samples = []
    for index in range(len(points)):
        start = points[index]
        end = points[(index + 1) % len(points)]
        length = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(length / spacing)))
        fractions = np.arange(count, dtype=float) / count
        samples.extend(start + fractions[:, None] * (end - start))
    return np.asarray(samples)


def sample_path_window(points, geometry, start_s, distance, spacing=0.05):
    """Sample a forward arc window using the reference path parameterization."""
    points = np.asarray(points, dtype=float)
    sample_s = start_s + np.arange(
        0.0, distance + 0.5 * spacing, spacing)
    wrapped = np.mod(sample_s, geometry.length)
    closed_x = np.concatenate((points[:, 0], [points[0, 0]]))
    closed_y = np.concatenate((points[:, 1], [points[0, 1]]))
    return np.column_stack((
        np.interp(wrapped, geometry.cumulative, closed_x),
        np.interp(wrapped, geometry.cumulative, closed_y),
    ))


def cluster_ordered_points(indexed_points, max_gap, min_points, max_diameter):
    """Cluster ordered scan endpoints while preserving missing-beam breaks."""
    clusters = []
    current = []
    previous_index = None
    previous_point = None

    def finish_cluster():
        if len(current) < min_points:
            return
        array = np.asarray(current, dtype=float)
        diameter = float(np.max(np.ptp(array, axis=0)))
        if diameter <= max_diameter:
            clusters.append(array)

    for beam_index, x_value, y_value in indexed_points:
        point = np.asarray([x_value, y_value], dtype=float)
        separated = (
            previous_index is None
            or beam_index != previous_index + 1
            or np.linalg.norm(point - previous_point) > max_gap)
        if separated and current:
            finish_cluster()
            current = []
        current.append(point)
        previous_index = beam_index
        previous_point = point

    if current:
        finish_cluster()
    return clusters


def nearest_clustered_corridor_distance(clusters, half_width):
    """
    Return the nearest forward point from a real scan cluster in a corridor.

    AEB must not react to an isolated range sample that was intentionally
    rejected by the obstacle cluster filter.  The caller supplies clusters in
    the vehicle base frame, so this helper remains independent of any map or
    track geometry.
    """
    nearest = float('inf')
    half_width = max(0.0, float(half_width))
    for cluster in clusters:
        points = np.asarray(cluster, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            continue
        in_corridor = (
            (points[:, 0] > 0.0)
            & (np.abs(points[:, 1]) < half_width))
        if np.any(in_corridor):
            nearest = min(nearest, float(np.min(points[in_corridor, 0])))
    return nearest


def update_tracked_obstacles(
        tracks, observations, now_seconds, next_id, match_distance,
        memory_seconds):
    """
    Update obstacle tracks with one-to-one scan associations.

    A new scan cluster starts with one hit. Repeated spatially consistent
    observations increase ``hits``; missed tracks remain only for the short
    configured memory window. This keeps one noisy scan from becoming a
    planned obstacle while preserving real obstacles across sparse scans.
    """
    tracks = [
        track for track in tracks
        if now_seconds - float(track['last_seen']) <= memory_seconds
    ]
    matched_indices = set()
    original_track_count = len(tracks)

    for observation in observations:
        center, radius, s_value, lateral = observation[:4]
        surface_points = (
            np.asarray(observation[4], dtype=float).reshape((-1, 2))
            if len(observation) >= 5 and observation[4] is not None
            else None)
        lateral_min = (
            float(observation[5]) if len(observation) >= 7
            else float(lateral) - float(radius))
        lateral_max = (
            float(observation[6]) if len(observation) >= 7
            else float(lateral) + float(radius))
        longitudinal_min = (
            float(observation[7]) if len(observation) >= 9
            else -float(radius))
        longitudinal_max = (
            float(observation[8]) if len(observation) >= 9
            else float(radius))
        center = np.asarray(center, dtype=float)
        best_index = None
        best_distance = float(match_distance)
        for index in range(original_track_count):
            if index in matched_indices:
                continue
            distance = float(np.linalg.norm(tracks[index]['center'] - center))
            if distance < best_distance:
                best_index = index
                best_distance = distance

        if best_index is None:
            tracks.append({
                'id': int(next_id),
                'center': center.copy(),
                'radius': float(radius),
                's': float(s_value),
                'lateral': float(lateral),
                'lateral_min': lateral_min,
                'lateral_max': lateral_max,
                'longitudinal_min': longitudinal_min,
                'longitudinal_max': longitudinal_max,
                'surface_points': (
                    surface_points.copy()
                    if surface_points is not None else None),
                'hits': 1,
                'last_seen': float(now_seconds),
            })
            next_id += 1
            continue

        matched_indices.add(best_index)
        track = tracks[best_index]
        # Static obstacles should not move as their visible LiDAR surface
        # changes. A running mean is map-independent and reduces path chatter.
        previous_hits = max(1, int(track.get('hits', 1)))
        observation_weight = 1.0 / (previous_hits + 1.0)
        track['center'] = (
            (1.0 - observation_weight) * track['center']
            + observation_weight * center)
        track['radius'] = max(float(track['radius']), float(radius))
        track['s'] = (
            (1.0 - observation_weight) * float(track['s'])
            + observation_weight * float(s_value))
        track['lateral'] = (
            (1.0 - observation_weight) * float(track['lateral'])
            + observation_weight * float(lateral))
        # Collision geometry must follow the currently observed surface.
        # Averaging a LiDAR surface centroid and then treating it as an object
        # centre double-counts the object's depth.  Keep the latest boundary
        # samples while the slowly averaged centroid remains useful only for
        # stable scan-to-scan association.
        track['lateral_min'] = lateral_min
        track['lateral_max'] = lateral_max
        track['longitudinal_min'] = longitudinal_min
        track['longitudinal_max'] = longitudinal_max
        track['surface_points'] = (
            surface_points.copy() if surface_points is not None else None)
        track['hits'] = previous_hits + 1
        track['last_seen'] = float(now_seconds)

    return tracks, next_id


def minimum_surface_clearance(path_points, surface_points, required_distance):
    """
    Return clearance between a path centreline and observed boundaries.

    ``surface_points`` are occupied LiDAR endpoints, not estimated object
    centres.  Measuring directly to that boundary avoids assuming a fixed
    obstacle diameter and works for compact obstacles of different shapes.
    The caller supplies the vehicle radius and independent safety margin.
    """
    path_points = np.asarray(path_points, dtype=float).reshape((-1, 2))
    surface_points = np.asarray(surface_points, dtype=float).reshape((-1, 2))
    if len(path_points) == 0 or len(surface_points) == 0:
        return float('inf')
    differences = (
        path_points[:, None, :] - surface_points[None, :, :])
    minimum_distance = float(np.min(np.linalg.norm(differences, axis=2)))
    return minimum_distance - max(0.0, float(required_distance))


def minimum_surface_footprint_clearance(
        path_points, surface_points, vehicle_length, vehicle_width,
        safety_margin=0.0):
    """
    Return signed clearance from scan endpoints to a swept vehicle box.

    The path is the vehicle centre trajectory.  At every sampled pose, scan
    endpoints are transformed into the local tangent frame and tested against
    the physical rectangular footprint expanded by ``safety_margin``.  A
    negative result means that an observed endpoint lies inside the expanded
    footprint.  This retains the measured-boundary model while accounting for
    the fact that an F1TENTH car is much longer than its lateral half-width.
    """
    path_points = np.asarray(path_points, dtype=float).reshape((-1, 2))
    surface_points = np.asarray(surface_points, dtype=float).reshape((-1, 2))
    if len(path_points) < 2 or len(surface_points) == 0:
        return float('inf')

    tangent = np.gradient(path_points, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    if np.any(tangent_norm < 1e-6):
        raise ValueError('Footprint path contains a degenerate tangent')
    tangent /= tangent_norm[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    relative = surface_points[None, :, :] - path_points[:, None, :]
    longitudinal = np.abs(np.einsum('ijk,ik->ij', relative, tangent))
    lateral = np.abs(np.einsum('ijk,ik->ij', relative, normal))
    margin = max(0.0, float(safety_margin))
    q_longitudinal = longitudinal - (
        0.5 * max(0.0, float(vehicle_length)) + margin)
    q_lateral = lateral - (
        0.5 * max(0.0, float(vehicle_width)) + margin)

    outside = np.hypot(
        np.maximum(q_longitudinal, 0.0),
        np.maximum(q_lateral, 0.0))
    inside = np.minimum(np.maximum(q_longitudinal, q_lateral), 0.0)
    return float(np.min(outside + inside))
