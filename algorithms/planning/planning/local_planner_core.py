"""Geometry helpers for the scan-based F1TENTH local planner."""

import math

import numpy as np


def angle_difference(target, source):
    """Return the wrapped signed angle from source to target."""
    return math.atan2(math.sin(target - source), math.cos(target - source))


def ordered_candidate_offsets(offsets, locked_offset, obstacle_lateral):
    """Prefer a stable avoidance side without excluding a safe fallback side."""
    offsets = list(offsets)
    if locked_offset is not None:
        locked_sign = math.copysign(1.0, locked_offset)
        offsets = [value for value in offsets if abs(value) > 1e-3]
        offsets.sort(key=lambda value: (
            math.copysign(1.0, value) != locked_sign,
            abs(value - locked_offset),
            abs(value),
        ))
    elif obstacle_lateral >= 0.0:
        offsets.sort(key=lambda value: (value > 0.0, abs(value)))
    else:
        offsets.sort(key=lambda value: (value < 0.0, abs(value)))
    return offsets


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

    for center, radius, s_value, lateral in observations:
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
                'hits': 1,
                'last_seen': float(now_seconds),
            })
            next_id += 1
            continue

        matched_indices.add(best_index)
        track = tracks[best_index]
        track['center'] = 0.65 * track['center'] + 0.35 * center
        track['radius'] = max(float(track['radius']), float(radius))
        track['s'] = 0.65 * float(track['s']) + 0.35 * float(s_value)
        track['lateral'] = (
            0.65 * float(track['lateral']) + 0.35 * float(lateral))
        track['hits'] = int(track.get('hits', 1)) + 1
        track['last_seen'] = float(now_seconds)

    return tracks, next_id
