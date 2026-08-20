"""
Deterministic static-obstacle helpers for the F1TENTH simulator.

The planner must not consume these ground-truth objects directly.  They are
used by the simulator to alter LaserScan data, report collisions, and publish
RViz/evaluation markers.  A real obstacle detector should reconstruct its own
obstacle representation from the scan.
"""

from dataclasses import dataclass
import csv
import math
import os
import secrets

import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
import yaml


@dataclass(frozen=True)
class StaticObstacle:
    """An oriented rectangular obstacle in the map frame."""

    obstacle_id: int
    x: float
    y: float
    yaw: float
    length: float
    width: float
    height: float


def lap_count_transition(previous_count, current_count):
    """Return the normalized lap count and newly completed lap count.

    A lower current count means the Gym environment was reset, not that a new
    lap was completed.
    """
    previous = max(0, int(previous_count))
    current = max(0, int(current_count))
    if current < previous:
        return current, 0
    return current, current - previous


class ClosedPathLapTracker:
    """Count laps from continuous progress along an ordered closed path.

    F1TENTH Gym's checkpoint counter can jump on tracks whose distant path
    sections are spatially close.  This tracker keeps the nearest-segment
    search local to the previous segment, so a nearby branch cannot be
    mistaken for forward progress around the circuit.
    """

    def __init__(self, points, start_xy, search_distance=1.5):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
            raise ValueError('Lap tracker requires a closed XY path')
        if np.linalg.norm(points[0] - points[-1]) < 1e-6:
            points = points[:-1]
        self.points = points
        self.segments = np.roll(points, -1, axis=0) - points
        self.lengths = np.linalg.norm(self.segments, axis=1)
        if np.any(self.lengths < 1e-6):
            raise ValueError('Lap tracker path contains a zero-length segment')
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.lengths)))
        self.path_length = float(self.cumulative[-1])
        self.search_distance = max(float(search_distance), 0.25)
        self.segment_index = 0
        self.previous_s = 0.0
        self.progress = 0.0
        self.lap_count = 0
        self.reset(start_xy)

    def _projection(self, position, indices):
        position = np.asarray(position, dtype=float)
        best = None
        for index in indices:
            index = int(index) % len(self.points)
            segment = self.segments[index]
            fraction = float(np.clip(
                np.dot(position - self.points[index], segment)
                / (self.lengths[index] ** 2), 0.0, 1.0))
            projected = self.points[index] + fraction * segment
            distance_squared = float(np.sum((position - projected) ** 2))
            if best is None or distance_squared < best[0]:
                best = (distance_squared, index, fraction)
        _, index, fraction = best
        return index, float(
            self.cumulative[index] + fraction * self.lengths[index])

    def _local_indices(self):
        indices = [self.segment_index]
        for direction in (-1, 1):
            distance = 0.0
            offset = 0
            while distance < self.search_distance:
                offset += direction
                index = (self.segment_index + offset) % len(self.points)
                indices.append(index)
                distance += self.lengths[index]
        return indices

    def reset(self, position):
        indices = range(len(self.points))
        self.segment_index, self.previous_s = self._projection(
            position, indices)
        self.progress = 0.0
        self.lap_count = 0

    def update(self, position):
        """Return the number of newly completed laps (normally zero or one)."""
        index, current_s = self._projection(position, self._local_indices())
        delta = (
            (current_s - self.previous_s + 0.5 * self.path_length)
            % self.path_length - 0.5 * self.path_length)
        self.segment_index = index
        self.previous_s = current_s

        # Backtracking must undo progress, while localization noise must not
        # manufacture distance. The local segment search rejects branch jumps.
        self.progress = max(0.0, self.progress + delta)
        completed = int(
            (self.progress + self.path_length * 1e-9) // self.path_length)
        if completed:
            self.progress = max(
                0.0, self.progress - completed * self.path_length)
            self.lap_count += completed
        return completed


def resolve_obstacle_seed(configured_seed, round_index, entropy=None):
    """Return a deterministic seed, or fresh entropy when seed is negative."""
    configured_seed = int(configured_seed)
    if configured_seed >= 0:
        return configured_seed + int(round_index)
    if entropy is None:
        entropy = secrets.randbits(32)
    return int(entropy) & 0xFFFFFFFF


class OccupancyMap:
    """Minimal ROS occupancy-map reader used for placement validation."""

    def __init__(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as stream:
            metadata = yaml.safe_load(stream)
        image_path = metadata['image']
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(yaml_path), image_path)
        self.image = np.asarray(Image.open(image_path).convert('L'))
        self.height, self.width = self.image.shape
        self.resolution = float(metadata['resolution'])
        self.origin_x = float(metadata['origin'][0])
        self.origin_y = float(metadata['origin'][1])
        self.negate = bool(metadata.get('negate', 0))
        self.occupied_threshold = float(metadata.get('occupied_thresh', 0.65))

    def is_free(self, x, y):
        column = int(math.floor((x - self.origin_x) / self.resolution))
        row_from_bottom = int(math.floor((y - self.origin_y) / self.resolution))
        row = self.height - 1 - row_from_bottom
        if row < 0 or row >= self.height or column < 0 or column >= self.width:
            return False
        normalized = float(self.image[row, column]) / 255.0
        occupancy = normalized if self.negate else 1.0 - normalized
        return occupancy <= self.occupied_threshold

    def rectangle_is_free(self, obstacle, sample_step=None):
        step = sample_step or max(0.02, self.resolution * 0.5)
        xs = np.arange(-obstacle.length * 0.5,
                       obstacle.length * 0.5 + step * 0.5, step)
        ys = np.arange(-obstacle.width * 0.5,
                       obstacle.width * 0.5 + step * 0.5, step)
        cosine = math.cos(obstacle.yaw)
        sine = math.sin(obstacle.yaw)
        for local_x in xs:
            for local_y in ys:
                world_x = obstacle.x + cosine * local_x - sine * local_y
                world_y = obstacle.y + sine * local_x + cosine * local_y
                if not self.is_free(world_x, world_y):
                    return False
        return True

    def circle_is_free(self, x, y, radius, samples=24):
        if not self.is_free(x, y):
            return False
        for ring in (0.5, 1.0):
            for index in range(samples):
                angle = 2.0 * math.pi * index / samples
                if not self.is_free(
                        x + radius * ring * math.cos(angle),
                        y + radius * ring * math.sin(angle)):
                    return False
        return True

    def segment_is_free(self, start_xy, end_xy, sample_step=None):
        """Return whether a simulated LiDAR ray is unobstructed by the map."""
        start = np.asarray(start_xy, dtype=float)
        end = np.asarray(end_xy, dtype=float)
        distance = float(np.linalg.norm(end - start))
        step = sample_step or max(0.01, self.resolution * 0.5)
        count = max(2, int(math.ceil(distance / step)) + 1)
        for fraction in np.linspace(0.0, 1.0, count):
            point = start + fraction * (end - start)
            if not self.is_free(float(point[0]), float(point[1])):
                return False
        return True


def point_before_path_index(points, path_index, distance):
    """Interpolate a point a given arc distance before a closed-path index."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError('Closed path must contain XY points')
    if np.linalg.norm(points[0] - points[-1]) < 1e-6:
        points = points[:-1]
    current_index = int(path_index) % len(points)
    current = points[current_index]
    remaining = max(0.0, float(distance))
    path_length = float(np.sum(np.linalg.norm(
        points - np.roll(points, 1, axis=0), axis=1)))
    remaining = min(remaining, path_length)
    while remaining > 1e-9:
        previous_index = (current_index - 1) % len(points)
        previous = points[previous_index]
        segment_length = float(np.linalg.norm(current - previous))
        if segment_length < 1e-9:
            current_index = previous_index
            current = previous
            continue
        if remaining <= segment_length:
            return current + (
                previous - current) * (remaining / segment_length)
        remaining -= segment_length
        current_index = previous_index
        current = previous
    return current.copy()


def load_path(path):
    """Load x/y/yaw from either the centerline or optimized raceline CSV."""
    with open(path, 'r', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise RuntimeError('Obstacle path CSV requires a header: ' + path)
        x_key = 'x_m' if 'x_m' in reader.fieldnames else 'x'
        y_key = 'y_m' if 'y_m' in reader.fieldnames else 'y'
        rows = [row for row in reader]
    if len(rows) < 3:
        raise RuntimeError('Obstacle path requires at least three points')
    points = np.asarray([
        [float(row[x_key]), float(row[y_key])] for row in rows
    ], dtype=float)
    # Path order is the authoritative direction used by the waypoint planner.
    # Some optimizer CSV variants store a normal angle (or use a different
    # heading convention) in ``psi_rad``. Using it here can rotate obstacle
    # placement by 90 degrees relative to the path followed by the car.
    following = np.roll(points, -1, axis=0)
    yaws = np.arctan2(following[:, 1] - points[:, 1],
                      following[:, 0] - points[:, 0])
    return points, yaws


def passage_candidate_is_feasible(
        occupancy_map, points, yaws, path_index, obstacle, passage_offset,
        vehicle_length, vehicle_width, wheelbase, max_steering_angle,
        before_distance=4.0, after_distance=4.0, wall_margin=0.02,
        sample_spacing=0.05, curvature_percentile=99.0, obstacle_side=1.0):
    """Check whether an Ackermann vehicle can use the intended passage.

    This is a simulator test-fixture check only.  It prevents random obstacle
    placement from producing a passage that is narrower than the vehicle or
    requires more curvature than the configured steering geometry permits.
    The online planner still receives obstacles exclusively through LaserScan.
    """
    points = np.asarray(points, dtype=float)
    yaws = np.asarray(yaws, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2
            or len(points) < 4 or len(points) != len(yaws)):
        raise ValueError('Passage validation requires matching closed XY/yaw')
    if np.linalg.norm(points[0] - points[-1]) < 1e-6:
        points = points[:-1]
        yaws = yaws[:-1]
    path_index = int(path_index) % len(points)

    segments = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths < 1e-5):
        return False
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    path_length = float(cumulative[-1])
    center_s = float(cumulative[path_index])
    point_s = cumulative[:-1]
    delta = (
        (point_s - center_s + 0.5 * path_length) % path_length
        - 0.5 * path_length)

    before_distance = max(0.1, float(before_distance))
    after_distance = max(0.1, float(after_distance))
    weights = np.zeros(len(points), dtype=float)
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

    normals = np.column_stack((-np.sin(yaws), np.cos(yaws)))
    target_offset = -math.copysign(
        abs(float(passage_offset)), float(obstacle_side))
    candidate = points + normals * (target_offset * weights[:, None])
    candidate_segments = np.roll(candidate, -1, axis=0) - candidate
    candidate_lengths = np.linalg.norm(candidate_segments, axis=1)
    if np.any(candidate_lengths < 1e-5):
        return False
    candidate_s = np.concatenate(([0.0], np.cumsum(candidate_lengths)))
    closed = np.vstack((candidate, candidate[0]))
    spline_x = CubicSpline(candidate_s, closed[:, 0], bc_type='periodic')
    spline_y = CubicSpline(candidate_s, closed[:, 1], bc_type='periodic')
    sample_count = max(
        len(candidate) * 4,
        int(math.ceil(candidate_s[-1] / max(float(sample_spacing), 1e-3))))
    sample_s = np.linspace(
        0.0, candidate_s[-1], sample_count, endpoint=False)
    sample_x = spline_x(sample_s)
    sample_y = spline_y(sample_s)
    dx = spline_x(sample_s, 1)
    dy = spline_y(sample_s, 1)
    ddx = spline_x(sample_s, 2)
    ddy = spline_y(sample_s, 2)
    denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    curvature = np.abs(dx * ddy - dy * ddx) / denominator

    # Candidate arc length differs slightly from the reference after offsetting.
    # Select only the manoeuvre window so unrelated track corners cannot reject
    # an otherwise valid obstacle position.
    center_candidate_s = float(candidate_s[path_index])
    sample_delta = (
        (sample_s - center_candidate_s + 0.5 * candidate_s[-1])
        % candidate_s[-1] - 0.5 * candidate_s[-1])
    affected = (
        (sample_delta >= -before_distance)
        & (sample_delta <= after_distance))
    if not np.any(affected):
        return False
    curvature_limit = math.tan(float(max_steering_angle)) / max(
        float(wheelbase), 1e-3)
    robust_curvature = float(np.percentile(
        curvature[affected], np.clip(curvature_percentile, 0.0, 100.0)))
    if robust_curvature > curvature_limit:
        return False

    inflated_length = float(vehicle_length) + 2.0 * float(wall_margin)
    inflated_width = float(vehicle_width) + 2.0 * float(wall_margin)
    for x_value, y_value, dx_value, dy_value in zip(
            sample_x[affected], sample_y[affected],
            dx[affected], dy[affected]):
        yaw = math.atan2(float(dy_value), float(dx_value))
        vehicle = StaticObstacle(
            obstacle_id=-1,
            x=float(x_value),
            y=float(y_value),
            yaw=yaw,
            length=inflated_length,
            width=inflated_width,
            height=0.0,
        )
        if not occupancy_map.rectangle_is_free(vehicle):
            return False
        if vehicle_hits_obstacle(
                (float(x_value), float(y_value), yaw),
                inflated_length, inflated_width, obstacle):
            return False
    return True


def generate_obstacles(
        points, yaws, *, count, seed, length, width, height,
        lateral_offset, start_xy, start_clearance, min_spacing,
        passage_offset, passage_radius, validator=None):
    """Generate reproducible obstacles alongside a closed reference path."""
    if count <= 0:
        return []
    if len(points) != len(yaws):
        raise ValueError('points and yaws must have equal lengths')

    generator = np.random.default_rng(int(seed))
    candidate_indices = generator.permutation(len(points))
    signs = generator.choice(np.asarray([-1.0, 1.0]), len(points))
    obstacles = []
    start = np.asarray(start_xy, dtype=float)
    segments = np.roll(points, -1, axis=0) - points
    segment_lengths = np.linalg.norm(segments, axis=1)
    path_cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    path_length = float(path_cumulative[-1])
    start_index = int(np.argmin(np.linalg.norm(points - start, axis=1)))
    start_s = float(path_cumulative[start_index])

    for index, sign in zip(candidate_indices, signs):
        yaw = float(yaws[index])
        normal = np.asarray([-math.sin(yaw), math.cos(yaw)])
        center = points[index] + sign * float(lateral_offset) * normal
        # Clearance on a closed course is forward arc distance, not Euclidean
        # distance. Nearby return branches may be almost a complete lap away.
        forward_from_start = float(
            (path_cumulative[index] - start_s) % path_length)
        if forward_from_start < float(start_clearance):
            continue
        if any(math.hypot(center[0] - item.x, center[1] - item.y)
               < float(min_spacing) for item in obstacles):
            continue

        obstacle = StaticObstacle(
            obstacle_id=len(obstacles),
            x=float(center[0]),
            y=float(center[1]),
            yaw=yaw,
            length=float(length),
            width=float(width),
            height=float(height),
        )
        passage = points[index] - sign * float(passage_offset) * normal
        if validator is not None and not validator(
                obstacle, float(passage[0]), float(passage[1]),
                float(passage_radius), int(index), float(sign)):
            continue
        obstacles.append(obstacle)
        if len(obstacles) == count:
            return obstacles

    raise RuntimeError(
        'Could only place %d of %d requested obstacles; adjust placement '
        'clearances or path' % (len(obstacles), count))


def rectangle_corners(x, y, yaw, length, width):
    local = np.asarray([
        [length * 0.5, width * 0.5],
        [length * 0.5, -width * 0.5],
        [-length * 0.5, -width * 0.5],
        [-length * 0.5, width * 0.5],
    ])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.asarray([x, y])


def rectangles_overlap(first_corners, second_corners):
    """Separating-axis test for two convex rectangles."""
    for corners in (first_corners, second_corners):
        for index in range(2):
            edge = corners[(index + 1) % 4] - corners[index]
            axis = np.asarray([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue
            axis /= norm
            first_projection = first_corners @ axis
            second_projection = second_corners @ axis
            if (first_projection.max() < second_projection.min()
                    or second_projection.max() < first_projection.min()):
                return False
    return True


def vehicle_hits_obstacle(pose, vehicle_length, vehicle_width, obstacle):
    vehicle = rectangle_corners(
        pose[0], pose[1], pose[2], vehicle_length, vehicle_width)
    target = rectangle_corners(
        obstacle.x, obstacle.y, obstacle.yaw,
        obstacle.length, obstacle.width)
    return rectangles_overlap(vehicle, target)


def inject_obstacles_into_scan(
        ranges, pose, angle_min, angle_increment, lidar_offset,
        obstacles, range_max):
    """Return ranges shortened by ray intersections with oriented boxes."""
    if not obstacles:
        return list(ranges)
    output = np.asarray(ranges, dtype=float).copy()
    output[~np.isfinite(output)] = float(range_max)
    output = np.clip(output, 0.0, float(range_max))

    vehicle_x, vehicle_y, vehicle_yaw = map(float, pose)
    origin_x = vehicle_x + math.cos(vehicle_yaw) * float(lidar_offset)
    origin_y = vehicle_y + math.sin(vehicle_yaw) * float(lidar_offset)
    beam_angles = (vehicle_yaw + float(angle_min)
                   + np.arange(len(output)) * float(angle_increment))
    world_dx = np.cos(beam_angles)
    world_dy = np.sin(beam_angles)

    for obstacle in obstacles:
        cosine = math.cos(obstacle.yaw)
        sine = math.sin(obstacle.yaw)
        relative_x = origin_x - obstacle.x
        relative_y = origin_y - obstacle.y
        local_origin_x = cosine * relative_x + sine * relative_y
        local_origin_y = -sine * relative_x + cosine * relative_y
        local_dx = cosine * world_dx + sine * world_dy
        local_dy = -sine * world_dx + cosine * world_dy

        tx_min, tx_max = _slab_intervals(
            local_origin_x, local_dx,
            -obstacle.length * 0.5, obstacle.length * 0.5)
        ty_min, ty_max = _slab_intervals(
            local_origin_y, local_dy,
            -obstacle.width * 0.5, obstacle.width * 0.5)
        entry = np.maximum(tx_min, ty_min)
        exit_distance = np.minimum(tx_max, ty_max)
        valid = (exit_distance >= np.maximum(entry, 0.0)) & (exit_distance >= 0.0)
        distances = np.where(entry >= 0.0, entry, 0.0)
        output[valid] = np.minimum(output[valid], distances[valid])
    return output.tolist()


def _slab_intervals(origin, direction, minimum, maximum):
    direction = np.asarray(direction, dtype=float)
    parallel = np.abs(direction) < 1e-12
    safe_direction = np.where(parallel, 1.0, direction)
    first = (minimum - origin) / safe_direction
    second = (maximum - origin) / safe_direction
    lower = np.minimum(first, second)
    upper = np.maximum(first, second)
    inside = minimum <= origin <= maximum
    lower = np.where(parallel & inside, -np.inf, lower)
    upper = np.where(parallel & inside, np.inf, upper)
    lower = np.where(parallel & ~inside, np.inf, lower)
    upper = np.where(parallel & ~inside, -np.inf, upper)
    return lower, upper
