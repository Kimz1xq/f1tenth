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


def load_path(path):
    """Load x/y/yaw from either the centerline or optimized raceline CSV."""
    with open(path, 'r', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise RuntimeError('Obstacle path CSV requires a header: ' + path)
        x_key = 'x_m' if 'x_m' in reader.fieldnames else 'x'
        y_key = 'y_m' if 'y_m' in reader.fieldnames else 'y'
        yaw_key = 'psi_rad' if 'psi_rad' in reader.fieldnames else 'yaw'
        rows = [row for row in reader]
    if len(rows) < 3:
        raise RuntimeError('Obstacle path requires at least three points')
    points = np.asarray([
        [float(row[x_key]), float(row[y_key])] for row in rows
    ], dtype=float)
    if yaw_key in rows[0] and rows[0][yaw_key] != '':
        yaws = np.asarray([float(row[yaw_key]) for row in rows], dtype=float)
    else:
        following = np.roll(points, -1, axis=0)
        yaws = np.arctan2(following[:, 1] - points[:, 1],
                          following[:, 0] - points[:, 0])
    return points, yaws


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

    for index, sign in zip(candidate_indices, signs):
        yaw = float(yaws[index])
        normal = np.asarray([-math.sin(yaw), math.cos(yaw)])
        center = points[index] + sign * float(lateral_offset) * normal
        if np.linalg.norm(center - start) < float(start_clearance):
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
                float(passage_radius)):
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
