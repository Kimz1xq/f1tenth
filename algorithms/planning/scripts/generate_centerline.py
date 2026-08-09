#!/usr/bin/env python3

import argparse
import csv
import math
from collections import deque
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage


NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS_DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def zhang_suen_thinning(mask):
    image = mask.astype(np.uint8).copy()
    changed = True

    while changed:
        changed = False

        for phase in (0, 1):
            padded = np.pad(image, 1)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = np.sum(
                np.stack(
                    [
                        (p2 == 0) & (p3 == 1),
                        (p3 == 0) & (p4 == 1),
                        (p4 == 0) & (p5 == 1),
                        (p5 == 0) & (p6 == 1),
                        (p6 == 0) & (p7 == 1),
                        (p7 == 0) & (p8 == 1),
                        (p8 == 0) & (p9 == 1),
                        (p9 == 0) & (p2 == 1),
                    ]
                ),
                axis=0,
            )

            if phase == 0:
                phase_condition = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                phase_condition = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            remove = (
                (image == 1)
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & phase_condition
            )

            if np.any(remove):
                image[remove] = 0
                changed = True

    return image.astype(bool)


def build_skeleton_graph(skeleton):
    points = {tuple(point) for point in np.argwhere(skeleton)}
    graph = {point: set() for point in points}

    for row, col in points:
        point = (row, col)

        for drow, dcol in NEIGHBORS_4:
            neighbor = (row + drow, col + dcol)
            if neighbor in points:
                graph[point].add(neighbor)

        for drow, dcol in NEIGHBORS_DIAGONAL:
            neighbor = (row + drow, col + dcol)
            if neighbor not in points:
                continue

            # Avoid triangular graph edges when the same pixels are already
            # connected orthogonally. This preserves a one-pixel-wide cycle.
            if (row + drow, col) not in points and (row, col + dcol) not in points:
                graph[point].add(neighbor)

    for point, neighbors in list(graph.items()):
        for neighbor in neighbors:
            graph[neighbor].add(point)

    return graph


def extract_cycle(graph):
    active = set(graph)
    queue = deque(point for point in active if len(graph[point]) < 2)

    # Iteratively remove spurs. The remaining 2-core must be the track loop.
    while queue:
        point = queue.popleft()
        if point not in active:
            continue

        active.remove(point)
        for neighbor in graph[point]:
            if neighbor in active:
                active_degree = sum(candidate in active for candidate in graph[neighbor])
                if active_degree < 2:
                    queue.append(neighbor)

    if not active:
        raise RuntimeError('No closed centerline cycle was found.')

    degrees = {
        point: sum(neighbor in active for neighbor in graph[point])
        for point in active
    }
    invalid = [point for point, degree in degrees.items() if degree != 2]
    if invalid:
        raise RuntimeError(
            f'Centerline core is not a single cycle: {len(invalid)} invalid graph nodes.'
        )

    start = next(iter(active))
    visited = set()
    stack = [start]
    while stack:
        point = stack.pop()
        if point in visited:
            continue
        visited.add(point)
        stack.extend(neighbor for neighbor in graph[point] if neighbor in active)

    if visited != active:
        raise RuntimeError('More than one closed skeleton cycle was found.')

    return active


def pixel_to_world(point, height, resolution, origin):
    row, col = point
    x = origin[0] + (col + 0.5) * resolution
    y = origin[1] + (height - row - 0.5) * resolution
    return np.array([x, y], dtype=float)


def order_cycle(cycle, graph, height, resolution, origin, start_pose):
    start_xy = np.array(start_pose[:2], dtype=float)
    start = min(
        cycle,
        key=lambda point: np.linalg.norm(
            pixel_to_world(point, height, resolution, origin) - start_xy
        ),
    )

    neighbors = [neighbor for neighbor in graph[start] if neighbor in cycle]
    heading = np.array([math.cos(start_pose[2]), math.sin(start_pose[2])])
    start_world = pixel_to_world(start, height, resolution, origin)
    first = max(
        neighbors,
        key=lambda point: np.dot(
            pixel_to_world(point, height, resolution, origin) - start_world,
            heading,
        ),
    )

    ordered = [start]
    previous = start
    current = first

    while current != start:
        ordered.append(current)
        candidates = [
            neighbor
            for neighbor in graph[current]
            if neighbor in cycle and neighbor != previous
        ]
        if len(candidates) != 1:
            raise RuntimeError('Failed to order the centerline cycle.')
        previous, current = current, candidates[0]

        if len(ordered) > len(cycle):
            raise RuntimeError('Centerline ordering did not return to its start.')

    if len(ordered) != len(cycle):
        raise RuntimeError('Ordered centerline does not contain every cycle point.')

    return np.asarray(
        [pixel_to_world(point, height, resolution, origin) for point in ordered]
    )


def smooth_and_resample(points, spacing, smoothing_sigma):
    x = ndimage.gaussian_filter1d(points[:, 0], smoothing_sigma, mode='wrap')
    y = ndimage.gaussian_filter1d(points[:, 1], smoothing_sigma, mode='wrap')
    smoothed = np.column_stack((x, y))

    closed = np.vstack((smoothed, smoothed[0]))
    segments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    total_length = cumulative[-1]
    sample_s = np.arange(0.0, total_length, spacing)

    sampled_x = np.interp(sample_s, cumulative, closed[:, 0])
    sampled_y = np.interp(sample_s, cumulative, closed[:, 1])
    return np.column_stack((sampled_x, sampled_y)), total_length


def path_geometry(points, spacing):
    dx = (np.roll(points[:, 0], -1) - np.roll(points[:, 0], 1)) / (2.0 * spacing)
    dy = (np.roll(points[:, 1], -1) - np.roll(points[:, 1], 1)) / (2.0 * spacing)
    ddx = (np.roll(points[:, 0], -1) - 2.0 * points[:, 0] + np.roll(points[:, 0], 1)) / (spacing ** 2)
    ddy = (np.roll(points[:, 1], -1) - 2.0 * points[:, 1] + np.roll(points[:, 1], 1)) / (spacing ** 2)

    yaw = np.arctan2(dy, dx)
    denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    curvature = (dx * ddy - dy * ddx) / denominator
    return yaw, curvature


def world_to_pixel(points, height, resolution, origin):
    cols = np.rint((points[:, 0] - origin[0]) / resolution - 0.5).astype(int)
    rows = np.rint(height - 0.5 - (points[:, 1] - origin[1]) / resolution).astype(int)
    return rows, cols


def measure_track_widths(points, yaw, track_mask, height, resolution, origin):
    sample_step = resolution * 0.25
    max_distance = max(track_mask.shape) * resolution

    def measure(point, direction):
        distance = 0.0
        while distance <= max_distance:
            sample = point + direction * distance
            row, col = world_to_pixel(
                sample.reshape(1, 2),
                height,
                resolution,
                origin,
            )
            if (
                row[0] < 0
                or row[0] >= track_mask.shape[0]
                or col[0] < 0
                or col[0] >= track_mask.shape[1]
                or not track_mask[row[0], col[0]]
            ):
                return max(0.0, distance - sample_step)
            distance += sample_step
        raise RuntimeError('Could not find a track boundary while measuring width.')

    widths_right = []
    widths_left = []
    for point, point_yaw in zip(points, yaw):
        right_normal = np.array([math.sin(point_yaw), -math.cos(point_yaw)])
        widths_right.append(measure(point, right_normal))
        widths_left.append(measure(point, -right_normal))

    widths_right = ndimage.gaussian_filter1d(
        np.asarray(widths_right), 2.0, mode='wrap'
    )
    widths_left = ndimage.gaussian_filter1d(
        np.asarray(widths_left), 2.0, mode='wrap'
    )
    return widths_right, widths_left


def write_preview(image, points, height, resolution, origin, output_path):
    preview = image.convert('RGB')
    draw = ImageDraw.Draw(preview)
    rows, cols = world_to_pixel(points, height, resolution, origin)
    pixels = [(int(col), int(row)) for row, col in zip(rows, cols)]
    if pixels:
        draw.line(pixels + [pixels[0]], fill=(0, 255, 0), width=2)
        col, row = pixels[0]
        draw.ellipse((col - 3, row - 3, col + 3, row + 3), fill=(255, 0, 0))
    preview.save(output_path)


def main():
    parser = argparse.ArgumentParser(description='Generate a closed F1TENTH centerline CSV.')
    parser.add_argument('--map-yaml', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--optimizer-output')
    parser.add_argument('--preview')
    parser.add_argument('--free-threshold', type=int, default=230)
    parser.add_argument('--spacing', type=float, default=0.10)
    parser.add_argument('--smoothing-sigma', type=float, default=2.0)
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument(
        '--max-half-width',
        type=float,
        default=0.65,
        help='Cap each optimizer track half-width to avoid crossed normals in concave sections.',
    )
    parser.add_argument('--start-x', type=float, default=0.0)
    parser.add_argument('--start-y', type=float, default=0.0)
    parser.add_argument('--start-yaw', type=float, default=0.0)
    args = parser.parse_args()

    yaml_path = Path(args.map_yaml).resolve()
    with yaml_path.open() as stream:
        metadata = yaml.safe_load(stream)

    image_path = Path(metadata['image'])
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path

    resolution = float(metadata['resolution'])
    origin = np.asarray(metadata['origin'], dtype=float)
    image = Image.open(image_path).convert('L')
    pixels = np.asarray(image)
    height, width = pixels.shape

    free_mask = pixels >= args.free_threshold
    labels, component_count = ndimage.label(free_mask, structure=np.ones((3, 3)))
    component_sizes = np.bincount(labels.ravel())[1:]
    if component_count == 0:
        raise RuntimeError('No free-space component was found in the map image.')

    track_label = 1 + int(np.argmax(component_sizes))
    track_mask = labels == track_label
    skeleton = zhang_suen_thinning(track_mask)
    graph = build_skeleton_graph(skeleton)
    cycle = extract_cycle(graph)
    raw_points = order_cycle(
        cycle,
        graph,
        height,
        resolution,
        origin,
        (args.start_x, args.start_y, args.start_yaw),
    )
    points, track_length = smooth_and_resample(
        raw_points,
        args.spacing,
        args.smoothing_sigma,
    )
    yaw, curvature = path_geometry(points, args.spacing)
    widths_right, widths_left = measure_track_widths(
        points,
        yaw,
        track_mask,
        height,
        resolution,
        origin,
    )
    widths_right = np.minimum(widths_right, args.max_half_width)
    widths_left = np.minimum(widths_left, args.max_half_width)

    rows, cols = world_to_pixel(points, height, resolution, origin)
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if not np.all(in_bounds):
        raise RuntimeError('Generated centerline contains points outside the map.')

    clearance_map = ndimage.distance_transform_edt(track_mask) * resolution
    clearances = clearance_map[rows, cols]
    if np.any(clearances <= 0.0):
        raise RuntimeError('Generated centerline leaves the detected track area.')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['x', 'y', 'yaw', 'curvature', 'speed'])
        for point, point_yaw, point_curvature in zip(points, yaw, curvature):
            writer.writerow(
                [
                    f'{point[0]:.6f}',
                    f'{point[1]:.6f}',
                    f'{point_yaw:.6f}',
                    f'{point_curvature:.6f}',
                    f'{args.speed:.3f}',
                ]
            )

    if args.optimizer_output:
        optimizer_output = Path(args.optimizer_output)
        optimizer_output.parent.mkdir(parents=True, exist_ok=True)
        with optimizer_output.open('w', newline='') as stream:
            stream.write('# x_m,y_m,w_tr_right_m,w_tr_left_m\n')
            writer = csv.writer(stream)
            for point, width_right, width_left in zip(
                points,
                widths_right,
                widths_left,
            ):
                writer.writerow(
                    [
                        f'{point[0]:.6f}',
                        f'{point[1]:.6f}',
                        f'{width_right:.6f}',
                        f'{width_left:.6f}',
                    ]
                )

    if args.preview:
        write_preview(
            image,
            points,
            height,
            resolution,
            origin,
            Path(args.preview),
        )

    closure_gap = np.linalg.norm(points[0] - points[-1])
    print(f'map_image={image_path}')
    print(f'map_size={width}x{height} resolution={resolution:.3f}')
    print(f'cycle_pixels={len(cycle)}')
    print(f'waypoints={len(points)} spacing={args.spacing:.3f}m')
    print(f'track_length={track_length:.3f}m')
    print(f'closure_gap={closure_gap:.3f}m')
    print(f'min_clearance={clearances.min():.3f}m')
    print(f'mean_clearance={clearances.mean():.3f}m')
    print(
        'track_width_range='
        f'{(widths_right + widths_left).min():.3f}..'
        f'{(widths_right + widths_left).max():.3f}m'
    )
    print(f'output={output_path}')
    if args.optimizer_output:
        print(f'optimizer_output={args.optimizer_output}')
    if args.preview:
        print(f'preview={args.preview}')


if __name__ == '__main__':
    main()
