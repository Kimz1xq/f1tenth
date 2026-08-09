#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage


def load_xy(csv_path):
    with Path(csv_path).open() as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise RuntimeError('Raceline CSV requires a header.')

        x_name = next(name for name in ('x', 'x_m') if name in reader.fieldnames)
        y_name = next(name for name in ('y', 'y_m') if name in reader.fieldnames)
        points = [(float(row[x_name]), float(row[y_name])) for row in reader]

    points = np.asarray(points)
    if len(points) < 3:
        raise RuntimeError('Raceline requires at least three points.')
    return points


def dense_closed_path(points, step):
    if np.linalg.norm(points[0] - points[-1]) > 1e-6:
        points = np.vstack((points, points[0]))

    samples = []
    for start, end in zip(points[:-1], points[1:]):
        count = max(2, int(np.ceil(np.linalg.norm(end - start) / step)) + 1)
        for ratio in np.linspace(0.0, 1.0, count, endpoint=False):
            samples.append(start + (end - start) * ratio)
    return np.asarray(samples)


def world_to_pixel(points, height, resolution, origin):
    cols = np.rint((points[:, 0] - origin[0]) / resolution - 0.5).astype(int)
    rows = np.rint(height - 0.5 - (points[:, 1] - origin[1]) / resolution).astype(int)
    return rows, cols


def draw_path(draw, points, height, resolution, origin, color, width):
    rows, cols = world_to_pixel(points, height, resolution, origin)
    pixels = [(int(col), int(row)) for row, col in zip(rows, cols)]
    draw.line(pixels, fill=color, width=width)


def main():
    parser = argparse.ArgumentParser(description='Validate and preview an optimized raceline.')
    parser.add_argument('--map-yaml', required=True)
    parser.add_argument('--raceline', required=True)
    parser.add_argument('--centerline')
    parser.add_argument('--preview', required=True)
    parser.add_argument('--free-threshold', type=int, default=230)
    parser.add_argument('--vehicle-width', type=float, default=0.296)
    parser.add_argument('--required-margin', type=float, default=0.05)
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
        raise RuntimeError('No free-space component was found.')
    track_mask = labels == (1 + int(np.argmax(component_sizes)))
    clearance_map = ndimage.distance_transform_edt(track_mask) * resolution

    raceline = load_xy(args.raceline)
    dense = dense_closed_path(raceline, step=0.01)
    rows, cols = world_to_pixel(dense, height, resolution, origin)
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if not np.all(in_bounds):
        raise RuntimeError('Raceline contains points outside the map.')

    clearances = clearance_map[rows, cols]
    if np.any(clearances <= 0.0):
        raise RuntimeError('Raceline leaves the detected track area.')

    vehicle_margin = clearances.min() - args.vehicle_width * 0.5
    if vehicle_margin < args.required_margin:
        raise RuntimeError(
            f'Raceline vehicle margin {vehicle_margin:.3f}m is below '
            f'{args.required_margin:.3f}m.'
        )

    preview = image.convert('RGB')
    draw = ImageDraw.Draw(preview)
    if args.centerline:
        centerline = load_xy(args.centerline)
        if np.linalg.norm(centerline[0] - centerline[-1]) > 1e-6:
            centerline = np.vstack((centerline, centerline[0]))
        draw_path(draw, centerline, height, resolution, origin, (0, 255, 0), 1)
    draw_path(draw, raceline, height, resolution, origin, (255, 0, 255), 2)
    preview.save(args.preview)

    length = np.linalg.norm(np.diff(raceline, axis=0), axis=1).sum()
    closure_gap = np.linalg.norm(raceline[0] - raceline[-1])
    print(f'raceline_points={len(raceline)}')
    print(f'raceline_length={length:.3f}m')
    print(f'closure_gap={closure_gap:.6f}m')
    print(f'min_center_clearance={clearances.min():.3f}m')
    print(f'min_vehicle_margin={vehicle_margin:.3f}m')
    print(f'preview={args.preview}')


if __name__ == '__main__':
    main()
