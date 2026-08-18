"""Scan-only obstacle detector and local path planner for F1TENTH."""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.time import Time
from scipy.ndimage import distance_transform_edt

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from planning.local_planner_core import (
    ClosedPathGeometry,
    cluster_ordered_points,
    ordered_candidate_offsets,
    path_curvature_percentile,
    sample_path_window,
    speed_dependent_horizon,
    update_tracked_obstacles,
)


class LocalObstaclePlannerNode(Node):
    """Detect unmapped scan clusters and publish a collision-free closed path."""

    def __init__(self):
        super().__init__('local_obstacle_planner_node')

        self.declare_parameter('global_path_topic', '/planning/global_path')
        self.declare_parameter('local_path_topic', '/planning/path')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('emergency_stop_topic', '/safety/emergency_stop')
        self.declare_parameter(
            'avoidance_active_topic', '/planning/avoidance_active')
        self.declare_parameter('speed_limit_topic', '/planning/speed_limit')
        self.declare_parameter('marker_topic', '/planning/local_markers')
        self.declare_parameter('status_topic', '/planning/local_status')
        self.declare_parameter('global_frame_id', 'map')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'ego_racecar/base_link')
        self.declare_parameter('planning_rate', 5.0)
        self.declare_parameter('scan_process_rate', 10.0)
        self.declare_parameter('scan_transform_delay', 0.08)
        self.declare_parameter('detection_range', 3.0)
        self.declare_parameter('map_endpoint_clearance', 0.30)
        self.declare_parameter('cluster_gap', 0.16)
        self.declare_parameter('cluster_min_points', 8)
        self.declare_parameter('cluster_max_diameter', 0.55)
        self.declare_parameter('path_obstacle_corridor', 0.55)
        self.declare_parameter('obstacle_memory_seconds', 0.8)
        self.declare_parameter('obstacle_confirmation_frames', 3)
        self.declare_parameter('obstacle_match_distance', 0.35)
        self.declare_parameter('obstacle_default_radius', 0.11)
        self.declare_parameter('obstacle_max_radius', 0.12)
        self.declare_parameter('obstacle_marker_radius_scale', 1.0)
        self.declare_parameter('obstacle_lookahead', 3.2)
        self.declare_parameter('maximum_planning_horizon', 6.0)
        self.declare_parameter('planning_reaction_time', 0.25)
        self.declare_parameter('planning_deceleration', 4.0)
        self.declare_parameter('planning_distance_margin', 0.50)
        self.declare_parameter(
            'candidate_offsets', [0.0, -0.20, 0.20, -0.24, 0.24,
                                  -0.28, 0.28,
                                  -0.36, 0.36, -0.44, 0.44,
                                  -0.52, 0.52, -0.60, 0.60])
        self.declare_parameter('avoidance_before_distance', 1.80)
        self.declare_parameter('avoidance_after_distance', 1.10)
        self.declare_parameter('maximum_avoidance_before_distance', 4.0)
        self.declare_parameter('maximum_avoidance_after_distance', 2.0)
        self.declare_parameter('avoidance_before_time', 0.60)
        self.declare_parameter('avoidance_after_time', 0.30)
        self.declare_parameter('path_sample_spacing', 0.04)
        self.declare_parameter('map_clearance', 0.19)
        self.declare_parameter('vehicle_clearance_radius', 0.17)
        self.declare_parameter('obstacle_safety_margin', 0.02)
        self.declare_parameter('aeb_corridor_half_width', 0.19)
        self.declare_parameter('aeb_reaction_time', 0.12)
        self.declare_parameter('aeb_max_deceleration', 2.0)
        self.declare_parameter('aeb_min_distance', 0.16)
        self.declare_parameter('aeb_critical_reaction_time', 0.05)
        self.declare_parameter('aeb_confirmation_frames', 2)
        self.declare_parameter('scan_timeout', 0.50)
        self.declare_parameter('maximum_planning_speed', 5.5)
        self.declare_parameter('minimum_avoidance_speed', 0.60)
        self.declare_parameter('max_lateral_acceleration', 1.50)
        self.declare_parameter('candidate_curvature_weight', 0.35)
        self.declare_parameter('candidate_clearance_weight', 0.10)
        self.declare_parameter('curvature_percentile', 90.0)

        self.global_frame = self.get_parameter('global_frame_id').value
        self.odom_frame = self.get_parameter('odom_frame_id').value
        self.base_frame = self.get_parameter('base_frame_id').value
        self.detection_range = float(
            self.get_parameter('detection_range').value)
        self.map_endpoint_clearance = float(
            self.get_parameter('map_endpoint_clearance').value)
        self.cluster_gap = float(self.get_parameter('cluster_gap').value)
        self.cluster_min_points = int(
            self.get_parameter('cluster_min_points').value)
        self.cluster_max_diameter = float(
            self.get_parameter('cluster_max_diameter').value)
        self.path_obstacle_corridor = float(
            self.get_parameter('path_obstacle_corridor').value)
        self.obstacle_memory = float(
            self.get_parameter('obstacle_memory_seconds').value)
        self.obstacle_confirmation_frames = int(
            self.get_parameter('obstacle_confirmation_frames').value)
        self.obstacle_match_distance = float(
            self.get_parameter('obstacle_match_distance').value)
        self.obstacle_default_radius = float(
            self.get_parameter('obstacle_default_radius').value)
        self.obstacle_max_radius = max(
            self.obstacle_default_radius,
            float(self.get_parameter('obstacle_max_radius').value))
        self.obstacle_marker_radius_scale = max(0.1, min(
            float(self.get_parameter('obstacle_marker_radius_scale').value),
            1.0))
        self.obstacle_lookahead = float(
            self.get_parameter('obstacle_lookahead').value)
        self.maximum_planning_horizon = max(
            self.obstacle_lookahead,
            float(self.get_parameter('maximum_planning_horizon').value))
        self.planning_reaction_time = float(
            self.get_parameter('planning_reaction_time').value)
        self.planning_deceleration = float(
            self.get_parameter('planning_deceleration').value)
        self.planning_distance_margin = float(
            self.get_parameter('planning_distance_margin').value)
        self.candidate_offsets = [
            float(value)
            for value in self.get_parameter('candidate_offsets').value]
        self.avoidance_before = float(
            self.get_parameter('avoidance_before_distance').value)
        self.avoidance_after = float(
            self.get_parameter('avoidance_after_distance').value)
        self.maximum_avoidance_before = max(
            self.avoidance_before,
            float(self.get_parameter(
                'maximum_avoidance_before_distance').value))
        self.maximum_avoidance_after = max(
            self.avoidance_after,
            float(self.get_parameter(
                'maximum_avoidance_after_distance').value))
        self.avoidance_before_time = float(
            self.get_parameter('avoidance_before_time').value)
        self.avoidance_after_time = float(
            self.get_parameter('avoidance_after_time').value)
        self.sample_spacing = float(
            self.get_parameter('path_sample_spacing').value)
        self.map_clearance = float(
            self.get_parameter('map_clearance').value)
        self.vehicle_clearance = float(
            self.get_parameter('vehicle_clearance_radius').value)
        self.obstacle_margin = float(
            self.get_parameter('obstacle_safety_margin').value)
        self.aeb_half_width = float(
            self.get_parameter('aeb_corridor_half_width').value)
        self.aeb_reaction_time = float(
            self.get_parameter('aeb_reaction_time').value)
        self.aeb_deceleration = float(
            self.get_parameter('aeb_max_deceleration').value)
        self.aeb_min_distance = float(
            self.get_parameter('aeb_min_distance').value)
        self.aeb_critical_reaction_time = float(
            self.get_parameter('aeb_critical_reaction_time').value)
        self.aeb_confirmation_frames = int(
            self.get_parameter('aeb_confirmation_frames').value)
        self.scan_timeout = float(self.get_parameter('scan_timeout').value)
        self.scan_process_period = 1.0 / max(
            float(self.get_parameter('scan_process_rate').value), 1.0)
        self.scan_transform_delay = float(
            self.get_parameter('scan_transform_delay').value)
        self.maximum_planning_speed = float(
            self.get_parameter('maximum_planning_speed').value)
        self.minimum_avoidance_speed = float(
            self.get_parameter('minimum_avoidance_speed').value)
        self.max_lateral_acceleration = float(
            self.get_parameter('max_lateral_acceleration').value)
        self.candidate_curvature_weight = float(
            self.get_parameter('candidate_curvature_weight').value)
        self.candidate_clearance_weight = float(
            self.get_parameter('candidate_clearance_weight').value)
        self.curvature_percentile = float(
            self.get_parameter('curvature_percentile').value)

        self.path_geometry = None
        self.map_clearance_grid = None
        self.map_info = None
        self.speed = 0.0
        self.last_scan_time = None
        self.pending_scans = deque(maxlen=20)
        self.ttc_stop = False
        self.nearest_corridor_distance = float('inf')
        self.aeb_detection_count = 0
        self.tracked_obstacles = []
        self.next_obstacle_id = 0
        self.last_safe_path = None
        self.last_status = ''
        self.locked_obstacle_id = None
        self.locked_offset = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        # Keep TF reception independent from the comparatively expensive
        # candidate-path evaluation. At racing speed a single-threaded TF
        # listener can lag behind the scan timestamp and create false timeouts.
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sensor_callback_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            Path,
            self.get_parameter('global_path_topic').value,
            self.global_path_callback,
            10)
        scan_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self.scan_callback,
            scan_qos,
            callback_group=self.sensor_callback_group)
        map_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(
            OccupancyGrid,
            self.get_parameter('map_topic').value,
            self.map_callback,
            map_qos)
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self.odom_callback,
            10,
            callback_group=self.sensor_callback_group)

        self.path_pub = self.create_publisher(
            Path, self.get_parameter('local_path_topic').value, 10)
        self.stop_pub = self.create_publisher(
            Bool, self.get_parameter('emergency_stop_topic').value, 10)
        self.avoidance_pub = self.create_publisher(
            Bool, self.get_parameter('avoidance_active_topic').value, 10)
        self.speed_limit_pub = self.create_publisher(
            Float32, self.get_parameter('speed_limit_topic').value, 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter('marker_topic').value, 10)
        self.status_pub = self.create_publisher(
            String, self.get_parameter('status_topic').value, 10)

        planning_rate = float(self.get_parameter('planning_rate').value)
        self.scan_timer = self.create_timer(
            self.scan_process_period,
            self.process_pending_scan,
            callback_group=self.sensor_callback_group)
        self.timer = self.create_timer(
            1.0 / max(planning_rate, 1.0), self.plan)
        self.get_logger().info(
            'Local obstacle planner ready: scan/map only, no ground-truth input')

    @staticmethod
    def quaternion_to_yaw(quaternion):
        return math.atan2(
            2.0 * (quaternion.w * quaternion.z
                   + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2))

    def global_path_callback(self, message):
        points = np.asarray([
            [pose.pose.position.x, pose.pose.position.y]
            for pose in message.poses
        ], dtype=float)
        try:
            new_geometry = ClosedPathGeometry(points)
        except ValueError as error:
            self.get_logger().error('Rejected global path: %s' % error)
            return
        if (self.path_geometry is None
                or len(new_geometry.points) != len(self.path_geometry.points)
                or np.max(np.abs(
                    new_geometry.points - self.path_geometry.points)) > 1e-6):
            self.path_geometry = new_geometry
            self.last_safe_path = new_geometry.points.copy()
            self.tracked_obstacles = []
            self.locked_obstacle_id = None
            self.locked_offset = None
            self.get_logger().info(
                'Global path received: %d points, %.2f m'
                % (len(new_geometry.points), new_geometry.length))

    def map_callback(self, message):
        values = np.asarray(message.data, dtype=np.int16).reshape(
            message.info.height, message.info.width)
        occupied = (values < 0) | (values >= 50)
        self.map_clearance_grid = (
            distance_transform_edt(~occupied) * message.info.resolution)
        origin = message.info.origin
        self.map_info = {
            'resolution': float(message.info.resolution),
            'width': int(message.info.width),
            'height': int(message.info.height),
            'origin_x': float(origin.position.x),
            'origin_y': float(origin.position.y),
            'origin_yaw': self.quaternion_to_yaw(origin.orientation),
        }

    def odom_callback(self, message):
        self.speed = max(0.0, float(message.twist.twist.linear.x))

    def current_map_base_pose(self):
        try:
            map_from_odom = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.odom_frame,
                Time(),
                timeout=Duration(seconds=0.03))
            odom_from_base = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.03))
        except TransformException as error:
            raise RuntimeError('localized vehicle TF unavailable: %s' % error)
        odom_x = odom_from_base.transform.translation.x
        odom_y = odom_from_base.transform.translation.y
        map_x, map_y = self.apply_transform(
            map_from_odom, np.asarray([odom_x]), np.asarray([odom_y]))
        return np.asarray([
            map_x[0],
            map_y[0],
            self.quaternion_to_yaw(map_from_odom.transform.rotation)
            + self.quaternion_to_yaw(odom_from_base.transform.rotation),
        ], dtype=float)

    def apply_transform(self, transform, x, y):
        translation = transform.transform.translation
        yaw = self.quaternion_to_yaw(transform.transform.rotation)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return (
            translation.x + cosine * x - sine * y,
            translation.y + sine * x + cosine * y,
        )

    def transform_scan_points(
            self, scan_frame, scan_time, map_from_odom, x, y):
        # Vehicle motion is sampled at the scan timestamp, while map->odom is
        # the correction captured by scan_callback when this scan arrived.
        odom_from_scan = self.tf_buffer.lookup_transform(
            self.odom_frame,
            scan_frame,
            scan_time,
            timeout=Duration(seconds=0.05))
        base_from_scan = self.tf_buffer.lookup_transform(
            self.base_frame,
            scan_frame,
            scan_time,
            timeout=Duration(seconds=0.05))
        odom_x, odom_y = self.apply_transform(odom_from_scan, x, y)
        map_x, map_y = self.apply_transform(map_from_odom, odom_x, odom_y)
        base_x, base_y = self.apply_transform(base_from_scan, x, y)
        return map_x, map_y, base_x, base_y

    def map_clearances(self, points):
        if self.map_clearance_grid is None or self.map_info is None:
            return np.zeros(len(points), dtype=float)
        points = np.asarray(points, dtype=float)
        dx = points[:, 0] - self.map_info['origin_x']
        dy = points[:, 1] - self.map_info['origin_y']
        yaw = self.map_info['origin_yaw']
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        columns = np.floor(local_x / self.map_info['resolution']).astype(int)
        rows = np.floor(local_y / self.map_info['resolution']).astype(int)
        valid = (
            (columns >= 0) & (columns < self.map_info['width'])
            & (rows >= 0) & (rows < self.map_info['height']))
        clearances = np.zeros(len(points), dtype=float)
        clearances[valid] = self.map_clearance_grid[
            rows[valid], columns[valid]]
        return clearances

    def scan_callback(self, message):
        # Keep the localization correction that was available when this scan
        # arrived.  Looking up the latest map->odom transform after the scan
        # delay can apply a correction from a newer AMCL update to older scan
        # points, which is noticeable at racing speed.
        try:
            map_from_odom = self.tf_buffer.lookup_transform(
                self.global_frame, self.odom_frame, Time())
        except TransformException:
            map_from_odom = None
        self.pending_scans.append(
            (self.get_clock().now(), message, map_from_odom))

    def process_pending_scan(self):
        now = self.get_clock().now()
        selected_index = None
        # A lower scan processing rate intentionally skips some laser frames.
        # Select the newest frame whose timestamped TF is already available;
        # blindly selecting the newest delayed frame can remain a few
        # milliseconds ahead of odometry and cause false scan timeouts.
        for index in range(len(self.pending_scans) - 1, -1, -1):
            received_time, message, map_from_odom = self.pending_scans[index]
            age = (now - received_time).nanoseconds * 1e-9
            if age < self.scan_transform_delay:
                continue
            if map_from_odom is None:
                continue
            scan_time = Time.from_msg(message.header.stamp)
            if (self.tf_buffer.can_transform(
                    self.odom_frame, message.header.frame_id, scan_time)
                    and self.tf_buffer.can_transform(
                        self.base_frame, message.header.frame_id, scan_time)):
                selected_index = index
                break
        if selected_index is None:
            return
        selected = None
        for _ in range(selected_index + 1):
            selected = self.pending_scans.popleft()
        received_time, message, map_from_odom = selected
        processed = self.process_scan(message, now, map_from_odom)
        retry_age = (now - received_time).nanoseconds * 1e-9
        if not processed and retry_age < 0.25:
            self.pending_scans.appendleft(selected)

    def process_scan(self, message, received_time, map_from_odom):
        ranges = np.asarray(message.ranges, dtype=float)
        indices = np.arange(len(ranges), dtype=int)
        valid = (
            np.isfinite(ranges)
            & (ranges >= message.range_min)
            & (ranges <= min(message.range_max, self.detection_range)))
        ranges = ranges[valid]
        indices = indices[valid]
        angles = message.angle_min + indices * message.angle_increment
        laser_x = ranges * np.cos(angles)
        laser_y = ranges * np.sin(angles)
        scan_time = Time.from_msg(message.header.stamp)
        try:
            map_x, map_y, base_x, base_y = self.transform_scan_points(
                message.header.frame_id, scan_time, map_from_odom,
                laser_x, laser_y)
        except TransformException as error:
            self.get_logger().warn(
                'Cannot transform timestamped scan for local planning: %s'
                % error,
                throttle_duration_sec=2.0)
            return False
        self.last_scan_time = received_time
        indexed_points = list(zip(
            indices.tolist(), map_x.tolist(), map_y.tolist()))

        if self.path_geometry is None or self.map_clearance_grid is None:
            return
        filtered = []
        unmapped_mask = np.zeros(len(indexed_points), dtype=bool)
        if indexed_points:
            points = np.asarray([
                [item[1], item[2]] for item in indexed_points])
            endpoint_clearance = self.map_clearances(points)
            # EDT clearances are quantized by the occupancy-grid cells.  Add
            # one diagonal cell as rasterization tolerance so a mapped wall
            # on the threshold cannot become an unmapped obstacle merely due
            # to pixel-to-world rounding.  This scales with any map resolution.
            raster_tolerance = (
                math.sqrt(2.0) * self.map_info['resolution'])
            unmapped_mask = endpoint_clearance > (
                self.map_endpoint_clearance + raster_tolerance)
            for item, is_unmapped in zip(indexed_points, unmapped_mask):
                if is_unmapped:
                    filtered.append(item)

        obstacle_forward = base_x[
            unmapped_mask
            & (base_x > 0.0)
            & (np.abs(base_y) < self.aeb_half_width)]
        self.nearest_corridor_distance = (
            float(np.min(obstacle_forward))
            if len(obstacle_forward) else float('inf'))
        stop_distance = (
            self.aeb_min_distance
            + self.speed * self.aeb_reaction_time
            + self.speed ** 2 / (2.0 * max(self.aeb_deceleration, 0.1)))
        raw_ttc_stop = bool(
            len(obstacle_forward) > 0
            and self.nearest_corridor_distance < stop_distance)
        self.aeb_detection_count = (
            self.aeb_detection_count + 1 if raw_ttc_stop else 0)
        self.ttc_stop = (
            self.aeb_detection_count >= self.aeb_confirmation_frames)

        clusters = cluster_ordered_points(
            filtered,
            max_gap=self.cluster_gap,
            min_points=self.cluster_min_points,
            max_diameter=self.cluster_max_diameter)
        observations = []
        for cluster in clusters:
            center = np.mean(cluster, axis=0)
            s_value, lateral, _ = self.path_geometry.project(center)
            if abs(lateral) > self.path_obstacle_corridor:
                continue
            observed_radius = float(np.max(np.linalg.norm(
                cluster - center, axis=1)))
            radius = min(
                self.obstacle_max_radius,
                max(self.obstacle_default_radius, observed_radius))
            observations.append((center, radius, s_value, lateral))
        self.update_obstacle_tracks(observations)
        return True

    def update_obstacle_tracks(self, observations):
        now_seconds = self.get_clock().now().nanoseconds * 1e-9
        self.tracked_obstacles, self.next_obstacle_id = (
            update_tracked_obstacles(
                self.tracked_obstacles,
                observations,
                now_seconds,
                self.next_obstacle_id,
                self.obstacle_match_distance,
                self.obstacle_memory))

    def vehicle_pose(self):
        return self.current_map_base_pose()[:2]

    def active_obstacles(self, vehicle_s):
        planning_horizon = self.current_planning_horizon()
        active = []
        for obstacle in self.tracked_obstacles:
            if int(obstacle.get('hits', 1)) < self.obstacle_confirmation_frames:
                continue
            forward = self.path_geometry.forward_distance(
                vehicle_s, obstacle['s'])
            if forward <= planning_horizon:
                active.append((forward, obstacle))
        active.sort(key=lambda item: item[0])
        return active

    def current_planning_horizon(self):
        # Plan for the commanded ceiling, not only the current odometry speed.
        # Otherwise a vehicle accelerating from rest discovers that its
        # previously short avoidance maneuver is infeasible only after it has
        # already reached racing speed.
        planning_speed = max(self.speed, self.maximum_planning_speed)
        return speed_dependent_horizon(
            planning_speed,
            self.planning_reaction_time,
            self.planning_deceleration,
            self.planning_distance_margin,
            self.obstacle_lookahead,
            self.maximum_planning_horizon)

    def current_avoidance_distances(self):
        planning_speed = max(self.speed, self.maximum_planning_speed)
        before = max(
            self.avoidance_before,
            min(self.maximum_avoidance_before,
                planning_speed * self.avoidance_before_time))
        after = max(
            self.avoidance_after,
            min(self.maximum_avoidance_after,
                planning_speed * self.avoidance_after_time))
        return before, after

    def candidate_is_safe(
            self, points, vehicle_s, obstacles, planning_horizon,
            avoidance_after):
        local = sample_path_window(
            points,
            self.path_geometry,
            vehicle_s,
            planning_horizon + avoidance_after,
            self.sample_spacing)

        map_values = self.map_clearances(local)
        minimum_map_clearance = float(np.min(map_values))
        map_extra_clearance = minimum_map_clearance - self.map_clearance
        if map_extra_clearance < 0.0:
            return False, map_extra_clearance

        minimum_obstacle_clearance = float('inf')
        for obstacle in obstacles:
            center_distance = np.linalg.norm(
                local - obstacle['center'], axis=1)
            required = (
                self.vehicle_clearance
                + obstacle['radius']
                + self.obstacle_margin)
            minimum = float(np.min(center_distance) - required)
            minimum_obstacle_clearance = min(
                minimum_obstacle_clearance, minimum)
            if minimum < 0.0:
                return False, minimum
        return True, min(map_extra_clearance, minimum_obstacle_clearance)

    def candidate_curvature(self, points, vehicle_s, distance):
        local = sample_path_window(
            points, self.path_geometry, vehicle_s, distance,
            max(self.sample_spacing, 0.05))
        return path_curvature_percentile(
            local, self.curvature_percentile)

    def publish_speed_limit(self, value):
        message = Float32()
        message.data = float(max(0.0, value))
        self.speed_limit_pub.publish(message)

    def publish_path(self, points):
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.global_frame
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            yaw = math.atan2(
                next_point[1] - point[1], next_point[0] - point[0])
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            message.poses.append(pose)
        self.path_pub.publish(message)

    def publish_markers(self, selected_path, active_obstacles):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        path_marker = Marker()
        path_marker.header.stamp = stamp
        path_marker.header.frame_id = self.global_frame
        path_marker.ns = 'selected_local_path'
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.065
        path_marker.color.r = 1.0
        path_marker.color.g = 0.0
        path_marker.color.b = 1.0
        path_marker.color.a = 0.9
        for xy in selected_path:
            point = Point()
            point.x = float(xy[0])
            point.y = float(xy[1])
            point.z = 0.07
            path_marker.points.append(point)
        if len(selected_path):
            path_marker.points.append(path_marker.points[0])
        marker_array.markers.append(path_marker)

        for _, obstacle in active_obstacles:
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.global_frame
            marker.ns = 'scan_detected_obstacles'
            marker.id = int(obstacle['id']) + 100
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(obstacle['center'][0])
            marker.pose.position.y = float(obstacle['center'][1])
            marker.pose.position.z = 0.12
            marker.pose.orientation.w = 1.0
            # Show the estimated physical obstacle only.  Collision checking
            # applies vehicle radius and safety margin separately below; they
            # must not make the RViz obstacle itself look larger.
            # Keep RViz close to the measured object footprint while collision
            # checks below retain the full conservative radius.
            diameter = (
                2.0 * obstacle['radius']
                * self.obstacle_marker_radius_scale)
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = diameter
            marker.color.r = 1.0
            marker.color.g = 0.8
            marker.color.b = 0.0
            marker.color.a = 0.9
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def set_status(self, text):
        message = String()
        message.data = text
        self.status_pub.publish(message)
        if text != self.last_status:
            self.get_logger().info(text)
            self.last_status = text

    def plan(self):
        stop = Bool()
        avoidance = Bool()
        if self.path_geometry is None or self.map_clearance_grid is None:
            stop.data = True
            self.stop_pub.publish(stop)
            self.avoidance_pub.publish(avoidance)
            self.publish_speed_limit(0.0)
            self.set_status('WAITING_FOR_GLOBAL_PATH_OR_MAP')
            return
        if (self.last_scan_time is None
                or (self.get_clock().now() - self.last_scan_time).nanoseconds
                * 1e-9 > self.scan_timeout):
            stop.data = True
            self.stop_pub.publish(stop)
            self.avoidance_pub.publish(avoidance)
            self.publish_speed_limit(0.0)
            self.set_status('AEB_SCAN_TIMEOUT')
            return

        try:
            vehicle_xy = self.vehicle_pose()
        except (TransformException, RuntimeError) as error:
            stop.data = True
            self.stop_pub.publish(stop)
            self.avoidance_pub.publish(avoidance)
            self.publish_speed_limit(0.0)
            self.set_status('AEB_TF_UNAVAILABLE: %s' % error)
            return

        vehicle_s, _, _ = self.path_geometry.project(vehicle_xy)
        planning_horizon = self.current_planning_horizon()
        avoidance_before, avoidance_after = (
            self.current_avoidance_distances())
        active = self.active_obstacles(vehicle_s)
        selected = self.path_geometry.points
        selected_offset = 0.0
        feasible = True
        candidate_results = []

        if active:
            primary_forward, primary = active[0]
            if self.locked_obstacle_id != primary['id']:
                self.locked_obstacle_id = primary['id']
                self.locked_offset = None
            collision_obstacles = [
                obstacle for forward, obstacle in active
                if forward <= primary_forward + avoidance_after]
            offsets = ordered_candidate_offsets(
                self.candidate_offsets,
                self.locked_offset,
                primary['lateral'])
            feasible = False
            best_score = float('inf')
            for offset in offsets:
                candidate = self.path_geometry.offset_bump(
                    primary['s'], offset,
                    avoidance_before, avoidance_after)
                safe, clearance = self.candidate_is_safe(
                    candidate, vehicle_s, collision_obstacles,
                    planning_horizon, avoidance_after)
                curvature = self.candidate_curvature(
                    candidate, vehicle_s, planning_horizon)
                candidate_results.append((offset, clearance, curvature))
                if not safe:
                    continue
                score = (
                    abs(offset)
                    + self.candidate_curvature_weight * curvature
                    - self.candidate_clearance_weight
                    * min(max(clearance, 0.0), 1.0))
                if self.locked_offset is not None:
                    score += 2.0 * abs(offset - self.locked_offset)
                if score < best_score:
                    selected = candidate
                    selected_offset = offset
                    best_score = score
                    feasible = True
            if feasible and abs(selected_offset) > 1e-3:
                self.locked_offset = selected_offset
        else:
            self.locked_obstacle_id = None
            self.locked_offset = None

        if feasible:
            self.last_safe_path = selected.copy()
            self.publish_path(selected)
        elif self.last_safe_path is not None:
            self.publish_path(self.last_safe_path)

        avoidance.data = bool(
            feasible and active and abs(selected_offset) > 1e-3)
        critical_distance = (
            self.aeb_min_distance
            + self.vehicle_clearance
            + self.obstacle_default_radius
            + self.obstacle_margin
            + self.speed * self.aeb_critical_reaction_time)
        critical_stop = bool(
            self.ttc_stop
            and self.nearest_corridor_distance < critical_distance)
        stop.data = bool(critical_stop or not feasible)
        self.stop_pub.publish(stop)
        self.avoidance_pub.publish(avoidance)
        speed_limit = self.maximum_planning_speed
        if avoidance.data:
            selected_curvature = self.candidate_curvature(
                selected, vehicle_s, planning_horizon)
            curve_speed = math.sqrt(
                self.max_lateral_acceleration
                / max(selected_curvature, 1e-3))
            speed_limit = min(speed_limit, max(
                self.minimum_avoidance_speed, curve_speed))
        if stop.data:
            speed_limit = 0.0
        self.publish_speed_limit(speed_limit)
        self.publish_markers(selected, active)
        if critical_stop:
            self.set_status('AEB_STOP')
        elif not feasible:
            details = ','.join(
                '%+.2f:clear=%+.2f:k=%.2f' % result
                for result in candidate_results)
            self.set_status('NO_COLLISION_FREE_PATH ' + details)
        elif active and abs(selected_offset) > 1e-3:
            self.set_status(
                'AVOIDING obstacle=%d offset=%+.2fm'
                % (active[0][1]['id'], selected_offset))
        elif active:
            self.set_status('OBSTACLE_CLEAR_OF_SELECTED_PATH')
        else:
            self.set_status('GLOBAL_PATH_CLEAR')


def main(args=None):
    rclpy.init(args=args)
    node = LocalObstaclePlannerNode()
    # One worker may be evaluating path candidates while another waits for a
    # timestamped scan transform.  Keep a third worker available so the TF
    # listener can receive the transform that unblocks that lookup.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
