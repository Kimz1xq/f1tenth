"""Scan-only obstacle detector and local path planner for F1TENTH."""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
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
    adaptive_candidate_offsets,
    adaptive_map_endpoint_threshold,
    cluster_ordered_points,
    minimum_clustered_path_clearance,
    nearest_clustered_corridor_distance,
    minimum_surface_footprint_clearance,
    ordered_candidate_offsets,
    sample_path_window,
    spline_path_curvature_percentile,
    speed_dependent_horizon,
    swept_rectangle_samples,
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
        self.declare_parameter('path_heartbeat_rate', 2.0)
        self.declare_parameter('marker_publish_rate', 5.0)
        self.declare_parameter('scan_transform_delay', 0.08)
        self.declare_parameter('detection_range', 3.0)
        self.declare_parameter('map_endpoint_clearance', 0.30)
        self.declare_parameter('map_registration_percentile', 60.0)
        self.declare_parameter('map_registration_margin', 0.08)
        self.declare_parameter('map_registration_max_extra', 0.25)
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
        self.declare_parameter('candidate_offset_spacing', 0.04)
        self.declare_parameter('candidate_offset_count', 6)
        self.declare_parameter('maximum_candidate_offset', 0.70)
        self.declare_parameter(
            'candidate_transition_scales', [1.0, 1.25, 1.50])
        self.declare_parameter('candidate_clearance_buffer', 0.05)
        self.declare_parameter('avoidance_before_distance', 1.80)
        self.declare_parameter('avoidance_after_distance', 1.10)
        self.declare_parameter('maximum_avoidance_before_distance', 4.0)
        self.declare_parameter('maximum_avoidance_after_distance', 4.0)
        self.declare_parameter('avoidance_before_time', 0.60)
        self.declare_parameter('avoidance_after_time', 0.30)
        self.declare_parameter('path_sample_spacing', 0.04)
        self.declare_parameter('map_clearance', 0.19)
        self.declare_parameter('vehicle_clearance_radius', 0.17)
        self.declare_parameter('vehicle_length', 0.58)
        self.declare_parameter('vehicle_width', 0.31)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('max_steering_angle', 0.4189)
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
        self.map_registration_percentile = float(
            self.get_parameter('map_registration_percentile').value)
        self.map_registration_margin = float(
            self.get_parameter('map_registration_margin').value)
        self.map_registration_max_extra = float(
            self.get_parameter('map_registration_max_extra').value)
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
        self.candidate_offset_spacing = float(
            self.get_parameter('candidate_offset_spacing').value)
        self.candidate_offset_count = int(
            self.get_parameter('candidate_offset_count').value)
        self.maximum_candidate_offset = float(
            self.get_parameter('maximum_candidate_offset').value)
        self.candidate_transition_scales = sorted(set(
            float(value) for value in self.get_parameter(
                'candidate_transition_scales').value
            if float(value) >= 1.0))
        if not self.candidate_transition_scales:
            raise RuntimeError(
                'candidate_transition_scales must contain a value >= 1.0')
        self.candidate_clearance_buffer = float(
            self.get_parameter('candidate_clearance_buffer').value)
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
        self.vehicle_length = float(
            self.get_parameter('vehicle_length').value)
        self.vehicle_width = float(
            self.get_parameter('vehicle_width').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.max_steering_angle = float(
            self.get_parameter('max_steering_angle').value)
        if self.wheelbase <= 0.0 or not (
                0.0 < self.max_steering_angle < 0.5 * math.pi):
            raise RuntimeError(
                'wheelbase and max_steering_angle must define a valid '
                'Ackermann vehicle')
        self.maximum_path_curvature = (
            math.tan(self.max_steering_angle) / self.wheelbase)
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
        self.path_heartbeat_period = 1.0 / max(
            float(self.get_parameter('path_heartbeat_rate').value), 0.1)
        self.marker_publish_period = 1.0 / max(
            float(self.get_parameter('marker_publish_rate').value), 0.1)
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
        self.latest_scan_clusters = []
        self.latest_scan_generation = 0
        self.last_aeb_scan_generation = -1
        self.aeb_detection_count = 0
        self.tracked_obstacles = []
        self.next_obstacle_id = 0
        self.last_safe_path = None
        self.last_published_path = None
        self.last_path_publish_time = None
        self.last_marker_publish_time = None
        self.last_status = ''
        self.locked_obstacle_id = None
        self.locked_offset = None
        self.locked_obstacle_s = None
        self.locked_vehicle_s = None
        self.locked_release_after = 0.0

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
            Path, self.get_parameter('local_path_topic').value, 1)
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
            self.locked_obstacle_s = None
            self.locked_vehicle_s = None
            self.locked_release_after = 0.0
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
            endpoint_threshold = adaptive_map_endpoint_threshold(
                endpoint_clearance,
                self.map_endpoint_clearance,
                self.map_registration_percentile,
                self.map_registration_margin,
                self.map_registration_max_extra)
            unmapped_mask = endpoint_clearance > (
                endpoint_threshold + raster_tolerance)
            for item, is_unmapped in zip(indexed_points, unmapped_mask):
                if is_unmapped:
                    filtered.append(item)

        filtered_base = [
            (beam_index, x_value, y_value)
            for beam_index, x_value, y_value, is_unmapped in zip(
                indices.tolist(), base_x.tolist(), base_y.tolist(),
                unmapped_mask.tolist())
            if is_unmapped
        ]
        base_clusters = cluster_ordered_points(
            filtered_base,
            max_gap=self.cluster_gap,
            min_points=self.cluster_min_points,
            max_diameter=self.cluster_max_diameter)
        self.nearest_corridor_distance = (
            nearest_clustered_corridor_distance(
                base_clusters, self.aeb_half_width))

        clusters = cluster_ordered_points(
            filtered,
            max_gap=self.cluster_gap,
            min_points=self.cluster_min_points,
            max_diameter=self.cluster_max_diameter)
        # The newest map-frame LaserScan surfaces feed the path-aligned AEB.
        # This is the same measured input in simulation and on the car.
        self.latest_scan_clusters = [cluster.copy() for cluster in clusters]
        self.latest_scan_generation += 1
        observations = []
        for cluster in clusters:
            center = np.mean(cluster, axis=0)
            s_value, lateral, _ = self.path_geometry.project(center)
            if abs(lateral) > self.path_obstacle_corridor:
                continue
            # Project the compact cluster in one local Frenet frame. On tracks
            # with nearby opposing branches, independently projecting every
            # surface point can send half of one object to the other branch
            # and inflate a 0.2 m object into a 0.5 m obstacle.
            segment_index = int(np.searchsorted(
                self.path_geometry.cumulative, s_value,
                side='right') - 1) % len(self.path_geometry.points)
            heading = float(self.path_geometry.yaw[segment_index])
            tangent = np.asarray([math.cos(heading), math.sin(heading)])
            normal = np.asarray([-math.sin(heading), math.cos(heading)])
            relative_surface = cluster - center
            surface_laterals = (
                lateral + relative_surface @ normal)
            surface_longitudinals = relative_surface @ tangent
            observed_radius = float(np.max(np.linalg.norm(
                cluster - center, axis=1)))
            radius = min(
                self.obstacle_max_radius,
                max(self.obstacle_default_radius, observed_radius))
            observations.append((
                center, radius, s_value, lateral, cluster,
                float(np.min(surface_laterals)),
                float(np.max(surface_laterals)),
                float(np.min(surface_longitudinals)),
                float(np.max(surface_longitudinals))))
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
        # The requested top speed is only a ceiling.  Planning every obstacle
        # at that ceiling makes a short track request unrealistically long,
        # low-curvature transitions which run into unrelated walls. Generate
        # geometry for the measured speed; candidate curvature below then
        # produces the physically valid avoidance speed limit.
        planning_speed = min(
            self.maximum_planning_speed,
            max(self.speed, self.minimum_avoidance_speed))
        stopping_horizon = speed_dependent_horizon(
            planning_speed,
            self.planning_reaction_time,
            self.planning_deceleration,
            self.planning_distance_margin,
            self.obstacle_lookahead,
            self.maximum_planning_horizon)
        # Detection must precede the start of the smooth lateral transition.
        # A stopping-distance-only horizon can activate an obstacle after the
        # vehicle has already passed that transition start, forcing the
        # controller to join a path whose tangent is no longer reachable.
        longest_transition = max(
            self.avoidance_before,
            min(self.maximum_avoidance_before,
                planning_speed * self.avoidance_before_time))
        transition_horizon = (
            longest_transition
            + planning_speed * self.planning_reaction_time
            + self.planning_distance_margin)
        return min(
            self.maximum_planning_horizon,
            max(stopping_horizon, transition_horizon))

    def current_avoidance_distances(self, offset=0.0):
        # Sample geometry independently from the requested velocity ceiling.
        # Ackermann feasibility is checked for every candidate and its
        # curvature is converted to a speed limit after selection.
        planning_speed = min(
            self.maximum_planning_speed,
            max(self.speed, self.minimum_avoidance_speed))
        before = max(
            self.avoidance_before,
            min(self.maximum_avoidance_before,
                planning_speed * self.avoidance_before_time))
        after = max(
            self.avoidance_after,
            min(self.maximum_avoidance_after,
                planning_speed * self.avoidance_after_time))
        return (
            min(before, self.maximum_avoidance_before),
            min(after, self.maximum_avoidance_after))

    def candidate_is_safe(
            self, points, vehicle_s, obstacles, planning_horizon,
            avoidance_after):
        local = sample_path_window(
            points,
            self.path_geometry,
            vehicle_s,
            planning_horizon + avoidance_after,
            self.sample_spacing)

        footprint = swept_rectangle_samples(
            local,
            self.vehicle_length,
            self.vehicle_width,
            self.candidate_clearance_buffer)
        map_values = self.map_clearances(footprint)
        minimum_map_clearance = float(np.min(map_values))
        # ``map_clearance`` historically represented half vehicle width plus
        # the wall margin. The footprint now models vehicle width explicitly,
        # so retain only that independent wall margin here.
        wall_margin = max(
            0.0, self.map_clearance - self.vehicle_clearance)
        map_extra_clearance = minimum_map_clearance - wall_margin
        if map_extra_clearance < 0.0:
            return False, map_extra_clearance, 'map'

        minimum_obstacle_clearance = float('inf')
        for obstacle in obstacles:
            surface_points = obstacle.get('surface_points')
            if surface_points is not None and len(surface_points):
                minimum = minimum_surface_footprint_clearance(
                    local,
                    surface_points,
                    self.vehicle_length,
                    self.vehicle_width,
                    self.obstacle_margin
                    + self.candidate_clearance_buffer)
            else:
                minimum = minimum_surface_footprint_clearance(
                    local,
                    np.asarray([obstacle['center']]),
                    self.vehicle_length + 2.0 * obstacle['radius'],
                    self.vehicle_width + 2.0 * obstacle['radius'],
                    self.obstacle_margin
                    + self.candidate_clearance_buffer)
            minimum_obstacle_clearance = min(
                minimum_obstacle_clearance, minimum)
            if minimum < 0.0:
                return False, minimum, 'obstacle'
        if map_extra_clearance <= minimum_obstacle_clearance:
            return True, map_extra_clearance, 'map'
        return True, minimum_obstacle_clearance, 'obstacle'

    def candidate_curvature(
            self, points, vehicle_s, distance, percentile=None):
        local = sample_path_window(
            points, self.path_geometry, vehicle_s, distance,
            max(self.sample_spacing, 0.05))
        return spline_path_curvature_percentile(
            local,
            self.curvature_percentile if percentile is None else percentile)

    def obstacle_plateau_distances(self, obstacle):
        """Return measured-object hold distances around an obstacle."""
        # Vehicle dimensions are already applied by candidate_is_safe() as a
        # swept footprint / clearance radius.  Adding half the vehicle length
        # here as well keeps the path displaced twice as long and can push its
        # return transition into an otherwise unrelated wall.  The plateau
        # therefore represents only the measured object's longitudinal span.
        fallback = obstacle['radius'] + self.obstacle_margin
        longitudinal_min = obstacle.get('longitudinal_min')
        longitudinal_max = obstacle.get('longitudinal_max')
        if longitudinal_min is None or longitudinal_max is None:
            return fallback, fallback
        hold_before = (
            max(0.0, -float(longitudinal_min))
            + self.obstacle_margin)
        hold_after = (
            max(0.0, float(longitudinal_max))
            + self.obstacle_margin)
        return hold_before, hold_after

    def obstacle_group_plateau_distances(self, primary, obstacles):
        """Cover consecutive detected obstacles with one lateral corridor."""
        relative_extents = []
        for obstacle in obstacles:
            longitudinal_min = obstacle.get('longitudinal_min')
            longitudinal_max = obstacle.get('longitudinal_max')
            if longitudinal_min is not None and longitudinal_max is not None:
                center_delta = float(
                    self.path_geometry.circular_delta(
                        [obstacle['s']], primary['s'])[0])
                relative_extents.extend((
                    center_delta + float(longitudinal_min),
                    center_delta + float(longitudinal_max)))
            else:
                center_delta = float(
                    self.path_geometry.circular_delta(
                        [obstacle['s']], primary['s'])[0])
                relative_extents.extend((
                    center_delta - obstacle['radius'],
                    center_delta + obstacle['radius']))
        if not relative_extents:
            return self.obstacle_plateau_distances(primary)
        return (
            max(0.0, -min(relative_extents)) + self.obstacle_margin,
            max(0.0, max(relative_extents)) + self.obstacle_margin,
        )

    def publish_speed_limit(self, value):
        message = Float32()
        message.data = float(max(0.0, value))
        self.speed_limit_pub.publish(message)

    def publish_path(self, points):
        """Publish path changes immediately and unchanged paths as heartbeat.

        A long raceline can contain thousands of poses. Rebuilding and
        serializing the identical Path at the planning rate consumes enough
        CPU to delay LaserScan and TF processing. The heartbeat keeps
        controller freshness checks valid, while an avoidance-path change is
        still delivered in the same planning cycle.
        """
        points = np.asarray(points, dtype=float)
        now = self.get_clock().now()
        changed = (
            self.last_published_path is None
            or self.last_published_path.shape != points.shape
            or np.max(np.abs(
                self.last_published_path - points), initial=0.0) > 1e-6)
        heartbeat_due = (
            self.last_path_publish_time is None
            or (now - self.last_path_publish_time).nanoseconds * 1e-9
            >= self.path_heartbeat_period)
        if not changed and not heartbeat_due:
            return False

        message = Path()
        message.header.stamp = now.to_msg()
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
        self.last_published_path = points.copy()
        self.last_path_publish_time = now
        return True

    def publish_markers(self, selected_path, active_obstacles):
        # Markers are diagnostic only. Avoid serializing a full raceline at
        # the control/planning rate, and do no work when RViz is not attached.
        if self.marker_pub.get_subscription_count() == 0:
            return
        now = self.get_clock().now()
        if (self.last_marker_publish_time is not None
                and (now - self.last_marker_publish_time).nanoseconds * 1e-9
                < self.marker_publish_period):
            return
        self.last_marker_publish_time = now
        marker_array = MarkerArray()
        stamp = now.to_msg()

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
        active = self.active_obstacles(vehicle_s)
        selected = self.path_geometry.points
        selected_offset = 0.0
        selected_after = self.avoidance_after
        feasible = True
        retained_avoidance = False
        candidate_results = []

        if active:
            primary_forward, primary = active[0]
            if self.locked_obstacle_id != primary['id']:
                self.locked_obstacle_id = primary['id']
                self.locked_offset = None
                self.locked_obstacle_s = float(primary['s'])
                # Keep the transition origin fixed for this manoeuvre. Moving
                # it to the latest vehicle pose every planning tick steadily
                # shortens the entry and eventually drives the swept vehicle
                # footprint back through the obstacle.
                self.locked_vehicle_s = float(vehicle_s)
                self.locked_release_after = self.avoidance_after
            collision_obstacles = [
                obstacle for forward, obstacle in active
                if forward <= (
                    primary_forward + self.maximum_avoidance_after)]
            primary_lateral_min = float(primary.get(
                'lateral_min', primary['lateral'] - primary['radius']))
            primary_lateral_max = float(primary.get(
                'lateral_max', primary['lateral'] + primary['radius']))
            surface_lateral = 0.5 * (
                primary_lateral_min + primary_lateral_max)
            offsets = []
            # Consecutive obstacles can occupy the same planning horizon. Use
            # the union of their lateral action sets, then keep one offset
            # through the complete group. This is the small action-set form of
            # a graph planner and avoids returning into the next obstacle.
            for obstacle in collision_obstacles:
                lateral_min = float(obstacle.get(
                    'lateral_min',
                    obstacle['lateral'] - obstacle['radius']))
                lateral_max = float(obstacle.get(
                    'lateral_max',
                    obstacle['lateral'] + obstacle['radius']))
                obstacle_lateral = 0.5 * (lateral_min + lateral_max)
                # A LiDAR initially sees only the nearest face. Use the
                # configured physical-radius estimate as the lower bound so
                # the selected side/offset does not change as side faces
                # become visible during the approach.
                surface_half_width = max(
                    float(obstacle['radius']),
                    0.5 * (lateral_max - lateral_min))
                required_lateral_clearance = (
                    self.vehicle_clearance
                    + surface_half_width
                    + self.obstacle_margin
                    + self.candidate_clearance_buffer)
                for value in adaptive_candidate_offsets(
                        obstacle_lateral,
                        required_lateral_clearance,
                        self.candidate_offset_spacing,
                        self.candidate_offset_count,
                        self.maximum_candidate_offset):
                    if not any(abs(value - existing) < 1e-6
                               for existing in offsets):
                        offsets.append(value)
            if (self.locked_offset is not None
                    and abs(self.locked_offset)
                    <= self.maximum_candidate_offset
                    and not any(abs(
                        self.locked_offset - value) < 1e-6
                        for value in offsets)):
                offsets.append(self.locked_offset)
            offsets = ordered_candidate_offsets(
                offsets,
                self.locked_offset,
                surface_lateral)
            feasible = False
            best_score = float('inf')
            hold_before, hold_after = (
                self.obstacle_group_plateau_distances(
                    primary, collision_obstacles))
            # The supplied global raceline is the proven driveable baseline.
            # Polyline curvature around a tight corner can slightly exceed
            # the analytic Ackermann limit even though the preview controller
            # tracks it successfully. Reject an avoidance candidate only when
            # it exceeds both the physical limit and the local baseline, so
            # ordinary corner curvature is not mistaken for added avoidance
            # curvature.
            baseline_curvature = self.candidate_curvature(
                self.path_geometry.points, vehicle_s, planning_horizon)
            allowed_curvature = max(
                self.maximum_path_curvature, baseline_curvature)
            for offset in offsets:
                base_before, base_after = (
                    self.current_avoidance_distances(offset))
                for transition_scale in self.candidate_transition_scales:
                    candidate_before = min(
                        self.maximum_planning_horizon,
                        base_before * transition_scale)
                    candidate_after = min(
                        self.maximum_planning_horizon,
                        base_after * transition_scale)
                    candidate = self.path_geometry.reachable_offset_plateau(
                        self.locked_vehicle_s,
                        primary['s'],
                        offset,
                        candidate_before,
                        candidate_after,
                        hold_before,
                        hold_after)
                    if candidate is None:
                        continue
                    evaluation_distance = (
                        planning_horizon + hold_after + candidate_after)
                    safe, clearance, clearance_source = self.candidate_is_safe(
                        candidate, vehicle_s, collision_obstacles,
                        planning_horizon + hold_after, candidate_after)
                    curvature = self.candidate_curvature(
                        candidate, vehicle_s, evaluation_distance)
                    # Use the configured robust local curvature for Ackermann
                    # feasibility. A near-maximum percentile on a piecewise
                    # linear path measures isolated waypoint impulses rather
                    # than the curvature followed by the controller.
                    peak_curvature = curvature
                    kinematically_feasible = (
                        peak_curvature <= allowed_curvature + 1e-9)
                    candidate_results.append((
                        offset, clearance, curvature, peak_curvature,
                        transition_scale, clearance_source))
                    if not safe or not kinematically_feasible:
                        continue
                    local_candidate = sample_path_window(
                        candidate, self.path_geometry, vehicle_s,
                        evaluation_distance, self.sample_spacing)
                    local_reference = sample_path_window(
                        self.path_geometry.points, self.path_geometry,
                        vehicle_s, evaluation_distance, self.sample_spacing)
                    deviation_cost = float(np.mean(np.linalg.norm(
                        local_candidate - local_reference, axis=1)))
                    score = (
                        deviation_cost
                        + self.candidate_curvature_weight * curvature
                        - self.candidate_clearance_weight
                        * min(max(clearance, 0.0), 1.0))
                    if self.locked_offset is not None:
                        score += 2.0 * abs(offset - self.locked_offset)
                    if score < best_score:
                        selected = candidate
                        selected_offset = offset
                        selected_after = hold_after + candidate_after
                        best_score = score
                        feasible = True
            if feasible and abs(selected_offset) > 1e-3:
                self.locked_offset = selected_offset
                self.locked_obstacle_s = float(primary['s'])
                self.locked_release_after = selected_after
        else:
            # A static obstacle can disappear from the scan while alongside
            # the vehicle. Keep the already validated avoidance path until
            # the vehicle has travelled beyond its smooth return section.
            # This uses path progress and the existing avoidance distance,
            # not a track coordinate or a timing guess.
            if (self.locked_obstacle_s is not None
                    and self.locked_offset is not None
                    and self.last_safe_path is not None):
                forward_to_obstacle = self.path_geometry.forward_distance(
                    vehicle_s, self.locked_obstacle_s)
                obstacle_is_behind = (
                    forward_to_obstacle > 0.5 * self.path_geometry.length)
                distance_past = self.path_geometry.forward_distance(
                    self.locked_obstacle_s, vehicle_s)
                retained_avoidance = bool(
                    not obstacle_is_behind
                    or distance_past < self.locked_release_after)
            if retained_avoidance:
                selected = self.last_safe_path
                selected_offset = self.locked_offset
            else:
                self.locked_obstacle_id = None
                self.locked_offset = None
                self.locked_obstacle_s = None
                self.locked_vehicle_s = None
                self.locked_release_after = 0.0

        if feasible:
            self.last_safe_path = selected.copy()
            self.publish_path(selected)
        elif self.last_safe_path is not None:
            self.publish_path(self.last_safe_path)

        avoidance.data = bool(
            feasible and (
                retained_avoidance
                or (active and abs(selected_offset) > 1e-3)))
        # Follow the selected path for the emergency check.  A fixed straight
        # corridor fights a valid avoidance manoeuvre: the bypassed object is
        # still physically in front of the car even though the selected path
        # has already curved around it.  The full current-speed stopping
        # distance remains fail-safe when that selected path still intersects
        # a dense LaserScan cluster.
        critical_distance = (
            self.aeb_min_distance
            + self.speed * self.aeb_reaction_time
            + self.speed ** 2 / (2.0 * max(self.aeb_deceleration, 0.1)))
        commanded_path = (
            selected if feasible or self.last_safe_path is None
            else self.last_safe_path)
        emergency_path = sample_path_window(
            commanded_path,
            self.path_geometry,
            vehicle_s,
            max(critical_distance, self.vehicle_length),
            self.sample_spacing)
        emergency_clearance = minimum_clustered_path_clearance(
            emergency_path,
            self.latest_scan_clusters,
            self.vehicle_length,
            self.vehicle_width,
            self.obstacle_margin)
        raw_path_stop = bool(emergency_clearance < 0.0)
        # A planning timer may run more often than LaserScan processing. Count
        # each sensor observation once so confirmation_frames represents real
        # independent measurements instead of repeated use of one scan.
        if self.latest_scan_generation != self.last_aeb_scan_generation:
            self.aeb_detection_count = (
                self.aeb_detection_count + 1 if raw_path_stop else 0)
            self.last_aeb_scan_generation = self.latest_scan_generation
        critical_stop = bool(
            self.aeb_detection_count >= self.aeb_confirmation_frames)
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
            self.set_status(
                'AEB_STOP path_clearance=%+.2fm' % emergency_clearance)
        elif not feasible:
            details = ','.join(
                '%+.2f:%s=%+.2f:k=%.2f:L=%.2f' % (
                    result[0], result[5], result[1], result[2], result[4])
                for result in candidate_results)
            self.set_status('NO_COLLISION_FREE_PATH ' + details)
        elif active and abs(selected_offset) > 1e-3:
            self.set_status(
                'AVOIDING obstacle=%d offset=%+.2fm'
                % (active[0][1]['id'], selected_offset))
        elif retained_avoidance:
            self.set_status(
                'AVOIDING_LOCKED_PATH offset=%+.2fm' % selected_offset)
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
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RCLError):
            pass


if __name__ == '__main__':
    main()
