# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from visualization_msgs.msg import Marker, MarkerArray

import gym
import numpy as np
from transforms3d import euler

from f1tenth_gym_ros.static_obstacles import (
    OccupancyMap,
    StaticObstacle,
    generate_obstacles,
    inject_obstacles_into_scan,
    lap_count_transition,
    load_path,
    resolve_obstacle_seed,
    vehicle_hits_obstacle,
)

class GymBridge(Node):
    def __init__(self):
        super().__init__('gym_bridge')

        self.declare_parameter('ego_namespace', 'ego_racecar')
        self.declare_parameter('ego_odom_topic', 'odom')
        self.declare_parameter('ego_opp_odom_topic', 'opp_odom')
        self.declare_parameter('ego_scan_topic', 'scan')
        self.declare_parameter('ego_drive_topic', 'drive')
        self.declare_parameter('opp_namespace', 'opp_racecar')
        self.declare_parameter('opp_odom_topic', 'odom')
        self.declare_parameter('opp_ego_odom_topic', 'opp_odom')
        self.declare_parameter('opp_scan_topic', 'opp_scan')
        self.declare_parameter('opp_drive_topic', 'opp_drive')
        self.declare_parameter('scan_distance_to_base_link', 0.275)
        self.declare_parameter('scan_fov', 4.7)
        self.declare_parameter('scan_beams', 1080)
        self.declare_parameter(
            'map_path', '/sim_ws/src/f1tenth_gym_ros/maps/track03')
        self.declare_parameter('map_img_ext', '.pgm')
        self.declare_parameter('num_agent', 1)
        self.declare_parameter('sx', 0.2985)
        self.declare_parameter('sy', 0.5926)
        self.declare_parameter('stheta', -0.6205)
        self.declare_parameter('sx1', 2.0)
        self.declare_parameter('sy1', 0.5)
        self.declare_parameter('stheta1', 0.0)
        self.declare_parameter('kb_teleop', True)
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('ground_truth_odom_topic', '/ground_truth/odom')
        self.declare_parameter('sim_reset_topic', '/sim_reset_pose')
        self.declare_parameter('collision_topic', '/ego_racecar/collision')
        self.declare_parameter('topic_publish_rate', 40.0)
        self.declare_parameter('random_obstacles_enabled', False)
        self.declare_parameter('random_obstacle_count', 2)
        self.declare_parameter('random_obstacle_seed', -1)
        self.declare_parameter('randomize_obstacles_on_reset', False)
        self.declare_parameter('randomize_obstacles_on_lap', True)
        self.declare_parameter(
            'obstacle_path_csv',
            '/sim_ws/src/planning/waypoints/track03_centerline.csv')
        self.declare_parameter('obstacle_length', 0.20)
        self.declare_parameter('obstacle_width', 0.12)
        self.declare_parameter('obstacle_height', 0.20)
        self.declare_parameter('obstacle_lateral_offset', 0.16)
        self.declare_parameter('obstacle_start_clearance', 1.50)
        self.declare_parameter('obstacle_min_spacing', 3.00)
        self.declare_parameter('obstacle_passage_offset', 0.18)
        self.declare_parameter('obstacle_passage_radius', 0.19)
        self.declare_parameter(
            'obstacle_marker_topic',
            '/simulation/obstacles_ground_truth')
        self.declare_parameter('vehicle_length', 0.58)
        self.declare_parameter('vehicle_width', 0.31)
        self.declare_parameter('friction_mu', 1.0489)

        # check num_agents
        num_agents = self.get_parameter('num_agent').value
        if num_agents < 1 or num_agents > 2:
            raise ValueError('num_agents should be either 1 or 2.')
        elif type(num_agents) != int:
            raise ValueError('num_agents should be an int.')

        # Keep the simulator vehicle model in one place. Experiments normally
        # override only friction_mu from the bringup command.
        vehicle_params = {
            'mu': float(self.get_parameter('friction_mu').value),
            'C_Sf': 4.718,
            'C_Sr': 5.4562,
            'lf': 0.15875,
            'lr': 0.17145,
            'h': 0.074,
            'm': 3.74,
            'I': 0.04712,
            's_min': -0.4189,
            's_max': 0.4189,
            'sv_min': -3.2,
            'sv_max': 3.2,
            'v_switch': 7.319,
            'a_max': 9.51,
            'v_min': -5.0,
            'v_max': 20.0,
            'width': float(self.get_parameter('vehicle_width').value),
            'length': float(self.get_parameter('vehicle_length').value),
        }

        # env backend
        self.env = gym.make('f110_gym:f110-v0',
                            map=self.get_parameter('map_path').value,
                            map_ext=self.get_parameter('map_img_ext').value,
                            num_agents=num_agents,
                            lidar_dist=self.get_parameter("scan_distance_to_base_link").value,
                            params=vehicle_params,
                            )
        self.get_logger().info(
            'Gym map=%s%s friction_mu=%.4f'
            % (
                self.get_parameter('map_path').value,
                self.get_parameter('map_img_ext').value,
                vehicle_params['mu'],
            )
        )

        sx = self.get_parameter('sx').value
        sy = self.get_parameter('sy').value
        stheta = self.get_parameter('stheta').value
        self.ego_pose = [sx, sy, stheta]
        self.ego_speed = [0.0, 0.0, 0.0]
        self.ego_requested_speed = 0.0
        self.ego_steer = 0.0
        self.ego_collision = False
        ego_scan_topic = self.get_parameter('ego_scan_topic').value
        ego_drive_topic = self.get_parameter('ego_drive_topic').value
        scan_fov = self.get_parameter('scan_fov').value
        scan_beams = self.get_parameter('scan_beams').value
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / scan_beams
        self.ego_namespace = self.get_parameter('ego_namespace').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        ego_odom_topic = self.ego_namespace + '/' + self.get_parameter('ego_odom_topic').value
        self.scan_distance_to_base_link = self.get_parameter('scan_distance_to_base_link').value
        self.obstacles = []
        self.obstacle_sequence = 0
        self.obstacle_round = 0
        self.obstacle_seed = int(
            self.get_parameter('random_obstacle_seed').value)
        self.random_obstacles_active = False
        self.obstacle_path = None
        self.obstacle_path_yaws = None
        self.occupancy_map = None
        self.vehicle_length = float(self.get_parameter('vehicle_length').value)
        self.vehicle_width = float(self.get_parameter('vehicle_width').value)
        
        if num_agents == 2:
            self.has_opp = True
            self.opp_namespace = self.get_parameter('opp_namespace').value
            sx1 = self.get_parameter('sx1').value
            sy1 = self.get_parameter('sy1').value
            stheta1 = self.get_parameter('stheta1').value
            self.opp_pose = [sx1, sy1, stheta1]
            self.opp_speed = [0.0, 0.0, 0.0]
            self.opp_requested_speed = 0.0
            self.opp_steer = 0.0
            self.opp_collision = False
            self.obs, _ , self.done, _ = self.env.reset(np.array([[sx, sy, stheta], [sx1, sy1, stheta1]]))
            self.ego_scan = list(self.obs['scans'][0])
            self.opp_scan = list(self.obs['scans'][1])

            opp_scan_topic = self.get_parameter('opp_scan_topic').value
            opp_odom_topic = self.opp_namespace + '/' + self.get_parameter('opp_odom_topic').value
            opp_drive_topic = self.get_parameter('opp_drive_topic').value

            ego_opp_odom_topic = self.ego_namespace + '/' + self.get_parameter('ego_opp_odom_topic').value
            opp_ego_odom_topic = self.opp_namespace + '/' + self.get_parameter('opp_ego_odom_topic').value
        else:
            self.has_opp = False
            self.obs, _ , self.done, _ = self.env.reset(np.array([[sx, sy, stheta]]))
            self.ego_scan = list(self.obs['scans'][0])

        self.last_ego_lap_count = int(self.obs['lap_counts'][0])

        # sim physical step timer
        self.drive_timer = self.create_timer(0.01, self.drive_timer_callback)
        # Publish simulated sensors at a realistic rate. The physics timer stays
        # at 100 Hz, while 1080-beam LaserScan serialization is kept bounded.
        topic_publish_rate = float(
            self.get_parameter('topic_publish_rate').value)
        self.timer = self.create_timer(
            1.0 / max(topic_publish_rate, 1.0), self.timer_callback)

        # transform broadcaster
        self.br = TransformBroadcaster(self)

        # publishers
        self.ego_scan_pub = self.create_publisher(LaserScan, ego_scan_topic, 10)
        self.ego_odom_pub = self.create_publisher(Odometry, ego_odom_topic, 10)
        self.ego_ground_truth_odom_pub = self.create_publisher(
            Odometry,
            self.get_parameter('ground_truth_odom_topic').value,
            10)
        self.ego_collision_pub = self.create_publisher(
            Bool,
            self.get_parameter('collision_topic').value,
            10)
        marker_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.obstacle_marker_pub = self.create_publisher(
            MarkerArray,
            self.get_parameter('obstacle_marker_topic').value,
            marker_qos)
        self.ego_drive_published = False
        if num_agents == 2:
            self.opp_scan_pub = self.create_publisher(LaserScan, opp_scan_topic, 10)
            self.ego_opp_odom_pub = self.create_publisher(Odometry, ego_opp_odom_topic, 10)
            self.opp_odom_pub = self.create_publisher(Odometry, opp_odom_topic, 10)
            self.opp_ego_odom_pub = self.create_publisher(Odometry, opp_ego_odom_topic, 10)
            self.opp_drive_published = False

        # subscribers
        self.ego_drive_sub = self.create_subscription(
            AckermannDriveStamped,
            ego_drive_topic,
            self.drive_callback,
            10)
        self.ego_reset_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter('sim_reset_topic').value,
            self.ego_reset_callback,
            10)
        self.clicked_point_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10)
        self.randomize_obstacles_service = self.create_service(
            Trigger,
            '/simulation/randomize_obstacles',
            self.randomize_obstacles_callback)
        self.clear_obstacles_service = self.create_service(
            Trigger,
            '/simulation/clear_obstacles',
            self.clear_obstacles_callback)
        if num_agents == 2:
            self.opp_drive_sub = self.create_subscription(
                AckermannDriveStamped,
                opp_drive_topic,
                self.opp_drive_callback,
                10)
            self.opp_reset_sub = self.create_subscription(
                PoseStamped,
                '/goal_pose',
                self.opp_reset_callback,
                10)

        if self.get_parameter('kb_teleop').value:
            self.teleop_sub = self.create_subscription(
                Twist,
                '/cmd_vel',
                self.teleop_callback,
                10)

        if self.get_parameter('random_obstacles_enabled').value:
            self.randomize_obstacles()
        else:
            self.publish_obstacle_markers()


    def drive_callback(self, drive_msg):
        self.ego_requested_speed = drive_msg.drive.speed
        self.ego_steer = drive_msg.drive.steering_angle
        self.ego_drive_published = True

    def opp_drive_callback(self, drive_msg):
        self.opp_requested_speed = drive_msg.drive.speed
        self.opp_steer = drive_msg.drive.steering_angle
        self.opp_drive_published = True

    def ego_reset_callback(self, pose_msg):
        rx = pose_msg.pose.pose.position.x
        ry = pose_msg.pose.pose.position.y
        rqx = pose_msg.pose.pose.orientation.x
        rqy = pose_msg.pose.pose.orientation.y
        rqz = pose_msg.pose.pose.orientation.z
        rqw = pose_msg.pose.pose.orientation.w
        _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
        if self.has_opp:
            opp_pose = [self.obs['poses_x'][1], self.obs['poses_y'][1], self.obs['poses_theta'][1]]
            self.obs, _ , self.done, _ = self.env.reset(np.array([[rx, ry, rtheta], opp_pose]))
        else:
            self.obs, _ , self.done, _ = self.env.reset(np.array([[rx, ry, rtheta]]))
        self._update_sim_state()
        if self.get_parameter('randomize_obstacles_on_reset').value:
            self.randomize_obstacles()

    def opp_reset_callback(self, pose_msg):
        if self.has_opp:
            rx = pose_msg.pose.position.x
            ry = pose_msg.pose.position.y
            rqx = pose_msg.pose.orientation.x
            rqy = pose_msg.pose.orientation.y
            rqz = pose_msg.pose.orientation.z
            rqw = pose_msg.pose.orientation.w
            _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
            self.obs, _ , self.done, _ = self.env.reset(np.array([list(self.ego_pose), [rx, ry, rtheta]]))
            self._update_sim_state()

    def teleop_callback(self, twist_msg):
        if not self.ego_drive_published:
            self.ego_drive_published = True

        self.ego_requested_speed = twist_msg.linear.x

        if twist_msg.angular.z > 0.0:
            self.ego_steer = 0.3
        elif twist_msg.angular.z < 0.0:
            self.ego_steer = -0.3
        else:
            self.ego_steer = 0.0

    def drive_timer_callback(self):
        if self.ego_drive_published and not self.has_opp:
            self.obs, _, self.done, _ = self.env.step(np.array([[self.ego_steer, self.ego_requested_speed]]))
        elif self.ego_drive_published and self.has_opp and self.opp_drive_published:
            self.obs, _, self.done, _ = self.env.step(np.array([[self.ego_steer, self.ego_requested_speed], [self.opp_steer, self.opp_requested_speed]]))
        self._update_sim_state()

    def timer_callback(self):
        ts = self.get_clock().now().to_msg()

        # Publish transforms before sensor messages so TF message filters can
        # resolve each scan at its exact timestamp.
        self._publish_odom(ts)
        self._publish_transforms(ts)
        self._publish_laser_transforms(ts)
        self._publish_wheel_transforms(ts)

        # pub scans
        scan = LaserScan()
        scan.header.stamp = ts
        scan.header.frame_id = self.ego_namespace + '/laser'
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = 0.
        scan.range_max = 30.
        scan.ranges = self.ego_scan
        self.ego_scan_pub.publish(scan)
        collision = Bool()
        collision.data = self.ego_collision
        self.ego_collision_pub.publish(collision)

        if self.has_opp:
            opp_scan = LaserScan()
            opp_scan.header.stamp = ts
            opp_scan.header.frame_id = self.opp_namespace + '/laser'
            opp_scan.angle_min = self.angle_min
            opp_scan.angle_max = self.angle_max
            opp_scan.angle_increment = self.angle_inc
            opp_scan.range_min = 0.
            opp_scan.range_max = 30.
            opp_scan.ranges = self.opp_scan
            self.opp_scan_pub.publish(opp_scan)

    def _update_sim_state(self):
        if self.has_opp:
            self.opp_scan = list(self.obs['scans'][1])
            self.opp_pose[0] = self.obs['poses_x'][1]
            self.opp_pose[1] = self.obs['poses_y'][1]
            self.opp_pose[2] = self.obs['poses_theta'][1]
            self.opp_speed[0] = self.obs['linear_vels_x'][1]
            self.opp_speed[1] = self.obs['linear_vels_y'][1]
            self.opp_speed[2] = self.obs['ang_vels_z'][1]

        self.ego_pose[0] = self.obs['poses_x'][0]
        self.ego_pose[1] = self.obs['poses_y'][0]
        self.ego_pose[2] = self.obs['poses_theta'][0]
        self.ego_speed[0] = self.obs['linear_vels_x'][0]
        self.ego_speed[1] = self.obs['linear_vels_y'][0]
        self.ego_speed[2] = self.obs['ang_vels_z'][0]
        self._randomize_obstacles_after_lap()
        self.ego_scan = inject_obstacles_into_scan(
            self.obs['scans'][0],
            self.ego_pose,
            self.angle_min,
            self.angle_inc,
            self.scan_distance_to_base_link,
            self.obstacles,
            30.0)
        obstacle_collision = any(
            vehicle_hits_obstacle(
                self.ego_pose,
                self.vehicle_length,
                self.vehicle_width,
                obstacle)
            for obstacle in self.obstacles)
        self.ego_collision = bool(
            self.obs['collisions'][0] or obstacle_collision)
        if obstacle_collision:
            self.ego_requested_speed = 0.0

    def _randomize_obstacles_after_lap(self):
        lap_counts = self.obs.get('lap_counts')
        if lap_counts is None or len(lap_counts) == 0:
            return

        current_lap, completed_laps = lap_count_transition(
            self.last_ego_lap_count, lap_counts[0])
        self.last_ego_lap_count = current_lap
        if completed_laps == 0:
            return
        if not self.random_obstacles_active:
            return
        if not self.get_parameter('randomize_obstacles_on_lap').value:
            return

        try:
            seed, summary = self.randomize_obstacles()
            self.get_logger().info(
                'Lap %d complete: obstacles moved with seed=%d: %s'
                % (current_lap, seed, summary))
        except Exception as error:  # pylint: disable=broad-except
            self.get_logger().error(
                'Lap %d obstacle randomization failed; keeping previous '
                'obstacles: %s' % (current_lap, error))

    def load_obstacle_placement_data(self):
        if self.obstacle_path is None:
            path = self.get_parameter('obstacle_path_csv').value
            self.obstacle_path, self.obstacle_path_yaws = load_path(path)
        if self.occupancy_map is None:
            yaml_path = self.get_parameter('map_path').value + '.yaml'
            self.occupancy_map = OccupancyMap(yaml_path)

    def obstacle_candidate_is_valid(
            self, obstacle, passage_x, passage_y, passage_radius):
        return (
            self.occupancy_map.rectangle_is_free(obstacle)
            and self.occupancy_map.circle_is_free(
                passage_x, passage_y, passage_radius))

    def randomize_obstacles(self):
        self.load_obstacle_placement_data()
        seed = resolve_obstacle_seed(
            self.obstacle_seed, self.obstacle_round)
        self.obstacle_round += 1
        self.obstacles = generate_obstacles(
            self.obstacle_path,
            self.obstacle_path_yaws,
            count=int(self.get_parameter('random_obstacle_count').value),
            seed=seed,
            length=float(self.get_parameter('obstacle_length').value),
            width=float(self.get_parameter('obstacle_width').value),
            height=float(self.get_parameter('obstacle_height').value),
            lateral_offset=float(
                self.get_parameter('obstacle_lateral_offset').value),
            start_xy=(self.ego_pose[0], self.ego_pose[1]),
            start_clearance=float(
                self.get_parameter('obstacle_start_clearance').value),
            min_spacing=float(
                self.get_parameter('obstacle_min_spacing').value),
            passage_offset=float(
                self.get_parameter('obstacle_passage_offset').value),
            passage_radius=float(
                self.get_parameter('obstacle_passage_radius').value),
            validator=self.obstacle_candidate_is_valid)
        self.obstacle_sequence = len(self.obstacles)
        self.random_obstacles_active = True
        self.publish_obstacle_markers()
        summary = ', '.join(
            '#%d=(%.2f, %.2f)' % (
                obstacle.obstacle_id, obstacle.x, obstacle.y)
            for obstacle in self.obstacles)
        self.get_logger().info(
            'Random obstacles seed=%d: %s' % (seed, summary))
        return seed, summary

    def randomize_obstacles_callback(self, request, response):
        del request
        try:
            seed, summary = self.randomize_obstacles()
            response.success = True
            response.message = 'seed=%d %s' % (seed, summary)
        except Exception as error:  # pylint: disable=broad-except
            response.success = False
            response.message = str(error)
            self.get_logger().error('Obstacle randomization failed: %s' % error)
        return response

    def clear_obstacles_callback(self, request, response):
        del request
        self.obstacles = []
        self.obstacle_sequence = 0
        self.random_obstacles_active = False
        self.publish_obstacle_markers()
        response.success = True
        response.message = 'Static obstacles cleared'
        return response

    def clicked_point_callback(self, point_msg):
        if point_msg.header.frame_id not in ('', 'map'):
            self.get_logger().warn(
                'Obstacle point must be in map frame, got %s'
                % point_msg.header.frame_id)
            return
        obstacle = StaticObstacle(
            obstacle_id=self.obstacle_sequence,
            x=float(point_msg.point.x),
            y=float(point_msg.point.y),
            yaw=0.0,
            length=float(self.get_parameter('obstacle_length').value),
            width=float(self.get_parameter('obstacle_width').value),
            height=float(self.get_parameter('obstacle_height').value))
        self.obstacle_sequence += 1
        self.obstacles.append(obstacle)
        self.publish_obstacle_markers()
        self.get_logger().info(
            'Added obstacle #%d at (%.2f, %.2f)'
            % (obstacle.obstacle_id, obstacle.x, obstacle.y))

    def publish_obstacle_markers(self):
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        if not self.obstacles:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.action = Marker.DELETEALL
            markers.markers.append(marker)
        for obstacle in self.obstacles:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = 'simulation_static_obstacles_ground_truth'
            marker.id = obstacle.obstacle_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = obstacle.x
            marker.pose.position.y = obstacle.y
            marker.pose.position.z = obstacle.height * 0.5
            quaternion = euler.euler2quat(
                0.0, 0.0, obstacle.yaw, axes='sxyz')
            marker.pose.orientation.x = quaternion[1]
            marker.pose.orientation.y = quaternion[2]
            marker.pose.orientation.z = quaternion[3]
            marker.pose.orientation.w = quaternion[0]
            marker.scale.x = obstacle.length
            marker.scale.y = obstacle.width
            marker.scale.z = obstacle.height
            marker.color.r = 0.95
            marker.color.g = 0.15
            marker.color.b = 0.10
            marker.color.a = 0.90
            markers.markers.append(marker)
        self.obstacle_marker_pub.publish(markers)

    def _publish_odom(self, ts):
        ego_odom = Odometry()
        ego_odom.header.stamp = ts
        ego_odom.header.frame_id = self.odom_frame_id
        ego_odom.child_frame_id = self.ego_namespace + '/base_link'
        ego_odom.pose.pose.position.x = self.ego_pose[0]
        ego_odom.pose.pose.position.y = self.ego_pose[1]
        ego_quat = euler.euler2quat(0., 0., self.ego_pose[2], axes='sxyz')
        ego_odom.pose.pose.orientation.x = ego_quat[1]
        ego_odom.pose.pose.orientation.y = ego_quat[2]
        ego_odom.pose.pose.orientation.z = ego_quat[3]
        ego_odom.pose.pose.orientation.w = ego_quat[0]
        ego_odom.twist.twist.linear.x = self.ego_speed[0]
        ego_odom.twist.twist.linear.y = self.ego_speed[1]
        ego_odom.twist.twist.angular.z = self.ego_speed[2]
        self.ego_odom_pub.publish(ego_odom)

        ground_truth_odom = Odometry()
        ground_truth_odom.header.stamp = ts
        ground_truth_odom.header.frame_id = 'map'
        ground_truth_odom.child_frame_id = self.ego_namespace + '/base_link'
        ground_truth_odom.pose = ego_odom.pose
        ground_truth_odom.twist = ego_odom.twist
        self.ego_ground_truth_odom_pub.publish(ground_truth_odom)

        if self.has_opp:
            opp_odom = Odometry()
            opp_odom.header.stamp = ts
            opp_odom.header.frame_id = self.odom_frame_id
            opp_odom.child_frame_id = self.opp_namespace + '/base_link'
            opp_odom.pose.pose.position.x = self.opp_pose[0]
            opp_odom.pose.pose.position.y = self.opp_pose[1]
            opp_quat = euler.euler2quat(0., 0., self.opp_pose[2], axes='sxyz')
            opp_odom.pose.pose.orientation.x = opp_quat[1]
            opp_odom.pose.pose.orientation.y = opp_quat[2]
            opp_odom.pose.pose.orientation.z = opp_quat[3]
            opp_odom.pose.pose.orientation.w = opp_quat[0]
            opp_odom.twist.twist.linear.x = self.opp_speed[0]
            opp_odom.twist.twist.linear.y = self.opp_speed[1]
            opp_odom.twist.twist.angular.z = self.opp_speed[2]
            self.opp_odom_pub.publish(opp_odom)
            self.opp_ego_odom_pub.publish(ego_odom)
            self.ego_opp_odom_pub.publish(opp_odom)

    def _publish_transforms(self, ts):
        ego_t = Transform()
        ego_t.translation.x = self.ego_pose[0]
        ego_t.translation.y = self.ego_pose[1]
        ego_t.translation.z = 0.0
        ego_quat = euler.euler2quat(0.0, 0.0, self.ego_pose[2], axes='sxyz')
        ego_t.rotation.x = ego_quat[1]
        ego_t.rotation.y = ego_quat[2]
        ego_t.rotation.z = ego_quat[3]
        ego_t.rotation.w = ego_quat[0]

        ego_ts = TransformStamped()
        ego_ts.transform = ego_t
        ego_ts.header.stamp = ts
        ego_ts.header.frame_id = self.odom_frame_id
        ego_ts.child_frame_id = self.ego_namespace + '/base_link'
        self.br.sendTransform(ego_ts)

        if self.has_opp:
            opp_t = Transform()
            opp_t.translation.x = self.opp_pose[0]
            opp_t.translation.y = self.opp_pose[1]
            opp_t.translation.z = 0.0
            opp_quat = euler.euler2quat(0.0, 0.0, self.opp_pose[2], axes='sxyz')
            opp_t.rotation.x = opp_quat[1]
            opp_t.rotation.y = opp_quat[2]
            opp_t.rotation.z = opp_quat[3]
            opp_t.rotation.w = opp_quat[0]

            opp_ts = TransformStamped()
            opp_ts.transform = opp_t
            opp_ts.header.stamp = ts
            opp_ts.header.frame_id = self.odom_frame_id
            opp_ts.child_frame_id = self.opp_namespace + '/base_link'
            self.br.sendTransform(opp_ts)

    def _publish_wheel_transforms(self, ts):
        ego_wheel_ts = TransformStamped()
        ego_wheel_quat = euler.euler2quat(0., 0., self.ego_steer, axes='sxyz')
        ego_wheel_ts.transform.rotation.x = ego_wheel_quat[1]
        ego_wheel_ts.transform.rotation.y = ego_wheel_quat[2]
        ego_wheel_ts.transform.rotation.z = ego_wheel_quat[3]
        ego_wheel_ts.transform.rotation.w = ego_wheel_quat[0]
        ego_wheel_ts.header.stamp = ts
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_left_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_left_wheel'
        self.br.sendTransform(ego_wheel_ts)
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_right_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_right_wheel'
        self.br.sendTransform(ego_wheel_ts)

        if self.has_opp:
            opp_wheel_ts = TransformStamped()
            opp_wheel_quat = euler.euler2quat(0., 0., self.opp_steer, axes='sxyz')
            opp_wheel_ts.transform.rotation.x = opp_wheel_quat[1]
            opp_wheel_ts.transform.rotation.y = opp_wheel_quat[2]
            opp_wheel_ts.transform.rotation.z = opp_wheel_quat[3]
            opp_wheel_ts.transform.rotation.w = opp_wheel_quat[0]
            opp_wheel_ts.header.stamp = ts
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_left_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_left_wheel'
            self.br.sendTransform(opp_wheel_ts)
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_right_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_right_wheel'
            self.br.sendTransform(opp_wheel_ts)

    def _publish_laser_transforms(self, ts):
        ego_scan_ts = TransformStamped()
        ego_scan_ts.transform.translation.x = self.scan_distance_to_base_link
        # ego_scan_ts.transform.translation.z = 0.04+0.1+0.025
        ego_scan_ts.transform.rotation.w = 1.
        ego_scan_ts.header.stamp = ts
        ego_scan_ts.header.frame_id = self.ego_namespace + '/base_link'
        ego_scan_ts.child_frame_id = self.ego_namespace + '/laser'
        self.br.sendTransform(ego_scan_ts)

        if self.has_opp:
            opp_scan_ts = TransformStamped()
            opp_scan_ts.transform.translation.x = self.scan_distance_to_base_link
            opp_scan_ts.transform.rotation.w = 1.
            opp_scan_ts.header.stamp = ts
            opp_scan_ts.header.frame_id = self.opp_namespace + '/base_link'
            opp_scan_ts.child_frame_id = self.opp_namespace + '/laser'
            self.br.sendTransform(opp_scan_ts)

def main(args=None):
    rclpy.init(args=args)
    gym_bridge = GymBridge()
    rclpy.spin(gym_bridge)

if __name__ == '__main__':
    main()
