import csv
import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


class WaypointPlannerNode(Node):
    def __init__(self):
        super().__init__('waypoint_planner_node')

        self.declare_parameter(
            'waypoint_csv',
            '')
        self.declare_parameter('path_topic', '/planning/path')
        self.declare_parameter('marker_topic', '/planning/markers')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate', 2.0)

        self.waypoint_csv = self.get_parameter('waypoint_csv').value
        if not self.waypoint_csv:
            raise RuntimeError(
                'waypoint_csv must be selected by the planning launch file')
        self.path_topic = self.get_parameter('path_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_rate = float(self.get_parameter('publish_rate').value)

        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.waypoints = self.load_waypoints(self.waypoint_csv)

        timer_period = 1.0 / max(self.publish_rate, 0.1)
        self.timer = self.create_timer(timer_period, self.publish)

        self.get_logger().info('waypoint_planner_node started')
        self.get_logger().info(f'waypoint_csv : {self.waypoint_csv}')
        self.get_logger().info(f'path_topic   : {self.path_topic}')
        self.get_logger().info(f'marker_topic : {self.marker_topic}')
        self.get_logger().info(f'num waypoints: {len(self.waypoints)}')

    def load_waypoints(self, csv_path):
        waypoints = []

        with open(csv_path, 'r') as f:
            rows = [row for row in csv.reader(f) if row and not row[0].strip().startswith('#')]

        if not rows:
            raise RuntimeError('Waypoint CSV is empty.')

        header = [cell.strip().lower() for cell in rows[0]]

        x_names = ['x', 'x_m']
        y_names = ['y', 'y_m']
        speed_names = ['speed', 'velocity', 'vx', 'vx_mps']

        x_idx = next((header.index(name) for name in x_names if name in header), None)
        y_idx = next((header.index(name) for name in y_names if name in header), None)

        if x_idx is not None and y_idx is not None:
            speed_idx = next((header.index(name) for name in speed_names if name in header), None)
            data_rows = rows[1:]
        else:
            x_idx = 0
            y_idx = 1
            speed_idx = 2 if len(rows[0]) > 2 else None
            data_rows = rows

        for row in data_rows:
            if len(row) <= max(x_idx, y_idx):
                continue

            x = float(row[x_idx])
            y = float(row[y_idx])
            speed = (
                float(row[speed_idx])
                if speed_idx is not None and len(row) > speed_idx
                else 1.0)
            waypoints.append((x, y, speed))

        if len(waypoints) < 2:
            raise RuntimeError('Waypoint CSV must contain at least 2 points.')

        return waypoints

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        return qz, qw

    def build_path_msg(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.frame_id

        for i, (x, y, speed) in enumerate(self.waypoints):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            if i < len(self.waypoints) - 1:
                nx, ny, _ = self.waypoints[i + 1]
            else:
                nx, ny, _ = self.waypoints[0]

            yaw = math.atan2(ny - y, nx - x)
            qz, qw = self.yaw_to_quaternion(yaw)

            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            path_msg.poses.append(pose)

        return path_msg

    def build_marker_msg(self):
        marker_array = MarkerArray()

        line = Marker()
        line.header.stamp = self.get_clock().now().to_msg()
        line.header.frame_id = self.frame_id
        line.ns = 'planning_path'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.05
        line.color.a = 1.0
        line.color.r = 0.0
        line.color.g = 1.0
        line.color.b = 0.0

        from geometry_msgs.msg import Point

        for x, y, speed in self.waypoints:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.05
            line.points.append(p)

        if len(self.waypoints) > 0:
            p = Point()
            p.x = self.waypoints[0][0]
            p.y = self.waypoints[0][1]
            p.z = 0.05
            line.points.append(p)

        marker_array.markers.append(line)
        return marker_array

    def publish(self):
        self.path_pub.publish(self.build_path_msg())
        self.marker_pub.publish(self.build_marker_msg())


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
