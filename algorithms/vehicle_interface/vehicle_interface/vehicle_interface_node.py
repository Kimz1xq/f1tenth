import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64, Bool


class VehicleInterfaceNode(Node):
    def __init__(self):
        super().__init__('vehicle_interface_node')

        self.declare_parameter('drive_mode', 'sim')

        self.declare_parameter('input_drive_topic', '/control/drive_cmd')

        self.declare_parameter('sim_drive_topic', '/drive')

        self.declare_parameter('real_speed_topic', '/commands/motor/speed')
        self.declare_parameter('real_servo_topic', '/commands/servo/position')

        self.declare_parameter('safety_stop_topic', '/safety/stop_required')
        self.declare_parameter('use_safety_stop', True)

        self.declare_parameter('watchdog_timeout', 0.5)
        self.declare_parameter('publish_rate', 50.0)

        self.declare_parameter('speed_to_erpm_gain', 3000.0)
        self.declare_parameter('speed_to_erpm_offset', 0.0)

        self.declare_parameter('servo_center', 0.5)
        self.declare_parameter('servo_gain', 1.0)
        self.declare_parameter('servo_min', 0.0)
        self.declare_parameter('servo_max', 1.0)

        self.drive_mode = self.get_parameter('drive_mode').value

        self.input_drive_topic = self.get_parameter('input_drive_topic').value

        self.sim_drive_topic = self.get_parameter('sim_drive_topic').value

        self.real_speed_topic = self.get_parameter('real_speed_topic').value
        self.real_servo_topic = self.get_parameter('real_servo_topic').value

        self.safety_stop_topic = self.get_parameter('safety_stop_topic').value
        self.use_safety_stop = bool(self.get_parameter('use_safety_stop').value)

        self.watchdog_timeout = float(self.get_parameter('watchdog_timeout').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)

        self.speed_to_erpm_gain = float(self.get_parameter('speed_to_erpm_gain').value)
        self.speed_to_erpm_offset = float(self.get_parameter('speed_to_erpm_offset').value)

        self.servo_center = float(self.get_parameter('servo_center').value)
        self.servo_gain = float(self.get_parameter('servo_gain').value)
        self.servo_min = float(self.get_parameter('servo_min').value)
        self.servo_max = float(self.get_parameter('servo_max').value)

        if self.drive_mode not in ['sim', 'real']:
            raise RuntimeError("drive_mode must be 'sim' or 'real'")

        self.latest_cmd = None
        self.latest_cmd_time = None
        self.safety_stop_required = False

        self.drive_sub = self.create_subscription(
            AckermannDriveStamped,
            self.input_drive_topic,
            self.drive_callback,
            10
        )

        self.safety_sub = self.create_subscription(
            Bool,
            self.safety_stop_topic,
            self.safety_callback,
            10
        )

        self.sim_drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.sim_drive_topic,
            10
        )

        self.real_speed_pub = self.create_publisher(
            Float64,
            self.real_speed_topic,
            10
        )

        self.real_servo_pub = self.create_publisher(
            Float64,
            self.real_servo_topic,
            10
        )

        self.timer = self.create_timer(
            1.0 / max(self.publish_rate, 1.0),
            self.publish_loop
        )

        self.get_logger().info('vehicle_interface_node started')
        self.get_logger().info(f'drive_mode        : {self.drive_mode}')
        self.get_logger().info(f'input_drive_topic : {self.input_drive_topic}')
        self.get_logger().info(f'sim_drive_topic   : {self.sim_drive_topic}')
        self.get_logger().info(f'real_speed_topic  : {self.real_speed_topic}')
        self.get_logger().info(f'real_servo_topic  : {self.real_servo_topic}')
        self.get_logger().info(f'use_safety_stop   : {self.use_safety_stop}')
        self.get_logger().info(f'watchdog_timeout  : {self.watchdog_timeout}')

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))

    def drive_callback(self, msg):
        self.latest_cmd = msg
        self.latest_cmd_time = self.get_clock().now()

    def safety_callback(self, msg):
        self.safety_stop_required = bool(msg.data)

    def is_cmd_stale(self):
        if self.latest_cmd is None or self.latest_cmd_time is None:
            return True

        now = self.get_clock().now()
        dt = (now - self.latest_cmd_time).nanoseconds * 1e-9

        return dt > self.watchdog_timeout

    def publish_stop(self):
        if self.drive_mode == 'sim':
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.drive.speed = 0.0
            msg.drive.steering_angle = 0.0
            self.sim_drive_pub.publish(msg)

        else:
            speed_msg = Float64()
            servo_msg = Float64()

            speed_msg.data = 0.0
            servo_msg.data = self.servo_center

            self.real_speed_pub.publish(speed_msg)
            self.real_servo_pub.publish(servo_msg)

    def publish_sim(self, cmd):
        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = cmd.header.frame_id
        out.drive = cmd.drive
        self.sim_drive_pub.publish(out)

    def publish_real(self, cmd):
        speed = float(cmd.drive.speed)
        steering = float(cmd.drive.steering_angle)

        speed_msg = Float64()
        servo_msg = Float64()

        speed_msg.data = self.speed_to_erpm_gain * speed + self.speed_to_erpm_offset

        servo = self.servo_center + self.servo_gain * steering
        servo_msg.data = self.clamp(servo, self.servo_min, self.servo_max)

        self.real_speed_pub.publish(speed_msg)
        self.real_servo_pub.publish(servo_msg)

    def publish_loop(self):
        if self.is_cmd_stale():
            self.publish_stop()
            return

        if self.use_safety_stop and self.safety_stop_required:
            self.publish_stop()
            return

        if self.drive_mode == 'sim':
            self.publish_sim(self.latest_cmd)
        else:
            self.publish_real(self.latest_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = VehicleInterfaceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
