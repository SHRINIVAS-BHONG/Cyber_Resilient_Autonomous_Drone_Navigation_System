import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32, String
import numpy as np


class EstimatorNode(Node):
    def __init__(self):
        super().__init__('estimator_node')

        self.declare_parameter('is_trusted', False)
        self.is_trusted = bool(self.get_parameter('is_trusted').value)

        output_topic = '/estimate/trusted' if self.is_trusted else '/estimate/main'
        self.odom_pub = self.create_publisher(Odometry, output_topic, 50)

        # Subscribers
        if not self.is_trusted:
            self.create_subscription(PointStamped, '/sensor/attacked/gps', self.gps_cb, 10)
        self.create_subscription(Imu, '/sensor/attacked/imu', self.imu_cb, 50)
        self.create_subscription(Float32, '/sensor/raw/baro', self.baro_cb, 20)
        self.create_subscription(PoseStamped, '/sensor/attacked/vision', self.vision_cb, 20)
        self.create_subscription(String, '/resilience/sensor_gate', self.sensor_gate_cb, 10)

        # State: px, py, pz, vx, vy, vz
        self.pos = np.array([0.0, 0.0, 10.0])
        self.vel = np.zeros(3)
        self.gps_enabled = not self.is_trusted

        self.create_timer(1.0 / 50.0, self.publish_odometry) # 50 Hz
        self.get_logger().info(f"Estimator Node started ({'TRUSTED' if self.is_trusted else 'MAIN'}). Publishing on {output_topic}")

    def sensor_gate_cb(self, msg: String):
        # Format: "gps:disable" or "gps:enable"
        parts = msg.data.split(':')
        if len(parts) == 2 and parts[0] == 'gps':
            self.gps_enabled = (parts[1] == 'enable') and (not self.is_trusted)
            self.get_logger().info(f"Sensor GPS gate changed: enabled={self.gps_enabled}")

    def gps_cb(self, msg: PointStamped):
        if self.gps_enabled:
            # Simple alpha complementary filter update
            meas = np.array([msg.point.x, msg.point.y, msg.point.z])
            self.pos[:2] = 0.8 * self.pos[:2] + 0.2 * meas[:2]

    def imu_cb(self, msg: Imu):
        dt = 0.02
        acc_world = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z - 9.81])
        self.vel += acc_world * dt
        self.pos += self.vel * dt

    def baro_cb(self, msg: Float32):
        self.pos[2] = 0.85 * self.pos[2] + 0.15 * msg.data

    def vision_cb(self, msg: PoseStamped):
        meas = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # Smooth with vision
        self.pos = 0.9 * self.pos + 0.1 * meas

    def publish_odometry(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(self.pos[0])
        msg.pose.pose.position.y = float(self.pos[1])
        msg.pose.pose.position.z = float(self.pos[2])
        msg.twist.twist.linear.x = float(self.vel[0])
        msg.twist.twist.linear.y = float(self.vel[1])
        msg.twist.twist.linear.z = float(self.vel[2])
        self.odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
