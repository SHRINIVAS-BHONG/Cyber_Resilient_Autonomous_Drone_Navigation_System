import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32
import numpy as np


class SensorSimNode(Node):
    def __init__(self):
        super().__init__('sensor_sim_node')

        # Publishers
        self.gps_pub = self.create_publisher(PointStamped, '/sensor/raw/gps', 10)
        self.imu_pub = self.create_publisher(Imu, '/sensor/raw/imu', 50)
        self.baro_pub = self.create_publisher(Float32, '/sensor/raw/baro', 20)
        self.vis_pub = self.create_publisher(PoseStamped, '/sensor/raw/vision', 20)
        self.lidar_pub = self.create_publisher(LaserScan, '/sensor/raw/lidar', 10)

        # Ground truth subscriber
        self.create_subscription(PoseStamped, '/truth/pose', self.truth_cb, 50)

        self.latest_truth = np.array([0.0, 0.0, 10.0])
        self.prev_truth = np.array([0.0, 0.0, 10.0])
        self.latest_time = 0.0

        # Timers for sensor rates
        self.create_timer(1.0 / 10.0, self.publish_gps)     # 10 Hz
        self.create_timer(1.0 / 100.0, self.publish_imu)    # 100 Hz
        self.create_timer(1.0 / 20.0, self.publish_baro)    # 20 Hz
        self.create_timer(1.0 / 20.0, self.publish_vision)  # 20 Hz
        self.create_timer(1.0 / 10.0, self.publish_lidar)   # 10 Hz

        self.get_logger().info("Sensor Simulator Node running.")

    def truth_cb(self, msg: PoseStamped):
        self.latest_truth = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

    def publish_gps(self):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        noise = np.random.normal(0.0, 0.5, size=3)
        p = self.latest_truth + noise
        msg.point.x, msg.point.y, msg.point.z = float(p[0]), float(p[1]), float(p[2])
        self.gps_pub.publish(msg)

    def publish_imu(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.linear_acceleration.x = float(np.random.normal(0.0, 0.05))
        msg.linear_acceleration.y = float(np.random.normal(0.0, 0.05))
        msg.linear_acceleration.z = float(9.81 + np.random.normal(0.0, 0.05))
        self.imu_pub.publish(msg)

    def publish_baro(self):
        msg = Float32()
        msg.data = float(self.latest_truth[2] + np.random.normal(0.0, 0.3))
        self.baro_pub.publish(msg)

    def publish_vision(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        noise = np.random.normal(0.0, 0.1, size=3)
        p = self.latest_truth + noise
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = float(p[0]), float(p[1]), float(p[2])
        msg.pose.orientation.w = 1.0
        self.vis_pub.publish(msg)

    def publish_lidar(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.range_min = 0.2
        msg.range_max = 30.0
        msg.ranges = [float(np.random.uniform(5.0, 25.0)) for _ in range(36)]
        self.lidar_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
