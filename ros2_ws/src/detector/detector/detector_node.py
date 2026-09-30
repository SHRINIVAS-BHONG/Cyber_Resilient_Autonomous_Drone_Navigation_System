import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from drone_interfaces.msg import AttackStatus
import numpy as np


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        self.status_pub = self.create_publisher(AttackStatus, '/resilience/attack_status', 10)

        self.create_subscription(PointStamped, '/sensor/attacked/gps', self.gps_cb, 10)
        self.create_subscription(Odometry, '/estimate/main', self.main_cb, 50)
        self.create_subscription(Odometry, '/estimate/trusted', self.trusted_cb, 50)

        self.latest_gps: np.ndarray = None
        self.latest_main: np.ndarray = None
        self.latest_trusted: np.ndarray = None

        self.suspicious_count = 0
        self.confirm_n = 5
        self.threshold_m = 4.5

        # Check loop at 10 Hz
        self.create_timer(0.1, self.check_loop)
        self.get_logger().info("Detector Node initialized (10 Hz).")

    def gps_cb(self, msg: PointStamped):
        self.latest_gps = np.array([msg.point.x, msg.point.y, msg.point.z])

    def main_cb(self, msg: Odometry):
        self.latest_main = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def trusted_cb(self, msg: Odometry):
        self.latest_trusted = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def check_loop(self):
        if self.latest_gps is None or self.latest_trusted is None:
            return

        diff = float(np.linalg.norm(self.latest_gps[:2] - self.latest_trusted[:2]))

        if diff > self.threshold_m:
            self.suspicious_count += 1
        else:
            self.suspicious_count = max(0, self.suspicious_count - 1)

        msg = AttackStatus()
        msg.start_time = self.get_clock().now().to_msg()
        msg.risk_score = min(1.0, diff / 10.0)

        if self.suspicious_count >= self.confirm_n:
            msg.attack_type = "gps_spoofing"
            msg.detected = True
            msg.confidence = 1.0
            msg.compromised_sensors = ["gps"]
        elif self.suspicious_count > 0:
            msg.attack_type = "suspicious_gps"
            msg.detected = False
            msg.confidence = 0.5
            msg.compromised_sensors = ["gps"]
        else:
            msg.attack_type = "none"
            msg.detected = False
            msg.confidence = 0.0
            msg.compromised_sensors = []

        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
