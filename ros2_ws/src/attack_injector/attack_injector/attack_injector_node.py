import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Imu, LaserScan
from drone_interfaces.msg import AttackStatus
import numpy as np


class AttackInjectorNode(Node):
    def __init__(self):
        super().__init__('attack_injector_node')

        # Attack configuration params
        self.declare_parameter('attack_type', 'none')
        self.declare_parameter('start_time', 30.0)
        self.declare_parameter('duration', 20.0)

        self.attack_type = self.get_parameter('attack_type').value
        self.start_time = float(self.get_parameter('start_time').value)
        self.duration = float(self.get_parameter('duration').value)

        # Publishers
        self.gps_pub = self.create_publisher(PointStamped, '/sensor/attacked/gps', 10)
        self.imu_pub = self.create_publisher(Imu, '/sensor/attacked/imu', 50)
        self.lidar_pub = self.create_publisher(LaserScan, '/sensor/attacked/lidar', 10)
        self.vis_pub = self.create_publisher(PoseStamped, '/sensor/attacked/vision', 20)
        self.attack_status_pub = self.create_publisher(AttackStatus, '/attack/ground_truth', 10)

        # Subscribers
        self.create_subscription(PointStamped, '/sensor/raw/gps', self.gps_cb, 10)
        self.create_subscription(Imu, '/sensor/raw/imu', self.imu_cb, 50)
        self.create_subscription(LaserScan, '/sensor/raw/lidar', self.lidar_cb, 10)
        self.create_subscription(PoseStamped, '/sensor/raw/vision', self.vis_cb, 20)

        self.sim_start_time = self.get_clock().now()
        self.create_timer(0.5, self.publish_ground_truth)
        self.get_logger().info(f"Attack Injector Node initialized. Type: {self.attack_type}")

    def get_elapsed_sec(self) -> float:
        now = self.get_clock().now()
        return (now - self.sim_start_time).nanoseconds / 1e9

    def is_attack_active(self) -> bool:
        t = self.get_elapsed_sec()
        return self.attack_type != 'none' and (self.start_time <= t < self.start_time + self.duration)

    def gps_cb(self, msg: PointStamped):
        t = self.get_elapsed_sec()
        if self.is_attack_active():
            if self.attack_type == 'gps_drift':
                elapsed = t - self.start_time
                ramp = min(1.0, elapsed / 5.0)
                msg.point.x += 15.0 * ramp
                msg.point.y -= 10.0 * ramp
            elif self.attack_type == 'gps_jump':
                msg.point.x += 22.0
                msg.point.y += 20.0
            elif self.attack_type == 'comm_disruption':
                # 30% drop
                if np.random.rand() < 0.3:
                    return
        self.gps_pub.publish(msg)

    def imu_cb(self, msg: Imu):
        if self.is_attack_active() and self.attack_type == 'imu_bias':
            msg.linear_acceleration.x += 0.8
            msg.linear_acceleration.y -= 0.5
        self.imu_pub.publish(msg)

    def lidar_cb(self, msg: LaserScan):
        if self.is_attack_active() and self.attack_type == 'lidar_corruption':
            msg.ranges = [r * 0.6 for r in msg.ranges]
        self.lidar_pub.publish(msg)

    def vis_cb(self, msg: PoseStamped):
        self.vis_pub.publish(msg)

    def publish_ground_truth(self):
        msg = AttackStatus()
        active = self.is_attack_active()
        msg.attack_type = self.attack_type if active else 'none'
        msg.detected = active
        msg.confidence = 1.0 if active else 0.0
        msg.risk_score = 1.0 if active else 0.0
        if active:
            if 'gps' in self.attack_type or 'comm' in self.attack_type:
                msg.compromised_sensors = ['gps']
            elif 'imu' in self.attack_type:
                msg.compromised_sensors = ['imu']
            elif 'lidar' in self.attack_type:
                msg.compromised_sensors = ['lidar']
        else:
            msg.compromised_sensors = []
        self.attack_status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AttackInjectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
