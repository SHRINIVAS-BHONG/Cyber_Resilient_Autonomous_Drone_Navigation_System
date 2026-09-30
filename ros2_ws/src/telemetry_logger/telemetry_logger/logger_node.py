import os
import csv
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from drone_interfaces.msg import AttackStatus, SensorTrust
import numpy as np


class TelemetryLoggerNode(Node):
    def __init__(self):
        super().__init__('telemetry_logger_node')

        self.declare_parameter('run_dir', 'runs/latest_ros2_run')
        self.run_dir = self.get_parameter('run_dir').value
        os.makedirs(self.run_dir, exist_ok=True)

        self.csv_path = os.path.join(self.run_dir, 'telemetry.csv')
        self.events_path = os.path.join(self.run_dir, 'events.json')
        self.trust_path = os.path.join(self.run_dir, 'trust.csv')

        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'time', 'truth_x', 'truth_y', 'truth_z',
            'main_est_x', 'main_est_y', 'main_est_z',
            'trusted_est_x', 'trusted_est_y', 'trusted_est_z',
            'gps_x', 'gps_y', 'gps_z', 'state'
        ])

        self.events = []
        self.latest_truth = np.zeros(3)
        self.latest_main = np.zeros(3)
        self.latest_trusted = np.zeros(3)
        self.latest_gps = np.zeros(3)
        self.current_state = 'NORMAL'

        self.start_time = self.get_clock().now()

        # Subscribers
        self.create_subscription(PoseStamped, '/truth/pose', self.truth_cb, 50)
        self.create_subscription(Odometry, '/estimate/main', self.main_cb, 50)
        self.create_subscription(Odometry, '/estimate/trusted', self.trusted_cb, 50)
        self.create_subscription(PointStamped, '/sensor/attacked/gps', self.gps_cb, 10)
        self.create_subscription(String, '/resilience/navigation_mode', self.mode_cb, 10)
        self.create_subscription(AttackStatus, '/resilience/attack_status', self.attack_cb, 10)

        # 20 Hz logging timer
        self.create_timer(0.05, self.log_tick)
        self.get_logger().info(f"Telemetry Logger active. Logging to: {self.run_dir}")

    def get_time_s(self) -> float:
        return (self.get_clock().now() - self.start_time).nanoseconds / 1e9

    def truth_cb(self, msg: PoseStamped):
        self.latest_truth = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def main_cb(self, msg: Odometry):
        self.latest_main = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def trusted_cb(self, msg: Odometry):
        self.latest_trusted = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def gps_cb(self, msg: PointStamped):
        self.latest_gps = np.array([msg.point.x, msg.point.y, msg.point.z])

    def mode_cb(self, msg: String):
        self.current_state = msg.data

    def attack_cb(self, msg: AttackStatus):
        if msg.detected:
            self.events.append({
                "time": self.get_time_s(),
                "event": "ATTACK_DETECTED",
                "type": msg.attack_type,
                "confidence": msg.confidence
            })

    def log_tick(self):
        t = round(self.get_time_s(), 3)
        self.csv_writer.writerow([
            t,
            round(self.latest_truth[0], 3), round(self.latest_truth[1], 3), round(self.latest_truth[2], 3),
            round(self.latest_main[0], 3), round(self.latest_main[1], 3), round(self.latest_main[2], 3),
            round(self.latest_trusted[0], 3), round(self.latest_trusted[1], 3), round(self.latest_trusted[2], 3),
            round(self.latest_gps[0], 3), round(self.latest_gps[1], 3), round(self.latest_gps[2], 3),
            self.current_state
        ])
        self.csv_file.flush()

    def destroy_node(self):
        self.csv_file.close()
        with open(self.events_path, 'w', encoding='utf-8') as f:
            json.dump(self.events, f, indent=2)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
