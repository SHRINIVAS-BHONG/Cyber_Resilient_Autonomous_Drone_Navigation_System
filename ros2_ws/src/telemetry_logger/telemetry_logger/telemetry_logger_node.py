"""
Telemetry Logger Node for Cyber-Resilient Drone Navigation.

Logs real-time flight telemetry, innovation residuals, sensor trust scores,
and cyber-attack detections to CSV and JSON for benchmarking.
"""

import csv
from datetime import datetime
from pathlib import Path
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PointStamped
from drone_interfaces.msg import ResidualTelemetry, SensorTrust, AttackStatus


class TelemetryLoggerNode(Node):
    def __init__(self):
        super().__init__("telemetry_logger_node")

        workspace_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        self.output_dir = workspace_dir / "reports" / "experiment_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"flight_telemetry_{timestamp_str}.csv"

        # CSV initialization
        self.csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp_sec",
            "est_x", "est_y", "est_z",
            "gps_raw_x", "gps_raw_y", "gps_raw_z",
            "last_sensor_nis", "anomaly_flag",
            "active_attack_type", "attack_detected", "attack_risk_score",
            "gps_trust_score"
        ])

        # State storage
        self.current_est = [0.0, 0.0, 0.0]
        self.current_gps = [0.0, 0.0, 0.0]
        self.last_nis = 0.0
        self.last_anomaly = False
        self.attack_type = "None"
        self.attack_detected = False
        self.risk_score = 0.0
        self.gps_trust = 1.0

        # Subscriptions
        self.est_sub = self.create_subscription(
            PoseStamped, "/state_estimator/pose", self.est_cb, 10
        )
        self.gps_sub = self.create_subscription(
            PointStamped, "/normalized/gps", self.gps_cb, 10
        )
        self.residual_sub = self.create_subscription(
            ResidualTelemetry, "/state_estimator/residuals", self.residual_cb, 10
        )
        self.trust_sub = self.create_subscription(
            SensorTrust, "/resilience/sensor_trust", self.trust_cb, 10
        )
        self.attack_sub = self.create_subscription(
            AttackStatus, "/resilience/attack_status", self.attack_cb, 10
        )

        # 10 Hz disk logging timer
        self.timer = self.create_timer(0.1, self.log_to_disk)
        self.get_logger().info(f"Telemetry Logger initialized. Logging to: {self.csv_path}")

    def est_cb(self, msg: PoseStamped):
        self.current_est = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

    def gps_cb(self, msg: PointStamped):
        self.current_gps = [msg.point.x, msg.point.y, msg.point.z]

    def residual_cb(self, msg: ResidualTelemetry):
        self.last_nis = float(msg.nis)
        self.last_anomaly = bool(msg.anomaly_flag)

    def trust_cb(self, msg: SensorTrust):
        if msg.sensor_name == "gps":
            self.gps_trust = float(msg.trust_score)

    def attack_cb(self, msg: AttackStatus):
        self.attack_type = msg.attack_type
        self.attack_detected = bool(msg.detected)
        self.risk_score = float(msg.risk_score)

    def log_to_disk(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.csv_writer.writerow([
            f"{now_sec:.3f}",
            f"{self.current_est[0]:.3f}", f"{self.current_est[1]:.3f}", f"{self.current_est[2]:.3f}",
            f"{self.current_gps[0]:.3f}", f"{self.current_gps[1]:.3f}", f"{self.current_gps[2]:.3f}",
            f"{self.last_nis:.3f}", int(self.last_anomaly),
            self.attack_type, int(self.attack_detected), f"{self.risk_score:.3f}",
            f"{self.gps_trust:.3f}"
        ])
        self.csv_file.flush()

    def destroy_node(self):
        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.close()
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


if __name__ == "__main__":
    main()
