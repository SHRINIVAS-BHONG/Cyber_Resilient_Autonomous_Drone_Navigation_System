"""
ROS 2 Cyber Attack Injector Node.

Simulates GPS spoofing, IMU bias injection, LiDAR range corruption,
and communication disruptions strictly within the simulation environment.
"""

import random
from pathlib import Path
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, Range
from geometry_msgs.msg import PointStamped
from drone_interfaces.msg import AttackStatus


class AttackInjectorNode(Node):
    def __init__(self):
        super().__init__("attack_injector_node")

        # Load default attack scenario: GPS spoofing
        workspace_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        config_path = workspace_dir / "config" / "attacks" / "gps_spoofing.yaml"

        self.attack_cfg = {
            "enabled": True,
            "type": "gps_spoofing",
            "start_time_sec": 20.0,
            "duration_sec": 30.0,
            "mode": "drift",
            "ramp_duration": 5.0,
            "offset_x": 18.0,
            "offset_y": -12.0
        }

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if loaded and "attack" in loaded:
                        atk = loaded["attack"]
                        params = atk.get("parameters", {})
                        self.attack_cfg.update({
                            "enabled": atk.get("enabled", True),
                            "type": atk.get("type", "gps_spoofing"),
                            "start_time_sec": atk.get("start_time_sec", 20.0),
                            "duration_sec": atk.get("duration_sec", 30.0),
                            "mode": params.get("mode", "drift"),
                            "ramp_duration": params.get("ramp_duration_sec", 5.0),
                            "offset_x": params.get("position_offset", {}).get("x", 18.0),
                            "offset_y": params.get("position_offset", {}).get("y", -12.0),
                        })
            except Exception as e:
                self.get_logger().error(f"Failed to parse attack yaml: {e}")

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 1. Subscribers to clean normalized topics
        self.clean_gps_sub = self.create_subscription(
            PointStamped, "/normalized/gps", self.gps_in_callback, best_effort_qos
        )
        self.clean_imu_sub = self.create_subscription(
            Imu, "/normalized/imu", self.imu_in_callback, best_effort_qos
        )

        # 2. Publishers for manipulated topics consumed by the drone pipeline
        self.corrupted_gps_pub = self.create_publisher(PointStamped, "/sensor/gps/attacked", 10)
        self.corrupted_imu_pub = self.create_publisher(Imu, "/sensor/imu/attacked", 10)
        self.ground_truth_pub = self.create_publisher(AttackStatus, "/attack/ground_truth", 10)

        self.start_clock = self.get_clock().now()
        self.get_logger().info(f"Attack Injector running scenario: {self.attack_cfg['type']}")

    def gps_in_callback(self, msg: PointStamped):
        elapsed_sec = (self.get_clock().now() - self.start_clock).nanoseconds * 1e-9
        is_attack_active = (
            self.attack_cfg["enabled"] and
            (self.attack_cfg["start_time_sec"] <= elapsed_sec < (self.attack_cfg["start_time_sec"] + self.attack_cfg["duration_sec"]))
        )

        attacked_msg = PointStamped()
        attacked_msg.header = msg.header
        attacked_msg.point = msg.point

        if is_attack_active and self.attack_cfg["type"] == "gps_spoofing":
            if self.attack_cfg["mode"] == "drift":
                ramp_progress = min(1.0, (elapsed_sec - self.attack_cfg["start_time_sec"]) / self.attack_cfg["ramp_duration"])
                attacked_msg.point.x += self.attack_cfg["offset_x"] * ramp_progress
                attacked_msg.point.y += self.attack_cfg["offset_y"] * ramp_progress
            else:
                attacked_msg.point.x += self.attack_cfg["offset_x"]
                attacked_msg.point.y += self.attack_cfg["offset_y"]

        self.corrupted_gps_pub.publish(attacked_msg)

        # Publish ground truth metadata
        gt_msg = AttackStatus()
        gt_msg.start_time = msg.header.stamp
        gt_msg.attack_type = self.attack_cfg["type"] if is_attack_active else "None"
        gt_msg.detected = is_attack_active
        gt_msg.confidence = 1.0 if is_attack_active else 0.0
        gt_msg.risk_score = 1.0 if is_attack_active else 0.0
        gt_msg.compromised_sensors = ["gps"] if is_attack_active else []
        self.ground_truth_pub.publish(gt_msg)

    def imu_in_callback(self, msg: Imu):
        elapsed_sec = (self.get_clock().now() - self.start_clock).nanoseconds * 1e-9
        is_attack_active = (
            self.attack_cfg["enabled"] and
            (self.attack_cfg["start_time_sec"] <= elapsed_sec < (self.attack_cfg["start_time_sec"] + self.attack_cfg["duration_sec"]))
        )

        attacked_imu = msg
        if is_attack_active and self.attack_cfg["type"] == "imu_manipulation":
            attacked_imu.linear_acceleration.x += 1.8
            attacked_imu.linear_acceleration.y -= 1.2
            attacked_imu.angular_velocity.z += 0.15

        self.corrupted_imu_pub.publish(attacked_imu)


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


if __name__ == "__main__":
    main()
