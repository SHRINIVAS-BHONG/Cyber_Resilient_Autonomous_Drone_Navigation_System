"""
ROS 2 Cyber Attack Injector Node for Drone Avionics Simulation.

Subscribes to clean normalized sensor topics, executes physically grounded
cyberattack perturbations via AttackSimulationEngine, and publishes attacked
sensor topics along with isolated ground-truth attack metadata.
"""

from pathlib import Path
from typing import Optional
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Imu, Range, MagneticField
    from geometry_msgs.msg import PointStamped, PoseStamped
    from drone_interfaces.msg import AttackStatus
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object

from attack_simulator.attack_engine import AttackSimulationEngine


class AttackInjectorNode(Node):
    def __init__(self):
        super().__init__("attack_injector_node")

        # Declare parameters
        self.declare_parameter("scenario_path", "")
        self.declare_parameter("attack_type", "gps_spoofing")
        self.declare_parameter("start_time_sec", 10.0)
        self.declare_parameter("duration_sec", 20.0)

        # Initialize simulation engine
        workspace_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        scenario_param = self.get_parameter("scenario_path").value

        if scenario_param and Path(scenario_param).exists():
            cfg_target = Path(scenario_param)
        else:
            default_scenario = workspace_dir / "simulation" / "scenarios" / "scenario_03.yaml"
            cfg_target = default_scenario if default_scenario.exists() else None

        if cfg_target:
            self.engine = AttackSimulationEngine(cfg_target)
            self.get_logger().info(f"Loaded Attack Scenario: {self.engine.config.get('name')}")
        else:
            self.engine = AttackSimulationEngine()
            self.get_logger().info("Running default baseline AttackSimulationEngine.")

        # Best effort QoS for sensor streaming
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 1. Subscribers to clean normalized streams
        self.gps_sub = self.create_subscription(
            PointStamped, "/normalized/gps", self.gps_callback, best_effort_qos
        )
        self.imu_sub = self.create_subscription(
            Imu, "/normalized/imu", self.imu_callback, best_effort_qos
        )
        self.lidar_sub = self.create_subscription(
            Range, "/normalized/lidar", self.lidar_callback, best_effort_qos
        )
        self.baro_sub = self.create_subscription(
            Range, "/normalized/barometer", self.baro_callback, best_effort_qos
        )
        self.mag_sub = self.create_subscription(
            MagneticField, "/normalized/magnetometer", self.mag_callback, best_effort_qos
        )
        self.vision_sub = self.create_subscription(
            PoseStamped, "/normalized/vision_pose", self.vision_callback, best_effort_qos
        )

        # 2. Publishers for corrupted / attacked streams
        self.attacked_gps_pub = self.create_publisher(PointStamped, "/sensor/gps/attacked", 10)
        self.attacked_imu_pub = self.create_publisher(Imu, "/sensor/imu/attacked", 10)
        self.attacked_lidar_pub = self.create_publisher(Range, "/sensor/lidar/attacked", 10)
        self.attacked_baro_pub = self.create_publisher(Range, "/sensor/barometer/attacked", 10)
        self.attacked_mag_pub = self.create_publisher(MagneticField, "/sensor/magnetometer/attacked", 10)
        self.attacked_vision_pub = self.create_publisher(PoseStamped, "/sensor/vision_pose/attacked", 10)

        # 3. Ground Truth Publisher (Strictly isolated from estimator/detector)
        self.ground_truth_pub = self.create_publisher(AttackStatus, "/attack/ground_truth", 10)

        self.start_clock = self.get_clock().now()
        self.get_logger().info("Attack Injector Node fully initialized with all 6 sensor pipelines.")

    def get_elapsed_sec(self) -> float:
        """Returns elapsed simulation flight time in seconds."""
        return (self.get_clock().now() - self.start_clock).nanoseconds * 1e-9

    def gps_callback(self, msg: PointStamped):
        t = self.get_elapsed_sec()

        # Check packet loss
        if self.engine.evaluate_packet_loss(t):
            self.publish_ground_truth(msg.header.stamp, t)
            return

        raw_pos = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)
        corrupted_pos = self.engine.inject_gps(raw_pos, t)

        if corrupted_pos is not None:
            atk_msg = PointStamped()
            atk_msg.header = msg.header
            atk_msg.point.x = float(corrupted_pos[0])
            atk_msg.point.y = float(corrupted_pos[1])
            atk_msg.point.z = float(corrupted_pos[2])
            self.attacked_gps_pub.publish(atk_msg)

        self.publish_ground_truth(msg.header.stamp, t)

    def imu_callback(self, msg: Imu):
        t = self.get_elapsed_sec()

        if self.engine.evaluate_packet_loss(t):
            return

        raw_accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=np.float64)
        raw_gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=np.float64)

        corrupted_accel, corrupted_gyro = self.engine.inject_imu(raw_accel, raw_gyro, t)

        atk_imu = Imu()
        atk_imu.header = msg.header
        atk_imu.orientation = msg.orientation
        atk_imu.linear_acceleration.x = float(corrupted_accel[0])
        atk_imu.linear_acceleration.y = float(corrupted_accel[1])
        atk_imu.linear_acceleration.z = float(corrupted_accel[2])
        atk_imu.angular_velocity.x = float(corrupted_gyro[0])
        atk_imu.angular_velocity.y = float(corrupted_gyro[1])
        atk_imu.angular_velocity.z = float(corrupted_gyro[2])
        atk_imu.linear_acceleration_covariance = msg.linear_acceleration_covariance
        atk_imu.angular_velocity_covariance = msg.angular_velocity_covariance

        self.attacked_imu_pub.publish(atk_imu)

    def lidar_callback(self, msg: Range):
        t = self.get_elapsed_sec()
        if self.engine.evaluate_packet_loss(t):
            return

        corrupted_range = self.engine.inject_lidar(msg.range, t)

        atk_lidar = Range()
        atk_lidar.header = msg.header
        atk_lidar.min_range = msg.min_range
        atk_lidar.max_range = msg.max_range
        atk_lidar.range = float(corrupted_range)
        self.attacked_lidar_pub.publish(atk_lidar)

    def baro_callback(self, msg: Range):
        t = self.get_elapsed_sec()
        if self.engine.evaluate_packet_loss(t):
            return

        corrupted_alt = self.engine.inject_barometer(msg.range, t)

        atk_baro = Range()
        atk_baro.header = msg.header
        atk_baro.min_range = msg.min_range
        atk_baro.max_range = msg.max_range
        atk_baro.range = float(corrupted_alt)
        self.attacked_baro_pub.publish(atk_baro)

    def mag_callback(self, msg: MagneticField):
        t = self.get_elapsed_sec()
        if self.engine.evaluate_packet_loss(t):
            return

        raw_mag = np.array([
            msg.magnetic_field.x,
            msg.magnetic_field.y,
            msg.magnetic_field.z
        ], dtype=np.float64)

        corrupted_mag = self.engine.inject_magnetometer(raw_mag, t)

        atk_mag = MagneticField()
        atk_mag.header = msg.header
        atk_mag.magnetic_field.x = float(corrupted_mag[0])
        atk_mag.magnetic_field.y = float(corrupted_mag[1])
        atk_mag.magnetic_field.z = float(corrupted_mag[2])
        atk_mag.magnetic_field_covariance = msg.magnetic_field_covariance
        self.attacked_mag_pub.publish(atk_mag)

    def vision_callback(self, msg: PoseStamped):
        t = self.get_elapsed_sec()
        if self.engine.evaluate_packet_loss(t):
            return

        # Optical / camera pose remains unattacked trusted source unless packet loss targets telemetry
        self.attacked_vision_pub.publish(msg)

    def publish_ground_truth(self, stamp, t: float):
        """Publishes isolated ground truth for benchmark performance evaluation."""
        gt_data = self.engine.get_ground_truth(t)

        gt_msg = AttackStatus()
        gt_msg.start_time = stamp
        gt_msg.attack_type = gt_data["attack_type"]
        gt_msg.detected = bool(gt_data["is_active"])
        gt_msg.confidence = float(gt_data["confidence"])
        gt_msg.risk_score = float(gt_data["risk_score"])
        gt_msg.compromised_sensors = list(gt_data["compromised_sensors"])

        self.ground_truth_pub.publish(gt_msg)


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
