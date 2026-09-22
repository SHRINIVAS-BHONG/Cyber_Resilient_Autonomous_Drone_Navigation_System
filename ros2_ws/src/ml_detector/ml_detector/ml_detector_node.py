"""
Machine Learning Cyber-Attack Detection ROS 2 Node.

Loads the trained Random Forest classifier and StandardScaler artifacts to execute
real-time 10 Hz multi-sensor attack classification as an independent second-opinion
voter alongside the statistical Chi-square Innovation residual detector.
"""

from collections import deque
import json
from pathlib import Path
import joblib
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped
    from sensor_msgs.msg import Imu, Range
    from drone_interfaces.msg import AttackStatus, ResidualTelemetry, SensorTrust
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object

CLASS_NAMES = {
    0: "NORMAL",
    1: "GPS_SPOOFING",
    2: "IMU_MANIPULATION",
    3: "LIDAR_CORRUPTION"
}


class MLDetectorEngine:
    """Pure mathematical/ML inference engine decoupled from ROS 2 middleware."""

    def __init__(self, model_path: Path, scaler_path: Path, feature_names_path: Path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(feature_names_path, "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        # Rolling buffers for feature calculations
        self.accel_buffer = deque(maxlen=15)
        self.gyro_buffer = deque(maxlen=15)
        self.latest_gps_nis = 0.5
        self.latest_lidar_nis = 0.2
        self.latest_lidar_residual_z = 0.02
        self.latest_pos_err = 0.1
        self.latest_vel_err = 0.05
        self.consecutive_anomalies = 0
        self.composite_trust = 1.0

    def update_sensor_telemetry(
        self,
        gps_pos: np.ndarray,
        vision_pos: np.ndarray,
        imu_accel: np.ndarray,
        imu_gyro_z: float,
        lidar_z: float,
        gps_nis: float,
        lidar_nis: float,
        trust: float,
    ):
        """Updates internal buffers with newest sensor fixes."""
        self.latest_pos_err = float(np.linalg.norm(gps_pos - vision_pos))
        self.latest_lidar_residual_z = float(abs(lidar_z - vision_pos[2]))
        self.latest_gps_nis = float(gps_nis)
        self.latest_lidar_nis = float(lidar_nis)
        self.composite_trust = float(trust)

        accel_norm = float(np.linalg.norm(imu_accel))
        self.accel_buffer.append(accel_norm)
        self.gyro_buffer.append(imu_gyro_z)

    def predict(self) -> dict:
        """Executes scaled inference and returns predicted class and confidence probabilities."""
        accel_norm = self.accel_buffer[-1] if self.accel_buffer else 9.80665
        accel_diff = float(abs(accel_norm - 9.80665))
        accel_var = float(np.var(self.accel_buffer)) if len(self.accel_buffer) > 1 else 0.008
        gyro_var = float(np.var(self.gyro_buffer)) if len(self.gyro_buffer) > 1 else 0.003

        is_anom = (self.latest_gps_nis > 16.27) or (accel_diff > 1.5) or (self.latest_lidar_nis > 10.83)
        if is_anom:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = max(0, self.consecutive_anomalies - 1)

        feat_vector = np.array([[
            self.latest_gps_nis,
            self.latest_pos_err,
            self.latest_vel_err,
            accel_diff,
            accel_var,
            gyro_var,
            self.latest_lidar_nis,
            self.latest_lidar_residual_z,
            self.latest_pos_err,
            self.latest_lidar_residual_z,
            0.05 if not is_anom else 2.0,
            float(self.consecutive_anomalies),
            self.composite_trust,
        ]])

        scaled = self.scaler.transform(feat_vector)
        pred_id = int(self.model.predict(scaled)[0])
        probabilities = self.model.predict_proba(scaled)[0]

        predicted_class = CLASS_NAMES.get(pred_id, "NORMAL")
        confidence = float(np.max(probabilities))

        compromised = []
        if predicted_class == "GPS_SPOOFING":
            compromised.append("gps")
        elif predicted_class == "IMU_MANIPULATION":
            compromised.append("imu")
        elif predicted_class == "LIDAR_CORRUPTION":
            compromised.append("lidar")

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_attack": predicted_class != "NORMAL",
            "anomaly_score": float(1.0 - probabilities[0]),
            "compromised_sensors": compromised,
            "probabilities": {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(probabilities))},
        }


class MLDetectorNode(Node):
    """ROS 2 Node wrapper for online real-time ML detection."""

    def __init__(self):
        super().__init__("ml_detector_node")
        self.get_logger().info("Initializing ML Cyber-Attack Detector Node...")

        workspace_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        models_dir = workspace_dir / "ml" / "models"

        rf_path = models_dir / "random_forest_detector.joblib"
        scaler_path = models_dir / "scaler.joblib"
        feat_path = models_dir / "feature_names.json"

        self.engine = MLDetectorEngine(rf_path, scaler_path, feat_path)

        # State tracking
        self.latest_gps_pos = np.array([0.0, 0.0, 10.0])
        self.latest_vision_pos = np.array([0.0, 0.0, 10.0])
        self.latest_imu_accel = np.array([0.0, 0.0, 9.80665])
        self.latest_gyro_z = 0.0
        self.latest_lidar_z = 10.0
        self.latest_gps_nis = 0.5
        self.latest_lidar_nis = 0.2
        self.latest_trust = 1.0

        # Subscriptions
        self.create_subscription(PointStamped, "/sensors/gps", self.gps_callback, 10)
        self.create_subscription(Imu, "/sensors/imu", self.imu_callback, 10)
        self.create_subscription(Range, "/sensors/lidar", self.lidar_callback, 10)
        self.create_subscription(PoseStamped, "/sensors/vision_pose", self.vision_callback, 10)
        self.create_subscription(ResidualTelemetry, "/telemetry/residuals", self.residual_callback, 10)
        self.create_subscription(SensorTrust, "/telemetry/sensor_trust", self.trust_callback, 10)

        # Publisher
        self.status_pub = self.create_publisher(AttackStatus, "/detector/ml_attack_status", 10)

        # 10 Hz Inference Timer
        self.timer = self.create_timer(0.10, self.inference_step)
        self.get_logger().info("ML Detector Node running at 10 Hz.")

    def gps_callback(self, msg: PointStamped):
        self.latest_gps_pos = np.array([msg.point.x, msg.point.y, msg.point.z])

    def imu_callback(self, msg: Imu):
        self.latest_imu_accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.latest_gyro_z = msg.angular_velocity.z

    def lidar_callback(self, msg: Range):
        self.latest_lidar_z = msg.range

    def vision_callback(self, msg: PoseStamped):
        self.latest_vision_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def residual_callback(self, msg: ResidualTelemetry):
        if msg.sensor_name == "gps":
            self.latest_gps_nis = msg.nis
        elif msg.sensor_name == "lidar":
            self.latest_lidar_nis = msg.nis

    def trust_callback(self, msg: SensorTrust):
        if msg.sensor_name == "gps":
            self.latest_trust = msg.trust_score

    def inference_step(self):
        self.engine.update_sensor_telemetry(
            gps_pos=self.latest_gps_pos,
            vision_pos=self.latest_vision_pos,
            imu_accel=self.latest_imu_accel,
            imu_gyro_z=self.latest_gyro_z,
            lidar_z=self.latest_lidar_z,
            gps_nis=self.latest_gps_nis,
            lidar_nis=self.latest_lidar_nis,
            trust=self.latest_trust,
        )

        result = self.engine.predict()

        msg = AttackStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.is_attack_active = result["is_attack"]
        msg.attack_type = result["predicted_class"]
        msg.confidence = result["confidence"]
        msg.anomaly_score = result["anomaly_score"]
        msg.compromised_sensors = result["compromised_sensors"]
        self.status_pub.publish(msg)


def main(args=None):
    if not ROS2_AVAILABLE:
        print("[!] rclpy is not installed on this host. Run within Docker ROS 2 container.")
        return
    rclpy.init(args=args)
    node = MLDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
