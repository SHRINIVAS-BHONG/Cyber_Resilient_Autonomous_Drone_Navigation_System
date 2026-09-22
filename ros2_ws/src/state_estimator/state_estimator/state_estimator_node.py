"""
ROS 2 State Estimator Node.

Runs 10-DOF Extended Kalman Filter (EKF), fusing IMU prediction with GPS,
LiDAR, and Vision updates. Exposes innovation residuals and NIS statistics
for real-time cyber-attack detection.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, Range
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from drone_interfaces.msg import ResidualTelemetry
from std_msgs.msg import String

from state_estimator.ekf_10dof import EKF10DOF


class StateEstimatorNode(Node):
    def __init__(self):
        super().__init__("state_estimator_node")

        # Initialize core EKF
        self.ekf = EKF10DOF()
        self.is_initialized = False
        self.active_sensors = {"gps", "imu", "lidar", "vision_pose"}

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 1. Subscribers
        self.imu_sub = self.create_subscription(
            Imu, "/normalized/imu", self.imu_callback, best_effort_qos
        )
        self.gps_sub = self.create_subscription(
            PointStamped, "/normalized/gps", self.gps_callback, best_effort_qos
        )
        self.lidar_sub = self.create_subscription(
            Range, "/normalized/lidar", self.lidar_callback, best_effort_qos
        )
        self.vision_sub = self.create_subscription(
            PoseStamped, "/normalized/vision_pose", self.vision_callback, best_effort_qos
        )
        self.active_sensor_sub = self.create_subscription(
            String, "/resilience/active_sensors", self.active_sensors_callback, 10
        )

        # 2. Publishers
        self.pose_pub = self.create_publisher(PoseStamped, "/state_estimator/pose", 10)
        self.odom_pub = self.create_publisher(Odometry, "/state_estimator/odometry", 10)
        self.residual_pub = self.create_publisher(ResidualTelemetry, "/state_estimator/residuals", 10)

        self.last_imu_time = None
        self.get_logger().info("10-DOF EKF State Estimator Node running.")

    def active_sensors_callback(self, msg: String):
        """Updates list of permitted sensors from Resilience Manager."""
        # Comma-separated list of active sensors
        self.active_sensors = set(msg.data.split(","))

    def imu_callback(self, msg: Imu):
        curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_imu_time is None:
            self.last_imu_time = curr_time
            return

        dt = curr_time - self.last_imu_time
        self.last_imu_time = curr_time

        if dt <= 0.0 or dt > 0.5:
            dt = 0.01  # Fallback to 100 Hz dt

        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        gyro_z = msg.angular_velocity.z

        # EKF Prediction step
        self.ekf.predict(accel=accel, gyro_z=gyro_z, dt=dt)
        self.publish_estimates(msg.header.stamp)

    def gps_callback(self, msg: PointStamped):
        pos_meas = np.array([msg.point.x, msg.point.y, msg.point.z])

        if not self.is_initialized:
            self.ekf.initialize_state(position=pos_meas)
            self.is_initialized = True
            self.get_logger().info(f"EKF initialized at position: {pos_meas}")
            return

        # Check if GPS is quarantined
        reject = "gps" not in self.active_sensors
        res = self.ekf.update_gps(pos_meas, reject_anomaly=reject)

        # Publish residual telemetry for cyber attack detection
        res_msg = ResidualTelemetry()
        res_msg.timestamp = msg.header.stamp
        res_msg.sensor_name = "gps"
        res_msg.innovation = [float(v) for v in res["position"]["innovation"]]
        res_msg.normalized_residual = [float(v) for v in res["position"]["normalized_residual"]]
        res_msg.nis = float(res["position"]["nis"])
        res_msg.threshold = 16.27
        res_msg.anomaly_flag = bool(res["position"]["is_gated"])
        self.residual_pub.publish(res_msg)

    def lidar_callback(self, msg: Range):
        if not self.is_initialized or "lidar" not in self.active_sensors:
            return
        res = self.ekf.update_lidar(msg.range, reject_anomaly=False)

        res_msg = ResidualTelemetry()
        res_msg.timestamp = msg.header.stamp
        res_msg.sensor_name = "lidar"
        res_msg.innovation = [float(res["innovation"][0])]
        res_msg.normalized_residual = [float(res["normalized_residual"][0])]
        res_msg.nis = float(res["nis"])
        res_msg.threshold = 10.83
        res_msg.anomaly_flag = bool(res["is_gated"])
        self.residual_pub.publish(res_msg)

    def vision_callback(self, msg: PoseStamped):
        if not self.is_initialized or "vision_pose" not in self.active_sensors:
            return
        pos_meas = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        res = self.ekf.update_vision_pose(pos_meas, reject_anomaly=False)

        res_msg = ResidualTelemetry()
        res_msg.timestamp = msg.header.stamp
        res_msg.sensor_name = "vision_pose"
        res_msg.innovation = [float(v) for v in res["innovation"]]
        res_msg.normalized_residual = [float(v) for v in res["normalized_residual"]]
        res_msg.nis = float(res["nis"])
        res_msg.threshold = 16.27
        res_msg.anomaly_flag = bool(res["is_gated"])
        self.residual_pub.publish(res_msg)

    def publish_estimates(self, stamp):
        state = self.ekf.get_state()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "map_enu"
        pose_msg.pose.position.x = float(state["position"][0])
        pose_msg.pose.position.y = float(state["position"][1])
        pose_msg.pose.position.z = float(state["position"][2])
        self.pose_pub.publish(pose_msg)

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "map_enu"
        odom_msg.child_frame_id = "base_link_enu"
        odom_msg.pose.pose.position.x = float(state["position"][0])
        odom_msg.pose.pose.position.y = float(state["position"][1])
        odom_msg.pose.pose.position.z = float(state["position"][2])
        odom_msg.twist.twist.linear.x = float(state["velocity"][0])
        odom_msg.twist.twist.linear.y = float(state["velocity"][1])
        odom_msg.twist.twist.linear.z = float(state["velocity"][2])
        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
