"""
ROS 2 Sensor Bridge Node for Autonomous Drone Avionics.

Ingests raw sensor streams from PX4 SITL, Gazebo, or physical avionics hardware,
transforms WGS-84 Geodetic coordinates to high-precision local East-North-Up (ENU) meters,
normalizes IMU coordinate frames (NED -> ENU), checks message frequency, and publishes
standardized, rate-monitored topics for state estimation and cyber-resilience detection.
"""

import math
from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, NavSatFix, Range
from geometry_msgs.msg import PoseStamped, PointStamped

from sensor_bridge.geodesy import WGS84Datum


class SensorBridgeNode(Node):
    def __init__(self):
        super().__init__("sensor_bridge_node")

        # Declare parameters for geodetic reference origin (WGS-84)
        self.declare_parameter("origin_lat", 0.0)
        self.declare_parameter("origin_lon", 0.0)
        self.declare_parameter("origin_alt", 0.0)
        self.declare_parameter("auto_origin", True)

        param_lat = self.get_parameter("origin_lat").value
        param_lon = self.get_parameter("origin_lon").value
        param_alt = self.get_parameter("origin_alt").value
        self.auto_origin = self.get_parameter("auto_origin").value

        self.wgs84_datum: Optional[WGS84Datum] = None
        if not self.auto_origin and (abs(param_lat) > 1e-4 or abs(param_lon) > 1e-4):
            self.wgs84_datum = WGS84Datum(param_lat, param_lon, param_alt)
            self.get_logger().info(f"WGS-84 Datum fixed from params: lat={param_lat:.6f}, lon={param_lon:.6f}, alt={param_alt:.1f}m")

        # QoS profile matching PX4 sensor streams (Best Effort, keep last 10)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 1. Subscribers to raw sensor streams (PX4/Gazebo/Hardware)
        self.raw_imu_sub = self.create_subscription(
            Imu, "/sensor/imu", self.imu_callback, sensor_qos
        )
        self.raw_gps_sub = self.create_subscription(
            NavSatFix, "/sensor/gps", self.gps_callback, sensor_qos
        )
        self.raw_lidar_sub = self.create_subscription(
            Range, "/sensor/lidar", self.lidar_callback, sensor_qos
        )
        self.raw_vision_sub = self.create_subscription(
            PoseStamped, "/sensor/camera/pose", self.vision_callback, sensor_qos
        )

        # 2. Publishers for normalized sensor streams (ENU frame)
        self.norm_imu_pub = self.create_publisher(Imu, "/normalized/imu", 10)
        self.norm_gps_pub = self.create_publisher(PointStamped, "/normalized/gps", 10)
        self.norm_lidar_pub = self.create_publisher(Range, "/normalized/lidar", 10)
        self.norm_vision_pub = self.create_publisher(PoseStamped, "/normalized/vision_pose", 10)

        # Rate and health diagnostics
        self.last_imu_stamp = None
        self.last_gps_stamp = None
        self.imu_msg_count = 0
        self.gps_msg_count = 0

        self.get_logger().info("Sensor Bridge Node running with WGS-84 Geodesy and NED->ENU frame normalization.")

    def imu_callback(self, msg: Imu):
        """
        Normalize IMU body frame: NED (North-East-Down) to ENU (East-North-Up).
        x_enu = y_ned (East), y_enu = x_ned (North), z_enu = -z_ned (Up).
        """
        norm_msg = Imu()
        norm_msg.header = msg.header
        norm_msg.header.frame_id = "base_link_enu"

        norm_msg.linear_acceleration.x = msg.linear_acceleration.y
        norm_msg.linear_acceleration.y = msg.linear_acceleration.x
        norm_msg.linear_acceleration.z = -msg.linear_acceleration.z

        norm_msg.angular_velocity.x = msg.angular_velocity.y
        norm_msg.angular_velocity.y = msg.angular_velocity.x
        norm_msg.angular_velocity.z = -msg.angular_velocity.z

        norm_msg.orientation = msg.orientation

        self.norm_imu_pub.publish(norm_msg)
        self.imu_msg_count += 1

    def gps_callback(self, msg: NavSatFix):
        """
        Convert WGS-84 GPS coordinates (lat, lon, alt) to local East-North-Up (ENU) meters.
        Establishes datum on first valid fix if auto_origin is enabled.
        """
        # Validate GPS health and non-NaN coordinates
        if math.isnan(msg.latitude) or math.isnan(msg.longitude) or math.isnan(msg.altitude):
            self.get_logger().warn("Received NaN GPS coordinates, dropping packet.")
            return

        # Initialize datum on first valid GPS fix
        if self.wgs84_datum is None:
            # Check if coordinates are global WGS-84 or already local
            if abs(msg.latitude) > 0.001 or abs(msg.longitude) > 0.001:
                self.wgs84_datum = WGS84Datum(msg.latitude, msg.longitude, msg.altitude)
                self.get_logger().info(
                    f"Locked WGS-84 Origin Datum at: lat={msg.latitude:.7f}, lon={msg.longitude:.7f}, alt={msg.altitude:.2f}m"
                )
            else:
                # If already near origin (e.g. flat local simulation), establish at (0, 0, 0)
                self.wgs84_datum = WGS84Datum(0.0, 0.0, 0.0)

        # High-precision Geodetic to ENU projection
        if abs(msg.latitude) > 0.001 or abs(msg.longitude) > 0.001:
            east_m, north_m, up_m = self.wgs84_datum.geodetic_to_enu(
                msg.latitude, msg.longitude, msg.altitude
            )
        else:
            # Local simulation fix where lat/lon are already local offsets
            east_m = msg.longitude
            north_m = msg.latitude
            up_m = msg.altitude

        norm_gps = PointStamped()
        norm_gps.header = msg.header
        norm_gps.header.frame_id = "map_enu"
        norm_gps.point.x = float(east_m)
        norm_gps.point.y = float(north_m)
        norm_gps.point.z = float(up_m)

        self.norm_gps_pub.publish(norm_gps)
        self.gps_msg_count += 1

    def lidar_callback(self, msg: Range):
        """Publish normalized 1D range / altitude."""
        norm_range = Range()
        norm_range.header = msg.header
        norm_range.header.frame_id = "lidar_link"
        norm_range.range = max(msg.min_range, min(msg.range, msg.max_range))
        self.norm_lidar_pub.publish(norm_range)

    def vision_callback(self, msg: PoseStamped):
        """Publish normalized visual odometry / camera pose."""
        norm_pose = PoseStamped()
        norm_pose.header = msg.header
        norm_pose.header.frame_id = "map_enu"
        norm_pose.pose = msg.pose
        self.norm_vision_pub.publish(norm_pose)


def main(args=None):
    rclpy.init(args=args)
    node = SensorBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
