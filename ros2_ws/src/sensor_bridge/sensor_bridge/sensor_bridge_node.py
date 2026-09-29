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

from sensor_msgs.msg import Imu, NavSatFix, Range, FluidPressure, MagneticField
from geometry_msgs.msg import PoseStamped, PointStamped

from sensor_bridge.geodesy import WGS84Datum


class SensorBridgeNode(Node):
    def __init__(self):
        super().__init__("sensor_bridge_node")

        # Source identifier tag
        self.source_id = "px4_sitl"

        # Declare parameters for geodetic reference origin (WGS-84)
        self.declare_parameter("origin_lat", 0.0)
        self.declare_parameter("origin_lon", 0.0)
        self.declare_parameter("origin_alt", 0.0)
        self.declare_parameter("auto_origin", True)

        # Magnetometer hard/soft iron calibration parameters
        self.declare_parameter("mag_hard_iron_x", 0.0)
        self.declare_parameter("mag_hard_iron_y", 0.0)
        self.declare_parameter("mag_hard_iron_z", 0.0)
        self.hard_iron = np.array([
            float(self.get_parameter("mag_hard_iron_x").value),
            float(self.get_parameter("mag_hard_iron_y").value),
            float(self.get_parameter("mag_hard_iron_z").value)
        ], dtype=np.float64)
        self.soft_iron = np.eye(3, dtype=np.float64)

        # Barometer QNH sea-level baseline pressure (default 101325.0 Pa)
        self.declare_parameter("baro_qnh_pressure", 101325.0)
        self.qnh_p0 = float(self.get_parameter("baro_qnh_pressure").value)

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
        self.raw_baro_sub = self.create_subscription(
            FluidPressure, "/sensor/barometer", self.baro_callback, sensor_qos
        )
        self.raw_mag_sub = self.create_subscription(
            MagneticField, "/sensor/magnetometer", self.mag_callback, sensor_qos
        )

        # 2. Publishers for normalized sensor streams (consistent ENU clock & coordinates)
        self.norm_imu_pub = self.create_publisher(Imu, "/normalized/imu", 10)
        self.norm_gps_pub = self.create_publisher(PointStamped, "/normalized/gps", 10)
        self.norm_lidar_pub = self.create_publisher(Range, "/normalized/lidar", 10)
        self.norm_vision_pub = self.create_publisher(PoseStamped, "/normalized/vision_pose", 10)
        self.norm_baro_pub = self.create_publisher(Range, "/normalized/barometer", 10)
        self.norm_mag_pub = self.create_publisher(MagneticField, "/normalized/magnetometer", 10)

        # Rate and health diagnostics
        self.last_imu_stamp = None
        self.last_gps_stamp = None
        self.imu_msg_count = 0
        self.gps_msg_count = 0
        self.baro_msg_count = 0
        self.mag_msg_count = 0

        # Frequency watchdog timer (1 Hz)
        self.watchdog_timer = self.create_timer(1.0, self.frequency_watchdog)

        self.get_logger().info("Sensor Bridge Node running with WGS-84 Geodesy, Baro/Mag support, and Watchdog.")

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

    def baro_callback(self, msg: FluidPressure):
        """
        Convert raw static atmospheric pressure (Pascals) to Barometric Altitude (meters)
        using the standard international hypsometric formula with QNH reference pressure:
        h = 44330.0 * (1.0 - (P / P0) ** 0.190295), where P0 is calibrated QNH baseline pressure.
        """
        p = msg.fluid_pressure
        if p <= 0.0 or math.isnan(p):
            return

        p0 = self.qnh_p0 if self.qnh_p0 > 0.0 else 101325.0
        alt_m = 44330.0 * (1.0 - (p / p0) ** 0.190295)

        norm_baro = Range()
        norm_baro.header = msg.header
        norm_baro.header.frame_id = "map_enu"
        norm_baro.range = float(alt_m)
        norm_baro.min_range = -500.0
        norm_baro.max_range = 10000.0
        self.norm_baro_pub.publish(norm_baro)
        self.baro_msg_count += 1

    def mag_callback(self, msg: MagneticField):
        """
        Normalize 3-axis magnetometer measurements in micro-Teslas with hard/soft-iron calibration.
        B_calib = S_soft @ (B_raw - b_hard), then normalize body frame from NED to ENU.
        """
        raw_b = np.array([
            msg.magnetic_field.x,
            msg.magnetic_field.y,
            msg.magnetic_field.z
        ], dtype=np.float64)

        # Apply hard/soft iron calibration
        calib_b = self.soft_iron @ (raw_b - self.hard_iron)

        norm_mag = MagneticField()
        norm_mag.header = msg.header
        norm_mag.header.frame_id = "base_link_enu"
        # NED to ENU: x_enu = y_ned, y_enu = x_ned, z_enu = -z_ned
        norm_mag.magnetic_field.x = float(calib_b[1])
        norm_mag.magnetic_field.y = float(calib_b[0])
        norm_mag.magnetic_field.z = -float(calib_b[2])
        norm_mag.magnetic_field_covariance = msg.magnetic_field_covariance

        self.norm_mag_pub.publish(norm_mag)
        self.mag_msg_count += 1

    def frequency_watchdog(self):
        """
        Monitors incoming sensor topic rates every 1.0 second.
        Flags sensor topic starvation / signal drop.
        Expected rates: IMU >= 25 Hz (nominal 50-100 Hz), GPS >= 2 Hz (nominal 5-10 Hz).
        """
        imu_hz = self.imu_msg_count
        gps_hz = self.gps_msg_count
        baro_hz = self.baro_msg_count
        mag_hz = self.mag_msg_count

        # Reset counters for the next window
        self.imu_msg_count = 0
        self.gps_msg_count = 0
        self.baro_msg_count = 0
        self.mag_msg_count = 0

        # Diagnostics warning only if node has been receiving data or expecting it
        if imu_hz > 0 and imu_hz < 25:
            self.get_logger().warn(f"IMU stream starvation: {imu_hz} Hz (nominal >= 50 Hz)")
        if gps_hz > 0 and gps_hz < 2:
            self.get_logger().warn(f"GPS stream starvation: {gps_hz} Hz (nominal >= 5 Hz)")



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
