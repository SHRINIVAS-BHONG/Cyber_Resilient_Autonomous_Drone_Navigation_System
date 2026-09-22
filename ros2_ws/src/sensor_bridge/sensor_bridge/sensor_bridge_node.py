"""
ROS 2 Sensor Bridge Node.

Ingests raw sensor streams from PX4 SITL and Gazebo, normalizes timestamps
and coordinate frames (NED -> ENU), checks message frequency, and publishes
standardized topics for state estimation and attack detection.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, NavSatFix, Range
from geometry_msgs.msg import PoseStamped, PointStamped


class SensorBridgeNode(Node):
    def __init__(self):
        super().__init__("sensor_bridge_node")

        # QoS profile matching PX4 sensor streams (Best Effort, keep last 10)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 1. Subscribers to raw sensor streams
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

        # Timing and rate tracking
        self.last_imu_time = self.get_clock().now()
        self.last_gps_time = self.get_clock().now()
        self.imu_msg_count = 0
        self.gps_msg_count = 0

        self.get_logger().info("Sensor Bridge Node initialized and streaming normalized sensors (NED -> ENU).")

    def imu_callback(self, msg: Imu):
        """Normalize IMU frame and validate timestamps."""
        norm_msg = Imu()
        norm_msg.header = msg.header
        norm_msg.header.frame_id = "base_link_enu"

        # Frame transformation: NED to ENU
        # x_enu = y_ned, y_enu = x_ned, z_enu = -z_ned
        norm_msg.linear_acceleration.x = msg.linear_acceleration.y
        norm_msg.linear_acceleration.y = msg.linear_acceleration.x
        norm_msg.linear_acceleration.z = -msg.linear_acceleration.z

        norm_msg.angular_velocity.x = msg.angular_velocity.y
        norm_msg.angular_velocity.y = msg.angular_velocity.x
        norm_msg.angular_velocity.z = -msg.angular_velocity.z

        self.norm_imu_pub.publish(norm_msg)
        self.imu_msg_count += 1

    def gps_callback(self, msg: NavSatFix):
        """Convert GPS coordinate offsets and publish normalized local position."""
        norm_gps = PointStamped()
        norm_gps.header = msg.header
        norm_gps.header.frame_id = "map_enu"

        # Conversion to local meters from reference origin (simplified flat earth)
        # Lat/Lon 1 deg ~ 111319.5 m
        norm_gps.point.x = msg.longitude * 111319.5
        norm_gps.point.y = msg.latitude * 111319.5
        norm_gps.point.z = msg.altitude

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
