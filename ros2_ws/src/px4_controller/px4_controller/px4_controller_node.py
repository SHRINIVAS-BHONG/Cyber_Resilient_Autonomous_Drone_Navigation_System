"""
PX4 Offboard Autonomous Controller Node.

Coordinates PX4 SITL flight control: arming, offboard mode transitions,
waypoint mission progression, and cyber-resilience failsafe overrides.
"""

from enum import Enum
import math
from typing import List, Tuple
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from drone_interfaces.msg import SafeTrajectory, AttackStatus


class FlightState(str, Enum):
    DISARMED = "DISARMED"
    ARMING = "ARMING"
    TAKEOFF = "TAKEOFF"
    WAYPOINT_MISSION = "WAYPOINT_MISSION"
    SAFE_REVISE_PATH = "SAFE_REVISE_PATH"
    HOLD_POSITION = "HOLD_POSITION"
    LANDING = "LANDING"


class PX4ControllerNode(Node):
    def __init__(self):
        super().__init__("px4_controller_node")

        # QoS configuration
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 1. Subscribers
        self.state_est_sub = self.create_subscription(
            PoseStamped, "/state_estimator/pose", self.estimator_pose_callback, best_effort_qos
        )
        self.safe_traj_sub = self.create_subscription(
            SafeTrajectory, "/planner/safe_trajectory", self.safe_trajectory_callback, 10
        )
        self.attack_status_sub = self.create_subscription(
            AttackStatus, "/resilience/attack_status", self.attack_status_callback, 10
        )

        # 2. Publishers for Offboard Control & Setpoints
        self.setpoint_pub = self.create_publisher(PoseStamped, "/fmu/in/trajectory_setpoint", 10)

        # Mission Waypoints (Local ENU coordinates: x, y, z)
        self.waypoints: List[Tuple[float, float, float]] = [
            (0.0, 0.0, 10.0),    # Takeoff hover
            (15.0, 0.0, 10.0),   # Waypoint 1
            (15.0, 15.0, 10.0),  # Waypoint 2
            (0.0, 15.0, 10.0),   # Waypoint 3
            (0.0, 0.0, 10.0)     # Home
        ]
        self.current_wp_idx = 0
        self.flight_state = FlightState.TAKEOFF
        self.current_pos = [0.0, 0.0, 0.0]

        # 20 Hz control loop timer
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("PX4 Offboard Controller Node initialized at 20 Hz.")

    def estimator_pose_callback(self, msg: PoseStamped):
        self.current_pos = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]

    def safe_trajectory_callback(self, msg: SafeTrajectory):
        """Overrides mission path with alternative safe path computed under cyber attack."""
        if msg.waypoints and self.flight_state != FlightState.LANDING:
            self.get_logger().warn("Engaging Alternative Safe Trajectory from Resilience Planner!")
            self.waypoints = [
                (wp.pose.position.x, wp.pose.position.y, wp.pose.position.z)
                for wp in msg.waypoints
            ]
            self.current_wp_idx = 0
            self.flight_state = FlightState.SAFE_REVISE_PATH

    def attack_status_callback(self, msg: AttackStatus):
        if msg.detected and msg.risk_score > 0.8:
            self.get_logger().error(f"Critical Attack Detected! Risk: {msg.risk_score:.2f}. Triggering Failsafe.")
            if "imu" in msg.compromised_sensors:
                self.flight_state = FlightState.HOLD_POSITION
            elif len(msg.compromised_sensors) >= 2:
                self.flight_state = FlightState.LANDING

    def control_loop(self):
        """Periodic 20 Hz setpoint publication."""
        target_setpoint = PoseStamped()
        target_setpoint.header.stamp = self.get_clock().now().to_msg()
        target_setpoint.header.frame_id = "map_enu"

        if self.flight_state == FlightState.HOLD_POSITION:
            # Command zero movement, hold current altitude
            target_setpoint.pose.position.x = self.current_pos[0]
            target_setpoint.pose.position.y = self.current_pos[1]
            target_setpoint.pose.position.z = self.current_pos[2]

        elif self.flight_state == FlightState.LANDING:
            # Descend gradually to ground
            target_setpoint.pose.position.x = self.current_pos[0]
            target_setpoint.pose.position.y = self.current_pos[1]
            target_setpoint.pose.position.z = 0.0

        else:
            # Track current waypoint
            target_wp = self.waypoints[self.current_wp_idx]
            target_setpoint.pose.position.x = target_wp[0]
            target_setpoint.pose.position.y = target_wp[1]
            target_setpoint.pose.position.z = target_wp[2]

            # Check waypoint acceptance radius (1.5 meters)
            dist_to_wp = math.hypot(
                self.current_pos[0] - target_wp[0],
                self.current_pos[1] - target_wp[1]
            )
            if dist_to_wp < 1.5:
                if self.current_wp_idx < len(self.waypoints) - 1:
                    self.current_wp_idx += 1
                    self.get_logger().info(f"Advancing to Waypoint {self.current_wp_idx}: {self.waypoints[self.current_wp_idx]}")
                else:
                    self.get_logger().info("Completed all mission waypoints. Returning to land.")
                    self.flight_state = FlightState.LANDING

        self.setpoint_pub.publish(target_setpoint)


def main(args=None):
    rclpy.init(args=args)
    node = PX4ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
