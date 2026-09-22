"""
ROS 2 Risk-Aware A* Path Planner Node.

Computes safe alternative trajectories circumventing physical obstacles and
high cyber-risk/jammed regions upon cyber attack detection.
"""

from typing import List, Tuple
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PoseStamped
from drone_interfaces.msg import SafeTrajectory
from path_planner.astar_planner import AStarPlanner


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")

        # Initialize core A* planner
        self.planner = AStarPlanner(
            grid_resolution=0.5,
            x_bounds=(-40.0, 40.0),
            y_bounds=(-40.0, 40.0),
            obstacle_inflation_radius=1.5,
            cyber_risk_weight=8.0,
            obstacle_proximity_weight=3.5
        )

        # Pre-seed world obstacles matching Gazebo SDF world
        # Tower at (8, 5) with radius 2.0m; Building at (-10, 8) with radius 3.0m
        self.planner.add_obstacle(8.0, 5.0, radius=2.0)
        self.planner.add_obstacle(-10.0, 8.0, radius=3.0)

        # Emergency Safe Landing Pad at (20, 15, 0)
        self.emergency_landing_zone = (20.0, 15.0, 10.0)

        # Current state storage
        self.current_pos = (0.0, 0.0, 10.0)
        self.current_mode = "NORMAL_MISSION"
        self.estimated_risk = 0.0

        # 1. Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, "/state_estimator/pose", self.pose_callback, 10
        )
        self.nav_mode_sub = self.create_subscription(
            String, "/resilience/navigation_mode", self.nav_mode_callback, 10
        )
        self.risk_sub = self.create_subscription(
            Float32, "/resilience/estimated_risk", self.risk_callback, 10
        )

        # 2. Publisher for safe trajectory setpoints to px4_controller
        self.safe_traj_pub = self.create_publisher(SafeTrajectory, "/planner/safe_trajectory", 10)

        # 1 Hz replanning evaluation timer
        self.timer = self.create_timer(1.0, self.replan_check)
        self.get_logger().info("Risk-Aware A* Path Planner Node initialized.")

    def pose_callback(self, msg: PoseStamped):
        self.current_pos = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        )

    def nav_mode_callback(self, msg: String):
        self.current_mode = msg.data

    def risk_callback(self, msg: Float32):
        self.estimated_risk = float(msg.data)

    def replan_check(self):
        """If under cyber attack or degraded mode, dynamically replan safe path."""
        if self.current_mode in ["DEGRADED_OPTICAL_LIDAR", "SAFE_RETURN_TO_BASE"]:
            self.get_logger().warn("Replanning alternative safe trajectory around cyber risk zones...")

            # Inject cyber-risk penalty around current suspicious GPS coordinates
            self.planner.add_cyber_risk_zone(
                x=self.current_pos[0],
                y=self.current_pos[1],
                radius=10.0,
                risk_level=self.estimated_risk
            )

            # Plan path from current position to designated emergency landing zone
            waypoints_3d = self.planner.plan(
                start_pos=self.current_pos,
                goal_pos=self.emergency_landing_zone
            )

            if waypoints_3d:
                traj_msg = SafeTrajectory()
                traj_msg.maximum_velocity = 2.0  # Reduced speed for degraded safety
                traj_msg.estimated_risk = self.estimated_risk
                traj_msg.planner = "A_STAR_CYBER_AWARE"
                traj_msg.emergency = True

                for wp in waypoints_3d:
                    pose_stamped = PoseStamped()
                    pose_stamped.header.stamp = self.get_clock().now().to_msg()
                    pose_stamped.header.frame_id = "map_enu"
                    pose_stamped.pose.position.x = float(wp[0])
                    pose_stamped.pose.position.y = float(wp[1])
                    pose_stamped.pose.position.z = float(wp[2])
                    traj_msg.waypoints.append(pose_stamped)

                self.safe_traj_pub.publish(traj_msg)
                self.get_logger().info(f"Published safe trajectory with {len(waypoints_3d)} waypoints to Emergency Zone.")


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
