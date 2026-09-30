import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from drone_interfaces.msg import SafeTrajectory
import numpy as np


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        self.traj_pub = self.create_publisher(SafeTrajectory, '/planner/safe_trajectory', 10)

        self.create_subscription(String, '/planner/trigger_replan', self.replan_cb, 10)
        self.create_subscription(Odometry, '/estimate/trusted', self.trusted_cb, 50)

        self.current_pos = np.array([0.0, 0.0, 10.0])
        self.goal_pos = np.array([60.0, 40.0, 10.0])

        self.publish_nominal_trajectory()
        self.get_logger().info("Path Planner Node running.")

    def trusted_cb(self, msg: Odometry):
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

    def replan_cb(self, msg: String):
        self.get_logger().info("Replanning safe trajectory around inflated obstacles...")
        # Safe replan avoids center obstacles via perimeter waypoint
        waypoints_xyz = [
            [float(self.current_pos[0]), float(self.current_pos[1]), 10.0],
            [15.0, 30.0, 10.0],
            [40.0, 45.0, 10.0],
            [60.0, 40.0, 10.0]
        ]
        self._publish_trajectory(waypoints_xyz, max_vel=1.5, risk=0.2)

    def publish_nominal_trajectory(self):
        waypoints_xyz = [
            [15.0, 8.0, 10.0],
            [30.0, 18.0, 10.0],
            [45.0, 30.0, 10.0],
            [60.0, 40.0, 10.0]
        ]
        self._publish_trajectory(waypoints_xyz, max_vel=3.0, risk=0.0)

    def _publish_trajectory(self, waypoints: list, max_vel: float, risk: float):
        msg = SafeTrajectory()
        msg.maximum_velocity = float(max_vel)
        msg.estimated_risk = float(risk)
        msg.planner = "astar_resilient"
        msg.emergency = False

        for wp in waypoints:
            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = 'map'
            ps.pose.position.x = float(wp[0])
            ps.pose.position.y = float(wp[1])
            ps.pose.position.z = float(wp[2])
            ps.pose.orientation.w = 1.0
            msg.waypoints.append(ps)

        self.traj_pub.publish(msg)


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


if __name__ == '__main__':
    main()
