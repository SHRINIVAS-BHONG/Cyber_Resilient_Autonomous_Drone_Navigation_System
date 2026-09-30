import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from drone_interfaces.msg import SafeTrajectory
import numpy as np


class PX4ControllerNode(Node):
    def __init__(self):
        super().__init__('px4_controller_node')

        # Control publishers
        self.cmd_pub = self.create_publisher(Twist, '/control/cmd_vel', 20)

        # Subscribers
        self.create_subscription(SafeTrajectory, '/planner/safe_trajectory', self.traj_cb, 10)
        self.create_subscription(Odometry, '/estimate/main', self.main_cb, 50)
        self.create_subscription(Odometry, '/estimate/trusted', self.trusted_cb, 50)
        self.create_subscription(String, '/resilience/navigation_mode', self.mode_cb, 10)

        self.waypoints = []
        self.current_wp_idx = 0
        self.max_speed = 3.0
        self.active_estimator = 'main'

        self.main_pos = np.array([0.0, 0.0, 10.0])
        self.trusted_pos = np.array([0.0, 0.0, 10.0])

        # 20 Hz control loop
        self.create_timer(1.0 / 20.0, self.control_loop)
        self.get_logger().info("PX4 Controller Node running at 20 Hz.")

    def mode_cb(self, msg: String):
        # Format: "state:...|estimator:trusted"
        if 'estimator:trusted' in msg.data:
            self.active_estimator = 'trusted'
            self.max_speed = 1.5
        elif 'estimator:main' in msg.data:
            self.active_estimator = 'main'
            self.max_speed = 3.0

    def main_cb(self, msg: Odometry):
        self.main_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def trusted_cb(self, msg: Odometry):
        self.trusted_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def traj_cb(self, msg: SafeTrajectory):
        self.waypoints = [
            np.array([wp.pose.position.x, wp.pose.position.y, wp.pose.position.z])
            for wp in msg.waypoints
        ]
        self.max_speed = float(msg.maximum_velocity)
        self.current_wp_idx = 0

    def control_loop(self):
        active_pos = self.trusted_pos if self.active_estimator == 'trusted' else self.main_pos

        if not self.waypoints or self.current_wp_idx >= len(self.waypoints):
            # Hover in place
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            return

        target = self.waypoints[self.current_wp_idx]
        diff = target - active_pos
        dist_horiz = np.linalg.norm(diff[:2])

        if dist_horiz < 2.0:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                return
            target = self.waypoints[self.current_wp_idx]
            diff = target - active_pos
            dist_horiz = np.linalg.norm(diff[:2])

        cmd = Twist()
        if dist_horiz > 0.1:
            cmd.linear.x = float((diff[0] / dist_horiz) * min(self.max_speed, dist_horiz))
            cmd.linear.y = float((diff[1] / dist_horiz) * min(self.max_speed, dist_horiz))
        cmd.linear.z = float(np.clip(diff[2] * 0.5, -1.0, 1.0))

        self.cmd_pub.publish(cmd)


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


if __name__ == '__main__':
    main()
