import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np


class FakeDroneNode(Node):
    def __init__(self):
        super().__init__('fake_drone_node')
        self.truth_pub = self.create_publisher(PoseStamped, '/truth/pose', 50)
        self.cmd_sub = self.create_subscription(Twist, '/control/cmd_vel', self.cmd_callback, 10)

        self.pos = np.array([0.0, 0.0, 10.0]) # Start hovering at 10m
        self.vel = np.zeros(3)
        self.cmd_vel = np.zeros(3)
        self.dt = 0.02 # 50 Hz

        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info("Fake Drone Node initialized at 50 Hz.")

    def cmd_callback(self, msg: Twist):
        self.cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.linear.z])

    def timer_callback(self):
        # Kinematic integration with acceleration limits
        accel_limit = 2.5
        vel_err = self.cmd_vel - self.vel
        accel = np.clip(vel_err * 2.0, -accel_limit, accel_limit)
        self.vel += accel * self.dt
        self.pos += self.vel * self.dt

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(self.pos[0])
        msg.pose.position.y = float(self.pos[1])
        msg.pose.position.z = float(self.pos[2])
        msg.pose.orientation.w = 1.0

        self.truth_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeDroneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
