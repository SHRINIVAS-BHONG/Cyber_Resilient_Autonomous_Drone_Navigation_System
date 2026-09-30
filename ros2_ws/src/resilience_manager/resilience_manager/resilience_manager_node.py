import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from drone_interfaces.msg import AttackStatus, SensorTrust


class ResilienceManagerNode(Node):
    def __init__(self):
        super().__init__('resilience_manager_node')

        self.mode_pub = self.create_publisher(String, '/resilience/navigation_mode', 10)
        self.trust_pub = self.create_publisher(SensorTrust, '/resilience/sensor_trust', 10)
        self.gate_pub = self.create_publisher(String, '/resilience/sensor_gate', 10)

        self.create_subscription(AttackStatus, '/resilience/attack_status', self.attack_status_cb, 10)

        self.state = 'NORMAL'
        self.gps_isolated = False
        self.active_estimator = 'main'
        self.replan_pub = self.create_publisher(String, '/planner/trigger_replan', 10)

        self.create_timer(0.2, self.status_loop) # 5 Hz
        self.get_logger().info("Resilience Manager Node running.")

    def attack_status_cb(self, msg: AttackStatus):
        if msg.detected and not self.gps_isolated:
            self.state = 'CONTAINMENT'
            self.gps_isolated = True
            self.active_estimator = 'trusted'

            # 1. Gate out GPS
            gate_cmd = String()
            gate_cmd.data = 'gps:disable'
            self.gate_pub.publish(gate_cmd)

            # 2. Update navigation mode
            mode_cmd = String()
            mode_cmd.data = 'safe_local_navigation'
            self.mode_pub.publish(mode_cmd)

            # 3. Trigger planner replan
            replan_msg = String()
            replan_msg.data = 'replan:trusted'
            self.replan_pub.publish(replan_msg)

            self.get_logger().warn("ATTACK CONFIRMED: GPS isolated, active estimator switched to TRUSTED filter.")

    def status_loop(self):
        trust_msg = SensorTrust()
        trust_msg.sensor_name = 'gps'
        trust_msg.trusted = not self.gps_isolated
        trust_msg.trust_score = 0.0 if self.gps_isolated else 1.0
        self.trust_pub.publish(trust_msg)

        mode_msg = String()
        mode_msg.data = f"state:{self.state}|estimator:{self.active_estimator}"
        self.mode_pub.publish(mode_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ResilienceManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
