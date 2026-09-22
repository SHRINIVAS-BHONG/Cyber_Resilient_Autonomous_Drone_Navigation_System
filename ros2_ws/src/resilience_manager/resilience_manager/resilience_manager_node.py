"""
ROS 2 Resilience Manager Node.

Autonomous decision engine managing dynamic sensor isolation, continuous trust scores,
and vehicle safety mode transitions during cyber attacks.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from drone_interfaces.msg import AttackStatus, SensorTrust
from resilience_manager.manager_engine import ResilienceManagerEngine


class ResilienceManagerNode(Node):
    def __init__(self):
        super().__init__("resilience_manager_node")

        # Initialize resilience engine
        self.engine = ResilienceManagerEngine(
            quarantine_threshold=0.35,
            recovery_threshold=0.85,
            penalty_per_anomaly=0.25,
            reward_per_normal=0.05
        )

        # 1. Subscriber to Attack Status from Detector
        self.attack_sub = self.create_subscription(
            AttackStatus, "/resilience/attack_status", self.attack_callback, 10
        )

        # 2. Publishers for State Estimator, Planner, and PX4 Controller
        self.active_sensors_pub = self.create_publisher(String, "/resilience/active_sensors", 10)
        self.nav_mode_pub = self.create_publisher(String, "/resilience/navigation_mode", 10)
        self.trust_pub = self.create_publisher(SensorTrust, "/resilience/sensor_trust", 10)

        # 10 Hz periodic publish timer
        self.timer = self.create_timer(0.1, self.periodic_publish)
        self.current_policy = self.engine.evaluate_resilience_policy("NORMAL", [])

        self.get_logger().info("Resilience Manager Node initialized.")

    def attack_callback(self, msg: AttackStatus):
        attack_state = "ATTACK_CONFIRMED" if msg.detected else "NORMAL"
        self.current_policy = self.engine.evaluate_resilience_policy(
            attack_status=attack_state,
            detected_compromised_sensors=msg.compromised_sensors
        )

        if self.current_policy["is_degraded"]:
            self.get_logger().warn(
                f"Resilience Action Engaged: Mode={self.current_policy['navigation_mode']}, "
                f"Active Sensors={self.current_policy['active_sensors']}, "
                f"Isolated={self.current_policy['isolated_sensors']}"
            )

    def periodic_publish(self):
        # 1. Active sensors string (comma-separated)
        active_str = ",".join(self.current_policy["active_sensors"])
        msg_active = String()
        msg_active.data = active_str
        self.active_sensors_pub.publish(msg_active)

        # 2. Navigation Mode
        msg_mode = String()
        msg_mode.data = self.current_policy["navigation_mode"]
        self.nav_mode_pub.publish(msg_mode)

        # 3. Publish individual sensor trust metrics
        for sensor_name, trust_val in self.current_policy["trust_scores"].items():
            trust_msg = SensorTrust()
            trust_msg.sensor_name = sensor_name
            trust_msg.trusted = sensor_name in self.current_policy["active_sensors"]
            trust_msg.trust_score = float(trust_val)
            self.trust_pub.publish(trust_msg)


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


if __name__ == "__main__":
    main()
