"""
ROS 2 Residual Cybersecurity Detection Node.

Subscribes to estimator innovation residuals, computes Normalized Innovation
Squared (NIS) hypothesis testing, and confirms cyber-attacks via a hysteresis state machine.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from drone_interfaces.msg import ResidualTelemetry, AttackStatus
from residual_detector.detector_engine import ResidualDetectorEngine


class ResidualDetectorNode(Node):
    def __init__(self):
        super().__init__("residual_detector_node")

        # Initialize detector engine
        self.engine = ResidualDetectorEngine(
            nis_warning_threshold=11.34,
            nis_alarm_threshold=16.27,
            window_size=15,
            consecutive_alarms_to_confirm=5,
            recovery_samples_to_clear=20
        )

        # 1. Subscriber to residuals
        self.residual_sub = self.create_subscription(
            ResidualTelemetry, "/state_estimator/residuals", self.residual_callback, 10
        )

        # 2. Publishers for cyber resilience manager
        self.attack_pub = self.create_publisher(AttackStatus, "/resilience/attack_status", 10)
        self.risk_pub = self.create_publisher(Float32, "/resilience/estimated_risk", 10)

        self.get_logger().info("Residual Cybersecurity Detector Node initialized.")

    def residual_callback(self, msg: ResidualTelemetry):
        # Process residual through statistical testing & hysteresis FSM
        eval_res = self.engine.process_sensor_residual(
            sensor_name=msg.sensor_name,
            nis=float(msg.nis)
        )

        # Publish attack status
        attack_msg = AttackStatus()
        attack_msg.start_time = msg.timestamp
        attack_msg.attack_type = f"{msg.sensor_name.upper()}_ANOMALY" if eval_res["is_anomaly"] else "None"
        attack_msg.detected = bool(eval_res["state"] in ["ATTACK_CONFIRMED", "CONTAINMENT"])
        attack_msg.confidence = float(eval_res["confidence"])
        attack_msg.risk_score = float(eval_res["risk_score"])
        attack_msg.compromised_sensors = eval_res["compromised_sensors"]

        self.attack_pub.publish(attack_msg)

        # Publish overall risk score
        risk_msg = Float32()
        risk_msg.data = float(eval_res["risk_score"])
        self.risk_pub.publish(risk_msg)

        if attack_msg.detected:
            self.get_logger().error(
                f"CYBER ATTACK CONFIRMED on sensors: {attack_msg.compromised_sensors}! Risk: {attack_msg.risk_score:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ResidualDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
