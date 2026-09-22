"""
Resilience & Recovery Manager Engine.

Dynamically computes sensor trust metrics, enforces sensor isolation,
and switches vehicle failsafe flight modes upon cyber attack detection.
"""

from enum import Enum
from typing import Dict, List, Optional, Set


class NavigationMode(str, Enum):
    NORMAL_MISSION = "NORMAL_MISSION"
    DEGRADED_OPTICAL_LIDAR = "DEGRADED_OPTICAL_LIDAR"
    HOLD_POSITION = "HOLD_POSITION"
    SAFE_RETURN_TO_BASE = "SAFE_RETURN_TO_BASE"
    EMERGENCY_LANDING = "EMERGENCY_LANDING"


class ResilienceManagerEngine:
    def __init__(
        self,
        quarantine_threshold: float = 0.35,
        recovery_threshold: float = 0.85,
        penalty_per_anomaly: float = 0.25,
        reward_per_normal: float = 0.05,
    ):
        self.quarantine_th = quarantine_threshold
        self.recovery_th = recovery_threshold
        self.penalty = penalty_per_anomaly
        self.reward = reward_per_normal

        # Trust score per sensor in range [0.0, 1.0]
        self.sensor_trust: Dict[str, float] = {
            "gps": 1.0,
            "imu": 1.0,
            "lidar": 1.0,
            "vision_pose": 1.0,
        }

        self.isolated_sensors: Set[str] = set()
        self.current_navigation_mode: NavigationMode = NavigationMode.NORMAL_MISSION
        self.speed_factor: float = 1.0

    def update_sensor_health(self, sensor_name: str, is_anomaly: bool) -> float:
        """Updates continuous trust score for a sensor and manages isolation status."""
        if sensor_name not in self.sensor_trust:
            self.sensor_trust[sensor_name] = 1.0

        current_score = self.sensor_trust[sensor_name]

        if is_anomaly:
            new_score = max(0.0, current_score - self.penalty)
        else:
            new_score = min(1.0, current_score + self.reward)

        self.sensor_trust[sensor_name] = new_score

        # Isolation policy
        if new_score <= self.quarantine_th:
            self.isolated_sensors.add(sensor_name)
        elif new_score >= self.recovery_th and sensor_name in self.isolated_sensors:
            self.isolated_sensors.remove(sensor_name)

        return new_score

    def evaluate_resilience_policy(
        self,
        attack_status: str,
        detected_compromised_sensors: List[str]
    ) -> Dict[str, any]:
        """
        Evaluates overall resilience and determines safe fallback flight modes.
        """
        # Register anomalies for all sensors flagged by the detector
        for sensor in detected_compromised_sensors:
            self.update_sensor_health(sensor, is_anomaly=True)

        num_compromised = len(self.isolated_sensors)

        # 1. Multiple critical sensors compromised -> Failsafe Emergency Landing
        if num_compromised >= 2 or ("gps" in self.isolated_sensors and "imu" in self.isolated_sensors):
            self.current_navigation_mode = NavigationMode.EMERGENCY_LANDING
            self.speed_factor = 0.2

        # 2. GPS Spoofing -> Isolate GPS, switch to Optical Flow + LiDAR odometry
        elif "gps" in self.isolated_sensors:
            self.current_navigation_mode = NavigationMode.DEGRADED_OPTICAL_LIDAR
            self.speed_factor = 0.6

        # 3. IMU Manipulation -> Increase uncertainty, hold position
        elif "imu" in self.isolated_sensors:
            self.current_navigation_mode = NavigationMode.HOLD_POSITION
            self.speed_factor = 0.0

        # 4. Attack cleared and all sensors healthy
        elif num_compromised == 0 and attack_status in ["NORMAL", "RECOVERED"]:
            self.current_navigation_mode = NavigationMode.NORMAL_MISSION
            self.speed_factor = 1.0

        active_sensors = [s for s in self.sensor_trust if s not in self.isolated_sensors]

        return {
            "navigation_mode": self.current_navigation_mode.value,
            "speed_factor": self.speed_factor,
            "active_sensors": active_sensors,
            "isolated_sensors": list(self.isolated_sensors),
            "trust_scores": dict(self.sensor_trust),
            "is_degraded": self.current_navigation_mode != NavigationMode.NORMAL_MISSION,
        }
