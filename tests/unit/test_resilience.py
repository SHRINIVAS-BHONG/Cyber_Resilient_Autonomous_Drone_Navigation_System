import sys
from pathlib import Path
import pytest

# Add resilience_manager package to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "ros2_ws" / "src" / "resilience_manager"))

from resilience_manager.manager_engine import ResilienceManagerEngine, NavigationMode


def test_resilience_initial_state():
    engine = ResilienceManagerEngine()
    policy = engine.evaluate_resilience_policy(attack_status="NORMAL", detected_compromised_sensors=[])
    assert policy["navigation_mode"] == NavigationMode.NORMAL_MISSION.value
    assert policy["speed_factor"] == 1.0
    assert len(policy["isolated_sensors"]) == 0
    assert "gps" in policy["active_sensors"]


def test_resilience_gps_spoofing_isolation_and_fallback():
    engine = ResilienceManagerEngine()

    # Repeated GPS anomaly triggers quarantine
    for _ in range(3):
        policy = engine.evaluate_resilience_policy(
            attack_status="ATTACK_CONFIRMED",
            detected_compromised_sensors=["gps"]
        )

    # GPS must be isolated and mode must be DEGRADED_OPTICAL_LIDAR
    assert "gps" in policy["isolated_sensors"]
    assert "gps" not in policy["active_sensors"]
    assert policy["navigation_mode"] == NavigationMode.DEGRADED_OPTICAL_LIDAR.value
    assert policy["speed_factor"] == 0.6
    assert "vision_pose" in policy["active_sensors"]
    assert "lidar" in policy["active_sensors"]


def test_resilience_multi_sensor_compromise_emergency_landing():
    engine = ResilienceManagerEngine()

    # Compromise both GPS and IMU
    for _ in range(4):
        policy = engine.evaluate_resilience_policy(
            attack_status="ATTACK_CONFIRMED",
            detected_compromised_sensors=["gps", "imu"]
        )

    assert policy["navigation_mode"] == NavigationMode.EMERGENCY_LANDING.value
    assert policy["speed_factor"] == 0.2


def test_resilience_sensor_recovery_hysteresis():
    engine = ResilienceManagerEngine()

    # 1. Drive GPS into isolation
    for _ in range(3):
        engine.evaluate_resilience_policy("ATTACK_CONFIRMED", ["gps"])
    assert "gps" in engine.isolated_sensors

    # 2. Feed healthy updates gradually
    for _ in range(20):
        engine.update_sensor_health("gps", is_anomaly=False)

    policy = engine.evaluate_resilience_policy("NORMAL", [])
    # Should recover above 0.85 and be reinstated
    assert "gps" not in policy["isolated_sensors"]
    assert policy["navigation_mode"] == NavigationMode.NORMAL_MISSION.value
