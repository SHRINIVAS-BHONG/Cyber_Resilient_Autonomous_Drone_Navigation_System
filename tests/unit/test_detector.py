import sys
from pathlib import Path
import pytest

# Add residual_detector package to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "ros2_ws" / "src" / "residual_detector"))

from residual_detector.detector_engine import ResidualDetectorEngine, DetectionState


def test_detector_normal_operation():
    engine = ResidualDetectorEngine(consecutive_alarms_to_confirm=5)
    # Feed normal low NIS values (< 11.34)
    for _ in range(10):
        res = engine.process_sensor_residual(sensor_name="gps", nis=2.1)
        assert res["state"] == DetectionState.NORMAL.value
        assert res["is_anomaly"] is False
        assert len(res["compromised_sensors"]) == 0


def test_detector_attack_confirmation():
    engine = ResidualDetectorEngine(consecutive_alarms_to_confirm=5)
    
    # 1. First alarm -> state becomes SUSPICIOUS
    res1 = engine.process_sensor_residual(sensor_name="gps", nis=45.0)
    assert res1["state"] == DetectionState.SUSPICIOUS.value
    assert res1["is_anomaly"] is True

    # 2. Feed consecutive high alarms -> transitions to ATTACK_CONFIRMED
    for _ in range(5):
        res = engine.process_sensor_residual(sensor_name="gps", nis=55.0)

    assert res["state"] == DetectionState.ATTACK_CONFIRMED.value
    assert "gps" in res["compromised_sensors"]
    assert res["confidence"] >= 1.0


def test_detector_recovery_after_attack_ends():
    engine = ResidualDetectorEngine(consecutive_alarms_to_confirm=3, recovery_samples_to_clear=5)

    # 1. Trigger attack
    for _ in range(4):
        engine.process_sensor_residual(sensor_name="gps", nis=60.0)
    assert engine.current_state == DetectionState.ATTACK_CONFIRMED

    # 2. Attack ceases, feed clean residuals
    for _ in range(6):
        res = engine.process_sensor_residual(sensor_name="gps", nis=1.5)

    # Should clear compromised list
    assert "gps" not in res["compromised_sensors"]
    assert res["state"] == DetectionState.NORMAL.value
