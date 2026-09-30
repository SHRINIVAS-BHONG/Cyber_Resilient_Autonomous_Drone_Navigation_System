import numpy as np
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyber_drone.anomaly_detector import AnomalyDetector


def test_detector_clean_baseline_zero_alarms():
    detector = AnomalyDetector()
    trusted_pos = np.array([20.0, 10.0, 10.0])
    trusted_cov = np.eye(3) * 0.1

    # Simulate 20 clean steps (2 seconds)
    for step in range(20):
        t = step * 0.1
        detector.record_sensor_message("gps", t)
        # Small noise
        gps_pos = trusted_pos + np.array([0.05, -0.05, 0.02])
        diag = detector.check_all(
            current_time=t,
            gps_pos=gps_pos,
            trusted_pos=trusted_pos,
            trusted_cov_pos=trusted_cov,
            main_gps_nis=1.5,
            imu_integrated_vel=np.array([1.0, 0.0, 0.0]),
            vision_vel=np.array([1.02, 0.01, 0.0]),
            lidar_alt=10.05,
            baro_alt=9.98
        )
        assert diag["status"] == "NORMAL"
        assert len(diag["compromised_sensors"]) == 0


def test_detector_gps_jump_confirmation():
    detector = AnomalyDetector()
    trusted_pos = np.array([20.0, 10.0, 10.0])
    trusted_cov = np.eye(3) * 0.1

    # Sudden 30m jump
    jump_gps = trusted_pos + np.array([30.0, 0.0, 0.0])

    for step in range(10):
        t = step * 0.1
        detector.record_sensor_message("gps", t)
        diag = detector.check_all(
            current_time=t,
            gps_pos=jump_gps,
            trusted_pos=trusted_pos,
            trusted_cov_pos=trusted_cov,
            main_gps_nis=45.0,  # High NIS
            imu_integrated_vel=np.array([1.0, 0.0, 0.0]),
            vision_vel=np.array([1.0, 0.0, 0.0]),
            lidar_alt=10.0,
            baro_alt=10.0
        )
        if step >= 5:
            assert diag["status"] == "ATTACK_CONFIRMED"
            assert "gps" in diag["compromised_sensors"]


def test_detector_watchdog_blackout():
    detector = AnomalyDetector()
    trusted_pos = np.array([0.0, 0.0, 10.0])
    trusted_cov = np.eye(3) * 0.1

    # Timestamp progresses but no GPS messages recorded
    t = 2.0
    diag = detector.check_all(
        current_time=t,
        gps_pos=None,
        trusted_pos=trusted_pos,
        trusted_cov_pos=trusted_cov,
        main_gps_nis=None,
        imu_integrated_vel=np.zeros(3),
        vision_vel=np.zeros(3),
        lidar_alt=10.0,
        baro_alt=10.0
    )
    assert "gps" in diag["compromised_sensors"]
    assert diag["attack_type"] == "communication_disruption"
