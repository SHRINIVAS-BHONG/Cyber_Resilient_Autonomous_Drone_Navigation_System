import numpy as np
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyber_drone.resilience_manager import ResilienceManager
from cyber_drone.ekf import DroneEKF


def test_resilience_state_machine_and_isolation():
    manager = ResilienceManager()
    main_ekf = DroneEKF(is_trusted=False)
    trusted_ekf = DroneEKF(is_trusted=True)

    assert manager.state == "NORMAL"
    assert manager.active_estimator == "main"

    # 1. Suspicious trigger
    detection_suspicious = {
        "status": "SUSPICIOUS",
        "compromised_sensors": ["gps"],
        "risk_score": 0.5,
        "attack_type": "gps_drift"
    }
    res = manager.update(10.0, detection_suspicious, main_ekf, trusted_ekf)
    assert manager.state == "SUSPICIOUS"
    assert manager.sensor_trust["gps"] < 1.0

    # 2. Confirmed Attack trigger -> Containment -> Safe Navigation
    detection_attack = {
        "status": "ATTACK_CONFIRMED",
        "compromised_sensors": ["gps"],
        "risk_score": 0.9,
        "attack_type": "gps_drift"
    }
    res = manager.update(10.5, detection_attack, main_ekf, trusted_ekf)
    assert manager.state == "SAFE_NAVIGATION"
    assert "gps" in manager.isolated_sensors
    assert manager.active_estimator == "trusted"
    assert not main_ekf.enabled_sensors["gps"]


def test_resilience_gradual_reintegration():
    manager = ResilienceManager()
    main_ekf = DroneEKF(is_trusted=False)
    trusted_ekf = DroneEKF(is_trusted=True)

    # Force into containment with isolated GPS
    manager.state = "SAFE_NAVIGATION"
    manager.isolated_sensors.add("gps")
    manager.gps_covariance_inflation = 10.0
    manager.sensor_trust["gps"] = 0.0
    manager.active_estimator = "trusted"

    # Attack ceases, environment is clean
    detection_clean = {
        "status": "NORMAL",
        "compromised_sensors": [],
        "risk_score": 0.0,
        "attack_type": "none"
    }

    # Step 1: Enters RECOVERY
    manager.update(50.0, detection_clean, main_ekf, trusted_ekf)
    assert manager.state == "RECOVERY"

    # 5.0 seconds dwell agreement requirement (t=50 to t=55.1)
    manager.update(55.1, detection_clean, main_ekf, trusted_ekf)
    # Step into ramp (t=56.0): ramp_elapsed > 0, covariance drops, trust rises
    manager.update(56.0, detection_clean, main_ekf, trusted_ekf)
    assert manager.gps_covariance_inflation < 10.0

    # 3.0 seconds ramp duration (t=55.1 to t=58.2)
    manager.update(58.5, detection_clean, main_ekf, trusted_ekf)
    # Once ramp completes, GPS is restored and system returns to normal
    assert "gps" not in manager.isolated_sensors
    assert manager.sensor_trust["gps"] >= 0.99
    assert manager.state == "RECOVERED"
