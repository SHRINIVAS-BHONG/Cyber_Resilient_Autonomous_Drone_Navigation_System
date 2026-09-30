import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyber_drone.ekf import DroneEKF


def test_ekf_initialization():
    ekf = DroneEKF(is_trusted=False)
    assert ekf.x.shape == (10,)
    assert ekf.P.shape == (10, 10)
    assert ekf.enabled_sensors["gps"] is True

    trusted_ekf = DroneEKF(is_trusted=True)
    assert trusted_ekf.enabled_sensors["gps"] is False


def test_ekf_prediction():
    ekf = DroneEKF()
    ekf.reset(np.array([0.0, 0.0, 10.0]), 0.0)

    # Hover acceleration in body frame: [0, 0, 9.81]
    accel_body = np.array([0.0, 0.0, 9.81])
    gyro_z = 0.0
    dt = 0.1

    ekf.predict(dt, accel_body, gyro_z)

    # Position and velocity should remain roughly unchanged during balanced hover
    assert np.allclose(ekf.position, [0.0, 0.0, 10.0], atol=0.05)
    assert np.allclose(ekf.velocity, [0.0, 0.0, 0.0], atol=0.05)


def test_ekf_gps_update_and_nis():
    ekf = DroneEKF()
    ekf.reset(np.array([10.0, 10.0, 10.0]))

    gps_meas = np.array([10.2, 9.9, 10.1])
    diag = ekf.update_gps(gps_meas)

    assert diag is not None
    assert "NIS" in diag
    assert "residual" in diag
    assert diag["NIS"] >= 0.0
    # State should move towards measurement
    assert np.allclose(ekf.position, gps_meas, atol=0.3)


def test_trusted_ekf_rejects_gps():
    trusted_ekf = DroneEKF(is_trusted=True)
    trusted_ekf.reset(np.array([0.0, 0.0, 10.0]))

    # Attempting to update GPS on trusted filter returns None and does not affect state
    diag = trusted_ekf.update_gps(np.array([50.0, 50.0, 10.0]))
    assert diag is None
    assert np.allclose(trusted_ekf.position, [0.0, 0.0, 10.0])
