import sys
from pathlib import Path
import numpy as np
import pytest

# Add state_estimator package to path for unit testing
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "ros2_ws" / "src" / "state_estimator"))

from state_estimator.ekf_10dof import EKF10DOF


def test_ekf_initialization():
    ekf = EKF10DOF()
    state = ekf.get_state()
    assert state["position"].shape == (3,)
    assert state["velocity"].shape == (3,)
    assert state["yaw"] == 0.0
    assert np.all(state["position_std"] > 0.0)


def test_ekf_prediction_stationary():
    ekf = EKF10DOF()
    # In stationary flight, accelerometer balances gravity (upward 9.80665 m/s^2)
    accel = np.array([0.0, 0.0, 9.80665])
    dt = 0.02
    for _ in range(50):
        ekf.predict(accel=accel, gyro_z=0.0, dt=dt)

    state = ekf.get_state()
    # Position and velocity should remain near zero
    assert np.allclose(state["position"], [0.0, 0.0, 0.0], atol=1e-3)
    assert np.allclose(state["velocity"], [0.0, 0.0, 0.0], atol=1e-3)


def test_ekf_gps_convergence():
    ekf = EKF10DOF()
    accel = np.array([0.0, 0.0, 9.80665])
    true_pos = np.array([10.0, 5.0, 15.0])
    ekf.initialize_state(position=true_pos)

    dt = 0.1
    for _ in range(30):
        ekf.predict(accel=accel, gyro_z=0.0, dt=dt)
        # Add normal small GPS noise
        noise = np.random.normal(0.0, 0.1, size=(3,))
        res = ekf.update_gps(true_pos + noise)
        assert res["position"]["nis"] < 16.27  # Chi2 gate check in steady state

    state = ekf.get_state()
    assert np.allclose(state["position"], true_pos, atol=0.5)


def test_ekf_gps_spoofing_detection_and_rejection():
    ekf = EKF10DOF()
    accel = np.array([0.0, 0.0, 9.80665])
    true_pos = np.array([10.0, 5.0, 15.0])

    # 1. Converge on true position
    for _ in range(25):
        ekf.predict(accel=accel, gyro_z=0.0, dt=0.1)
        ekf.update_gps(true_pos)

    # 2. Inject large spoofing jump (35m offset)
    spoofed_pos = true_pos + np.array([35.0, -25.0, 0.0])
    res = ekf.update_gps(spoofed_pos, reject_anomaly=True)

    # Should detect anomaly and trigger gating
    assert res["position"]["is_gated"] is True
    assert res["position"]["nis"] > 16.27

    # Since reject_anomaly is True, state should NOT jump to the spoofed location
    state = ekf.get_state()
    assert not np.allclose(state["position"], spoofed_pos, atol=10.0)
    assert np.allclose(state["position"], true_pos, atol=1.0)
