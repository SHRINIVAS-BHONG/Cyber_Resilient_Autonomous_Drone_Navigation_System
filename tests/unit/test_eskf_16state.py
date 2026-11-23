"""
Unit tests for 16-State / 15-Error-State Quaternion Error-State Kalman Filter (ESKF).
"""

import numpy as np
import pytest
from state_estimator.eskf_16state import ESKF16State, quaternion_to_rotation_matrix, quaternion_multiply


def test_quaternion_rotation_and_multiply():
    # Identity quaternion [1, 0, 0, 0]
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    R_id = quaternion_to_rotation_matrix(q_id)
    assert np.allclose(R_id, np.eye(3), atol=1e-6)

    # 90-degree yaw rotation around Z axis
    cy = np.cos(np.pi / 4.0)
    sy = np.sin(np.pi / 4.0)
    q_yaw90 = np.array([cy, 0.0, 0.0, sy])
    R_yaw90 = quaternion_to_rotation_matrix(q_yaw90)

    # Vector along X [1, 0, 0] should rotate to [0, 1, 0]
    v_rot = R_yaw90 @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-5)


def test_eskf_initialization_and_euler():
    eskf = ESKF16State()
    init_pos = np.array([10.0, -5.0, 15.0])
    init_vel = np.array([1.0, 0.5, -0.2])
    eskf.initialize_state(position=init_pos, velocity=init_vel, yaw=np.radians(45.0))

    state = eskf.get_state()
    assert np.allclose(state["position"], init_pos, atol=1e-5)
    assert np.allclose(state["velocity"], init_vel, atol=1e-5)
    assert abs(state["euler_angles_deg"][2] - 45.0) < 1.0


def test_eskf_prediction_stationary():
    eskf = ESKF16State()
    eskf.initialize_state(position=np.array([0.0, 0.0, 10.0]), velocity=np.zeros(3))

    # Stationary accelerometer reading gravity opposing vector [0, 0, 9.80665]
    accel = np.array([0.0, 0.0, 9.80665])
    gyro = np.zeros(3)

    for _ in range(10):
        eskf.predict(accel=accel, gyro=gyro, dt=0.1)

    state = eskf.get_state()
    # Height should remain roughly constant at 10.0m
    assert abs(state["position"][2] - 10.0) < 0.05
    assert np.linalg.norm(state["velocity"]) < 0.05


def test_eskf_gps_convergence_and_gating():
    eskf = ESKF16State()
    true_pos = np.array([20.0, 30.0, 15.0])
    eskf.initialize_state(position=true_pos, velocity=np.zeros(3))

    # Feed nominal GPS measurements to verify convergence
    for _ in range(15):
        eskf.predict(accel=np.array([0.0, 0.0, 9.80665]), gyro=np.zeros(3), dt=0.1)
        res = eskf.update_gps(true_pos + np.random.normal(0.0, 0.05, size=3), reject_anomaly=True)
        assert not res["position"]["is_gated"]

    # Estimated position should converge near true_pos
    assert np.linalg.norm(eskf.p - true_pos) < 0.5

    # Inject sudden massive spoofing offset (35m)
    spoofed_pos = true_pos + np.array([35.0, 0.0, 0.0])
    res_spoof = eskf.update_gps(spoofed_pos, reject_anomaly=True)

    # Anomaly must be detected and rejected
    assert res_spoof["position"]["is_gated"]
    assert res_spoof["position"]["nis"] > 16.27
    # Position must NOT jump to 35m
    assert abs(eskf.p[0] - true_pos[0]) < 1.0


def test_eskf_joseph_covariance_symmetry():
    eskf = ESKF16State()
    eskf.predict(accel=np.array([0.1, -0.1, 9.8]), gyro=np.array([0.01, 0.0, 0.02]), dt=0.1)
    eskf.update_gps(np.array([1.0, 1.0, 10.0]), reject_anomaly=False)
    eskf.update_lidar(10.0, reject_anomaly=False)

    # Covariance P must remain strictly symmetric and positive semi-definite
    assert np.allclose(eskf.P, eskf.P.T, atol=1e-8)
    eigenvals = np.linalg.eigvals(eskf.P)
    assert np.all(eigenvals >= -1e-10)
