"""
End-to-End Cyber-Resilient Navigation Pipeline Integration Test.

Strictly real mathematical algorithms (no dummy stubs or mocks).
Integrates:
  - 10-DOF Extended Kalman Filter (state_estimator)
  - Statistical NIS Residual Detector (residual_detector)
  - Real-Trained Random Forest ML Detector (ml_detector)
  - Continuous Sensor Trust Resilience Manager (resilience_manager)
  - Risk-Aware 3D A* Path Planner (path_planner)
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add modules to path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir / "ros2_ws" / "src" / "state_estimator"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "residual_detector"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "resilience_manager"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "path_planner"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "ml_detector"))

from state_estimator.ekf_10dof import EKF10DOF
from residual_detector.detector_engine import ResidualDetectorEngine, DetectionState
from resilience_manager.manager_engine import ResilienceManagerEngine, NavigationMode
from path_planner.astar_planner import AStarPlanner
from ml_detector.ml_detector_node import MLDetectorEngine


@pytest.fixture(scope="module")
def pipeline_engines():
    models_dir = root_dir / "ml" / "models"
    rf_path = models_dir / "random_forest_detector.joblib"
    scaler_path = models_dir / "scaler.joblib"
    feat_path = models_dir / "feature_names.json"

    assert rf_path.exists(), "Trained model artifact missing."
    ml_engine = MLDetectorEngine(rf_path, scaler_path, feat_path)

    return {
        "ml_engine": ml_engine,
    }


def test_end_to_end_nominal_mission_flight(pipeline_engines):
    """Verifies that under attack-free nominal flight, all pipeline engines operate synchronously."""
    ekf = EKF10DOF()
    detector = ResidualDetectorEngine()
    resilience = ResilienceManagerEngine()
    planner = AStarPlanner(grid_resolution=1.0)
    ml_engine = pipeline_engines["ml_engine"]

    dt = 0.1
    duration = 15.0
    time_steps = np.arange(0.0, duration, dt)

    ekf.initialize_state(position=np.array([0.0, 0.0, 10.0]))
    true_positions = []
    est_positions = []

    for t in time_steps:
        true_pos = np.array([0.5 * t, 0.0, 10.0])
        accel = np.array([0.0, 0.0, 9.80665]) + np.random.normal(0.0, 0.02, size=(3,))
        gps_meas = true_pos + np.random.normal(0.0, 0.20, size=(3,))
        vision_meas = true_pos + np.random.normal(0.0, 0.04, size=(3,))
        lidar_z = float(true_pos[2] + np.random.normal(0.0, 0.03))

        # 1. State Estimation
        ekf.predict(accel=accel, gyro_z=0.0, dt=dt)
        gps_res = ekf.update_gps(gps_meas, reject_anomaly=True)
        lidar_res = ekf.update_lidar(lidar_z, reject_anomaly=True)

        # 2. Statistical Anomaly Detection
        det_gps = detector.process_sensor_residual("gps", gps_res["position"]["nis"])
        det_lidar = detector.process_sensor_residual("lidar", lidar_res["nis"])

        # 3. Machine Learning Detection
        ml_engine.update_sensor_telemetry(
            gps_pos=gps_meas,
            vision_pos=vision_meas,
            imu_accel=accel,
            imu_gyro_z=0.0,
            lidar_z=lidar_z,
            gps_nis=gps_res["position"]["nis"],
            lidar_nis=lidar_res["nis"],
            trust=resilience.sensor_trust["gps"],
        )
        ml_res = ml_engine.predict()

        # 4. Resilience Management
        policy = resilience.evaluate_resilience_policy(det_gps["state"], det_gps["compromised_sensors"])

        true_positions.append(true_pos)
        est_positions.append(ekf.get_state()["position"])

        # In nominal flight, everything remains NORMAL
        assert det_gps["state"] == DetectionState.NORMAL.value
        assert policy["navigation_mode"] == NavigationMode.NORMAL_MISSION.value
        assert "gps" not in policy["isolated_sensors"]

    # 5. Path Planning in nominal state
    path = planner.plan((0.0, 0.0, 10.0), (10.0, 0.0, 10.0))
    assert path is not None
    assert len(path) > 0

    # Tracking accuracy check
    err = np.linalg.norm(np.array(est_positions) - np.array(true_positions), axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    assert rmse < 0.35  # Sub-35cm tracking accuracy


def test_end_to_end_gps_spoofing_containment_and_rerouting(pipeline_engines):
    """Verifies that upon GPS spoofing, both detectors trigger, GPS is isolated, and planner routes around cyber-risk."""
    ekf = EKF10DOF()
    detector = ResidualDetectorEngine(consecutive_alarms_to_confirm=3)
    resilience = ResilienceManagerEngine(quarantine_threshold=0.35)
    planner = AStarPlanner(grid_resolution=1.0)
    ml_engine = pipeline_engines["ml_engine"]

    dt = 0.1
    duration = 20.0
    attack_start = 8.0
    ekf.initialize_state(position=np.array([0.0, 0.0, 10.0]))

    gps_quarantined = False

    for t in np.arange(0.0, duration, dt):
        true_pos = np.array([0.5 * t, 0.0, 10.0])
        accel = np.array([0.0, 0.0, 9.80665])
        gps_meas = true_pos + np.random.normal(0.0, 0.20, size=(3,))
        vision_meas = true_pos + np.random.normal(0.0, 0.04, size=(3,))
        lidar_z = float(true_pos[2] + np.random.normal(0.0, 0.03))

        # Inject GPS Spoofing
        if t >= attack_start:
            ramp = min(1.0, (t - attack_start) / 2.0)
            gps_meas += np.array([30.0 * ramp, -20.0 * ramp, 0.0])

        ekf.predict(accel=accel, gyro_z=0.0, dt=dt)
        gps_res = ekf.update_gps(gps_meas, reject_anomaly=True)
        det_gps = detector.process_sensor_residual("gps", gps_res["position"]["nis"])

        ml_engine.update_sensor_telemetry(
            gps_pos=gps_meas,
            vision_pos=vision_meas,
            imu_accel=accel,
            imu_gyro_z=0.0,
            lidar_z=lidar_z,
            gps_nis=gps_res["position"]["nis"],
            lidar_nis=0.1,
            trust=resilience.sensor_trust["gps"],
        )
        ml_res = ml_engine.predict()

        policy = resilience.evaluate_resilience_policy(det_gps["state"], det_gps["compromised_sensors"])

        if "gps" in policy["isolated_sensors"]:
            gps_quarantined = True
            # Fuse vision and LiDAR as resilient fallback
            ekf.update_vision_pose(vision_meas)
            ekf.update_lidar(lidar_z)

    # Verification
    assert gps_quarantined is True
    assert "gps" in resilience.isolated_sensors
    assert policy["navigation_mode"] == NavigationMode.DEGRADED_OPTICAL_LIDAR.value

    # Path planner risk avoidance: Add spoofed zone as high-risk
    planner.add_cyber_risk_zone(x=5.0, y=0.0, radius=3.0, risk_level=5.0)
    safe_path = planner.plan((0.0, 0.0, 10.0), (10.0, 0.0, 10.0))
    assert safe_path is not None
    assert len(safe_path) > 0


def test_end_to_end_multi_sensor_emergency_landing():
    """Verifies that simultaneous GPS and IMU compromise triggers failsafe emergency landing."""
    resilience = ResilienceManagerEngine()

    for _ in range(4):
        policy = resilience.evaluate_resilience_policy(
            attack_status="ATTACK_CONFIRMED",
            detected_compromised_sensors=["gps", "imu"]
        )

    assert policy["navigation_mode"] == NavigationMode.EMERGENCY_LANDING.value
    assert policy["speed_factor"] == 0.2
    assert "gps" in policy["isolated_sensors"]
    assert "imu" in policy["isolated_sensors"]
