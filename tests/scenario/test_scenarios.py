"""
Full-Pipeline Scenario Benchmark Tests.

Simulates end-to-end multi-sensor flight missions against cyber-attacks,
validating Time to Detect (TTD), Time to Contain (TTC), RMSE improvements,
and autonomous resilience state transitions.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add packages to path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir / "ros2_ws" / "src" / "state_estimator"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "residual_detector"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "resilience_manager"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "path_planner"))

from state_estimator.ekf_10dof import EKF10DOF
from residual_detector.detector_engine import ResidualDetectorEngine, DetectionState
from resilience_manager.manager_engine import ResilienceManagerEngine, NavigationMode
from path_planner.astar_planner import AStarPlanner


def test_scenario_01_baseline_normal_flight():
    """Scenario 01: Attack-free normal waypoint flight."""
    ekf = EKF10DOF()
    detector = ResidualDetectorEngine()
    resilience = ResilienceManagerEngine()

    dt = 0.1
    duration = 30.0
    time_steps = np.arange(0.0, duration, dt)

    # Initialize at true position at t=0: [0.0, 5.0, 10.0]
    ekf.initialize_state(position=np.array([0.0, 5.0, 10.0]))
    true_positions = []
    est_positions = []

    for t in time_steps:
        # True waypoint flight
        true_pos = np.array([5.0 * np.sin(0.2 * t), 5.0 * np.cos(0.2 * t), 10.0])
        accel = np.array([0.0, 0.0, 9.80665]) + np.random.normal(0.0, 0.02, size=(3,))
        gps_meas = true_pos + np.random.normal(0.0, 0.25, size=(3,))

        # Pipeline execution
        ekf.predict(accel=accel, gyro_z=0.0, dt=dt)
        res = ekf.update_gps(gps_meas, reject_anomaly=False)
        det = detector.process_sensor_residual(sensor_name="gps", nis=res["position"]["nis"])
        policy = resilience.evaluate_resilience_policy(det["state"], det["compromised_sensors"])

        # Record
        true_positions.append(true_pos)
        est_positions.append(ekf.get_state()["position"])

        # In normal flight, state should stay NORMAL with 0 false alarms
        assert det["state"] == DetectionState.NORMAL.value
        assert policy["navigation_mode"] == NavigationMode.NORMAL_MISSION.value

    # Evaluate RMSE
    err = np.linalg.norm(np.array(est_positions) - np.array(true_positions), axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    assert rmse < 0.45  # Sub-half-meter tracking accuracy


def test_scenario_03_slow_gps_drift_spoofing():
    """Scenario 03: 15-meter gradual GPS drift spoofing starting at t=10s."""
    ekf = EKF10DOF()
    detector = ResidualDetectorEngine(consecutive_alarms_to_confirm=4)
    resilience = ResilienceManagerEngine(quarantine_threshold=0.35)

    dt = 0.1
    duration = 25.0
    attack_start = 10.0
    attack_offset = 25.0

    ekf.initialize_state(position=np.array([0.0, 0.0, 10.0]))
    time_to_detect = None
    time_to_contain = None

    for t in np.arange(0.0, duration, dt):
        true_pos = np.array([t * 0.5, 0.0, 10.0])
        accel = np.array([0.0, 0.0, 9.80665])
        gps_meas = true_pos + np.random.normal(0.0, 0.2, size=(3,))
        vision_meas = true_pos + np.random.normal(0.0, 0.05, size=(3,))

        # Inject GPS ramp spoofing
        if t >= attack_start:
            ramp = min(1.0, (t - attack_start) / 3.0)
            gps_meas += np.array([attack_offset * ramp, -15.0 * ramp, 0.0])

        ekf.predict(accel=accel, gyro_z=0.0, dt=dt)
        # Enable chi-square gating so anomalous drift is rejected and innovation is exposed
        gps_res = ekf.update_gps(gps_meas, reject_anomaly=True)
        det = detector.process_sensor_residual(sensor_name="gps", nis=gps_res["position"]["nis"])
        policy = resilience.evaluate_resilience_policy(det["state"], det["compromised_sensors"])

        # If GPS quarantined, fuse vision
        if "gps" in policy["isolated_sensors"]:
            ekf.update_vision_pose(vision_meas)
            if time_to_contain is None:
                time_to_contain = t - attack_start

        if det["state"] == DetectionState.ATTACK_CONFIRMED.value and time_to_detect is None:
            time_to_detect = t - attack_start

    # Verify cyber-resilience metrics
    assert time_to_detect is not None
    assert time_to_detect < 2.5  # Detected in under 2.5 seconds
    assert time_to_contain is not None
    assert time_to_contain < 3.5  # Fully isolated in under 3.5 seconds
    assert "gps" in resilience.isolated_sensors
    assert policy["navigation_mode"] == NavigationMode.DEGRADED_OPTICAL_LIDAR.value


def test_scenario_04_sudden_gps_jump_spoofing():
    """Scenario 04: Abrupt 30-meter GPS step jump offset."""
    ekf = EKF10DOF()
    detector = ResidualDetectorEngine()

    ekf.initialize_state(position=np.array([10.0, 5.0, 10.0]))
    true_pos = np.array([10.0, 5.0, 10.0])

    # Settle
    for _ in range(15):
        ekf.predict(accel=np.array([0.0, 0.0, 9.80665]), gyro_z=0.0, dt=0.1)
        ekf.update_gps(true_pos)

    # Abrupt 30m spoofed jump
    spoofed_pos = true_pos + np.array([30.0, -20.0, 0.0])
    res = ekf.update_gps(spoofed_pos, reject_anomaly=True)

    # Must immediately trigger chi-square gating and NIS alarm
    assert res["position"]["is_gated"] is True
    assert res["position"]["nis"] > 50.0  # Massive statistical deviation

    # State must be protected from jumping 30 meters
    est_pos = ekf.get_state()["position"]
    assert np.linalg.norm(est_pos - true_pos) < 1.0


def test_scenario_14_coordinated_multi_sensor_attack():
    """Scenario 14: Multi-sensor attack compromising both GPS and IMU."""
    resilience = ResilienceManagerEngine()

    # Repeated multi-sensor anomalies
    for _ in range(3):
        policy = resilience.evaluate_resilience_policy(
            attack_status="ATTACK_CONFIRMED",
            detected_compromised_sensors=["gps", "imu"]
        )

    # Under multi-sensor compromise, system must enter EMERGENCY_LANDING
    assert policy["navigation_mode"] == NavigationMode.EMERGENCY_LANDING.value
    assert policy["speed_factor"] == 0.2


def test_scenario_06_imu_manipulation_containment():
    """Scenario 06: IMU constant bias manipulation leading to HOLD_POSITION."""
    detector = ResidualDetectorEngine(consecutive_alarms_to_confirm=3)
    resilience = ResilienceManagerEngine()

    # Inject 4 high NIS alerts for IMU
    for _ in range(4):
        det = detector.process_sensor_residual("imu", nis=35.0)

    assert det["state"] == DetectionState.ATTACK_CONFIRMED.value
    assert "imu" in det["compromised_sensors"]

    policy = resilience.evaluate_resilience_policy(det["state"], det["compromised_sensors"])
    for _ in range(3):
        policy = resilience.evaluate_resilience_policy(det["state"], det["compromised_sensors"])

    assert "imu" in policy["isolated_sensors"]
    assert policy["navigation_mode"] == NavigationMode.HOLD_POSITION.value
    assert policy["speed_factor"] == 0.0


# ---------------------------------------------------------------------
# Parameterized Test for All 15 YAML Scenarios (Section 9)
# ---------------------------------------------------------------------
import yaml

scenarios_dir = root_dir / "simulation" / "scenarios"
scenario_files = sorted(scenarios_dir.glob("scenario_*.yaml"))


@pytest.mark.parametrize("scenario_path", scenario_files, ids=[p.stem for p in scenario_files])
def test_all_15_scenarios_pipeline(scenario_path):
    """Executes each of the 15 standardized YAML scenario files through the full CPS pipeline."""
    with open(scenario_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(int(cfg.get("random_seed", 42)))

    ekf = EKF10DOF()
    detector = ResidualDetectorEngine(
        nis_warning_threshold=11.34,
        nis_alarm_threshold=16.27,
        consecutive_alarms_to_confirm=4,
        recovery_samples_to_clear=15
    )
    resilience = ResilienceManagerEngine(quarantine_threshold=0.35, recovery_threshold=0.85)

    duration = 25.0
    dt = 0.1
    time_steps = np.arange(0.0, duration, dt)

    attack_type = cfg.get("attack_type", "none")
    a_start = float(cfg.get("attack_start_time", 10.0))
    a_dur = float(cfg.get("attack_duration", 15.0))
    is_attack_run = (attack_type != "none")

    ekf.initialize_state(position=np.array([0.0, 0.0, 10.0]))
    confirmed_attack = False

    for t in time_steps:
        true_pos = np.array([t * 0.6, 2.0 * np.sin(0.2 * t), 10.0])
        accel = np.array([0.0, 0.0, 9.80665]) + np.random.normal(0.0, 0.015, size=(3,))
        gps_pos = true_pos + np.random.normal(0.0, 0.2, size=(3,))
        vision_pos = true_pos + np.random.normal(0.0, 0.05, size=(3,))
        lidar_z = 10.0 + float(np.random.normal(0.0, 0.03))

        # Inject attack if within window
        target_sensor = cfg.get("attack_parameters", {}).get("target_sensor", "")
        params_str = str(cfg.get("attack_parameters", {}))
        mag_heading_error = 0.0

        if is_attack_run and (a_start <= t < (a_start + a_dur)):
            if "gps" in attack_type or target_sensor == "gps" or "gps" in params_str:
                ramp = min(1.0, (t - a_start) / 3.0)
                gps_pos += np.array([25.0 * ramp, -12.0 * ramp, 0.0])

            if target_sensor == "imu" or "imu" in params_str or "IMU" in cfg.get("name", ""):
                accel += np.array([3.0, 1.5, 0.0])

            if target_sensor == "lidar" or "lidar" in params_str or "LiDAR" in cfg.get("name", ""):
                lidar_z -= 6.0

            if target_sensor == "magnetometer" or "magnetometer" in params_str or "Magnetometer" in cfg.get("name", ""):
                mag_heading_error = 45.0

            if attack_type == "communication_disruption":
                gps_pos += np.random.normal(0.0, 6.0, size=(3,))

        # Prediction
        used_accel = np.array([0.0, 0.0, 9.80665]) if "imu" in resilience.isolated_sensors else accel
        ekf.predict(accel=used_accel, gyro_z=0.0, dt=dt)

        # Update
        gps_res = ekf.update_gps(gps_pos, reject_anomaly=("gps" in resilience.isolated_sensors))
        lidar_res = ekf.update_lidar(lidar_z, reject_anomaly=("lidar" in resilience.isolated_sensors))

        # Calculate IMU kinematic innovation residual
        expected_accel_norm = 9.80665
        imu_accel_diff = abs(float(np.linalg.norm(accel)) - expected_accel_norm)
        imu_nis = float((imu_accel_diff / 0.15) ** 2)

        # Calculate Magnetometer heading innovation residual
        mag_nis = float((mag_heading_error / 5.0) ** 2) if mag_heading_error > 0 else 1.0

        det_gps = detector.process_sensor_residual("gps", gps_res["position"]["nis"])
        det_lidar = detector.process_sensor_residual("lidar", lidar_res["nis"])
        det_imu = detector.process_sensor_residual("imu", imu_nis)
        det_mag = detector.process_sensor_residual("magnetometer", mag_nis)

        # Aggregate confirmed compromised sensors
        all_compromised = list(set(
            det_gps["compromised_sensors"] +
            det_lidar["compromised_sensors"] +
            det_imu["compromised_sensors"] +
            det_mag["compromised_sensors"]
        ))
        highest_state = DetectionState.NORMAL.value
        for d in [det_gps, det_lidar, det_imu, det_mag]:
            if d["state"] == DetectionState.ATTACK_CONFIRMED.value:
                highest_state = DetectionState.ATTACK_CONFIRMED.value
                confirmed_attack = True
                break
            elif d["state"] == DetectionState.SUSPICIOUS.value and highest_state == DetectionState.NORMAL.value:
                highest_state = DetectionState.SUSPICIOUS.value

        policy = resilience.evaluate_resilience_policy(highest_state, all_compromised)

        # Fuse vision when GPS is isolated
        if "gps" in policy["isolated_sensors"]:
            ekf.update_vision_pose(vision_pos)

    # Assertions based on scenario type
    if not is_attack_run:
        assert not confirmed_attack, f"{cfg['name']} raised false alarm"
        assert policy["navigation_mode"] == NavigationMode.NORMAL_MISSION.value
    else:
        # Attack scenarios must detect anomaly or contain
        assert confirmed_attack or (policy["navigation_mode"] != NavigationMode.NORMAL_MISSION.value), \
            f"{cfg['name']} failed to detect or contain attack"


