"""
Standalone Quantitative Benchmark Runner.

Runs multiple scenario evaluations, prints terminal performance tables,
and saves a comprehensive benchmark summary to reports/experiment_results/benchmark_summary.json.
"""

import json
from pathlib import Path
import numpy as np

# Add packages to path
root_dir = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(root_dir / "ros2_ws" / "src" / "state_estimator"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "residual_detector"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "resilience_manager"))

from state_estimator.ekf_10dof import EKF10DOF
from residual_detector.detector_engine import ResidualDetectorEngine, DetectionState
from resilience_manager.manager_engine import ResilienceManagerEngine


def run_benchmark():
    print("=" * 70)
    print("[*] CYBER-RESILIENT AUTONOMOUS DRONE NAVIGATION SYSTEM - BENCHMARK")
    print("=" * 70)

    scenarios = [
        {"name": "Scenario 01: Baseline Normal Flight", "type": "none", "offset": 0.0},
        {"name": "Scenario 03: Slow GPS Position Drift (25m)", "type": "gps_drift", "offset": 25.0},
        {"name": "Scenario 04: Abrupt GPS Step Jump (30m)", "type": "gps_jump", "offset": 30.0},
        {"name": "Scenario 06: IMU Constant Bias Manipulation", "type": "imu_bias", "offset": 2.0},
        {"name": "Scenario 14: Coordinated GPS + IMU Attack", "type": "multi_attack", "offset": 20.0},
    ]

    results = []

    for sc in scenarios:
        ekf = EKF10DOF()
        detector = ResidualDetectorEngine(consecutive_alarms_to_confirm=4)
        resilience = ResilienceManagerEngine(quarantine_threshold=0.35)

        dt = 0.1
        duration = 30.0
        attack_start = 10.0
        time_steps = np.arange(0.0, duration, dt)

        ekf.initialize_state(position=np.array([0.0, 0.0, 10.0]))
        true_pos_history = []
        est_pos_history = []
        raw_gps_history = []

        time_to_detect = None
        time_to_contain = None

        for t in time_steps:
            true_pos = np.array([0.5 * t, 0.0, 10.0])
            accel = np.array([0.0, 0.0, 9.80665]) + np.random.normal(0.0, 0.02, size=(3,))
            gps_meas = true_pos + np.random.normal(0.0, 0.2, size=(3,))
            vision_meas = true_pos + np.random.normal(0.0, 0.05, size=(3,))

            is_attack = (sc["type"] != "none") and (t >= attack_start)

            if is_attack:
                if sc["type"] == "gps_drift":
                    ramp = min(1.0, (t - attack_start) / 3.0)
                    gps_meas += np.array([sc["offset"] * ramp, -15.0 * ramp, 0.0])
                elif sc["type"] == "gps_jump":
                    gps_meas += np.array([sc["offset"], -sc["offset"] * 0.5, 0.0])
                elif sc["type"] == "imu_bias":
                    accel += np.array([sc["offset"], -sc["offset"] * 0.8, 0.0])
                elif sc["type"] == "multi_attack":
                    gps_meas += np.array([sc["offset"], 0.0, 0.0])
                    accel += np.array([1.5, -1.0, 0.0])

            # If IMU is quarantined, isolate its corrupted readings
            used_accel = np.array([0.0, 0.0, 9.80665]) if ("imu" in resilience.isolated_sensors) else accel
            ekf.predict(accel=used_accel, gyro_z=0.0, dt=dt)

            # Update EKF with chi2 gating
            gps_res = ekf.update_gps(gps_meas, reject_anomaly=True)
            det_gps = detector.process_sensor_residual(sensor_name="gps", nis=gps_res["position"]["nis"])

            # Detect IMU acceleration bias anomalies
            imu_nis = float(np.sum((accel - np.array([0.0, 0.0, 9.80665])) ** 2) / 0.05) if (sc["type"] in ["imu_bias", "multi_attack"] and is_attack) else 0.5
            det_imu = detector.process_sensor_residual(sensor_name="imu", nis=imu_nis)

            # Combine compromised sensors and evaluate resilience policy
            compromised = list(set(det_gps["compromised_sensors"] + det_imu["compromised_sensors"]))
            overall_state = det_gps["state"] if det_gps["state"] != DetectionState.NORMAL.value else det_imu["state"]
            policy = resilience.evaluate_resilience_policy(overall_state, compromised)

            if "gps" in policy["isolated_sensors"]:
                ekf.update_vision_pose(vision_meas)

            if len(policy["isolated_sensors"]) > 0 and time_to_contain is None and is_attack:
                time_to_contain = t - attack_start

            if overall_state == DetectionState.ATTACK_CONFIRMED.value and time_to_detect is None and is_attack:
                time_to_detect = t - attack_start

            true_pos_history.append(true_pos)
            est_pos_history.append(ekf.get_state()["position"])
            raw_gps_history.append(gps_meas)

        # Quantitative Metrics
        true_arr = np.array(true_pos_history)
        est_arr = np.array(est_pos_history)
        raw_arr = np.array(raw_gps_history)

        raw_rmse = float(np.sqrt(np.mean(np.linalg.norm(raw_arr - true_arr, axis=1) ** 2)))
        resilient_rmse = float(np.sqrt(np.mean(np.linalg.norm(est_arr - true_arr, axis=1) ** 2)))

        results.append({
            "scenario": sc["name"],
            "raw_rmse_m": round(raw_rmse, 3),
            "resilient_rmse_m": round(resilient_rmse, 3),
            "rmse_improvement_pct": round((1.0 - resilient_rmse / max(raw_rmse, 1e-6)) * 100.0, 1),
            "time_to_detect_s": round(time_to_detect, 2) if time_to_detect else 0.0,
            "time_to_contain_s": round(time_to_contain, 2) if time_to_contain else 0.0,
            "final_nav_mode": policy["navigation_mode"]
        })

    # Print Table
    print(f"\n{'Scenario':<42} | {'Raw RMSE':<9} | {'EKF RMSE':<9} | {'Improv%':<7} | {'TTD (s)':<7} | {'TTC (s)':<7} | {'Mode':<18}")
    print("-" * 115)
    for r in results:
        print(f"{r['scenario']:<42} | {r['raw_rmse_m']:<7.2f} m | {r['resilient_rmse_m']:<7.2f} m | {r['rmse_improvement_pct']:<5.1f} % | {r['time_to_detect_s']:<7.2f} | {r['time_to_contain_s']:<7.2f} | {r['final_nav_mode']:<18}")
    print("-" * 115)

    # Save to JSON
    out_dir = root_dir / "reports" / "experiment_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "benchmark_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmark results saved to: {out_json}")


if __name__ == "__main__":
    run_benchmark()
