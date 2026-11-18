"""
Flight Log & Telemetry Replay Engine.

Supports loading and streaming real-world UAV flight logs (CSV / PX4 ULog exported telemetry)
or high-fidelity ground truth logs through the cyber-resilience pipeline. Enables repeatable
benchmarking of real hardware flights under simulated cyber-physical attacks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class TelemetryFrame:
    timestamp: float
    true_pos_enu: np.ndarray       # 3D [px, py, pz]
    true_vel_enu: np.ndarray       # 3D [vx, vy, vz]
    imu_accel: np.ndarray          # 3D [ax, ay, az] (specific force + gravity)
    imu_gyro: np.ndarray           # 3D [gx, gy, gz]
    gps_pos_meas: np.ndarray       # 3D raw GPS measurement
    gps_vel_meas: np.ndarray       # 3D raw GPS velocity
    lidar_alt_meas: float          # 1D ground LiDAR distance
    vision_pos_meas: np.ndarray    # 3D secondary visual odometry


class FlightLogReplayer:
    """
    Loads, processes, and replays UAV telemetry logs through estimation & defense pipelines.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.frames: List[TelemetryFrame] = []
        self.total_duration_s = 0.0

    def load_from_dataframe(self, df: pd.DataFrame) -> int:
        """Parses a DataFrame containing standard telemetry columns."""
        self.frames.clear()

        required = ["timestamp", "pos_x", "pos_y", "pos_z"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Telemetry log missing required column: '{col}'")

        # Extract optional fields with sensible defaults
        has_vel = all(c in df.columns for c in ["vel_x", "vel_y", "vel_z"])
        has_acc = all(c in df.columns for c in ["accel_x", "accel_y", "accel_z"])
        has_gyro = all(c in df.columns for c in ["gyro_x", "gyro_y", "gyro_z"])

        for _, row in df.iterrows():
            t = float(row["timestamp"])
            pos = np.array([float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])])
            vel = np.array([float(row["vel_x"]), float(row["vel_y"]), float(row["vel_z"])]) if has_vel else np.zeros(3)
            acc = np.array([float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"])]) if has_acc else np.array([0.0, 0.0, 9.80665])
            gyro = np.array([float(row["gyro_x"]), float(row["gyro_y"]), float(row["gyro_z"])]) if has_gyro else np.zeros(3)

            # Simulated noisy sensor readings from ground truth
            gps_pos = pos + np.random.normal(0.0, 0.25, size=3)
            gps_vel = vel + np.random.normal(0.0, 0.05, size=3)
            lidar_z = float(pos[2] + np.random.normal(0.0, 0.03))
            vision_pos = pos + np.random.normal(0.0, 0.08, size=3)

            self.frames.append(TelemetryFrame(
                timestamp=t,
                true_pos_enu=pos,
                true_vel_enu=vel,
                imu_accel=acc,
                imu_gyro=gyro,
                gps_pos_meas=gps_pos,
                gps_vel_meas=gps_vel,
                lidar_alt_meas=lidar_z,
                vision_pos_meas=vision_pos
            ))

        if self.frames:
            self.total_duration_s = self.frames[-1].timestamp - self.frames[0].timestamp
        return len(self.frames)

    def load_from_csv(self, file_path: Union[str, Path]) -> int:
        """Loads telemetry from a CSV file."""
        df = pd.read_csv(file_path)
        return self.load_from_dataframe(df)

    def generate_synthetic_mission(
        self,
        duration_s: float = 30.0,
        waypoints: Optional[List[np.ndarray]] = None,
        speed_ms: float = 4.0
    ) -> int:
        """Generates a continuous realistic 3D quadcopter waypoint mission log."""
        if waypoints is None:
            waypoints = [
                np.array([0.0, 0.0, 15.0]),
                np.array([80.0, 0.0, 15.0]),
                np.array([120.0, 60.0, 18.0]),
                np.array([0.0, 100.0, 15.0]),
                np.array([0.0, 0.0, 15.0])
            ]

        self.frames.clear()
        time_steps = np.arange(0.0, duration_s, self.dt)
        curr_pos = np.copy(waypoints[0])
        curr_wp_idx = 1
        rng = np.random.RandomState(42)

        for t in time_steps:
            target_wp = waypoints[curr_wp_idx]
            diff = target_wp - curr_pos
            dist = float(np.linalg.norm(diff))

            if dist < 2.0:
                curr_wp_idx = (curr_wp_idx + 1) % len(waypoints)
                target_wp = waypoints[curr_wp_idx]
                diff = target_wp - curr_pos
                dist = float(np.linalg.norm(diff))

            unit_dir = diff / max(dist, 1e-4)
            vel = unit_dir * min(speed_ms, dist)
            curr_pos = curr_pos + vel * self.dt

            acc = np.array([0.0, 0.0, 9.80665]) + rng.normal(0.0, 0.02, size=3)
            gyro = np.array([0.0, 0.0, 0.0]) + rng.normal(0.0, 0.005, size=3)

            gps_pos = curr_pos + rng.normal(0.0, 0.20, size=3)
            gps_vel = vel + rng.normal(0.0, 0.04, size=3)
            lidar_z = float(curr_pos[2] + rng.normal(0.0, 0.02))
            vision_pos = curr_pos + rng.normal(0.0, 0.06, size=3)

            self.frames.append(TelemetryFrame(
                timestamp=float(t),
                true_pos_enu=np.copy(curr_pos),
                true_vel_enu=np.copy(vel),
                imu_accel=acc,
                imu_gyro=gyro,
                gps_pos_meas=gps_pos,
                gps_vel_meas=gps_vel,
                lidar_alt_meas=lidar_z,
                vision_pos_meas=vision_pos
            ))

        self.total_duration_s = duration_s
        return len(self.frames)

    def stream(self) -> Generator[TelemetryFrame, None, None]:
        """Yields sequential telemetry frames representing real-time playback."""
        for frame in self.frames:
            yield frame

    def execute_playback_with_defense(
        self,
        ekf_instance,
        detector_instance,
        resilience_instance,
        attack_config: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Executes full pipeline playback over the loaded telemetry log with an optional attack.
        Returns comprehensive performance metrics and tracking histories.
        """
        true_traj, est_traj, raw_gps_traj = [], [], []
        nav_modes = []
        is_attack_active = False
        detected_time = None
        contained_time = None

        atk_type = attack_config.get("type", "none") if attack_config else "none"
        atk_start = attack_config.get("start_time", 10.0) if attack_config else 10.0
        atk_mag = attack_config.get("magnitude", 25.0) if attack_config else 25.0

        if self.frames and hasattr(ekf_instance, "initialize_state"):
            ekf_instance.initialize_state(
                position=self.frames[0].true_pos_enu,
                velocity=self.frames[0].true_vel_enu
            )

        for frame in self.stream():
            t = frame.timestamp
            true_p = frame.true_pos_enu
            accel = np.copy(frame.imu_accel)
            gyro_z = float(frame.imu_gyro[2])
            gps_p = np.copy(frame.gps_pos_meas)
            gps_v = np.copy(frame.gps_vel_meas)
            lidar_z = frame.lidar_alt_meas
            vision_p = np.copy(frame.vision_pos_meas)

            # Attack Injection Logic
            if atk_type != "none" and t >= atk_start:
                is_attack_active = True
                if atk_type == "gps_spoofing":
                    gps_p[0] += atk_mag
                    gps_p[1] += atk_mag * 0.5
                elif atk_type == "gps_replay":
                    # Replay lag: inject position from 5 seconds ago
                    lag_idx = max(0, int((t - 5.0) / self.dt))
                    if lag_idx < len(self.frames):
                        gps_p = np.copy(self.frames[lag_idx].true_pos_enu)
                elif atk_type == "imu_bias":
                    accel[0] += atk_mag
                elif atk_type == "lidar_corruption":
                    lidar_z += atk_mag

            # EKF Prediction
            ekf_instance.predict(accel=accel, gyro_z=gyro_z, dt=self.dt)

            # EKF Measurement Update with Anomaly Rejection
            gps_res = ekf_instance.update_gps(gps_p, vel_meas=gps_v, reject_anomaly=True)
            gps_nis = gps_res["position"]["nis"]

            # Residual Detection
            det_result = detector_instance.process_sensor_residual("gps", gps_nis)
            policy = resilience_instance.evaluate_resilience_policy(
                det_result["state"], det_result["compromised_sensors"]
            )

            # Sensor switching & failover
            if "gps" in policy["isolated_sensors"]:
                ekf_instance.update_vision_pose(vision_p)
                if contained_time is None and is_attack_active:
                    contained_time = t - atk_start

            if det_result["state"] == "ATTACK_CONFIRMED" and detected_time is None and is_attack_active:
                detected_time = t - atk_start

            # LiDAR Update
            ekf_instance.update_lidar(lidar_z, reject_anomaly=True)

            # Record States
            est_p = ekf_instance.get_state()["position"]
            true_traj.append(true_p)
            est_traj.append(est_p)
            raw_gps_traj.append(gps_p)
            nav_modes.append(policy["navigation_mode"])

        true_traj = np.array(true_traj)
        est_traj = np.array(est_traj)
        raw_gps_traj = np.array(raw_gps_traj)

        resilient_err = np.linalg.norm(est_traj - true_traj, axis=1)
        raw_err = np.linalg.norm(raw_gps_traj - true_traj, axis=1)

        resilient_rmse = float(np.sqrt(np.mean(resilient_err ** 2)))
        raw_rmse = float(np.sqrt(np.mean(raw_err ** 2)))
        error_reduction = max(0.0, (1.0 - resilient_rmse / max(raw_rmse, 1e-4)) * 100.0)

        return {
            "total_frames": len(self.frames),
            "resilient_rmse_m": round(resilient_rmse, 3),
            "raw_rmse_m": round(raw_rmse, 3),
            "error_reduction_pct": round(error_reduction, 2),
            "ttd_s": round(detected_time, 2) if detected_time is not None else None,
            "ttc_s": round(contained_time, 2) if contained_time is not None else None,
            "final_nav_mode": nav_modes[-1] if nav_modes else "NORMAL_MISSION",
            "mission_success": resilient_rmse < 1.0
        }
