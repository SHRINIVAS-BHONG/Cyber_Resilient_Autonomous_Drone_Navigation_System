"""
Extended Kalman Filter (EKF) for Drone Navigation State Estimation
10-State: [px, py, pz, vx, vy, vz, bax, bay, baz, yaw]
Supports Main Filter (GPS + multi-sensor) and Trusted Filter (No GPS).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class DroneEKF:
    def __init__(self, is_trusted: bool = False, config: Optional[Dict[str, Any]] = None):
        self.is_trusted = is_trusted
        self.config = config or {}

        # 10 State vector: [px, py, pz, vx, vy, vz, bax, bay, baz, yaw]
        self.x = np.zeros(10)
        self.P = np.eye(10) * 1.0
        self.P[0:3, 0:3] *= 1.0    # pos cov
        self.P[3:6, 3:6] *= 0.5    # vel cov
        self.P[6:9, 6:9] *= 0.1    # bias cov
        self.P[9, 9] = 0.1         # yaw cov

        # Process noise Q
        self.Q = np.zeros((10, 10))
        self.Q[0:3, 0:3] = np.eye(3) * 0.01
        self.Q[3:6, 3:6] = np.eye(3) * 0.05
        self.Q[6:9, 6:9] = np.eye(3) * 0.001
        self.Q[9, 9] = 0.005

        meas_cfg = self.config.get("measurement_noise", {})
        gps_pos_sigma = float(meas_cfg.get("gps_pos", 0.5))
        gps_vel_sigma = float(meas_cfg.get("gps_vel", 0.1))
        baro_sigma = float(meas_cfg.get("baro_alt", 0.3))
        vision_sigma = float(meas_cfg.get("vision_pos", 0.15))
        lidar_sigma = float(meas_cfg.get("lidar_alt", 0.05))

        # Measurement noise covariance defaults (R)
        # Main EKF prioritizes GPS; Trusted EKF (no GPS) prioritizes Vision
        self.R_gps_pos = np.eye(3) * (gps_pos_sigma ** 2) if not is_trusted else np.eye(3) * (0.5 ** 2)
        self.R_gps_vel = np.eye(3) * (gps_vel_sigma ** 2)
        self.R_baro = np.array([[baro_sigma ** 2]])
        self.R_vision_pos = np.eye(3) * (2.5 ** 2) if not is_trusted else np.eye(3) * (vision_sigma ** 2)
        self.R_lidar_alt = np.array([[lidar_sigma ** 2]])

        # Covariance inflation multipliers
        self.gps_inflation = 1.0

        # Enable flags per sensor
        self.enabled_sensors = {
            "gps": not is_trusted,  # Trusted filter NEVER enables GPS
            "imu": True,
            "baro": True,
            "vision": True,
            "lidar": True
        }

        # Last innovation diagnostics: sensor -> (residual, S, NIS)
        self.latest_diagnostics: Dict[str, Dict[str, Any]] = {}
        self.last_timestamp = 0.0

    def reset(self, initial_pos: np.ndarray = np.array([0.0, 0.0, 0.0]), yaw: float = 0.0):
        self.x = np.zeros(10)
        self.x[0:3] = initial_pos
        self.x[9] = yaw
        self.P = np.eye(10) * 1.0
        self.latest_diagnostics.clear()

    def set_sensor_enabled(self, sensor: str, enabled: bool):
        if sensor == "gps" and self.is_trusted:
            self.enabled_sensors["gps"] = False
            return
        if sensor in self.enabled_sensors:
            self.enabled_sensors[sensor] = enabled

    def set_gps_inflation(self, factor: float):
        self.gps_inflation = max(1.0, factor)

    def predict(self, dt: float, accel_body: np.ndarray, gyro_z: float):
        if dt <= 0.0 or not self.enabled_sensors["imu"]:
            return

        psi = self.x[9]
        c, s = np.cos(psi), np.sin(psi)
        R_body_to_world = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Bias-corrected body acceleration
        b_a = self.x[6:9]
        a_corrected_body = accel_body - b_a
        a_world = R_body_to_world @ a_corrected_body - np.array([0.0, 0.0, 9.81])

        # State propagation (Euler integration)
        self.x[0:3] += self.x[3:6] * dt + 0.5 * a_world * (dt ** 2)
        self.x[3:6] += a_world * dt
        self.x[9] = (self.x[9] + gyro_z * dt + np.pi) % (2 * np.pi) - np.pi

        # State transition Jacobian F
        F = np.eye(10)
        F[0:3, 3:6] = np.eye(3) * dt
        # d(a_world)/d(psi)
        da_dpsi = np.array([
            -s * a_corrected_body[0] - c * a_corrected_body[1],
             c * a_corrected_body[0] - s * a_corrected_body[1],
            0.0
        ])
        F[0:3, 9] = 0.5 * da_dpsi * (dt ** 2)
        F[3:6, 9] = da_dpsi * dt
        # d(a_world)/d(b_a) = -R_body_to_world
        F[0:3, 6:9] = -0.5 * R_body_to_world * (dt ** 2)
        F[3:6, 6:9] = -R_body_to_world * dt

        # Covariance propagation with process noise scaled by dt
        Q_dt = self.Q * dt
        self.P = F @ self.P @ F.T + Q_dt
        self.P = 0.5 * (self.P + self.P.T)

    def _kalman_update(self, z: np.ndarray, z_pred: np.ndarray, H: np.ndarray, R: np.ndarray, sensor_name: str) -> Tuple[np.ndarray, np.ndarray, float]:
        r = z - z_pred
        S = H @ self.P @ H.T + R

        try:
            S_inv = np.linalg.inv(S)
            nis = float(r.T @ S_inv @ r)
            K = self.P @ H.T @ S_inv
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            nis = float(r.T @ S_inv @ r)
            K = self.P @ H.T @ S_inv

        # Joseph form update for numerical stability
        I_KH = np.eye(10) - K @ H
        self.x = self.x + K @ r
        # Wrap yaw
        self.x[9] = (self.x[9] + np.pi) % (2 * np.pi) - np.pi
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.latest_diagnostics[sensor_name] = {
            "residual": r,
            "S": S,
            "NIS": nis
        }
        return r, S, nis

    def update_gps(self, pos_xyz: np.ndarray, vel_xyz: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        if not self.enabled_sensors["gps"] or self.is_trusted:
            return None

        R = self.R_gps_pos * self.gps_inflation
        H = np.zeros((3, 10))
        H[0:3, 0:3] = np.eye(3)
        z_pred = self.x[0:3]
        r, S, nis = self._kalman_update(pos_xyz, z_pred, H, R, "gps")

        if vel_xyz is not None:
            H_v = np.zeros((3, 10))
            H_v[0:3, 3:6] = np.eye(3)
            R_v = self.R_gps_vel * self.gps_inflation
            self._kalman_update(vel_xyz, self.x[3:6], H_v, R_v, "gps_vel")

        return {"residual": r, "S": S, "NIS": nis}

    def update_baro(self, altitude_z: float) -> Optional[Dict[str, Any]]:
        if not self.enabled_sensors["baro"]:
            return None

        H = np.zeros((1, 10))
        H[0, 2] = 1.0
        z = np.array([altitude_z])
        z_pred = np.array([self.x[2]])
        r, S, nis = self._kalman_update(z, z_pred, H, self.R_baro, "baro")
        return {"residual": r, "S": S, "NIS": nis}

    def update_vision(self, pos_xyz: np.ndarray, yaw: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if not self.enabled_sensors["vision"]:
            return None

        H = np.zeros((3, 10))
        H[0:3, 0:3] = np.eye(3)
        r, S, nis = self._kalman_update(pos_xyz, self.x[0:3], H, self.R_vision_pos, "vision")

        if yaw is not None:
            H_yaw = np.zeros((1, 10))
            H_yaw[0, 9] = 1.0
            r_yaw = (yaw - self.x[9] + np.pi) % (2 * np.pi) - np.pi
            R_yaw = np.array([[0.05 ** 2]])
            self._kalman_update(np.array([yaw]), np.array([self.x[9]]), H_yaw, R_yaw, "vision_yaw")

        return {"residual": r, "S": S, "NIS": nis}

    def update_lidar_altitude(self, altitude_z: float) -> Optional[Dict[str, Any]]:
        if not self.enabled_sensors["lidar"]:
            return None

        H = np.zeros((1, 10))
        H[0, 2] = 1.0
        z = np.array([altitude_z])
        z_pred = np.array([self.x[2]])
        r, S, nis = self._kalman_update(z, z_pred, H, self.R_lidar_alt, "lidar_alt")
        return {"residual": r, "S": S, "NIS": nis}

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def accel_bias(self) -> np.ndarray:
        return self.x[6:9].copy()

    @property
    def yaw(self) -> float:
        return float(self.x[9])

    @property
    def position_covariance(self) -> np.ndarray:
        return self.P[0:3, 0:3].copy()
