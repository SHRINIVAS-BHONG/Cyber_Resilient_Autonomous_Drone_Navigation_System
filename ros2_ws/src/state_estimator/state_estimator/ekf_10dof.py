"""
Extended Kalman Filter (10-DOF) for Autonomous Drone State Estimation.

State Vector (10x1):
  x[0:3] = Position (px, py, pz) in NED/ENU frame [m]
  x[3:6] = Velocity (vx, vy, vz) [m/s]
  x[6:9] = Accelerometer Bias (bax, bay, baz) [m/s^2]
  x[9]   = Heading / Yaw angle (psi) [rad]
"""

from typing import Dict, Optional, Tuple
import numpy as np


class EKF10DOF:
    def __init__(
        self,
        q_pos: float = 0.01,
        q_vel: float = 0.1,
        q_bias: float = 0.001,
        q_yaw: float = 0.005,
        r_gps_pos: float = 0.25,
        r_gps_vel: float = 0.01,
        r_lidar_z: float = 0.0009,
        r_vision_pos: float = 0.0025,
        r_baro_z: float = 0.04,
        chi2_gate_3d: float = 16.27,  # 99.9% confidence threshold for 3-DOF
        chi2_gate_1d: float = 10.83,  # 99.9% confidence threshold for 1-DOF
    ):
        self.dim_x = 10

        # State vector [px, py, pz, vx, vy, vz, bax, bay, baz, yaw]
        self.x = np.zeros((self.dim_x, 1), dtype=np.float64)

        # State covariance matrix P (10x10)
        self.P = np.eye(self.dim_x, dtype=np.float64)
        self.P[0:3, 0:3] *= 0.1
        self.P[3:6, 3:6] *= 0.1
        self.P[6:9, 6:9] *= 0.01
        self.P[9, 9] = 0.05

        # Process noise covariance Q (10x10)
        self.Q = np.eye(self.dim_x, dtype=np.float64)
        self.Q[0:3, 0:3] *= q_pos
        self.Q[3:6, 3:6] *= q_vel
        self.Q[6:9, 6:9] *= q_bias
        self.Q[9, 9] = q_yaw

        # Measurement noise parameters
        self.R_gps_pos = np.eye(3, dtype=np.float64) * r_gps_pos
        self.R_gps_vel = np.eye(3, dtype=np.float64) * r_gps_vel
        self.R_lidar_z = np.array([[r_lidar_z]], dtype=np.float64)
        self.R_vision_pos = np.eye(3, dtype=np.float64) * r_vision_pos
        self.R_baro_z = np.array([[r_baro_z]], dtype=np.float64)

        # Chi-square gating thresholds
        self.chi2_gate_3d = chi2_gate_3d
        self.chi2_gate_1d = chi2_gate_1d

        # Gravity vector in ENU [m/s^2]
        self.gravity = np.array([[0.0], [0.0], [-9.80665]], dtype=np.float64)

    def initialize_state(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        yaw: Optional[float] = None
    ) -> None:
        """Initializes the filter state with initial sensor lock (e.g. initial GPS fix)."""
        if position is not None:
            self.x[0:3] = np.asarray(position, dtype=np.float64).reshape(3, 1)
            self.P[0:3, 0:3] = np.eye(3) * 0.5
        if velocity is not None:
            self.x[3:6] = np.asarray(velocity, dtype=np.float64).reshape(3, 1)
            self.P[3:6, 3:6] = np.eye(3) * 0.2
        if yaw is not None:
            self.x[9, 0] = float(yaw)
            self.P[9, 9] = 0.05

    def predict(self, accel: np.ndarray, gyro_z: float, dt: float) -> None:
        """
        IMU prediction step using body acceleration and yaw rate.
        accel: 3x1 body accelerometer measurements [ax, ay, az]
        gyro_z: yaw rate [rad/s]
        dt: sampling interval [s]
        """
        if dt <= 0.0:
            return

        accel = np.asarray(accel, dtype=np.float64).reshape(3, 1)

        yaw = self.x[9, 0]
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        # 2D Rotation matrix around Z-axis (body to nav frame)
        R_z = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)

        # Correct acceleration by estimated bias
        accel_unbiased = accel - self.x[6:9]
        accel_nav = R_z @ accel_unbiased + self.gravity

        # State propagation (strapdown kinematic integration)
        self.x[0:3] += self.x[3:6] * dt + 0.5 * accel_nav * (dt ** 2)
        self.x[3:6] += accel_nav * dt
        self.x[9, 0] = (self.x[9, 0] + gyro_z * dt + np.pi) % (2.0 * np.pi) - np.pi

        # State transition Jacobian F = d(f)/dx (10x10)
        F = np.eye(self.dim_x, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt

        # Partial derivative of accel_nav with respect to bias
        F[3:6, 6:9] = -R_z * dt

        # Partial derivative of accel_nav with respect to yaw
        d_Rz_dyaw = np.array([
            [-sin_y, -cos_y, 0.0],
            [ cos_y, -sin_y, 0.0],
            [ 0.0,    0.0,   0.0]
        ], dtype=np.float64)
        F[3:6, 9:10] = (d_Rz_dyaw @ accel_unbiased) * dt

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q * dt

    def _update(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        gate_threshold: float,
        reject_on_gate: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Generic Kalman update with innovation covariance and chi-square NIS calculation.
        Returns: (innovation, normalized_residual, NIS, is_gated)
        """
        z = np.asarray(z, dtype=np.float64)
        y = z - H @ self.x  # Innovation / residual

        # Innovation covariance S
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.pinv(S)

        # Normalized Innovation Squared (NIS)
        nis = float((y.T @ S_inv @ y).item())
        is_gated = nis > gate_threshold

        # Calculate normalized residual (component-wise z-scores)
        diag_S = np.diag(S)
        diag_S = np.where(diag_S > 1e-12, diag_S, 1e-12)
        normalized_residual = y.flatten() / np.sqrt(diag_S)

        # If gating rejection is enabled and anomaly detected, do NOT mutate state
        if reject_on_gate and is_gated:
            return y.flatten(), normalized_residual, nis, True

        # Kalman Gain
        K = self.P @ H.T @ S_inv

        # State and Covariance Update (Joseph form for numerical stability)
        self.x += K @ y
        I_KH = np.eye(self.dim_x, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        return y.flatten(), normalized_residual, nis, is_gated

    def update_gps(
        self,
        pos_meas: np.ndarray,
        vel_meas: Optional[np.ndarray] = None,
        reject_anomaly: bool = False
    ) -> Dict[str, any]:
        """Update with GPS 3D position and optional velocity."""
        pos_meas = np.asarray(pos_meas, dtype=np.float64).reshape(3, 1)

        # Position measurement matrix H_pos (3x10)
        H_pos = np.zeros((3, self.dim_x), dtype=np.float64)
        H_pos[0:3, 0:3] = np.eye(3)

        innov, norm_res, nis, is_gated = self._update(
            pos_meas, H_pos, self.R_gps_pos, self.chi2_gate_3d, reject_anomaly
        )

        vel_res = None
        if vel_meas is not None:
            vel_meas = np.asarray(vel_meas, dtype=np.float64).reshape(3, 1)
            H_vel = np.zeros((3, self.dim_x), dtype=np.float64)
            H_vel[0:3, 3:6] = np.eye(3)
            vel_innov, vel_norm_res, vel_nis, vel_gated = self._update(
                vel_meas, H_vel, self.R_gps_vel, self.chi2_gate_3d, reject_anomaly
            )
            vel_res = {
                "innovation": vel_innov,
                "normalized_residual": vel_norm_res,
                "nis": vel_nis,
                "is_gated": vel_gated
            }

        return {
            "position": {
                "innovation": innov,
                "normalized_residual": norm_res,
                "nis": nis,
                "is_gated": is_gated
            },
            "velocity": vel_res
        }

    def update_lidar(self, altitude_meas: float, reject_anomaly: bool = False) -> Dict[str, any]:
        """Update with 1D LiDAR altitude measurement."""
        z = np.array([[altitude_meas]], dtype=np.float64)
        H = np.zeros((1, self.dim_x), dtype=np.float64)
        H[0, 2] = 1.0  # Measures pz

        innov, norm_res, nis, is_gated = self._update(
            z, H, self.R_lidar_z, self.chi2_gate_1d, reject_anomaly
        )

        return {
            "innovation": innov,
            "normalized_residual": norm_res,
            "nis": nis,
            "is_gated": is_gated
        }

    def update_vision_pose(self, pos_meas: np.ndarray, reject_anomaly: bool = False) -> Dict[str, any]:
        """Update with Visual Odometry / Camera position."""
        pos_meas = np.asarray(pos_meas, dtype=np.float64).reshape(3, 1)
        H = np.zeros((3, self.dim_x), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        innov, norm_res, nis, is_gated = self._update(
            pos_meas, H, self.R_vision_pos, self.chi2_gate_3d, reject_anomaly
        )

        return {
            "innovation": innov,
            "normalized_residual": norm_res,
            "nis": nis,
            "is_gated": is_gated
        }

    def get_state(self) -> Dict[str, np.ndarray]:
        """Returns readable dictionary of current estimates and uncertainties."""
        return {
            "position": self.x[0:3].flatten(),
            "velocity": self.x[3:6].flatten(),
            "accel_bias": self.x[6:9].flatten(),
            "yaw": float(self.x[9, 0]),
            "position_std": np.sqrt(np.diag(self.P)[0:3]),
            "velocity_std": np.sqrt(np.diag(self.P)[3:6]),
        }
