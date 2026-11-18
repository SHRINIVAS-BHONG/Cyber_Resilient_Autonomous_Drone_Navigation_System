"""
Full 16-State / 15-Error-State Quaternion Error-State Kalman Filter (ESKF).

State Vector (16 states):
  - Position p: 3x1 [px, py, pz] in ENU frame [m]
  - Velocity v: 3x1 [vx, vy, vz] in ENU frame [m/s]
  - Quaternion q: 4x1 [qw, qx, qy, qz] (body to ENU rotation)
  - Accelerometer Bias ba: 3x1 [bax, bay, baz] [m/s^2]
  - Gyroscope Bias bw: 3x1 [bwx, bwy, bwz] [rad/s]

Error State Vector delta_x (15 states):
  [delta_p (3), delta_v (3), delta_theta (3), delta_ba (3), delta_bw (3)]
Covariance Matrix P (15x15) propagated via Joseph-stabilized Riccati equation.
"""

from typing import Dict, Optional, Tuple
import numpy as np


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Computes 3x3 skew-symmetric cross-product matrix of a 3-element vector."""
    v = v.flatten()
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Converts unit quaternion [qw, qx, qy, qz] to 3x3 rotation matrix R (body to nav)."""
    qw, qx, qy, qz = q.flatten()
    return np.array([
        [1.0 - 2.0 * (qy ** 2 + qz ** 2), 2.0 * (qx * qy - qw * qz),       2.0 * (qx * qz + qw * qy)],
        [2.0 * (qx * qy + qw * qz),       1.0 - 2.0 * (qx ** 2 + qz ** 2), 2.0 * (qy * qz - qw * qx)],
        [2.0 * (qx * qz - qw * qy),       2.0 * (qy * qz + qw * qx),       1.0 - 2.0 * (qx ** 2 + qy ** 2)]
    ], dtype=np.float64)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiplies two quaternions [qw, qx, qy, qz]."""
    w1, x1, y1, z1 = q1.flatten()
    w2, x2, y2, z2 = q2.flatten()
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=np.float64)


class ESKF16State:
    """
    16-State Quaternion Error-State Kalman Filter with Joseph-stabilized Riccati update
    and statistical Chi-Square innovation gating for autonomous cyber resilience.
    """

    def __init__(
        self,
        q_pos: float = 0.005,
        q_vel: float = 0.05,
        q_rot: float = 0.001,
        q_acc_bias: float = 0.0005,
        q_gyro_bias: float = 0.0001,
        r_gps_pos: float = 0.25,
        r_gps_vel: float = 0.01,
        r_lidar_z: float = 0.0009,
        r_vision_pos: float = 0.0025,
        chi2_gate_3d: float = 16.27,
        chi2_gate_1d: float = 10.83,
    ):
        self.dim_error = 15

        # Nominal States
        self.p = np.zeros(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Unit quaternion [qw, qx, qy, qz]
        self.ba = np.zeros(3, dtype=np.float64)
        self.bw = np.zeros(3, dtype=np.float64)

        # 15x15 Error Covariance Matrix P
        self.P = np.eye(self.dim_error, dtype=np.float64)
        self.P[0:3, 0:3] *= 0.1     # pos variance
        self.P[3:6, 3:6] *= 0.1     # vel variance
        self.P[6:9, 6:9] *= 0.02    # attitude error variance
        self.P[9:12, 9:12] *= 0.01  # accel bias variance
        self.P[12:15, 12:15] *= 0.005  # gyro bias variance

        # Process Noise Matrix Q (15x15)
        self.Q = np.eye(self.dim_error, dtype=np.float64)
        self.Q[0:3, 0:3] *= q_pos
        self.Q[3:6, 3:6] *= q_vel
        self.Q[6:9, 6:9] *= q_rot
        self.Q[9:12, 9:12] *= q_acc_bias
        self.Q[12:15, 12:15] *= q_gyro_bias

        # Measurement Covariances
        self.R_gps_pos = np.eye(3, dtype=np.float64) * r_gps_pos
        self.R_gps_vel = np.eye(3, dtype=np.float64) * r_gps_vel
        self.R_lidar_z = np.array([[r_lidar_z]], dtype=np.float64)
        self.R_vision_pos = np.eye(3, dtype=np.float64) * r_vision_pos

        self.chi2_gate_3d = chi2_gate_3d
        self.chi2_gate_1d = chi2_gate_1d
        self.gravity = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

    def initialize_state(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        yaw: Optional[float] = None
    ) -> None:
        """Initializes position, velocity, and orientation quaternion."""
        if position is not None:
            self.p = np.asarray(position, dtype=np.float64).flatten()
            self.P[0:3, 0:3] = np.eye(3) * 0.3
        if velocity is not None:
            self.v = np.asarray(velocity, dtype=np.float64).flatten()
            self.P[3:6, 3:6] = np.eye(3) * 0.1
        if yaw is not None:
            # Set yaw rotation around Z axis
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            self.q = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
            self.P[6:9, 6:9] = np.eye(3) * 0.02

    def get_euler_angles(self) -> Tuple[float, float, float]:
        """Returns (roll, pitch, yaw) in radians."""
        qw, qx, qy, qz = self.q
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = np.sign(sinp) * (np.pi / 2.0)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return float(roll), float(pitch), float(yaw)

    def predict(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        """
        Propagates the nominal state using strapdown IMU equations and
        propagates the 15x15 error covariance matrix P.
        accel: 3D body accelerometer measurements [ax, ay, az]
        gyro: 3D body angular velocity measurements [gx, gy, gz]
        dt: sampling period [s]
        """
        if dt <= 0.0:
            return

        accel = np.asarray(accel, dtype=np.float64).flatten()
        gyro = np.asarray(gyro, dtype=np.float64).flatten()

        # Unbiased measurements
        a_unbiased = accel - self.ba
        w_unbiased = gyro - self.bw

        # Current rotation matrix (body to nav)
        R = quaternion_to_rotation_matrix(self.q)

        # Accelerometer specific force in ENU frame
        accel_nav = R @ a_unbiased + self.gravity

        # 1. Propagate Nominal Position & Velocity
        self.p += self.v * dt + 0.5 * accel_nav * (dt ** 2)
        self.v += accel_nav * dt

        # 2. Propagate Nominal Quaternion (Zero-order hold on angular rate)
        angle = np.linalg.norm(w_unbiased) * dt
        if angle > 1e-10:
            axis = w_unbiased / np.linalg.norm(w_unbiased)
            delta_q = np.array([
                np.cos(angle * 0.5),
                axis[0] * np.sin(angle * 0.5),
                axis[1] * np.sin(angle * 0.5),
                axis[2] * np.sin(angle * 0.5)
            ])
            self.q = quaternion_multiply(self.q, delta_q)
        # Normalize quaternion
        self.q /= np.linalg.norm(self.q)

        # 3. Compute Discrete Error Transition Matrix F_d (15x15)
        F_d = np.eye(self.dim_error, dtype=np.float64)
        F_d[0:3, 3:6] = np.eye(3) * dt

        # d(v_nav)/d(theta) = -R * [a_unbiased]_x * dt
        F_d[3:6, 6:9] = -R @ skew_symmetric(a_unbiased) * dt
        # d(v_nav)/d(ba) = -R * dt
        F_d[3:6, 9:12] = -R * dt

        # d(theta)/d(theta) = I - [w_unbiased]_x * dt
        F_d[6:9, 6:9] = np.eye(3) - skew_symmetric(w_unbiased) * dt
        # d(theta)/d(bw) = -I * dt
        F_d[6:9, 12:15] = -np.eye(3) * dt

        # 4. Propagate Covariance
        self.P = F_d @ self.P @ F_d.T + self.Q * dt

    def _update_error_state(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R_cov: np.ndarray,
        gate_threshold: float,
        reject_anomaly: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Computes innovation, Chi-Square NIS, and updates the nominal state via error injection.
        Employs Joseph-stabilized Riccati formulation: P = (I-KH)P(I-KH)' + KRK'.
        """
        z = np.asarray(z, dtype=np.float64).flatten()
        # Innovation y = z - h(x)
        # For direct linear measurements, h(x) is H @ error_state_pred (which is 0)
        y = z  # caller passes pre-subtracted residual (z_meas - z_pred)

        # Innovation Covariance S (m x m)
        S = H @ self.P @ H.T + R_cov
        S_inv = np.linalg.pinv(S)

        # Normalized Innovation Squared (NIS)
        nis = float((y.T @ S_inv @ y).item())
        is_gated = nis > gate_threshold

        diag_S = np.diag(S)
        diag_S = np.where(diag_S > 1e-12, diag_S, 1e-12)
        normalized_residual = y / np.sqrt(diag_S)

        if reject_anomaly and is_gated:
            return y, normalized_residual, nis, True

        # Kalman Gain (15 x m)
        K = self.P @ H.T @ S_inv

        # Compute error state delta_x (15 x 1)
        delta_x = K @ y

        # --- Error State Injection into Nominal States ---
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]

        # Orientation error angle vector delta_theta
        delta_theta = delta_x[6:9]
        dq = np.array([1.0, 0.5 * delta_theta[0], 0.5 * delta_theta[1], 0.5 * delta_theta[2]])
        dq /= np.linalg.norm(dq)
        self.q = quaternion_multiply(self.q, dq)
        self.q /= np.linalg.norm(self.q)

        # Biases
        self.ba += delta_x[9:12]
        self.bw += delta_x[12:15]

        # --- Joseph Stabilized Covariance Update ---
        I_KH = np.eye(self.dim_error, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_cov @ K.T

        return y, normalized_residual, nis, is_gated

    def update_gps(
        self,
        pos_meas: np.ndarray,
        vel_meas: Optional[np.ndarray] = None,
        reject_anomaly: bool = False
    ) -> Dict[str, any]:
        """Updates with GPS 3D position and optional velocity."""
        pos_meas = np.asarray(pos_meas, dtype=np.float64).flatten()
        pos_residual = pos_meas - self.p

        H_pos = np.zeros((3, self.dim_error), dtype=np.float64)
        H_pos[0:3, 0:3] = np.eye(3)

        innov_p, norm_p, nis_p, gated_p = self._update_error_state(
            pos_residual, H_pos, self.R_gps_pos, self.chi2_gate_3d, reject_anomaly
        )

        vel_res = None
        if vel_meas is not None:
            vel_meas = np.asarray(vel_meas, dtype=np.float64).flatten()
            vel_residual = vel_meas - self.v
            H_vel = np.zeros((3, self.dim_error), dtype=np.float64)
            H_vel[0:3, 3:6] = np.eye(3)

            innov_v, norm_v, nis_v, gated_v = self._update_error_state(
                vel_residual, H_vel, self.R_gps_vel, self.chi2_gate_3d, reject_anomaly
            )
            vel_res = {
                "innovation": innov_v,
                "normalized_residual": norm_v,
                "nis": nis_v,
                "is_gated": gated_v
            }

        return {
            "position": {
                "innovation": innov_p,
                "normalized_residual": norm_p,
                "nis": nis_p,
                "is_gated": gated_p
            },
            "velocity": vel_res
        }

    def update_lidar(self, lidar_alt: float, reject_anomaly: bool = False) -> Dict[str, any]:
        """Updates with 1D ground LiDAR distance."""
        z_residual = np.array([float(lidar_alt) - self.p[2]])
        H_lidar = np.zeros((1, self.dim_error), dtype=np.float64)
        H_lidar[0, 2] = 1.0

        innov, norm_res, nis, is_gated = self._update_error_state(
            z_residual, H_lidar, self.R_lidar_z, self.chi2_gate_1d, reject_anomaly
        )
        return {
            "innovation": float(innov[0]),
            "normalized_residual": float(norm_res[0]),
            "nis": nis,
            "is_gated": is_gated
        }

    def update_vision_pose(self, vision_pos: np.ndarray, reject_anomaly: bool = False) -> Dict[str, any]:
        """Updates with secondary Visual Odometry 3D position."""
        vision_pos = np.asarray(vision_pos, dtype=np.float64).flatten()
        pos_residual = vision_pos - self.p

        H_vis = np.zeros((3, self.dim_error), dtype=np.float64)
        H_vis[0:3, 0:3] = np.eye(3)

        innov, norm_res, nis, is_gated = self._update_error_state(
            pos_residual, H_vis, self.R_vision_pos, self.chi2_gate_3d, reject_anomaly
        )
        return {
            "innovation": innov,
            "normalized_residual": norm_res,
            "nis": nis,
            "is_gated": is_gated
        }

    def get_state(self) -> Dict[str, any]:
        """Returns the full current state and covariance diagonals."""
        roll, pitch, yaw = self.get_euler_angles()
        return {
            "position": self.p.copy(),
            "velocity": self.v.copy(),
            "quaternion": self.q.copy(),
            "euler_angles_rad": (roll, pitch, yaw),
            "euler_angles_deg": (float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))),
            "accel_bias": self.ba.copy(),
            "gyro_bias": self.bw.copy(),
            "covariance_diag": np.diag(self.P).copy(),
            "pos_std": np.sqrt(np.maximum(0.0, np.diag(self.P)[0:3])),
            "vel_std": np.sqrt(np.maximum(0.0, np.diag(self.P)[3:6])),
            "att_std_deg": np.degrees(np.sqrt(np.maximum(0.0, np.diag(self.P)[6:9])))
        }
