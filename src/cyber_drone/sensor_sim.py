"""
Sensor Simulation Module
Generates realistic sensor streams from ground truth kinematic state with:
- GPS (pos sigma 0.5m, vel sigma 0.1m/s)
- IMU (accel sigma 0.05m/s^2 with constant bias, gyro noise 0.01 rad/s)
- Barometer (alt sigma 0.3m)
- Vision Odometry (pos sigma 0.1m with slow linear drift 0.02 m/s)
- LiDAR (range noise 0.03m, altitude measurement)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class SensorSimulator:
    def __init__(self, config: Optional[Dict[str, Any]] = None, seed: int = 42):
        self.config = config or {}
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # GPS noise parameters
        gps_cfg = self.config.get("gps", {})
        self.gps_pos_sigma = float(gps_cfg.get("position_noise_sigma", 0.5))
        self.gps_vel_sigma = float(gps_cfg.get("velocity_noise_sigma", 0.1))

        # IMU noise parameters
        imu_cfg = self.config.get("imu", {})
        self.imu_accel_sigma = float(imu_cfg.get("accel_noise_sigma", 0.05))
        self.imu_gyro_sigma = float(imu_cfg.get("gyro_noise_sigma", 0.01))
        self.imu_accel_bias = np.array(imu_cfg.get("accel_bias_constant", [0.02, -0.01, 0.01]))

        # Baro noise parameters
        baro_cfg = self.config.get("barometer", {})
        self.baro_sigma = float(baro_cfg.get("altitude_noise_sigma", 0.3))

        # Vision noise & drift parameters
        vis_cfg = self.config.get("vision", {})
        self.vision_sigma = float(vis_cfg.get("position_noise_sigma", 0.1))
        self.vision_drift_rate = float(vis_cfg.get("drift_rate_per_sec", 0.02))
        # Random drift direction vector in 2D
        drift_angle = self.rng.uniform(0, 2 * np.pi)
        self.drift_dir = np.array([np.cos(drift_angle), np.sin(drift_angle), 0.0])

        # LiDAR noise parameters
        lidar_cfg = self.config.get("lidar", {})
        self.lidar_sigma = float(lidar_cfg.get("noise_sigma", 0.03))

    def generate_gps(self, truth_pos: np.ndarray, truth_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pos_noise = self.rng.normal(0.0, self.gps_pos_sigma, size=3)
        vel_noise = self.rng.normal(0.0, self.gps_vel_sigma, size=3)
        return truth_pos + pos_noise, truth_vel + vel_noise

    def generate_imu(self, truth_accel_world: np.ndarray, truth_yaw: float, truth_yaw_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        # Convert world acceleration + gravity into body frame
        c, s = np.cos(truth_yaw), np.sin(truth_yaw)
        R_world_to_body = np.array([
            [c,  s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        a_gravity_world = truth_accel_world + np.array([0.0, 0.0, 9.81])
        a_body = R_world_to_body @ a_gravity_world

        # Add bias and Gaussian noise
        accel_noise = self.rng.normal(0.0, self.imu_accel_sigma, size=3)
        measured_accel = a_body + self.imu_accel_bias + accel_noise

        gyro_noise = self.rng.normal(0.0, self.imu_gyro_sigma, size=3)
        measured_gyro = np.array([0.0, 0.0, truth_yaw_rate]) + gyro_noise
        return measured_accel, measured_gyro

    def generate_baro(self, truth_altitude: float) -> float:
        noise = self.rng.normal(0.0, self.baro_sigma)
        return float(truth_altitude + noise)

    def generate_vision(self, current_time: float, truth_pos: np.ndarray, truth_yaw: float, truth_vel: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        noise = self.rng.normal(0.0, self.vision_sigma, size=3)
        # Slow linear drift accumulation
        drift = self.drift_dir * (self.vision_drift_rate * current_time)
        measured_pos = truth_pos + noise + drift
        vel_base = truth_vel if truth_vel is not None else np.zeros(3)
        measured_vel = vel_base + self.rng.normal(0.0, 0.05, size=3)
        measured_yaw = float(truth_yaw + self.rng.normal(0.0, 0.02))
        return measured_pos, measured_vel, measured_yaw

    def generate_lidar(self, truth_altitude: float, obstacles: list, drone_pos: np.ndarray) -> Tuple[np.ndarray, float]:
        alt_noise = self.rng.normal(0.0, self.lidar_sigma)
        measured_alt = float(max(0.0, truth_altitude + alt_noise))

        # 360-degree scan simulation (sparse 36 beams for efficiency)
        num_beams = 36
        angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)
        ranges = np.ones(num_beams) * 30.0 # max range

        for i, ang in enumerate(angles):
            ray_dir = np.array([np.cos(ang), np.sin(ang)])
            for obs in obstacles:
                ox, oy = obs["x"], obs["y"]
                r = obs.get("radius", 3.0)
                d_vec = np.array([ox - drone_pos[0], oy - drone_pos[1]])
                proj = np.dot(d_vec, ray_dir)
                if proj > 0:
                    perp_dist = np.linalg.norm(d_vec - proj * ray_dir)
                    if perp_dist <= r:
                        hit_dist = proj - np.sqrt(max(0.0, r**2 - perp_dist**2))
                        if 0.2 <= hit_dist < ranges[i]:
                            ranges[i] = hit_dist

        ranges += self.rng.normal(0.0, self.lidar_sigma, size=num_beams)
        ranges = np.clip(ranges, 0.2, 30.0)
        return ranges, measured_alt
