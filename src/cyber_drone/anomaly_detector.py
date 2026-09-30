"""
Residual-Based Anomaly and Cyberattack Detector
Implements multi-sensor statistical checks and hysteresis confirmation.
Strictly decoupled from ground truth (uses only sensor feeds and filter innovations).
"""

import collections
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class AnomalyDetector:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Thresholds
        thresh = self.config.get("thresholds", {})
        self.th_mahalanobis = float(thresh.get("gps_mahalanobis_dist", 4.5))
        self.th_nis = float(thresh.get("gps_nis_chi2_99", 11.345))
        self.th_rate_drop_fraction = float(thresh.get("watchdog_rate_drop_fraction", 0.70))
        self.th_max_age = float(thresh.get("watchdog_max_age_sec", 0.300))
        self.th_imu_vision_vel = float(thresh.get("imu_vision_vel_divergence", 0.8))
        self.th_lidar_baro_alt = float(thresh.get("lidar_baro_alt_diff", 1.5))

        # Hysteresis parameters
        hyst = self.config.get("hysteresis", {})
        self.confirm_n = int(hyst.get("confirm_n", 5))
        self.high_thresh = float(hyst.get("high_threshold", 0.75))
        self.low_thresh = float(hyst.get("low_threshold", 0.25))

        # Sliding window buffers
        self.window_gps_sec = 2.0
        self.gps_discrepancies = collections.deque()  # (timestamp, mahalanobis_d)
        
        # Sensor message timestamp trackers for watchdogs
        self.sensor_timestamps: Dict[str, collections.deque] = {
            "gps": collections.deque(),
            "imu": collections.deque(),
            "baro": collections.deque(),
            "vision": collections.deque(),
            "lidar": collections.deque()
        }
        self.nominal_rates = {
            "gps": 10.0,
            "imu": 100.0,
            "baro": 20.0,
            "vision": 20.0,
            "lidar": 10.0
        }

        # Anomaly scoring and hysteresis state
        self.suspicious_count = 0
        self.attack_status = "NORMAL" # NORMAL, SUSPICIOUS, ATTACK_CONFIRMED
        self.latest_anomaly_type: Optional[str] = None
        self.compromised_sensors: List[str] = []
        self.confidence: float = 0.0
        self.risk_score: float = 0.0
        self.first_detected_time: Optional[float] = None

    def record_sensor_message(self, sensor: str, timestamp: float):
        if sensor in self.sensor_timestamps:
            buf = self.sensor_timestamps[sensor]
            buf.append(timestamp)
            # Retain only last 3 seconds
            while buf and timestamp - buf[0] > 3.0:
                buf.popleft()

    def check_all(
        self,
        current_time: float,
        gps_pos: Optional[np.ndarray],
        trusted_pos: np.ndarray,
        trusted_cov_pos: np.ndarray,
        main_gps_nis: Optional[float],
        imu_integrated_vel: np.ndarray,
        vision_vel: np.ndarray,
        lidar_alt: Optional[float],
        baro_alt: Optional[float]
    ) -> Dict[str, Any]:
        """
        Executes all residual and statistical tests.
        Returns diagnosis dict with scores and attack classification.
        """
        active_checks = {}
        compromised = set()
        instant_risk_scores = []

        # 1. GPS vs Trusted Filter Position Check (Mahalanobis Distance)
        gps_dist_score = 0.0
        if gps_pos is not None:
            diff = gps_pos - trusted_pos
            # Combined covariance: GPS measurement cov + filter position cov
            combined_cov = np.eye(3) * (0.5 ** 2) + trusted_cov_pos
            try:
                cov_inv = np.linalg.inv(combined_cov)
                d_m = float(np.sqrt(np.maximum(0.0, diff.T @ cov_inv @ diff)))
            except np.linalg.LinAlgError:
                d_m = float(np.linalg.norm(diff) / 0.5)

            self.gps_discrepancies.append((current_time, d_m))
            while self.gps_discrepancies and current_time - self.gps_discrepancies[0][0] > self.window_gps_sec:
                self.gps_discrepancies.popleft()

            mean_dm = np.mean([item[1] for item in self.gps_discrepancies]) if self.gps_discrepancies else d_m
            gps_dist_score = min(1.0, mean_dm / self.th_mahalanobis)
            active_checks["gps_mahalanobis"] = {"value": mean_dm, "threshold": self.th_mahalanobis}
            if mean_dm >= self.th_mahalanobis:
                compromised.add("gps")
                instant_risk_scores.append(min(1.0, mean_dm / (self.th_mahalanobis * 1.5)))

        # 2. GPS NIS Check (Instant Innovation Jump)
        if main_gps_nis is not None:
            active_checks["gps_nis"] = {"value": main_gps_nis, "threshold": self.th_nis}
            if main_gps_nis > self.th_nis:
                compromised.add("gps")
                instant_risk_scores.append(min(1.0, main_gps_nis / (self.th_nis * 2.0)))

        # 3. Watchdog Check (Rate drop, delay, blackout)
        watchdog_failed = False
        gps_buf = self.sensor_timestamps["gps"]
        if gps_buf:
            age = current_time - gps_buf[-1]
            # Calculate rate in past 1.0 second
            recent_count = sum(1 for t in gps_buf if current_time - t <= 1.0)
            expected_count = self.nominal_rates["gps"] * 1.0
            rate_ratio = recent_count / expected_count
            active_checks["gps_watchdog"] = {"age": age, "rate_ratio": rate_ratio}
            rate_failed = (current_time >= 1.0 and rate_ratio < self.th_rate_drop_fraction)
            age_failed = (age > self.th_max_age)
            if rate_failed or age_failed:
                compromised.add("gps")
                watchdog_failed = True
                instant_risk_scores.append(0.85)
        else:
            if current_time > 1.0: # If running for >1s and no GPS received
                compromised.add("gps")
                watchdog_failed = True
                instant_risk_scores.append(0.95)

        # 4. IMU vs Vision Velocity Divergence
        vel_div = float(np.linalg.norm(imu_integrated_vel - vision_vel))
        active_checks["imu_vision_div"] = {"value": vel_div, "threshold": self.th_imu_vision_vel}
        if vel_div > self.th_imu_vision_vel:
            compromised.add("imu")
            instant_risk_scores.append(min(1.0, vel_div / (self.th_imu_vision_vel * 2.0)))

        # 5. LiDAR vs Barometer Altitude Check
        if lidar_alt is not None and baro_alt is not None:
            alt_diff = abs(lidar_alt - baro_alt)
            active_checks["lidar_baro_alt"] = {"value": alt_diff, "threshold": self.th_lidar_baro_alt}
            if alt_diff > self.th_lidar_baro_alt:
                compromised.add("lidar")
                instant_risk_scores.append(min(1.0, alt_diff / (self.th_lidar_baro_alt * 2.0)))

        # Overall composite raw score
        raw_score = max(instant_risk_scores) if instant_risk_scores else 0.0

        # Hysteresis confirmation logic
        if raw_score >= self.high_thresh or len(compromised) > 0:
            self.suspicious_count += 1
        elif raw_score < self.low_thresh and len(compromised) == 0:
            self.suspicious_count = max(0, self.suspicious_count - 1)

        # State decision
        if self.suspicious_count >= self.confirm_n:
            self.attack_status = "ATTACK_CONFIRMED"
            if self.first_detected_time is None:
                self.first_detected_time = current_time
        elif self.suspicious_count >= 3:
            self.attack_status = "SUSPICIOUS"
        else:
            self.attack_status = "NORMAL"

        # Determine attack classification
        if "gps" in compromised and len(compromised) == 1:
            if watchdog_failed:
                self.latest_anomaly_type = "communication_disruption"
            elif main_gps_nis is not None and main_gps_nis > self.th_nis * 2.5:
                self.latest_anomaly_type = "gps_jump"
            else:
                self.latest_anomaly_type = "gps_drift"
        elif "imu" in compromised and len(compromised) == 1:
            self.latest_anomaly_type = "imu_bias"
        elif "lidar" in compromised and len(compromised) == 1:
            self.latest_anomaly_type = "lidar_corruption"
        elif len(compromised) > 1:
            self.latest_anomaly_type = "combined_attack"
        else:
            if self.attack_status == "NORMAL":
                self.latest_anomaly_type = "none"

        self.compromised_sensors = sorted(list(compromised))
        self.risk_score = round(float(raw_score), 3)
        self.confidence = round(min(1.0, self.suspicious_count / float(self.confirm_n)), 3)

        return {
            "status": self.attack_status,
            "attack_type": self.latest_anomaly_type,
            "compromised_sensors": self.compromised_sensors,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "suspicious_count": self.suspicious_count,
            "first_detected_time": self.first_detected_time,
            "checks": active_checks
        }
