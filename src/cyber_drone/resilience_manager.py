"""
Resilience and Containment Manager (Core of Deliverable 2)
State Machine:
NORMAL -> SUSPICIOUS -> ATTACK_CONFIRMED -> CONTAINMENT -> SAFE_NAVIGATION -> RECOVERY -> RECOVERED
Handles sensor gating, covariance inflation, estimator switching, quarantine logging, and safe recovery.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class ResilienceManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.state = "NORMAL"
        self.active_estimator = "main" # "main" or "trusted"
        self.navigation_mode = "normal_navigation"
        
        # Sensor trust scores (0.0 to 1.0)
        self.sensor_trust = {
            "gps": 1.0,
            "imu": 1.0,
            "baro": 1.0,
            "vision": 1.0,
            "lidar": 1.0
        }
        
        # Isolated / gated sensors
        self.isolated_sensors = set()
        
        # Quarantine log
        self.quarantine_log: List[Dict[str, Any]] = []
        
        # Reintegration tracking (sensor -> continuous agreement duration)
        self.reintegration_dwell_sec = float(self.config.get("containment", {}).get("reintegration_dwell_sec", 5.0))
        self.reintegration_ramp_sec = float(self.config.get("containment", {}).get("reintegration_ramp_sec", 3.0))
        self.sensor_agreement_start: Dict[str, Optional[float]] = {s: None for s in self.sensor_trust}
        self.reintegration_start_time: Dict[str, Optional[float]] = {s: None for s in self.sensor_trust}
        
        # Covariance inflation
        self.gps_covariance_inflation = 1.0
        
        # Events history
        self.events: List[Dict[str, Any]] = []

    def record_event(self, event_type: str, current_time: float, details: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": round(current_time, 3),
            "event": event_type,
            "state": self.state,
            "active_estimator": self.active_estimator,
            "details": details or {}
        }
        self.events.append(entry)

    def log_quarantine(self, sensor: str, current_time: float, data: Any, reason: str):
        entry = {
            "timestamp": round(current_time, 3),
            "sensor": sensor,
            "data": data,
            "reason": reason
        }
        self.quarantine_log.append(entry)

    def update(
        self,
        current_time: float,
        detection_result: Dict[str, Any],
        main_ekf,
        trusted_ekf
    ) -> Dict[str, Any]:
        """
        Executes resilience management logic given current detection diagnostics.
        Adjusts state machine, sensor gates, covariance inflation, and active estimator.
        """
        status = detection_result["status"]
        compromised = detection_result["compromised_sensors"]
        risk_score = detection_result["risk_score"]
        attack_type = detection_result.get("attack_type", "none")

        prev_state = self.state

        # State Machine Transitions
        if self.state == "NORMAL":
            if status == "SUSPICIOUS":
                self.state = "SUSPICIOUS"
                self.record_event("SUSPICIOUS_ANOMALY_DETECTED", current_time, {"compromised": compromised})
            elif status == "ATTACK_CONFIRMED":
                self.state = "ATTACK_CONFIRMED"
                self.record_event("ATTACK_CONFIRMED", current_time, {"type": attack_type, "compromised": compromised})

        elif self.state == "SUSPICIOUS":
            if status == "ATTACK_CONFIRMED":
                self.state = "ATTACK_CONFIRMED"
                self.record_event("ATTACK_CONFIRMED", current_time, {"type": attack_type, "compromised": compromised})
            elif status == "NORMAL" and risk_score < 0.1:
                self.state = "NORMAL"
                self.record_event("RETURN_TO_NORMAL", current_time)

        if self.state == "ATTACK_CONFIRMED":
            # Transition to CONTAINMENT immediately to isolate threats
            self.state = "CONTAINMENT"
            self.record_event("CONTAINMENT_INITIATED", current_time, {"isolating": compromised})

        if self.state == "CONTAINMENT":
            # Once isolated and stabilized, enter SAFE_NAVIGATION
            self.state = "SAFE_NAVIGATION"
            self.record_event("SAFE_NAVIGATION_ACTIVE", current_time, {"estimator": self.active_estimator})

        elif self.state in ["SAFE_NAVIGATION", "CONTAINMENT"]:
            # If all checks return to clean (no active compromised sensors)
            if len(compromised) == 0 and status == "NORMAL":
                self.state = "RECOVERY"
                self.record_event("RECOVERY_EVALUATION_STARTED", current_time)

        elif self.state == "RECOVERY":
            if len(compromised) > 0 or status != "NORMAL":
                # Relapse to containment if attack recurs
                self.state = "CONTAINMENT"
                self.record_event("RECOVERY_ABORTED_ANOMALY_RETURNED", current_time, {"compromised": compromised})
            else:
                # Check if all sensors have completed reintegration
                all_reintegrated = all(s not in self.isolated_sensors for s in self.sensor_trust)
                if all_reintegrated and all(self.sensor_trust[s] >= 0.95 for s in self.sensor_trust):
                    self.state = "RECOVERED"
                    self.record_event("SYSTEM_FULLY_RECOVERED", current_time)

        elif self.state == "RECOVERED":
            if status == "NORMAL":
                self.state = "NORMAL"
                self.active_estimator = "main"

        # Apply Actions According to State and Compromised Sensors
        replan_required = False

        # 1. Suspicious: lower trust, inflate covariance
        if self.state == "SUSPICIOUS":
            for s in compromised:
                self.sensor_trust[s] = max(0.4, self.sensor_trust[s] - 0.15)
            if "gps" in compromised:
                self.gps_covariance_inflation = 10.0
                main_ekf.set_gps_inflation(self.gps_covariance_inflation)

        # 2. Containment and Safe Navigation: strict isolation and switching
        elif self.state in ["ATTACK_CONFIRMED", "CONTAINMENT", "SAFE_NAVIGATION"]:
            # Multi-sensor check
            if len(compromised) >= 2 or len(self.isolated_sensors) >= 2:
                self.navigation_mode = "multi_sensor_containment"
            
            # GPS Spoofing / Disruption Containment
            if "gps" in compromised or "gps" in self.isolated_sensors:
                self.gps_covariance_inflation = 10.0
                if "gps" not in self.isolated_sensors:
                    self.isolated_sensors.add("gps")
                    self.sensor_trust["gps"] = 0.0
                    main_ekf.set_sensor_enabled("gps", False)
                    self.active_estimator = "trusted"
                    self.navigation_mode = "safe_local_navigation"
                    replan_required = True
                    self.record_event("GPS_ISOLATED_AND_SWITCHED_TO_TRUSTED", current_time)

            # IMU Bias Containment
            if "imu" in compromised:
                if "imu" not in self.isolated_sensors:
                    self.isolated_sensors.add("imu")
                    self.sensor_trust["imu"] = 0.0
                    main_ekf.set_sensor_enabled("imu", False)
                    trusted_ekf.set_sensor_enabled("imu", False)
                    self.navigation_mode = "vision_lidar_fallback"
                    replan_required = True
                    self.record_event("IMU_REJECTED_VISION_LIDAR_FALLBACK", current_time)

            # LiDAR Corruption Containment
            if "lidar" in compromised:
                if "lidar" not in self.isolated_sensors:
                    self.isolated_sensors.add("lidar")
                    self.sensor_trust["lidar"] = 0.0
                    main_ekf.set_sensor_enabled("lidar", False)
                    trusted_ekf.set_sensor_enabled("lidar", False)
                    self.navigation_mode = "safe_inflation_navigation"
                    replan_required = True
                    self.record_event("LIDAR_ISOLATED_OBSTACLE_INFLATION_INCREASED", current_time)

        # 3. Gradual Reintegration During Recovery (Never instant!)
        elif self.state == "RECOVERY":
            for s in list(self.isolated_sensors):
                if s not in compromised:
                    if self.sensor_agreement_start[s] is None:
                        self.sensor_agreement_start[s] = current_time

                    dwell = current_time - self.sensor_agreement_start[s]
                    if dwell >= self.reintegration_dwell_sec:
                        # 5s continuous agreement met, begin 3s covariance ramp
                        if self.reintegration_start_time[s] is None:
                            self.reintegration_start_time[s] = current_time
                            self.record_event(f"BEGIN_REINTEGRATION_RAMP_{s.upper()}", current_time)

                        ramp_elapsed = current_time - self.reintegration_start_time[s]
                        alpha = min(1.0, ramp_elapsed / self.reintegration_ramp_sec)
                        
                        # Trust ramps 0.0 -> 1.0
                        self.sensor_trust[s] = alpha
                        
                        if s == "gps":
                            # Covariance inflation ramps 10.0 -> 1.0
                            self.gps_covariance_inflation = 10.0 - 9.0 * alpha
                            main_ekf.set_gps_inflation(self.gps_covariance_inflation)
                            main_ekf.set_sensor_enabled("gps", True)

                        if alpha >= 1.0:
                            self.isolated_sensors.remove(s)
                            self.sensor_trust[s] = 1.0
                            self.sensor_agreement_start[s] = None
                            self.reintegration_start_time[s] = None
                            self.record_event(f"SENSOR_FULLY_REINTEGRATED_{s.upper()}", current_time)
                            if len(self.isolated_sensors) == 0:
                                self.active_estimator = "main"
                                self.navigation_mode = "normal_navigation"
                                self.state = "RECOVERED"
                                self.record_event("SYSTEM_FULLY_RECOVERED", current_time)
                else:
                    # Agreement broken, reset dwell
                    self.sensor_agreement_start[s] = None
                    self.reintegration_start_time[s] = None

        return {
            "state": self.state,
            "active_estimator": self.active_estimator,
            "navigation_mode": self.navigation_mode,
            "sensor_trust": dict(self.sensor_trust),
            "isolated_sensors": list(self.isolated_sensors),
            "gps_covariance_inflation": self.gps_covariance_inflation,
            "replan_required": replan_required,
            "events": self.events
        }
