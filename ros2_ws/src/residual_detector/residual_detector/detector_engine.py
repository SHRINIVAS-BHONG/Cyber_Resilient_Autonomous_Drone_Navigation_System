"""
Cyber Attack Residual Detection Engine.

Implements statistical hypothesis testing using Normalized Innovation Squared (NIS)
and a hysteresis-based finite state machine (FSM) to confirm navigation cyber attacks.
"""

from collections import deque
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class DetectionState(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    ATTACK_CONFIRMED = "ATTACK_CONFIRMED"
    CONTAINMENT = "CONTAINMENT"
    SAFE_NAVIGATION = "SAFE_NAVIGATION"
    RECOVERY = "RECOVERY"
    RECOVERED = "RECOVERED"


class ResidualDetectorEngine:
    def __init__(
        self,
        nis_warning_threshold: float = 11.34,  # Chi2 99% threshold for 3-DOF
        nis_alarm_threshold: float = 16.27,    # Chi2 99.9% threshold for 3-DOF
        window_size: int = 15,
        consecutive_alarms_to_confirm: int = 5,
        recovery_samples_to_clear: int = 20,
    ):
        self.nis_warning = nis_warning_threshold
        self.nis_alarm = nis_alarm_threshold
        self.window_size = window_size
        self.confirm_count = consecutive_alarms_to_confirm
        self.recovery_count = recovery_samples_to_clear

        # Sliding windows for sensor NIS values
        self.nis_history: Dict[str, deque] = {
            "gps": deque(maxlen=window_size),
            "imu": deque(maxlen=window_size),
            "lidar": deque(maxlen=window_size),
            "vision": deque(maxlen=window_size)
        }

        # State machine tracking per sensor
        self.suspicious_counters: Dict[str, int] = {k: 0 for k in self.nis_history}
        self.healthy_counters: Dict[str, int] = {k: 0 for k in self.nis_history}
        self.current_state: DetectionState = DetectionState.NORMAL
        self.compromised_sensors: List[str] = []

    def process_sensor_residual(
        self,
        sensor_name: str,
        nis: float,
        timestamp_delay_sec: float = 0.0
    ) -> Dict[str, any]:
        """
        Processes a single sensor innovation residual and updates the cyber detection state machine.
        """
        if sensor_name not in self.nis_history:
            self.nis_history[sensor_name] = deque(maxlen=self.window_size)
            self.suspicious_counters[sensor_name] = 0
            self.healthy_counters[sensor_name] = 0

        self.nis_history[sensor_name].append(nis)

        # Hysteresis update logic
        is_alarm = (nis > self.nis_alarm) or (timestamp_delay_sec > 0.25)
        is_warning = nis > self.nis_warning

        if is_alarm:
            self.suspicious_counters[sensor_name] += 1
            self.healthy_counters[sensor_name] = 0
        elif is_warning:
            # Slower increment on warnings
            self.suspicious_counters[sensor_name] += 0.5
            self.healthy_counters[sensor_name] = max(0, self.healthy_counters[sensor_name] - 1)
        else:
            self.suspicious_counters[sensor_name] = max(0.0, self.suspicious_counters[sensor_name] - 1.0)
            self.healthy_counters[sensor_name] += 1

        # Check for attack confirmation
        if self.suspicious_counters[sensor_name] >= self.confirm_count:
            if sensor_name not in self.compromised_sensors:
                self.compromised_sensors.append(sensor_name)

        # Check for recovery
        if self.healthy_counters[sensor_name] >= self.recovery_count:
            if sensor_name in self.compromised_sensors:
                self.compromised_sensors.remove(sensor_name)

        # State machine transitions
        self._update_global_state()

        # Compute risk score (0.0 to 1.0)
        risk_score = min(1.0, len(self.compromised_sensors) * 0.4 + (0.3 if is_alarm else 0.0))
        confidence = min(1.0, float(self.suspicious_counters[sensor_name]) / self.confirm_count)

        return {
            "state": self.current_state.value,
            "sensor": sensor_name,
            "nis": nis,
            "is_anomaly": is_alarm,
            "confidence": confidence,
            "risk_score": risk_score,
            "consecutive_anomalies": int(self.suspicious_counters.get(sensor_name, 0)),
            "compromised_sensors": list(self.compromised_sensors),
        }

    def _update_global_state(self) -> None:
        """Evaluates overall vehicle cyber detection state from individual sensor tracks."""
        if len(self.compromised_sensors) > 0:
            if self.current_state in [DetectionState.NORMAL, DetectionState.SUSPICIOUS]:
                self.current_state = DetectionState.ATTACK_CONFIRMED
        else:
            any_suspicious = any(c >= 1.0 for c in self.suspicious_counters.values())
            if any_suspicious:
                if self.current_state == DetectionState.NORMAL:
                    self.current_state = DetectionState.SUSPICIOUS
            else:
                if self.current_state in [DetectionState.SUSPICIOUS, DetectionState.RECOVERY, DetectionState.ATTACK_CONFIRMED]:
                    self.current_state = DetectionState.NORMAL

    def transition_to_mode(self, new_state: DetectionState) -> None:
        """Manual or resilience-triggered state transition (e.g. into CONTAINMENT or SAFE_NAVIGATION)."""
        self.current_state = new_state
