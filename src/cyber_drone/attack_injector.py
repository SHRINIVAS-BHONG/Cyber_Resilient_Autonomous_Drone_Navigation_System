"""
Attack Injector Module
Simulates cyberattacks on sensor streams:
- GPS drift (ramped offset)
- GPS jump (instant step)
- IMU bias (constant acceleration offset)
- LiDAR corruption (range scaling factor)
- Message drop (packet loss)
- Message delay (latency buffering)
- GPS blackout (topic silence)
"""

import collections
import random
import numpy as np
from typing import Dict, Any, Optional, Tuple, List


class AttackInjector:
    def __init__(self, scenario_config: Dict[str, Any], seed: int = 42):
        self.scenario = scenario_config
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.attack_cfg = self.scenario.get("attack", {})
        self.enabled = bool(self.attack_cfg.get("enabled", False))
        self.attack_type = self.attack_cfg.get("type", "none")
        self.start_time = float(self.attack_cfg.get("start_time", 30.0))
        self.duration = float(self.attack_cfg.get("duration", 20.0))
        self.end_time = self.start_time + self.duration

        # Delay buffer for GPS: deque of (delivery_time, payload)
        self.gps_delay_buffer = collections.deque()
        self.delay_sec = float(self.attack_cfg.get("delay_ms", 0.0)) / 1000.0

        # Ground truth status output
        self.is_attack_active = False

    def is_active(self, current_time: float) -> bool:
        if not self.enabled:
            return False
        return self.start_time <= current_time < self.end_time

    def get_attack_ground_truth(self, current_time: float) -> Dict[str, Any]:
        active = self.is_active(current_time)
        return {
            "attack_type": self.attack_type if active else "none",
            "active": active,
            "start_time": self.start_time,
            "duration": self.duration,
            "compromised_sensors": self._get_compromised_sensors_gt() if active else []
        }

    def _get_compromised_sensors_gt(self) -> List[str]:
        if "gps" in self.attack_type or "comm" in self.attack_type:
            return ["gps"]
        elif "imu" in self.attack_type:
            return ["imu"]
        elif "lidar" in self.attack_type:
            return ["lidar"]
        return []

    def process_gps(self, current_time: float, raw_pos: np.ndarray, raw_vel: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Applies GPS attacks: drift, jump, drop, delay, blackout.
        Returns (attacked_pos, attacked_vel) or None if dropped/blacked out.
        """
        active = self.is_active(current_time)
        pos = raw_pos.copy()
        vel = raw_vel.copy()

        if active:
            elapsed = current_time - self.start_time

            # 1. GPS Blackout
            if self.attack_type == "comm_disruption":
                blackout_start = float(self.attack_cfg.get("blackout_start_time", 40.0))
                blackout_dur = float(self.attack_cfg.get("blackout_duration", 10.0))
                if blackout_start <= current_time < blackout_start + blackout_dur:
                    return None  # Total silence

            # 2. Message Drop (Packet loss)
            drop_rate = float(self.attack_cfg.get("drop_rate", 0.0))
            if drop_rate > 0.0 and self.rng.random() < drop_rate:
                return None  # Dropped packet

            # 3. GPS Drift
            if self.attack_type in ["gps_drift", "combined_gps_drift_and_delay"]:
                ramp_dur = float(self.attack_cfg.get("ramp_duration", 5.0))
                bias_dict = self.attack_cfg.get("position_bias", {"x": 15.0, "y": -10.0, "z": 0.0})
                max_bias = np.array([bias_dict["x"], bias_dict["y"], bias_dict.get("z", 0.0)])
                alpha = min(1.0, elapsed / ramp_dur)
                current_bias = max_bias * alpha
                pos += current_bias
                if ramp_dur > 0 and elapsed < ramp_dur:
                    vel += (max_bias / ramp_dur)

            # 4. GPS Jump
            elif self.attack_type == "gps_jump":
                offset_dict = self.attack_cfg.get("offset", {"x": 22.0, "y": 20.0, "z": 0.0})
                pos += np.array([offset_dict["x"], offset_dict["y"], offset_dict.get("z", 0.0)])

            # 5. Delay simulation
            if self.delay_sec > 0.0 or "delay" in self.attack_type or self.attack_type == "comm_disruption":
                delay = self.delay_sec if self.delay_sec > 0.0 else 0.500
                delivery_time = current_time + delay
                self.gps_delay_buffer.append((delivery_time, (pos, vel)))
                # Check if earliest buffered item is ready
                if self.gps_delay_buffer and self.gps_delay_buffer[0][0] <= current_time:
                    _, ready_payload = self.gps_delay_buffer.popleft()
                    return ready_payload
                return None

        # If not active or no delay, deliver immediately (or drain any remaining buffer)
        if self.gps_delay_buffer and self.gps_delay_buffer[0][0] <= current_time:
            _, ready_payload = self.gps_delay_buffer.popleft()
            return ready_payload

        return pos, vel

    def process_imu(self, current_time: float, raw_accel: np.ndarray, raw_gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies IMU attacks: constant acceleration bias injection.
        """
        active = self.is_active(current_time)
        accel = raw_accel.copy()
        gyro = raw_gyro.copy()

        if active and self.attack_type == "imu_bias":
            bias_dict = self.attack_cfg.get("accel_bias", {"x": 0.8, "y": -0.5, "z": 0.2})
            accel += np.array([bias_dict["x"], bias_dict["y"], bias_dict["z"]])

        return accel, gyro

    def process_lidar(self, current_time: float, raw_ranges: np.ndarray, raw_alt: float) -> Tuple[np.ndarray, float]:
        """
        Applies LiDAR attacks: range corruption scaling factor.
        """
        active = self.is_active(current_time)
        ranges = raw_ranges.copy()
        alt = raw_alt

        if active and self.attack_type == "lidar_corruption":
            scale = float(self.attack_cfg.get("scale_factor", 0.6))
            ranges *= scale
            alt *= scale

        return ranges, alt
