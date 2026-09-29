"""
Realistic Cyberattack Injection Engine for Drone Avionics.

Decoupled, mathematically grounded attack simulation engine supporting:
1. GPS Spoofing (Ramp / Slow Drift, Step Jump, Latch / Freeze, Jamming / Drop, False Velocity)
2. IMU Manipulation (DC Bias, Dynamic Scale Factor, High-G Noise Increase)
3. LiDAR Deception (Range Scale Compression, Distance Offsets)
4. Barometric Pressure Manipulation (Atmospheric Offset, False Descent / Climb)
5. Magnetometer Spoofing (Magnetic Flux Distortion, Yaw Heading Bias)
6. Communication Disruption (Packet Loss, Network Latency Delay Queue, Topic Drop)
7. Coordinated Multi-Sensor Attacks (Simultaneous Compound Attacks)

Strictly separates ground-truth metadata from injected sensor streams.
"""

from collections import deque
import copy
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml


class AttackSimulationEngine:
    """
    Pure Python & NumPy attack injection engine.
    Can be used within ROS 2 nodes, pytest pipelines, and SITL simulations.
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], Path, str]] = None):
        # Default baseline configuration (attack disabled)
        self.config: Dict[str, Any] = {
            "id": "scenario_01",
            "name": "Scenario 01: Baseline Normal Flight",
            "category": "Baseline",
            "attack_type": "none",
            "attack_start_time": 0.0,
            "attack_duration": 0.0,
            "attack_parameters": {},
            "random_seed": 42
        }

        # Internal state tracking
        self.frozen_gps_pos: Optional[np.ndarray] = None
        self.delay_queue: deque = deque()
        self.packet_drop_rng = random.Random(42)

        if config is not None:
            self.load_scenario(config)

    def load_scenario(self, config: Union[Dict[str, Any], Path, str]) -> None:
        """Loads attack scenario from YAML file or configuration dictionary."""
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Scenario configuration file not found: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if not isinstance(loaded, dict):
                    raise ValueError(f"Invalid YAML content in {config_path}")
                self.config.update(loaded)
        elif isinstance(config, dict):
            self.config.update(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        # Reset attack random seed for deterministic reproducibility
        seed = self.config.get("random_seed", 42)
        self.packet_drop_rng = random.Random(seed)
        np.random.seed(seed)
        self.frozen_gps_pos = None
        self.delay_queue.clear()

    @property
    def attack_type(self) -> str:
        return str(self.config.get("attack_type", "none")).lower()

    @property
    def start_time(self) -> float:
        return float(self.config.get("attack_start_time", 0.0))

    @property
    def duration(self) -> float:
        return float(self.config.get("attack_duration", 0.0))

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.config.get("attack_parameters", {})

    def is_attack_active(self, t: float) -> bool:
        """Evaluates if the cyber attack is actively executing at time t."""
        if self.attack_type in ("none", "baseline") or self.duration <= 0.0:
            return False
        return self.start_time <= t < self.end_time

    # -------------------------------------------------------------------------
    # 1. GPS ATTACK INJECTION
    # -------------------------------------------------------------------------
    def inject_gps(self, pos: np.ndarray, t: float) -> Optional[np.ndarray]:
        """
        Injects realistic GPS attack into 3D position [East, North, Up] in meters.
        Returns corrupted 3D position or None if signal is completely jammed / dropped.
        """
        pos = np.asarray(pos, dtype=np.float64).copy()
        if not self.is_attack_active(t):
            self.frozen_gps_pos = None
            return pos

        atk_type = self.attack_type
        params = self.parameters
        elapsed_attack = t - self.start_time

        # Check if GPS is targeted in compound attack or standalone
        is_gps_targeted = (
            "gps" in atk_type or
            params.get("target_sensor") == "gps" or
            params.get("primary_attack") == "gps_spoofing" or
            any(a.get("sensor") == "gps" for a in params.get("attacks", []))
        )

        if not is_gps_targeted:
            return pos

        # Extract specific GPS attack parameters
        gps_params = params
        if "attacks" in params:
            for sub_atk in params["attacks"]:
                if sub_atk.get("sensor") == "gps":
                    gps_params = sub_atk
                    break

        mode = gps_params.get("mode", "slow_drift").lower()

        # Handle GPS Jamming / Complete Topic Interruption
        if mode in ("jamming", "denial_of_service", "signal_loss", "topic_interruption"):
            return None

        # Handle GPS Freeze / Replay Attack
        if mode in ("freeze", "replay", "latch"):
            if self.frozen_gps_pos is None:
                self.frozen_gps_pos = pos.copy()
            return self.frozen_gps_pos.copy()

        # Determine target offset vector
        bias_cfg = gps_params.get("position_bias") or gps_params.get("bias") or gps_params.get("offset")
        if isinstance(bias_cfg, dict):
            dx = float(bias_cfg.get("x", 15.0))
            dy = float(bias_cfg.get("y", -10.0))
            dz = float(bias_cfg.get("z", 0.0))
        elif isinstance(bias_cfg, (list, tuple)) and len(bias_cfg) >= 3:
            dx, dy, dz = float(bias_cfg[0]), float(bias_cfg[1]), float(bias_cfg[2])
        else:
            dx = float(gps_params.get("offset_x", 15.0))
            dy = float(gps_params.get("offset_y", -10.0))
            dz = float(gps_params.get("offset_z", 0.0))

        offset_target = np.array([dx, dy, dz], dtype=np.float64)

        # Handle Step Jump (instantaneous offset)
        if mode in ("step_jump", "step", "instant"):
            return pos + offset_target

        # Handle Slow Drift / Ramp Spoofing (stealthy ramp designed to evade simple gating)
        ramp_dur = float(gps_params.get("ramp_duration", gps_params.get("ramp_duration_sec", 5.0)))
        ramp_progress = min(1.0, max(0.0, elapsed_attack / max(ramp_dur, 0.1)))
        return pos + offset_target * ramp_progress

    def inject_gps_velocity(self, vel: np.ndarray, t: float) -> Optional[np.ndarray]:
        """Injects false GPS velocity bias in m/s."""
        vel = np.asarray(vel, dtype=np.float64).copy()
        if not self.is_attack_active(t):
            return vel

        params = self.parameters
        if "velocity_bias" in params:
            v_bias = params["velocity_bias"]
            vx = float(v_bias.get("vx", 0.0))
            vy = float(v_bias.get("vy", 0.0))
            vz = float(v_bias.get("vz", 0.0))
            return vel + np.array([vx, vy, vz])
        return vel

    # -------------------------------------------------------------------------
    # 2. IMU ATTACK INJECTION
    # -------------------------------------------------------------------------
    def inject_imu(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        t: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Injects realistic IMU manipulation into accelerometer (m/s^2) and gyroscope (rad/s).
        Simulates DC bias injection, resonant frequency acoustic injection, or scaling.
        """
        accel = np.asarray(accel, dtype=np.float64).copy()
        gyro = np.asarray(gyro, dtype=np.float64).copy()
        if not self.is_attack_active(t):
            return accel, gyro

        params = self.parameters
        is_imu_targeted = (
            "imu" in self.attack_type or
            params.get("target_sensor") == "imu" or
            any(a.get("sensor") == "imu" for a in params.get("attacks", []))
        )

        if not is_imu_targeted:
            return accel, gyro

        imu_params = params
        if "attacks" in params:
            for sub_atk in params["attacks"]:
                if sub_atk.get("sensor") == "imu":
                    imu_params = sub_atk
                    break

        # Accelerometer Bias
        acc_bias = imu_params.get("accel_bias") or imu_params.get("bias")
        if isinstance(acc_bias, dict):
            ax = float(acc_bias.get("ax", acc_bias.get("x", 2.2)))
            ay = float(acc_bias.get("ay", acc_bias.get("y", -1.5)))
            az = float(acc_bias.get("az", acc_bias.get("z", 0.5)))
            accel += np.array([ax, ay, az])
        elif isinstance(acc_bias, (list, tuple)) and len(acc_bias) >= 3:
            accel += np.array([acc_bias[0], acc_bias[1], acc_bias[2]], dtype=np.float64)

        # Gyroscope Bias
        gyro_bias = imu_params.get("gyro_bias")
        if isinstance(gyro_bias, dict):
            gx = float(gyro_bias.get("gx", gyro_bias.get("x", 0.0)))
            gy = float(gyro_bias.get("gy", gyro_bias.get("y", 0.0)))
            gz = float(gyro_bias.get("gz", gyro_bias.get("z", 0.05)))
            gyro += np.array([gx, gy, gz])
        elif isinstance(gyro_bias, (list, tuple)) and len(gyro_bias) >= 3:
            gyro += np.array([gyro_bias[0], gyro_bias[1], gyro_bias[2]], dtype=np.float64)

        # Scale factor manipulation
        scale = imu_params.get("scale_factor")
        if scale is not None:
            accel *= float(scale)

        # Noise increase (Scenario 07)
        noise_var = float(imu_params.get("noise_variance", 0.0))
        noise_mult = float(imu_params.get("noise_multiplier", 1.0))
        if noise_var > 0.0 or noise_mult > 1.0:
            std = math.sqrt(noise_var) if noise_var > 0.0 else 0.05 * noise_mult
            accel += np.random.normal(0.0, std, size=(3,))

        return accel, gyro

    # -------------------------------------------------------------------------
    # 3. LIDAR ATTACK INJECTION
    # -------------------------------------------------------------------------
    def inject_lidar(self, range_m: float, t: float) -> float:
        """
        Injects range deception or deceptive compression into 1D LiDAR distance.
        """
        if not self.is_attack_active(t):
            return range_m

        params = self.parameters
        is_lidar_targeted = (
            "lidar" in self.attack_type or
            params.get("target_sensor") == "lidar" or
            any(a.get("sensor") == "lidar" for a in params.get("attacks", []))
        )

        if not is_lidar_targeted:
            return range_m

        lidar_params = params
        if "attacks" in params:
            for sub_atk in params["attacks"]:
                if sub_atk.get("sensor") == "lidar":
                    lidar_params = sub_atk
                    break

        scale = float(lidar_params.get("range_compression_ratio", lidar_params.get("scale_factor", 1.0)))
        offset = float(lidar_params.get("offset_meters", lidar_params.get("distance_bias", lidar_params.get("offset", 0.0))))

        corrupted = range_m * scale + offset
        return max(0.05, float(corrupted))

    # -------------------------------------------------------------------------
    # 4. BAROMETER ATTACK INJECTION
    # -------------------------------------------------------------------------
    def inject_barometer(self, altitude_m: float, t: float) -> float:
        """Injects false atmospheric static pressure offset into barometric altitude."""
        if not self.is_attack_active(t):
            return altitude_m

        params = self.parameters
        is_baro_targeted = (
            "barometer" in self.attack_type or
            params.get("target_sensor") == "barometer" or
            any(a.get("sensor") == "barometer" for a in params.get("attacks", []))
        )

        if not is_baro_targeted:
            return altitude_m

        baro_params = params
        if "attacks" in params:
            for sub_atk in params["attacks"]:
                if sub_atk.get("sensor") == "barometer":
                    baro_params = sub_atk
                    break

        bias = float(baro_params.get("altitude_bias", baro_params.get("pressure_bias_pa", baro_params.get("offset", 15.0))))
        return altitude_m + bias

    # -------------------------------------------------------------------------
    # 5. MAGNETOMETER ATTACK INJECTION
    # -------------------------------------------------------------------------
    def inject_magnetometer(self, mag_field: np.ndarray, t: float) -> np.ndarray:
        """
        Injects magnetic flux distortion or heading bias into 3-axis magnetometer measurements.
        """
        mag_field = np.asarray(mag_field, dtype=np.float64).copy()
        if not self.is_attack_active(t):
            return mag_field

        params = self.parameters
        is_mag_targeted = (
            "magnetometer" in self.attack_type or
            params.get("target_sensor") == "magnetometer" or
            any(a.get("sensor") == "magnetometer" for a in params.get("attacks", []))
        )

        if not is_mag_targeted:
            return mag_field

        mag_params = params
        if "attacks" in params:
            for sub_atk in params["attacks"]:
                if sub_atk.get("sensor") == "magnetometer":
                    mag_params = sub_atk
                    break

        heading_bias_deg = float(mag_params.get("heading_bias_deg", 45.0))
        rad = math.radians(heading_bias_deg)

        # Rotate horizontal magnetic field vector in XY plane
        bx, by, bz = mag_field[0], mag_field[1], mag_field[2]
        bx_rot = bx * math.cos(rad) - by * math.sin(rad)
        by_rot = bx * math.sin(rad) + by * math.cos(rad)

        flux_distortion = float(mag_params.get("flux_distortion_gauss", 0.0))
        return np.array([bx_rot + flux_distortion, by_rot - flux_distortion, bz], dtype=np.float64)

    # -------------------------------------------------------------------------
    # 6. COMMUNICATION DISRUPTION & LATENCY
    # -------------------------------------------------------------------------
    def evaluate_packet_loss(self, t: float) -> bool:
        """
        Returns True if packet should be DROPPED due to cyber communication disruption.
        """
        if not self.is_attack_active(t):
            return False

        params = self.parameters
        is_comm_targeted = (
            "communication" in self.attack_type or
            params.get("target_channel") == "telemetry" or
            params.get("packet_loss_rate") is not None or
            any(a.get("type") == "packet_loss" or a.get("packet_loss_rate") is not None for a in params.get("attacks", []))
        )

        if not is_comm_targeted:
            return False

        comm_params = params
        if "attacks" in params:
            for sub_atk in params["attacks"]:
                if sub_atk.get("type") in ("packet_loss", "communication_disruption") or "packet_loss_rate" in sub_atk:
                    comm_params = sub_atk
                    break

        drop_rate = float(comm_params.get("packet_loss_rate", comm_params.get("drop_rate", 0.5)))
        return self.packet_drop_rng.random() < drop_rate

    def queue_delayed_packet(self, payload: Any, t: float) -> List[Tuple[float, Any]]:
        """
        Enqueues payload with network transmission delay.
        Returns list of (release_time, payload) ready for dispatch at time t.
        """
        ready_packets = []
        delay_sec = 0.0

        if self.is_attack_active(t):
            params = self.parameters
            is_delay_targeted = (
                "delay" in self.attack_type or
                params.get("delay_ms") is not None or
                params.get("delay_sec") is not None or
                any(a.get("delay_ms") is not None or a.get("delay_sec") is not None for a in params.get("attacks", []))
            )
            if is_delay_targeted:
                if "delay_ms" in params:
                    delay_sec = float(params["delay_ms"]) / 1000.0
                elif "delay_sec" in params:
                    delay_sec = float(params["delay_sec"])
                else:
                    delay_sec = 0.35

        release_time = t + delay_sec
        self.delay_queue.append((release_time, copy.deepcopy(payload)))

        # Pop all packets whose release time has arrived
        while self.delay_queue and self.delay_queue[0][0] <= t:
            ready_packets.append(self.delay_queue.popleft())

        return ready_packets

    # -------------------------------------------------------------------------
    # 7. ISOLATED GROUND-TRUTH TELEMETRY
    # -------------------------------------------------------------------------
    def get_ground_truth(self, t: float) -> Dict[str, Any]:
        """
        Returns authentic ground-truth metadata strictly isolated from estimation.
        Used solely by logging and scoring benchmarks.
        """
        is_active = self.is_attack_active(t)
        compromised = []

        if is_active:
            atk_type = self.attack_type
            params = self.parameters

            if "gps" in atk_type or params.get("target_sensor") == "gps" or params.get("primary_attack") == "gps_spoofing":
                compromised.append("gps")
            if "imu" in atk_type or params.get("target_sensor") == "imu":
                compromised.append("imu")
            if "lidar" in atk_type or params.get("target_sensor") == "lidar":
                compromised.append("lidar")
            if "barometer" in atk_type or params.get("target_sensor") == "barometer":
                compromised.append("barometer")
            if "magnetometer" in atk_type or params.get("target_sensor") == "magnetometer":
                compromised.append("magnetometer")
            if "communication" in atk_type or "delay" in atk_type or "packet_loss_rate" in params:
                compromised.append("telemetry_link")

            if "attacks" in params:
                for sub in params["attacks"]:
                    s = sub.get("sensor") or sub.get("target_channel")
                    if s and s not in compromised:
                        compromised.append(s)

        return {
            "scenario_id": self.config.get("id", "unknown"),
            "scenario_name": self.config.get("name", "unknown"),
            "attack_type": self.config.get("attack_type", "none") if is_active else "None",
            "is_active": is_active,
            "confidence": 1.0 if is_active else 0.0,
            "risk_score": 1.0 if is_active else 0.0,
            "compromised_sensors": compromised,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "current_time": t
        }
