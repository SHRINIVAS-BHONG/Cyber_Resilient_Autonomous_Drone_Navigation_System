"""
Dryden Wind Turbulence and Atmospheric Environmental Disturbance Model.

Implements military aerospace-grade Dryden continuous turbulence spectra (MIL-F-8785C)
and discrete "1 - cosine" gust profiles to simulate realistic aerodynamic disturbances
acting on autonomous quadcopters during low-altitude urban and open-field flight.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class AtmosphericCondition:
    air_density: float = 1.225  # kg/m^3 at sea level
    mean_speed: float = 3.0     # m/s nominal ambient wind
    wind_direction_deg: float = 45.0  # degrees from True North (wind blowing FROM)
    turbulence_severity: str = "light"  # "light", "moderate", "severe"
    enable_discrete_gusts: bool = True
    drone_mass_kg: float = 1.8  # kg
    drag_coefficient: float = 0.45
    projected_area_m2: float = 0.08  # effective frontal cross-section


class DrydenWindModel:
    """
    Generates realistic 3D stochastic wind fields and aerodynamic drag accelerations.
    """

    def __init__(self, config: Optional[AtmosphericCondition] = None, seed: Optional[int] = 42):
        self.cfg = config or AtmosphericCondition()
        self.rng = np.random.RandomState(seed)

        # Turbulence intensities based on severity (m/s)
        intensities = {
            "light": {"sigma_w": 0.5, "sigma_u": 1.0, "sigma_v": 1.0},
            "moderate": {"sigma_w": 1.2, "sigma_u": 2.0, "sigma_v": 2.0},
            "severe": {"sigma_w": 2.5, "sigma_u": 4.0, "sigma_v": 4.0},
        }
        self.sigma = intensities.get(self.cfg.turbulence_severity, intensities["light"])

        # Filter states for colored Gaussian noise integration
        self._gust_state = np.zeros(3, dtype=np.float64)

        # Discrete gust state
        self._discrete_gust_active = False
        self._discrete_gust_timer = 0.0
        self._discrete_gust_duration = 2.5
        self._discrete_gust_amplitude = np.zeros(3)

        # Compute mean wind vector in ENU coordinates
        dir_rad = np.radians(self.cfg.wind_direction_deg)
        # Wind blowing TOWARDS direction = dir_rad + 180 deg
        self.mean_wind_enu = np.array([
            -self.cfg.mean_speed * np.sin(dir_rad),
            -self.cfg.mean_speed * np.cos(dir_rad),
            0.0
        ], dtype=np.float64)

    def trigger_discrete_gust(self, amplitude: Tuple[float, float, float] = (4.0, 3.0, 1.5), duration_s: float = 2.5):
        """Triggers a 1-cosine discrete gust (e.g. urban canyon wind shear)."""
        self._discrete_gust_active = True
        self._discrete_gust_timer = 0.0
        self._discrete_gust_duration = max(0.5, duration_s)
        self._discrete_gust_amplitude = np.array(amplitude, dtype=np.float64)

    def step(self, altitude_m: float, drone_velocity_enu: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
        """
        Advances the atmospheric model by dt seconds.
        Returns total wind velocity and resulting aerodynamic acceleration on the drone.
        """
        alt = max(1.0, float(altitude_m))
        drone_vel = np.asarray(drone_velocity_enu, dtype=np.float64)

        # Altitude-dependent Dryden scale lengths (MIL-F-8785C low-altitude model)
        # L_w = h, L_u = L_v = h / (0.177 + 0.000823 * h)^1.2
        L_w = alt
        L_u = alt / ((0.177 + 0.000823 * alt) ** 1.2)
        L_v = L_u

        # Airspeed magnitude (relative to mean wind)
        airspeed = float(np.linalg.norm(drone_vel - self.mean_wind_enu))
        airspeed = max(1.0, airspeed)

        # Dryden first-order discrete approximations for u, v, w
        # tau = L / airspeed
        tau_u = max(0.1, L_u / airspeed)
        tau_v = max(0.1, L_v / airspeed)
        tau_w = max(0.1, L_w / airspeed)

        # Discrete colored noise state update: state(k) = (1 - dt/tau)*state(k-1) + sigma*sqrt(2*dt/tau)*white_noise
        white_noise = self.rng.normal(0.0, 1.0, size=3)
        decay = np.array([
            max(0.0, 1.0 - dt / tau_u),
            max(0.0, 1.0 - dt / tau_v),
            max(0.0, 1.0 - dt / tau_w)
        ])
        gain = np.array([
            self.sigma["sigma_u"] * np.sqrt(2.0 * dt / tau_u),
            self.sigma["sigma_v"] * np.sqrt(2.0 * dt / tau_v),
            self.sigma["sigma_w"] * np.sqrt(2.0 * dt / tau_w)
        ])
        self._gust_state = decay * self._gust_state + gain * white_noise

        # Discrete 1-cosine gust computation
        discrete_gust = np.zeros(3, dtype=np.float64)
        if self._discrete_gust_active:
            self._discrete_gust_timer += dt
            if self._discrete_gust_timer <= self._discrete_gust_duration:
                phase = 2.0 * np.pi * self._discrete_gust_timer / self._discrete_gust_duration
                # (1 - cos(2*pi*t/T)) profile
                discrete_gust = 0.5 * self._discrete_gust_amplitude * (1.0 - np.cos(phase))
            else:
                self._discrete_gust_active = False

        # Total atmospheric wind velocity in ENU
        total_wind_enu = self.mean_wind_enu + self._gust_state + discrete_gust

        # Aerodynamic relative velocity
        rel_vel = drone_vel - total_wind_enu
        rel_speed = float(np.linalg.norm(rel_vel))

        # Quadratic aerodynamic drag acceleration: a_aero = -0.5 * rho * Cd * A / m * |v_rel| * v_rel
        drag_const = 0.5 * self.cfg.air_density * self.cfg.drag_coefficient * self.cfg.projected_area_m2 / self.cfg.drone_mass_kg
        aero_accel_enu = -drag_const * rel_speed * rel_vel

        return {
            "wind_enu": total_wind_enu,
            "stochastic_gust_enu": self._gust_state.copy(),
            "discrete_gust_enu": discrete_gust,
            "aerodynamic_accel_enu": aero_accel_enu,
            "relative_airspeed_ms": rel_speed
        }
