"""
High-Precision WGS-84 Geodesy Module for Drone Avionics.

Converts Geodetic coordinates (Latitude, Longitude, Ellipsoidal Height) to
Earth-Centered Earth-Fixed (ECEF) and Local East-North-Up (ENU) tangent plane
with sub-millimeter geometric accuracy.
"""

import math
from typing import Optional, Tuple
import numpy as np


class WGS84Datum:
    """
    Standard World Geodetic System 1984 (WGS-84) Reference Ellipsoid.
    """
    # WGS-84 Ellipsoid constants
    A: float = 6378137.0                # Semi-major axis [m]
    F: float = 1.0 / 298.257223563      # Flattening
    B: float = A * (1.0 - F)            # Semi-minor axis ~ 6356752.3142 m
    E2: float = 2.0 * F - F * F         # First eccentricity squared ~ 0.00669437999014
    E_PRIME2: float = (A * A - B * B) / (B * B) # Second eccentricity squared

    def __init__(self, lat0: float, lon0: float, alt0: float = 0.0):
        """
        Initialize Local ENU Tangent Plane Origin.
        lat0: Origin latitude in degrees [-90, 90]
        lon0: Origin longitude in degrees [-180, 180]
        alt0: Origin altitude above WGS-84 ellipsoid in meters
        """
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.alt0 = float(alt0)

        # Radians
        self.phi0 = math.radians(self.lat0)
        self.lam0 = math.radians(self.lon0)

        # Precompute trigonometric terms for origin
        self.sin_phi0 = math.sin(self.phi0)
        self.cos_phi0 = math.cos(self.phi0)
        self.sin_lam0 = math.sin(self.lam0)
        self.cos_lam0 = math.cos(self.lam0)

        # Origin ECEF coordinates
        self.x0_ecef, self.y0_ecef, self.z0_ecef = self.geodetic_to_ecef(
            self.lat0, self.lon0, self.alt0
        )

        # Rotation matrix from ECEF to ENU
        # [ -sin(lam),          cos(lam),          0        ]
        # [ -sin(phi)*cos(lam), -sin(phi)*sin(lam), cos(phi) ]
        # [  cos(phi)*cos(lam),  cos(phi)*sin(lam), sin(phi) ]
        self.R_ecef_to_enu = np.array([
            [-self.sin_lam0, self.cos_lam0, 0.0],
            [-self.sin_phi0 * self.cos_lam0, -self.sin_phi0 * self.sin_lam0, self.cos_phi0],
            [ self.cos_phi0 * self.cos_lam0,  self.cos_phi0 * self.sin_lam0, self.sin_phi0]
        ], dtype=np.float64)

    @classmethod
    def geodetic_to_ecef(cls, lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
        """Convert WGS-84 Geodetic coordinates to ECEF (meters)."""
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        sin_lam = math.sin(lam)
        cos_lam = math.cos(lam)

        # Prime vertical radius of curvature N(phi)
        n = cls.A / math.sqrt(1.0 - cls.E2 * sin_phi * sin_phi)

        x = (n + alt_m) * cos_phi * cos_lam
        y = (n + alt_m) * cos_phi * sin_lam
        z = (n * (1.0 - cls.E2) + alt_m) * sin_phi
        return x, y, z

    @classmethod
    def ecef_to_geodetic(cls, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert ECEF coordinates to WGS-84 Geodetic (Bowring's algorithm)."""
        p = math.hypot(x, y)
        if p < 1e-6:
            lat = 90.0 if z > 0 else -90.0
            lon = 0.0
            alt = abs(z) - cls.B
            return lat, lon, alt

        theta = math.atan2(z * cls.A, p * cls.B)
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        phi = math.atan2(
            z + cls.E_PRIME2 * cls.B * (sin_theta ** 3),
            p - cls.E2 * cls.A * (cos_theta ** 3)
        )
        lam = math.atan2(y, x)

        sin_phi = math.sin(phi)
        n = cls.A / math.sqrt(1.0 - cls.E2 * sin_phi * sin_phi)
        alt = p / math.cos(phi) - n

        return math.degrees(phi), math.degrees(lam), alt

    def geodetic_to_enu(self, lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
        """
        Convert WGS-84 Geodetic coordinates to local East-North-Up (ENU) tangent plane.
        Returns: (east_meters, north_meters, up_meters)
        """
        x_ecef, y_ecef, z_ecef = self.geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        dx = x_ecef - self.x0_ecef
        dy = y_ecef - self.y0_ecef
        dz = z_ecef - self.z0_ecef

        d_ecef = np.array([dx, dy, dz], dtype=np.float64)
        enu = self.R_ecef_to_enu @ d_ecef
        return float(enu[0]), float(enu[1]), float(enu[2])

    def enu_to_geodetic(self, east_m: float, north_m: float, up_m: float) -> Tuple[float, float, float]:
        """
        Convert Local ENU offsets back to WGS-84 Geodetic coordinates.
        Returns: (latitude_deg, longitude_deg, altitude_m)
        """
        enu = np.array([east_m, north_m, up_m], dtype=np.float64)
        d_ecef = self.R_ecef_to_enu.T @ enu

        x_ecef = self.x0_ecef + float(d_ecef[0])
        y_ecef = self.y0_ecef + float(d_ecef[1])
        z_ecef = self.z0_ecef + float(d_ecef[2])

        return self.ecef_to_geodetic(x_ecef, y_ecef, z_ecef)
