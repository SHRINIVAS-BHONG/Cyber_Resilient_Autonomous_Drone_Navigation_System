"""
Unit tests for WGS-84 Geodesy Module.
Validates Geodetic to ECEF and Local ENU conversions against standard aviation benchmarks.
"""

import math
import pytest
from sensor_bridge.geodesy import WGS84Datum


def test_wgs84_datum_origin_ecef():
    # Equator and Prime Meridian (0 deg N, 0 deg E, 0 m)
    datum = WGS84Datum(0.0, 0.0, 0.0)
    # ECEF at (0, 0, 0) should be (A, 0, 0) = (6378137.0, 0.0, 0.0)
    assert pytest.approx(datum.x0_ecef, abs=1.0) == 6378137.0
    assert pytest.approx(datum.y0_ecef, abs=1.0) == 0.0
    assert pytest.approx(datum.z0_ecef, abs=1.0) == 0.0


def test_wgs84_enu_local_offsets():
    # Origin at standard test airfield: 47.397742 deg N, 8.545594 deg E, 488.0 m
    lat0, lon0, alt0 = 47.397742, 8.545594, 488.0
    datum = WGS84Datum(lat0, lon0, alt0)

    # Convert origin to ENU -> must be (0, 0, 0)
    e, n, u = datum.geodetic_to_enu(lat0, lon0, alt0)
    assert pytest.approx(e, abs=1e-3) == 0.0
    assert pytest.approx(n, abs=1e-3) == 0.0
    assert pytest.approx(u, abs=1e-3) == 0.0

    # Small displacement North (approx 1 arcsecond ~ 30.87 meters)
    lat_displaced = lat0 + (1.0 / 3600.0)
    e, n, u = datum.geodetic_to_enu(lat_displaced, lon0, alt0)
    assert pytest.approx(e, abs=0.1) == 0.0
    assert 30.0 < n < 32.0
    assert pytest.approx(u, abs=0.1) == 0.0

    # Roundtrip conversion ENU -> Geodetic -> ENU
    lat_rt, lon_rt, alt_rt = datum.enu_to_geodetic(15.0, -25.0, 10.0)
    e_rt, n_rt, u_rt = datum.geodetic_to_enu(lat_rt, lon_rt, alt_rt)
    assert pytest.approx(e_rt, abs=1e-3) == 15.0
    assert pytest.approx(n_rt, abs=1e-3) == -25.0
    assert pytest.approx(u_rt, abs=1e-3) == 10.0
