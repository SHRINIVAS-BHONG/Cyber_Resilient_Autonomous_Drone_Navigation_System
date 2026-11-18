"""
Unit tests for Dryden Wind Turbulence and Atmospheric Disturbance Model.
"""

import numpy as np
import pytest
from simulation.environment.wind_turbulence import DrydenWindModel, AtmosphericCondition


def test_dryden_initialization():
    cond = AtmosphericCondition(mean_speed=4.0, wind_direction_deg=90.0, turbulence_severity="moderate")
    model = DrydenWindModel(cond, seed=100)

    # Wind blowing FROM 90 deg (East) blows TOWARDS West (negative X)
    assert model.mean_wind_enu[0] < -3.5
    assert abs(model.mean_wind_enu[2]) < 1e-5


def test_dryden_turbulence_step():
    model = DrydenWindModel(seed=42)
    drone_vel = np.array([3.0, 0.0, 0.0])

    step_res = model.step(altitude_m=15.0, drone_velocity_enu=drone_vel, dt=0.1)

    assert "wind_enu" in step_res
    assert "aerodynamic_accel_enu" in step_res
    assert "relative_airspeed_ms" in step_res

    # Acceleration should oppose relative motion
    assert step_res["relative_airspeed_ms"] > 0.0
    # Values should be finite and bounded
    assert np.all(np.isfinite(step_res["aerodynamic_accel_enu"]))
    assert np.all(np.abs(step_res["aerodynamic_accel_enu"]) < 10.0)


def test_discrete_gust_trigger():
    model = DrydenWindModel(seed=42)
    model.trigger_discrete_gust(amplitude=(5.0, 0.0, 2.0), duration_s=2.0)

    drone_vel = np.array([0.0, 0.0, 0.0])
    # Mid-gust (t = 1.0s, half-phase) should exhibit positive gust velocity
    step1 = model.step(altitude_m=10.0, drone_velocity_enu=drone_vel, dt=1.0)
    assert step1["discrete_gust_enu"][0] > 1.0

    # Past duration (t > 2.0s), discrete gust should cease
    step2 = model.step(altitude_m=10.0, drone_velocity_enu=drone_vel, dt=2.0)
    assert step2["discrete_gust_enu"][0] == 0.0
