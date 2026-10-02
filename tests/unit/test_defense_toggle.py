"""
Unit tests for the Cyber-Defense Toggle (ON / OFF) Mechanism.
Verifies state transitions, unprotected spoofed ingestion when disabled,
and automatic resilience containment when enabled.
"""
import sys
from pathlib import Path
import pytest
import numpy as np

# Root path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from api.server import SystemStateManager, SystemSettingsModel


def test_defense_initial_state():
    mgr = SystemStateManager()
    assert mgr.defense_enabled is True
    assert mgr.settings.defense_enabled is True


def test_defense_toggle_transitions():
    mgr = SystemStateManager()
    
    # Toggle OFF
    state = mgr.toggle_defense()
    assert state is False
    assert mgr.defense_enabled is False
    assert mgr.settings.defense_enabled is False
    assert len(mgr.resilience.isolated_sensors) == 0

    # Explicit set ON
    state = mgr.toggle_defense(enabled=True)
    assert state is True
    assert mgr.defense_enabled is True
    assert mgr.settings.defense_enabled is True

    # Explicit set OFF
    state = mgr.toggle_defense(enabled=False)
    assert state is False
    assert mgr.defense_enabled is False


def test_defense_disabled_nominal_without_attack():
    mgr = SystemStateManager()
    mgr.toggle_defense(enabled=False)
    assert mgr.active_attack is None

    # Step simulation without any attack
    for _ in range(5):
        mgr.step_simulation(dt=0.1)

    telem = mgr.get_latest_telemetry()
    assert telem["defense_enabled"] is False
    assert telem["is_attack_active"] is False
    assert telem["navigation_mode"] == "NORMAL_MISSION"
    assert telem["chi2_attack_state"] == "NORMAL"
    assert len(telem["isolated_sensors"]) == 0


def test_defense_disabled_unprotected_ingestion():
    mgr = SystemStateManager()
    mgr.toggle_defense(enabled=False)

    # Ingest attack
    mgr.active_attack = {
        "attack_type": "gps_spoofing",
        "magnitude": 30.0,
        "duration_sec": 20.0
    }
    mgr.attack_start_time = mgr.start_time

    # Run simulation step
    mgr.step_simulation(dt=0.1)

    telem = mgr.get_latest_telemetry()
    assert telem["defense_enabled"] is False
    assert telem["is_attack_active"] is True
    assert telem["navigation_mode"] == "DEFENSE_DISABLED"
    assert len(telem["isolated_sensors"]) == 0
    assert "gps" in telem["active_sensors"]


def test_defense_enabled_active_quarantine():
    mgr = SystemStateManager()
    mgr.toggle_defense(enabled=True)

    # Ingest attack
    mgr.active_attack = {
        "attack_type": "gps_spoofing",
        "magnitude": 30.0,
        "duration_sec": 20.0
    }
    mgr.attack_start_time = mgr.start_time

    # Step simulation multiple frames to confirm attack
    for _ in range(5):
        mgr.step_simulation(dt=0.1)

    telem = mgr.get_latest_telemetry()
    assert telem["defense_enabled"] is True
    assert telem["chi2_attack_state"] == "ATTACK_CONFIRMED"
    assert "gps" in telem["isolated_sensors"]
    assert "gps" not in telem["active_sensors"]
    assert telem["navigation_mode"] == "DEGRADED_OPTICAL_LIDAR"


def test_imu_attack_physical_divergence_when_defense_off_vs_on():
    # 1. Defense OFF: IMU attack physically drives the drone into lateral drift and attitude wobble
    mgr_off = SystemStateManager()
    mgr_off.flight_phase = "CRUISE"
    mgr_off.is_airborne = True
    mgr_off.drone_pos = np.array([0.0, 0.0, 12.0])
    mgr_off.drone_vel = np.array([0.0, 0.0, 0.0])
    mgr_off.toggle_defense(enabled=False)
    mgr_off.active_attack = {"attack_type": "imu_manipulation", "magnitude": 2.5, "duration_sec": 20.0}
    mgr_off.attack_start_time = mgr_off.start_time

    for _ in range(10):
        mgr_off.step_simulation(dt=0.1)

    telem_off = mgr_off.get_latest_telemetry()
    assert telem_off["defense_enabled"] is False
    assert telem_off["is_attack_active"] is True
    # Physical velocity and position must show the effect of unmitigated IMU drift
    assert abs(telem_off["velocity_enu"][0]) > 0.5 or abs(telem_off["velocity_enu"][1]) > 0.5

    # 2. Defense ON: IMU attack is isolated, drone holds stable hover
    mgr_on = SystemStateManager()
    mgr_on.flight_phase = "CRUISE"
    mgr_on.is_airborne = True
    mgr_on.drone_pos = np.array([0.0, 0.0, 12.0])
    mgr_on.drone_vel = np.array([0.0, 0.0, 0.0])
    mgr_on.toggle_defense(enabled=True)
    mgr_on.active_attack = {"attack_type": "imu_manipulation", "magnitude": 2.5, "duration_sec": 20.0}
    mgr_on.attack_start_time = mgr_on.start_time

    for _ in range(10):
        mgr_on.step_simulation(dt=0.1)

    telem_on = mgr_on.get_latest_telemetry()
    assert telem_on["defense_enabled"] is True
    assert "imu" in telem_on["isolated_sensors"]
    assert "imu" not in telem_on["active_sensors"]


def test_lidar_attack_altitude_hunting_when_defense_off_vs_on():
    # 1. Defense OFF: LiDAR attack forces altitude surge/hunting
    mgr_off = SystemStateManager()
    mgr_off.flight_phase = "CRUISE"
    mgr_off.is_airborne = True
    mgr_off.drone_pos = np.array([0.0, 0.0, 12.0])
    mgr_off.drone_vel = np.array([0.0, 0.0, 0.0])
    mgr_off.toggle_defense(enabled=False)
    mgr_off.active_attack = {"attack_type": "lidar_corruption", "magnitude": 7.0, "duration_sec": 20.0}
    mgr_off.attack_start_time = mgr_off.start_time

    for _ in range(10):
        mgr_off.step_simulation(dt=0.1)

    telem_off = mgr_off.get_latest_telemetry()
    assert telem_off["defense_enabled"] is False
    assert telem_off["is_attack_active"] is True
    # Altitude or vertical climb speed must reflect the unmitigated laser tamper
    assert telem_off["altitude_m"] > 12.3 or abs(telem_off["velocity_enu"][2]) > 0.5

    # 2. Defense ON: LiDAR attack is isolated, altitude is smoothly maintained
    mgr_on = SystemStateManager()
    mgr_on.flight_phase = "CRUISE"
    mgr_on.is_airborne = True
    mgr_on.drone_pos = np.array([0.0, 0.0, 12.0])
    mgr_on.drone_vel = np.array([0.0, 0.0, 0.0])
    mgr_on.toggle_defense(enabled=True)
    mgr_on.active_attack = {"attack_type": "lidar_corruption", "magnitude": 7.0, "duration_sec": 20.0}
    mgr_on.attack_start_time = mgr_on.start_time

    for _ in range(10):
        mgr_on.step_simulation(dt=0.1)

    telem_on = mgr_on.get_latest_telemetry()
    assert telem_on["defense_enabled"] is True
    assert "lidar" in telem_on["isolated_sensors"]
    assert "lidar" not in telem_on["active_sensors"]


def test_multi_attack_when_defense_off_vs_on():
    # 1. Defense OFF: Multi-attack causes simultaneous lateral & vertical divergence
    mgr_off = SystemStateManager()
    mgr_off.flight_phase = "CRUISE"
    mgr_off.is_airborne = True
    mgr_off.drone_pos = np.array([0.0, 0.0, 12.0])
    mgr_off.drone_vel = np.array([0.0, 0.0, 0.0])
    mgr_off.toggle_defense(enabled=False)
    mgr_off.active_attack = {"attack_type": "multi_attack", "magnitude": 20.0, "duration_sec": 20.0}
    mgr_off.attack_start_time = mgr_off.start_time

    for _ in range(10):
        mgr_off.step_simulation(dt=0.1)

    telem_off = mgr_off.get_latest_telemetry()
    assert telem_off["defense_enabled"] is False
    assert telem_off["is_attack_active"] is True
    assert len(telem_off["isolated_sensors"]) == 0

    # 2. Defense ON: Multi-attack isolates all compromised sensors and triggers contingency
    mgr_on = SystemStateManager()
    mgr_on.flight_phase = "CRUISE"
    mgr_on.is_airborne = True
    mgr_on.drone_pos = np.array([0.0, 0.0, 12.0])
    mgr_on.drone_vel = np.array([0.0, 0.0, 0.0])
    mgr_on.toggle_defense(enabled=True)
    mgr_on.active_attack = {"attack_type": "multi_attack", "magnitude": 20.0, "duration_sec": 20.0}
    mgr_on.attack_start_time = mgr_on.start_time

    for _ in range(10):
        mgr_on.step_simulation(dt=0.1)

    telem_on = mgr_on.get_latest_telemetry()
    assert telem_on["defense_enabled"] is True
    assert "gps" in telem_on["isolated_sensors"]
    assert "imu" in telem_on["isolated_sensors"]
    assert "lidar" in telem_on["isolated_sensors"]

