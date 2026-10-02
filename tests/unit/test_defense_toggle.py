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
