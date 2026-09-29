"""
Unit tests for Step 3: Realistic Cyberattack Injection Engine.

Validates physics-grounded mathematical attack models:
- GPS drift, step, freeze, and jamming
- IMU accelerometer & gyroscope bias
- LiDAR range scale compression
- Barometer pressure & magnetometer disturbance
- Communication packet loss and latency queue
- Coordinated multi-sensor compound attacks
- All 15 scenario YAML configurations
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add attack_simulator package to path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir / "ros2_ws" / "src" / "attack_simulator"))

from attack_simulator.attack_engine import AttackSimulationEngine


def test_attack_engine_initialization():
    engine = AttackSimulationEngine()
    assert engine.attack_type == "none"
    assert engine.is_attack_active(5.0) is False
    assert engine.is_attack_active(15.0) is False

    # Under baseline, sensor streams must pass through unaltered
    clean_gps = np.array([10.0, 20.0, 5.0])
    injected_gps = engine.inject_gps(clean_gps, t=12.0)
    assert np.allclose(clean_gps, injected_gps)


def test_attack_engine_load_all_15_scenarios():
    scenarios_dir = root_dir / "simulation" / "scenarios"
    scenario_files = sorted(scenarios_dir.glob("scenario_*.yaml"))
    assert len(scenario_files) == 15, f"Expected 15 scenario YAML files, found {len(scenario_files)}"

    for sc_file in scenario_files:
        engine = AttackSimulationEngine(sc_file)
        assert engine.config["id"] == sc_file.stem
        assert engine.start_time >= 0.0
        assert engine.duration >= 0.0
        assert engine.end_time >= engine.start_time


def test_gps_slow_drift_attack():
    """Scenario 03: Verify gradual ramp spoofing physics."""
    sc_path = root_dir / "simulation" / "scenarios" / "scenario_03.yaml"
    engine = AttackSimulationEngine(sc_path)

    t_start = engine.start_time
    clean_pos = np.array([0.0, 0.0, 10.0])

    # 1. Before attack: clean
    p_before = engine.inject_gps(clean_pos, t=t_start - 1.0)
    assert np.allclose(p_before, clean_pos)

    # 2. At ramp midpoint (e.g. 50% ramp progress)
    ramp_mid = t_start + 2.5
    p_mid = engine.inject_gps(clean_pos, t=ramp_mid)
    # Target bias in scenario_03 is [15.0, -10.0, 0.0]
    expected_mid = clean_pos + np.array([7.5, -5.0, 0.0])
    assert np.allclose(p_mid, expected_mid, atol=0.2)

    # 3. After ramp completion: full offset
    p_full = engine.inject_gps(clean_pos, t=t_start + 6.0)
    expected_full = clean_pos + np.array([15.0, -10.0, 0.0])
    assert np.allclose(p_full, expected_full, atol=0.1)

    # 4. After attack ends: recovers to clean
    p_after = engine.inject_gps(clean_pos, t=engine.end_time + 1.0)
    assert np.allclose(p_after, clean_pos)


def test_gps_step_jump_attack():
    """Scenario 04: Sudden step spoofing jump."""
    sc_path = root_dir / "simulation" / "scenarios" / "scenario_04.yaml"
    engine = AttackSimulationEngine(sc_path)

    clean_pos = np.array([5.0, 5.0, 10.0])
    # Jump must be instantaneous at t = start_time
    p_jump = engine.inject_gps(clean_pos, t=engine.start_time + 0.1)
    target_offset = np.array([30.0, 0.0, 0.0])
    assert np.allclose(p_jump, clean_pos + target_offset, atol=0.1)


def test_gps_freeze_and_jamming():
    """Scenario 05 & Jamming: GPS signal freeze and loss."""
    # Test Freeze
    freeze_cfg = {
        "attack_type": "gps_spoofing",
        "attack_start_time": 10.0,
        "attack_duration": 15.0,
        "attack_parameters": {"mode": "freeze"}
    }
    engine = AttackSimulationEngine(freeze_cfg)

    # Moving drone position
    p1 = engine.inject_gps(np.array([10.0, 0.0, 10.0]), t=10.0)
    p2 = engine.inject_gps(np.array([15.0, 5.0, 10.0]), t=12.0)
    p3 = engine.inject_gps(np.array([20.0, 10.0, 10.0]), t=14.0)

    # All outputs during attack must be frozen to p1
    assert np.allclose(p1, [10.0, 0.0, 10.0])
    assert np.allclose(p2, [10.0, 0.0, 10.0])
    assert np.allclose(p3, [10.0, 0.0, 10.0])

    # Test Jamming (Signal Loss)
    jam_cfg = {
        "attack_type": "gps_spoofing",
        "attack_start_time": 10.0,
        "attack_duration": 15.0,
        "attack_parameters": {"mode": "jamming"}
    }
    jam_engine = AttackSimulationEngine(jam_cfg)
    assert jam_engine.inject_gps(np.array([1.0, 2.0, 3.0]), t=12.0) is None


def test_imu_manipulation_attack():
    """Scenario 06: IMU constant bias injection."""
    sc_path = root_dir / "simulation" / "scenarios" / "scenario_06.yaml"
    engine = AttackSimulationEngine(sc_path)

    accel = np.array([0.0, 0.0, 9.80665])
    gyro = np.array([0.0, 0.0, 0.0])

    # Active attack at t=12.0
    atk_accel, atk_gyro = engine.inject_imu(accel, gyro, t=12.0)
    # Scenario 06 bias: ax: 2.2, ay: -1.5, az: 0.5, gz: 0.05
    expected_accel = accel + np.array([2.2, -1.5, 0.5])
    expected_gyro = gyro + np.array([0.0, 0.0, 0.05])

    assert np.allclose(atk_accel, expected_accel, atol=1e-3)
    assert np.allclose(atk_gyro, expected_gyro, atol=1e-3)


def test_lidar_range_compression():
    """Scenario 08: LiDAR range deception."""
    sc_path = root_dir / "simulation" / "scenarios" / "scenario_08.yaml"
    engine = AttackSimulationEngine(sc_path)

    clean_range = 10.0
    # Active attack at t=12.0
    atk_range = engine.inject_lidar(clean_range, t=12.0)
    # Scenario 08: range_compression_ratio: 0.5, offset_meters: -4.0 -> 10 * 0.5 - 4.0 = 1.0m
    assert pytest.approx(atk_range, abs=0.1) == 1.0


def test_barometer_and_magnetometer_attacks():
    """Magnetometer heading bias and barometer altitude bias."""
    # Test Magnetometer (Scenario 09)
    sc09_path = root_dir / "simulation" / "scenarios" / "scenario_09.yaml"
    engine_mag = AttackSimulationEngine(sc09_path)
    clean_mag = np.array([20.0, 0.0, -40.0])  # micro-Tesla
    atk_mag = engine_mag.inject_magnetometer(clean_mag, t=12.0)
    # Should rotate horizontal components by 45 degrees
    assert atk_mag[0] != clean_mag[0]
    assert atk_mag[1] != clean_mag[1]
    assert atk_mag[2] == clean_mag[2]  # Vertical component preserved

    # Test Barometer manipulation
    baro_cfg = {
        "attack_type": "sensor_manipulation",
        "attack_start_time": 10.0,
        "attack_duration": 15.0,
        "attack_parameters": {
            "target_sensor": "barometer",
            "altitude_bias": 18.5
        }
    }
    engine_baro = AttackSimulationEngine(baro_cfg)
    clean_alt = 15.0
    atk_alt = engine_baro.inject_barometer(clean_alt, t=12.0)
    assert pytest.approx(atk_alt, abs=0.01) == 33.5


def test_communication_disruption_and_packet_loss():
    """Scenarios 10 & 11: Packet drop and transmission latency queue."""
    # Test Packet Loss (Scenario 10: 30% drop)
    sc10_path = root_dir / "simulation" / "scenarios" / "scenario_10.yaml"
    engine_drop = AttackSimulationEngine(sc10_path)

    drops = [engine_drop.evaluate_packet_loss(t=12.0) for _ in range(200)]
    drop_rate = sum(drops) / len(drops)
    # Expected drop rate is ~0.30 in scenario 10
    assert 0.15 < drop_rate < 0.45

    # Test Delay Queue (Scenario 11: 500 ms delay)
    sc11_path = root_dir / "simulation" / "scenarios" / "scenario_11.yaml"
    engine_delay = AttackSimulationEngine(sc11_path)

    # Queue packet at t=10.0 with 0.5s delay
    ready_initial = engine_delay.queue_delayed_packet(payload="pkt1", t=10.0)
    assert len(ready_initial) == 0  # Delayed, should not pop immediately

    # Check at t=10.2s (still within delay window)
    ready_mid = engine_delay.queue_delayed_packet(payload="pkt2", t=10.2)
    assert len(ready_mid) == 0

    # Check at t=10.6s (pkt1 release time is 10.5s, so it must release!)
    ready_released = engine_delay.queue_delayed_packet(payload="pkt3", t=10.6)
    assert len(ready_released) >= 1
    assert ready_released[0][1] == "pkt1"


def test_coordinated_multi_sensor_attack():
    """Scenario 14: IMU Bias Plus GPS Spoofing compound attack."""
    sc14_path = root_dir / "simulation" / "scenarios" / "scenario_14.yaml"
    engine = AttackSimulationEngine(sc14_path)

    clean_gps = np.array([0.0, 0.0, 10.0])
    clean_accel = np.array([0.0, 0.0, 9.80665])
    clean_gyro = np.array([0.0, 0.0, 0.0])

    # In active attack, both GPS and IMU must be simultaneously perturbed
    atk_gps = engine.inject_gps(clean_gps, t=12.0)
    atk_accel, atk_gyro = engine.inject_imu(clean_accel, clean_gyro, t=12.0)

    assert not np.allclose(atk_gps, clean_gps)
    assert not np.allclose(atk_accel, clean_accel)

    gt = engine.get_ground_truth(t=12.0)
    assert gt["is_active"] is True
    assert "gps" in gt["compromised_sensors"]
    assert "imu" in gt["compromised_sensors"]
