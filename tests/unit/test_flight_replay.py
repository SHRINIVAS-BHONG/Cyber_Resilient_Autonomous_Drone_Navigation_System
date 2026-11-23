"""
Unit tests for Flight Log & Telemetry Replay Engine.
"""

from pathlib import Path
import numpy as np
import pytest
from simulation.replay.flight_log_replayer import FlightLogReplayer
from state_estimator.ekf_10dof import EKF10DOF
from residual_detector.detector_engine import ResidualDetectorEngine
from resilience_manager.manager_engine import ResilienceManagerEngine


def test_synthetic_mission_generation():
    replayer = FlightLogReplayer(dt=0.1)
    num_frames = replayer.generate_synthetic_mission(duration_s=10.0, speed_ms=5.0)

    assert num_frames == 100
    assert len(replayer.frames) == 100
    assert replayer.total_duration_s == 10.0

    first_frame = replayer.frames[0]
    assert first_frame.true_pos_enu.shape == (3,)
    assert first_frame.imu_accel.shape == (3,)
    assert first_frame.gps_pos_meas.shape == (3,)


def test_playback_with_defense_nominal():
    replayer = FlightLogReplayer(dt=0.1)
    replayer.generate_synthetic_mission(duration_s=8.0)

    ekf = EKF10DOF()
    detector = ResidualDetectorEngine()
    resilience = ResilienceManagerEngine()

    result = replayer.execute_playback_with_defense(
        ekf_instance=ekf,
        detector_instance=detector,
        resilience_instance=resilience,
        attack_config={"type": "none"}
    )

    assert result["mission_success"]
    assert result["resilient_rmse_m"] < 0.50
    assert result["final_nav_mode"] == "NORMAL_MISSION"


def test_playback_with_defense_under_gps_replay_attack():
    replayer = FlightLogReplayer(dt=0.1)
    replayer.generate_synthetic_mission(duration_s=15.0)

    ekf = EKF10DOF()
    detector = ResidualDetectorEngine(consecutive_alarms_to_confirm=3)
    resilience = ResilienceManagerEngine()

    result = replayer.execute_playback_with_defense(
        ekf_instance=ekf,
        detector_instance=detector,
        resilience_instance=resilience,
        attack_config={"type": "gps_replay", "start_time": 6.0}
    )

    # Under attack, raw unmitigated error is large, while resilient RMSE is protected
    assert result["resilient_rmse_m"] < 1.0
    assert result["error_reduction_pct"] > 60.0
    assert result["ttd_s"] is not None
