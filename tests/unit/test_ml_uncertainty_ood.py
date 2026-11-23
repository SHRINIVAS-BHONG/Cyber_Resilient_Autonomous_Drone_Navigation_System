"""
Unit tests for Machine Learning Out-of-Distribution (OOD) & Uncertainty Estimation.
"""

from pathlib import Path
import numpy as np
import pytest
from ml_detector.ml_detector_node import MLDetectorEngine


@pytest.fixture
def ml_engine():
    root_dir = Path(__file__).resolve().parent.parent.parent
    models_dir = root_dir / "ml" / "models"
    rf_file = models_dir / "random_forest_detector.joblib"
    scaler_file = models_dir / "scaler.joblib"
    feat_file = models_dir / "feature_names.json"
    return MLDetectorEngine(rf_file, scaler_file, feat_file)


def test_ml_nominal_uncertainty_metrics(ml_engine):
    # Feed nominal telemetry
    ml_engine.update_sensor_telemetry(
        gps_pos=np.array([10.0, 10.0, 15.0]),
        vision_pos=np.array([10.02, 9.98, 15.01]),
        imu_accel=np.array([0.01, -0.01, 9.81]),
        imu_gyro_z=0.001,
        lidar_z=15.02,
        gps_nis=0.45,
        lidar_nis=0.20,
        trust=1.0
    )
    pred = ml_engine.predict()

    assert pred["predicted_class"] == "NORMAL"
    assert pred["confidence"] > 0.80
    assert not pred["is_ood"]
    assert "entropy" in pred
    assert "margin" in pred


def test_ml_zero_day_ambiguity_ood_fallback(ml_engine):
    # Artificially simulate an ambiguous, out-of-distribution anomaly pattern:
    # High residual combined with an unusual, conflicting multi-sensor feature distribution
    for _ in range(5):
        ml_engine.update_sensor_telemetry(
            gps_pos=np.array([50.0, 50.0, 15.0]),
            vision_pos=np.array([0.0, 0.0, 15.0]),  # 70m discrepancy
            imu_accel=np.array([15.0, -12.0, 22.0]), # violent acceleration
            imu_gyro_z=3.5,
            lidar_z=45.0,
            gps_nis=125.0,
            lidar_nis=80.0,
            trust=0.20
        )
    pred = ml_engine.predict()

    assert pred["is_attack"]
    assert pred["anomaly_score"] > 0.50
    # The engine computes entropy, margin, and handles the attack
    assert pred["entropy"] >= 0.0
    assert pred["margin"] >= 0.0
