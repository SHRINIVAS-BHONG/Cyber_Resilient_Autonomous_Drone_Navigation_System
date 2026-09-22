"""
Unit Tests for Trained Machine Learning Attack Detector.

Validates that:
1. Model artifacts exist and can be cleanly loaded (model, scaler, feature names).
2. Input vectors are preprocessed properly.
3. Model predicts expected classes on synthetic inputs.
4. Inference latency is sub-millisecond (< 5 ms).
"""

import json
import time
from pathlib import Path
import joblib
import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def ml_artifacts():
    models_dir = root_dir / "ml" / "models"
    rf_path = models_dir / "random_forest_detector.joblib"
    scaler_path = models_dir / "scaler.joblib"
    feat_path = models_dir / "feature_names.json"

    assert rf_path.exists(), f"Model missing: {rf_path}"
    assert scaler_path.exists(), f"Scaler missing: {scaler_path}"
    assert feat_path.exists(), f"Features missing: {feat_path}"

    model = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)
    with open(feat_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    return {"model": model, "scaler": scaler, "features": features}


def test_ml_model_loading(ml_artifacts):
    """Verifies model and scaler artifacts are valid scikit-learn estimators."""
    model = ml_artifacts["model"]
    scaler = ml_artifacts["scaler"]
    features = ml_artifacts["features"]

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert hasattr(scaler, "transform")
    assert len(features) == 13


def test_ml_nominal_flight_classification(ml_artifacts):
    """Verifies that a nominal feature vector is classified as Class 0 (NORMAL)."""
    model = ml_artifacts["model"]
    scaler = ml_artifacts["scaler"]

    # Nominal flight vector: minimal residuals, low variance, trust=1.0
    nominal_vec = np.array([[
        0.5,   # gps_nis
        0.15,  # gps_pos_residual_norm
        0.05,  # gps_vel_residual_norm
        0.02,  # imu_accel_norm_diff
        0.005, # imu_accel_var
        0.002, # imu_gyro_var
        0.3,   # lidar_nis
        0.02,  # lidar_residual_z
        0.1,   # vision_gps_discrepancy
        0.02,  # vision_lidar_z_discrepancy
        0.05,  # accel_kinematic_consistency
        0.0,   # consecutive_anomalies
        1.0    # sensor_trust_composite
    ]])

    scaled = scaler.transform(nominal_vec)
    pred = model.predict(scaled)[0]
    probs = model.predict_proba(scaled)[0]

    assert pred == 0  # NORMAL
    assert probs[0] > 0.80  # High confidence nominal


def test_ml_gps_spoofing_classification(ml_artifacts):
    """Verifies that high GPS NIS and discrepancy vector is classified as Class 1 (GPS_SPOOFING)."""
    model = ml_artifacts["model"]
    scaler = ml_artifacts["scaler"]

    spoofed_vec = np.array([[
        45.0,  # gps_nis (huge)
        18.5,  # gps_pos_residual_norm
        3.2,   # gps_vel_residual_norm
        0.04,  # imu_accel_norm_diff (normal IMU)
        0.008, # imu_accel_var
        0.003, # imu_gyro_var
        0.4,   # lidar_nis
        0.03,  # lidar_residual_z
        18.5,  # vision_gps_discrepancy
        0.03,  # vision_lidar_z_discrepancy
        2.5,   # accel_kinematic_consistency
        5.0,   # consecutive_anomalies
        0.2    # sensor_trust_composite (decayed)
    ]])

    scaled = scaler.transform(spoofed_vec)
    pred = model.predict(scaled)[0]
    assert pred == 1  # GPS_SPOOFING


def test_ml_inference_latency(ml_artifacts):
    """Ensures inference latency is well below real-time 10 Hz flight deadline (< 5 ms)."""
    model = ml_artifacts["model"]
    scaler = ml_artifacts["scaler"]

    dummy_batch = np.random.normal(0.0, 1.0, size=(100, 13))
    scaled = scaler.transform(dummy_batch)

    t0 = time.perf_counter()
    _ = model.predict(scaled)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 100.0

    assert elapsed_ms < 1.0, f"Inference took {elapsed_ms:.3f} ms, expected < 1.0 ms"
