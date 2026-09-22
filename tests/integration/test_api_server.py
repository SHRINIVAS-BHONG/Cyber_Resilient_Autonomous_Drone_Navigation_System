"""
Integration tests for FastAPI Cloud REST & WebSocket Telemetry Gateway.
Verifies genuine telemetry ingestion, real dataset replay, and attack triggers.
"""

import pytest
from fastapi.testclient import TestClient
from api.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OPERATIONAL"
    assert "version" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "active_sensors" in data
    assert "sensor_trust" in data


def test_latest_telemetry_endpoint(client):
    response = client.get("/telemetry/latest")
    assert response.status_code == 200
    data = response.json()
    assert len(data["position_enu"]) == 3
    assert len(data["velocity_enu"]) == 3
    assert "gps_nis" in data
    assert "ml_predicted_class" in data
    assert "sensor_trust" in data


def test_telemetry_ingest_endpoint(client):
    payload = {
        "timestamp": "2026-09-22T12:00:00Z",
        "position_enu": [12.5, -4.2, 10.0],
        "velocity_enu": [1.2, -0.3, 0.0],
        "yaw_deg": 45.0,
        "raw_gps_enu": [12.8, -4.1, 10.1],
        "gps_nis": 2.15,
        "chi2_attack_state": "NORMAL",
        "ml_predicted_class": "NORMAL",
        "ml_confidence": 0.995,
        "navigation_mode": "NORMAL_MISSION",
        "active_sensors": ["gps", "imu", "lidar", "vision_pose"],
        "isolated_sensors": [],
        "sensor_trust": {"gps": 1.0, "imu": 1.0, "lidar": 1.0, "vision_pose": 1.0}
    }
    response = client.post("/telemetry/ingest", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "INGESTED"

    # Verify latest telemetry reflects ingested packet
    latest = client.get("/telemetry/latest").json()
    assert latest["position_enu"] == [12.5, -4.2, 10.0]
    assert latest["gps_nis"] == 2.15


def test_attack_injection_and_clear(client):
    attack_payload = {
        "attack_type": "gps_spoofing",
        "magnitude": 25.0,
        "duration_sec": 15.0,
        "target_sensor": "gps"
    }
    response = client.post("/attack/inject", json=attack_payload)
    assert response.status_code == 202
    assert response.json()["status"] == "ATTACK_INJECTED"

    clear_resp = client.post("/attack/clear")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["status"] == "ATTACK_CLEARED"


def test_ml_metrics_endpoint(client):
    response = client.get("/ml/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "overall_accuracy" in data
    assert "macro_f1" in data
    assert data["overall_accuracy"] >= 0.95


def test_benchmarks_endpoint(client):
    response = client.get("/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "scenario" in data[0]
    assert "rmse_improvement_pct" in data[0]
