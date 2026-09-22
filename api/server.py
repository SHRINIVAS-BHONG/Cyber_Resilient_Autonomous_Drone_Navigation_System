"""
Production FastAPI Cloud REST & WebSocket Telemetry Gateway.

Exposes vehicle telemetry, health status, benchmark summaries, and attack injection triggers
for remote Ground Control Stations (GCS) and cloud dashboards over REST and WebSockets.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Root path
root_dir = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Cyber-Resilient Drone Navigation Gateway",
    description="Production REST & WebSocket Telemetry API for Autonomous Drones under Cyber Attacks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for remote GCS web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# Data Models
# -------------------------------------------------------------
class AttackInjectionRequest(BaseModel):
    attack_type: str = Field(..., description="Type of cyber attack: gps_spoofing, imu_manipulation, lidar_corruption, multi_attack")
    magnitude: float = Field(..., ge=0.5, le=100.0, description="Severity or bias magnitude (meters or m/s²)")
    duration_sec: float = Field(default=20.0, ge=1.0, le=300.0, description="Duration of attack in seconds")
    target_sensor: Optional[str] = Field(default="gps", description="Target sensor name")


class VehicleHealthResponse(BaseModel):
    timestamp: str
    status: str
    navigation_mode: str
    speed_factor: float
    active_sensors: List[str]
    isolated_sensors: List[str]
    sensor_trust: Dict[str, float]
    is_degraded: bool


class TelemetryPacket(BaseModel):
    timestamp: str
    position_enu: List[float]
    velocity_enu: List[float]
    yaw_deg: float
    raw_gps_enu: List[float]
    gps_nis: float
    chi2_attack_state: str
    ml_predicted_class: str
    ml_confidence: float
    navigation_mode: str
    active_sensors: List[str]


# -------------------------------------------------------------
# In-Memory State Manager
# -------------------------------------------------------------
class SystemStateManager:
    def __init__(self):
        self.active_attack: Optional[Dict] = None
        self.attack_start_time: Optional[float] = None
        self.sim_time: float = 0.0

    def get_latest_telemetry(self) -> Dict:
        # Ground truth circular patrol trajectory
        t = self.sim_time
        true_x = float(10.0 * np.sin(0.2 * t))
        true_y = float(10.0 * np.cos(0.2 * t))
        true_z = float(10.0 + 1.0 * np.sin(0.1 * t))

        # Check active attack
        gps_x, gps_y, gps_z = true_x, true_y, true_z
        gps_nis = 0.45
        chi2_state = "NORMAL"
        ml_class = "NORMAL"
        ml_conf = 0.95
        nav_mode = "NORMAL_MISSION"
        active_sensors = ["gps", "imu", "lidar", "vision_pose"]
        isolated = []

        if self.active_attack:
            elapsed = self.sim_time - self.attack_start_time
            if elapsed < self.active_attack["duration_sec"]:
                atype = self.active_attack["attack_type"]
                mag = self.active_attack["magnitude"]
                if "gps" in atype:
                    gps_x += mag
                    gps_y -= mag * 0.7
                    gps_nis = 35.8
                    chi2_state = "ATTACK_CONFIRMED"
                    ml_class = "GPS_SPOOFING"
                    ml_conf = 0.98
                    nav_mode = "DEGRADED_OPTICAL_LIDAR"
                    active_sensors.remove("gps")
                    isolated.append("gps")
                elif "imu" in atype:
                    chi2_state = "ATTACK_CONFIRMED"
                    ml_class = "IMU_MANIPULATION"
                    ml_conf = 0.96
                    nav_mode = "HOLD_POSITION"
                    active_sensors.remove("imu")
                    isolated.append("imu")
            else:
                self.active_attack = None

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "position_enu": [true_x, true_y, true_z],
            "velocity_enu": [float(2.0 * np.cos(0.2 * t)), float(-2.0 * np.sin(0.2 * t)), 0.0],
            "yaw_deg": float(np.degrees(0.2 * t) % 360),
            "raw_gps_enu": [gps_x, gps_y, gps_z],
            "gps_nis": round(gps_nis, 2),
            "chi2_attack_state": chi2_state,
            "ml_predicted_class": ml_class,
            "ml_confidence": round(ml_conf, 3),
            "navigation_mode": nav_mode,
            "active_sensors": active_sensors,
            "isolated_sensors": isolated,
            "sensor_trust": {
                "gps": 0.15 if "gps" in isolated else 1.0,
                "imu": 0.20 if "imu" in isolated else 1.0,
                "lidar": 1.0,
                "vision_pose": 1.0
            }
        }


state_manager = SystemStateManager()


# -------------------------------------------------------------
# REST Endpoints
# -------------------------------------------------------------
@app.get("/", tags=["System"])
def get_root():
    return {
        "system": "Cyber-Resilient Autonomous Drone Navigation System",
        "version": "1.0.0",
        "status": "OPERATIONAL",
        "documentation": "/docs",
        "websocket_stream": "/ws/telemetry"
    }


@app.get("/health", response_model=VehicleHealthResponse, tags=["Health & Status"])
def get_health():
    telem = state_manager.get_latest_telemetry()
    is_degraded = telem["navigation_mode"] != "NORMAL_MISSION"
    return VehicleHealthResponse(
        timestamp=telem["timestamp"],
        status="ATTACK_CONTAINMENT" if is_degraded else "HEALTHY_NORMAL",
        navigation_mode=telem["navigation_mode"],
        speed_factor=0.6 if is_degraded else 1.0,
        active_sensors=telem["active_sensors"],
        isolated_sensors=telem["isolated_sensors"],
        sensor_trust=telem["sensor_trust"],
        is_degraded=is_degraded
    )


@app.get("/telemetry/latest", response_model=TelemetryPacket, tags=["Telemetry"])
def get_latest_telemetry():
    return state_manager.get_latest_telemetry()


@app.get("/benchmarks", tags=["Evaluation"])
def get_benchmark_results():
    bench_file = root_dir / "reports" / "experiment_results" / "benchmark_summary.json"
    if not bench_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark summary file not found. Run scripts/run_benchmarks.py first.")
    with open(bench_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/ml/metrics", tags=["Machine Learning"])
def get_ml_metrics():
    metrics_file = root_dir / "reports" / "experiment_results" / "ml_metrics.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="ML metrics file not found. Run ml/evaluate.py first.")
    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/attack/inject", status_code=status.HTTP_202_ACCEPTED, tags=["Cyber Attack Injection"])
def inject_attack(req: AttackInjectionRequest):
    valid_attacks = ["gps_spoofing", "imu_manipulation", "lidar_corruption", "multi_attack"]
    if req.attack_type.lower() not in valid_attacks:
        raise HTTPException(status_code=400, detail=f"Invalid attack type. Must be one of: {valid_attacks}")

    state_manager.active_attack = req.dict()
    state_manager.attack_start_time = state_manager.sim_time
    return {
        "status": "ATTACK_INJECTED",
        "attack_details": req.dict(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/attack/clear", tags=["Cyber Attack Injection"])
def clear_attack():
    state_manager.active_attack = None
    return {
        "status": "ATTACK_CLEARED",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# -------------------------------------------------------------
# Streaming WebSocket Endpoint (10 Hz)
# -------------------------------------------------------------
@app.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            state_manager.sim_time += 0.1
            packet = state_manager.get_latest_telemetry()
            await websocket.send_json(packet)
            await asyncio.sleep(0.10)  # 10 Hz broadcast
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
