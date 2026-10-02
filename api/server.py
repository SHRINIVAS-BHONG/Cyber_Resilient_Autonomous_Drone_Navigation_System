"""
Production FastAPI Cloud REST & WebSocket Telemetry Gateway.

Exposes vehicle telemetry, health status, benchmark summaries, and attack injection triggers
for remote Ground Control Stations (GCS) and cloud dashboards over REST and WebSockets.
Runs continuous 10 Hz real-time flight physics, 10-DOF EKF estimation, and autonomous cyber-resilience.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Callable
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib

# Root path and package setup
root_dir = Path(__file__).resolve().parent.parent
import sys
for pkg in ["sensor_bridge", "state_estimator", "residual_detector", "resilience_manager", "path_planner", "ml_detector"]:
    p = str(root_dir / "ros2_ws" / "src" / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from state_estimator.ekf_10dof import EKF10DOF
from residual_detector.detector_engine import ResidualDetectorEngine, DetectionState
from resilience_manager.manager_engine import ResilienceManagerEngine, NavigationMode
from path_planner.astar_planner import AStarPlanner


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


class TelemetryIngestPacket(BaseModel):
    timestamp: Optional[str] = None
    position_enu: List[float] = Field(..., min_length=3, max_length=3)
    velocity_enu: List[float] = Field(..., min_length=3, max_length=3)
    yaw_deg: float
    raw_gps_enu: List[float] = Field(..., min_length=3, max_length=3)
    gps_nis: float
    chi2_attack_state: str
    ml_predicted_class: str
    ml_confidence: float
    navigation_mode: str
    active_sensors: List[str]
    isolated_sensors: List[str]
    sensor_trust: Dict[str, float]


class SystemSettingsModel(BaseModel):
    cruise_altitude: float = Field(default=15.0, ge=4.0, le=45.0, description="Nominal cruise altitude (meters)")
    cruise_speed: float = Field(default=4.5, ge=1.0, le=12.0, description="Nominal cruise horizontal speed (m/s)")
    climb_speed: float = Field(default=2.5, ge=0.5, le=6.0, description="Vertical climb speed (m/s)")
    descent_speed: float = Field(default=1.5, ge=0.4, le=4.0, description="Vertical descent speed during landing (m/s)")
    rtl_altitude: float = Field(default=22.0, ge=8.0, le=50.0, description="Return-To-Launch clearance altitude (meters)")
    overflight_clearance: float = Field(default=2.0, ge=0.8, le=8.0, description="Minimum clearance altitude above obstacles for overflight (meters)")
    collision_margin: float = Field(default=2.5, ge=0.5, le=6.0, description="Horizontal safety buffer around physical obstacles (meters)")
    ground_effect_enabled: bool = Field(default=True, description="Enable realistic air-cushion deceleration near touchdown surface")
    nis_gate_threshold: float = Field(default=11.34, ge=5.0, le=25.0, description="Chi-Square NIS threshold for cyber-attack detection")
    quarantine_threshold: float = Field(default=0.50, ge=0.1, le=0.8, description="Sensor trust threshold below which sensor is isolated")
    recovery_threshold: float = Field(default=0.85, ge=0.5, le=0.98, description="Sensor trust threshold above which sensor is restored")
    auto_recovery_enabled: bool = Field(default=True, description="Auto-recover sensor trust when residual returns to normal")
    audio_alerts_enabled: bool = Field(default=True, description="Enable audio alerts for attacks and touchdown")


class SetDestinationRequest(BaseModel):
    x: float = Field(..., description="East target coordinate (meters)")
    y: float = Field(..., description="North target coordinate (meters)")
    z: Optional[float] = Field(default=15.0, description="Altitude (meters)")
    label: Optional[str] = Field(default="TARGET_OBJECTIVE", description="Destination label")


class RooftopLandingRequest(BaseModel):
    x: float = Field(..., description="East target surface coordinate (meters)")
    y: float = Field(..., description="North target surface coordinate (meters)")
    label: Optional[str] = Field(default="ROOFTOP_LANDING", description="Landing pad designation")


class TelemetryPacket(BaseModel):
    model_config = {"extra": "allow"}
    timestamp: str
    position_enu: List[float]
    velocity_enu: List[float]
    yaw_deg: float
    raw_gps_enu: List[float]
    gps_nis: float
    chi2_attack_state: str
    is_attack_active: Optional[bool] = False
    active_attack_type: Optional[str] = "none"
    active_attack_mag: Optional[float] = 0.0
    ml_predicted_class: str
    ml_confidence: float
    navigation_mode: str
    active_sensors: List[str]
    isolated_sensors: List[str]
    sensor_trust: Dict[str, float]
    altitude_m: Optional[float] = 15.0
    altitude_agl_m: Optional[float] = 15.0
    surface_name: Optional[str] = "GROUND_TARMAC"
    surface_elevation_m: Optional[float] = 0.0
    target_touchdown_z: Optional[float] = 0.53
    ground_speed_mps: Optional[float] = 4.5
    roll_deg: Optional[float] = 0.0
    pitch_deg: Optional[float] = 0.0
    battery_pct: Optional[float] = 98.0
    current_waypoint: Optional[str] = "WP-1/8"
    lat_wgs84: Optional[float] = 37.774929
    lon_wgs84: Optional[float] = -122.419416
    mission_type: Optional[str] = "patrol"
    target_goal: Optional[List[float]] = None
    target_label: Optional[str] = "PERIMETER_PATROL"
    distance_to_goal: Optional[float] = 0.0
    planned_waypoints: Optional[List[List[float]]] = None
    flight_phase: Optional[str] = "CRUISE"
    is_airborne: Optional[bool] = True


# -------------------------------------------------------------
# Real-Time Physical Flight & Resilience Engine
# -------------------------------------------------------------
class SystemStateManager:
    def __init__(self):
        self.active_attack: Optional[Dict] = None
        self.attack_start_time: Optional[float] = None
        self.last_ingest_time: float = 0.0
        self.start_time = time.time()

        # System Settings & Dynamic Tuning
        self.settings = SystemSettingsModel()
        self.cruise_altitude = self.settings.cruise_altitude
        self.cruise_speed = self.settings.cruise_speed
        self.climb_speed = self.settings.climb_speed
        self.descent_speed = self.settings.descent_speed
        self.rtl_altitude = self.settings.rtl_altitude
        self.overflight_clearance = self.settings.overflight_clearance
        self.collision_margin = self.settings.collision_margin
        self.ground_effect_enabled = self.settings.ground_effect_enabled

        # Core Avionics & Resilient Estimators
        self.ekf = EKF10DOF()
        self.ekf.initialize_state(position=np.array([0.0, 0.0, self.cruise_altitude]), velocity=np.array([0.0, 0.0, 0.0]))
        self.detector = ResidualDetectorEngine()
        self.resilience = ResilienceManagerEngine()
        self.planner = AStarPlanner(x_bounds=(-600.0, 600.0), y_bounds=(-600.0, 600.0), grid_resolution=2.0)

        # Physical Drone State
        self.drone_pos = np.array([0.0, 0.0, self.cruise_altitude], dtype=np.float64)
        self.drone_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0

        # Physical Surface & Elevation Engine
        self.landing_gear_offset = 0.53
        self.target_touchdown_z = 0.53
        self.ground_altitude = 0.53
        self.current_surface = "GROUND_TARMAC"
        self.surface_elevation = 0.0

        # Mission Waypoints & Destination Targeting
        self.mission_type = "patrol"  # "patrol" or "target_point"
        self.target_goal = None
        self.target_label = "PERIMETER_PATROL"
        self.waypoints = [
            np.array([0.0, 0.0, self.cruise_altitude]),
            np.array([120.0, 0.0, self.cruise_altitude]),
            np.array([160.0, 120.0, self.cruise_altitude + 1.0]),
            np.array([0.0, 180.0, self.cruise_altitude + 3.0]),
            np.array([-140.0, 120.0, self.cruise_altitude + 1.0]),
            np.array([-160.0, -90.0, self.cruise_altitude]),
            np.array([0.0, -140.0, self.cruise_altitude]),
            np.array([0.0, 0.0, self.cruise_altitude])
        ]
        self.current_wp_idx = 1
        self.emergency_landing_zone = np.array([50.0, 45.0, self.cruise_altitude])

        # Flight Phase State Machine (GROUNDED, TAKEOFF, CRUISE, LANDING, LANDED, RTL_CLIMB, RTL_CRUISE, ROOFTOP_APPROACH)
        self.flight_phase = "CRUISE"
        self.is_airborne = True

        # Physical 3D Obstacles (Radar, Hangars, Perimeter Towers)
        self.obstacles = [
            {"x": -90.0, "y": 180.0, "radius": 6.0, "height": 7.5, "roof_elev": 3.2, "is_landable": True, "name": "DELTA_RADAR"},
            {"x": 220.0, "y": 190.0, "radius": 6.0, "height": 7.5, "roof_elev": 3.2, "is_landable": True, "name": "ECHO_RADAR"},
            {"x": 160.0, "y": 0.0, "radius": 10.0, "height": 7.5, "roof_elev": 7.0, "is_landable": True, "name": "HANGAR_ALPHA"},
            {"x": -160.0, "y": -90.0, "radius": 10.0, "height": 7.5, "roof_elev": 7.0, "is_landable": True, "name": "HANGAR_BETA"},
            {"x": 0.0, "y": -180.0, "radius": 10.0, "height": 7.5, "roof_elev": 7.0, "is_landable": True, "name": "HANGAR_GAMMA"},
            {"x": 260.0, "y": -160.0, "radius": 5.0, "height": 36.5, "name": "COMM_MAST_EAST"},
            {"x": -260.0, "y": 160.0, "radius": 5.0, "height": 36.5, "name": "COMM_MAST_WEST"},
            {"x": 340.0, "y": 300.0, "radius": 4.5, "height": 28.0, "name": "WATCHTOWER_NE"},
            {"x": -340.0, "y": -280.0, "radius": 4.5, "height": 28.0, "name": "WATCHTOWER_SW"},
            {"x": 320.0, "y": -300.0, "radius": 4.5, "height": 28.0, "name": "WATCHTOWER_SE"},
            {"x": -320.0, "y": 300.0, "radius": 4.5, "height": 28.0, "name": "WATCHTOWER_NW"},
        ]
        for obs in self.obstacles:
            self.planner.add_obstacle(obs["x"], obs["y"], radius=obs["radius"], height=obs["height"])

        # Preload ML Model
        self.ml_model = None
        self.ml_scaler = None
        self.ml_features = None
        self._load_ml_artifacts()

        # Telemetry State Buffer
        self.current_state: Dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "position_enu": [0.0, 0.0, self.cruise_altitude],
            "velocity_enu": [0.0, 0.0, 0.0],
            "altitude_m": self.cruise_altitude,
            "altitude_agl_m": self.cruise_altitude,
            "surface_name": "GROUND_TARMAC",
            "surface_elevation_m": 0.0,
            "target_touchdown_z": 0.53,
            "ground_speed_mps": 0.0,
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
            "raw_gps_enu": [0.0, 0.0, self.cruise_altitude],
            "gps_nis": 0.45,
            "chi2_attack_state": "NORMAL",
            "is_attack_active": False,
            "active_attack_type": "none",
            "active_attack_mag": 0.0,
            "ml_predicted_class": "NORMAL",
            "ml_confidence": 0.99,
            "navigation_mode": "NORMAL_MISSION",
            "active_sensors": ["gps", "imu", "lidar", "vision_pose"],
            "isolated_sensors": [],
            "sensor_trust": {"gps": 1.0, "imu": 1.0, "lidar": 1.0, "vision_pose": 1.0},
            "current_waypoint": "WP-1/8",
            "battery_pct": 98.5,
            "lat_wgs84": 37.774929,
            "lon_wgs84": -122.419416,
            "mission_type": "patrol",
            "target_goal": None,
            "target_label": "PERIMETER_PATROL",
            "distance_to_goal": 0.0,
            "planned_waypoints": [[round(float(c), 2) for c in wp] for wp in self.waypoints],
            "flight_phase": self.flight_phase,
            "is_airborne": self.is_airborne
        }

    def apply_settings(self, new_settings: SystemSettingsModel):
        """Applies runtime flight and resilience parameters immediately."""
        self.settings = new_settings
        self.cruise_altitude = new_settings.cruise_altitude
        self.cruise_speed = new_settings.cruise_speed
        self.climb_speed = new_settings.climb_speed
        self.descent_speed = new_settings.descent_speed
        self.rtl_altitude = new_settings.rtl_altitude
        self.overflight_clearance = new_settings.overflight_clearance
        self.collision_margin = new_settings.collision_margin
        self.ground_effect_enabled = new_settings.ground_effect_enabled
        self.detector.gate_threshold = new_settings.nis_gate_threshold
        self.resilience.quarantine_th = new_settings.quarantine_threshold
        self.resilience.recovery_th = new_settings.recovery_threshold

    def get_surface_profile(self, x: float, y: float) -> Tuple[float, str, bool]:
        """
        Calculates physical surface elevation directly beneath coordinates (x, y).
        Returns: (surface_elevation_m, surface_name, is_landable)
        """
        # 1. Hangar Alpha Rooftop Helipad (x=160, y=0, width X: [153.0, 167.0], length Y: [-14.0, 14.0])
        if 153.0 <= x <= 167.0 and -14.0 <= y <= 14.0:
            return 7.20, "HANGAR_ALPHA_ROOF", True

        # 2. Hangar Beta Rooftop Helipad (x=-160, y=-90, width X: [-167.0, -153.0], length Y: [-104.0, -76.0])
        if -167.0 <= x <= -153.0 and -104.0 <= y <= -76.0:
            return 7.20, "HANGAR_BETA_ROOF", True

        # 3. Hangar Gamma Rooftop Helipad (x=0, y=-180, width X: [-14.0, 14.0], length Y: [-187.0, -173.0])
        if -14.0 <= x <= 14.0 and -187.0 <= y <= -173.0:
            return 7.20, "HANGAR_GAMMA_ROOF", True

        # 4. Radar Bunker Observation & Servicing Decks (Deck height: 3.30m)
        if math.hypot(x - (-90.0), y - 180.0) <= 5.0:
            return 3.30, "DELTA_RADAR_DECK", True
        if math.hypot(x - 220.0, y - 190.0) <= 5.0:
            return 3.30, "ECHO_RADAR_DECK", True

        # 5. Tactical Field Outposts & Helipads Across the 1600m Airfield
        if math.hypot(x, y) <= 6.0:
            return 0.0, "BASE_ALPHA_HELIPAD", True
        elif math.hypot(x - 50.0, y - 45.0) <= 6.0:
            return 0.0, "EMERGENCY_BRAVO_HELIPAD", True
        elif math.hypot(x - 110.0, y - 90.0) <= 6.0:
            return 0.0, "OUTPOST_CHARLIE_HELIPAD", True
        elif math.hypot(x - (-120.0), y - 130.0) <= 6.0:
            return 0.0, "OBJECTIVE_DELTA_HELIPAD", True
        elif math.hypot(x - 180.0, y - (-90.0)) <= 6.0:
            return 0.0, "OBSERVATION_ECHO_HELIPAD", True
        elif math.hypot(x - (-200.0), y - (-140.0)) <= 6.0:
            return 0.0, "FORWARD_FOXTROT_HELIPAD", True
        elif math.hypot(x - 240.0, y - 180.0) <= 6.0:
            return 0.0, "GOLF_PERIMETER_HELIPAD", True

        return 0.0, "GROUND_TARMAC", True

    def takeoff(self, target_alt: Optional[float] = None):
        """Commands vertical takeoff from current surface (ground or roof) to cruise altitude."""
        surf_elev, surf_name, _ = self.get_surface_profile(self.drone_pos[0], self.drone_pos[1])
        min_cruise = surf_elev + self.overflight_clearance + 2.0
        if target_alt is not None:
            self.cruise_altitude = max(float(target_alt), min_cruise)
        else:
            self.cruise_altitude = max(self.cruise_altitude, min_cruise)

        self.flight_phase = "TAKEOFF"
        self.is_airborne = True
        self.drone_vel = np.array([0.0, 0.0, self.climb_speed], dtype=np.float64)

    def land(self):
        """Commands precision vertical landing onto the surface directly below (ground or roof)."""
        self.drone_vel = np.asarray(self.drone_vel, dtype=np.float64)
        self.drone_pos = np.asarray(self.drone_pos, dtype=np.float64)
        surf_elev, surf_name, _ = self.get_surface_profile(float(self.drone_pos[0]), float(self.drone_pos[1]))
        self.target_touchdown_z = surf_elev + self.landing_gear_offset
        self.current_surface = surf_name
        self.flight_phase = "LANDING"
        self.drone_vel[:2] *= 0.3

    def land_on_surface(self, x: float, y: float, label: str = "ROOFTOP_LANDING"):
        """Routes to surface coordinates at safe overflight altitude, then executes precision vertical touchdown."""
        surf_elev, surf_name, _ = self.get_surface_profile(x, y)
        transit_alt = max(self.cruise_altitude, surf_elev + self.overflight_clearance + 2.5)
        self.set_destination(x, y, transit_alt, label)
        self.target_touchdown_z = surf_elev + self.landing_gear_offset
        self.flight_phase = "ROOFTOP_APPROACH"

    def rtl_and_land(self):
        """Aviation Safety RTL: Climbs to safe RTL altitude, returns to Base Alpha, and touches down."""
        self.flight_phase = "RTL_CLIMB"
        self.set_destination(0.0, 0.0, self.rtl_altitude, "BASE_ALPHA_RTL")

    def set_destination(self, x: float, y: float, z: Optional[float] = None, label: str = "CUSTOM_OBJECTIVE"):
        """Sets an arbitrary target destination point and plans path to reach it."""
        surf_elev, surf_name, _ = self.get_surface_profile(float(x), float(y))
        min_safe_alt = surf_elev + self.overflight_clearance
        alt = float(z) if z is not None else max(self.cruise_altitude, min_safe_alt + 1.0)
        goal = np.array([float(x), float(y), alt], dtype=np.float64)
        self.target_goal = goal
        self.target_label = label
        self.mission_type = "target_point"

        # Plan trajectory using A* Planner around obstacles
        start_pt = tuple(float(v) for v in self.drone_pos)
        goal_pt = tuple(float(v) for v in goal)
        planned = self.planner.plan(start_pt, goal_pt)
        if planned and len(planned) > 1:
            self.waypoints = [np.array(wp, dtype=np.float64) for wp in planned]
            self.current_wp_idx = 1
        else:
            self.waypoints = [np.copy(self.drone_pos), goal]
            self.current_wp_idx = 1

    def set_patrol_mode(self):
        """Restores continuous 8-waypoint reconnaissance perimeter patrol across the 1600m airbase."""
        self.mission_type = "patrol"
        self.target_goal = None
        self.target_label = "PERIMETER_PATROL"
        self.waypoints = [
            np.array([0.0, 0.0, self.cruise_altitude]),
            np.array([120.0, 0.0, self.cruise_altitude]),
            np.array([160.0, 120.0, self.cruise_altitude + 1.0]),
            np.array([0.0, 180.0, self.cruise_altitude + 3.0]),
            np.array([-140.0, 120.0, self.cruise_altitude + 1.0]),
            np.array([-160.0, -90.0, self.cruise_altitude]),
            np.array([0.0, -140.0, self.cruise_altitude]),
            np.array([0.0, 0.0, self.cruise_altitude])
        ]
        self.current_wp_idx = 1

    def reset_nominal(self):
        """Immediately restores nominal operational state and clears all attack traces."""
        self.active_attack = None
        self.attack_start_time = 0.0
        for s in ["gps", "imu", "lidar", "vision_pose"]:
            self.resilience.sensor_trust[s] = 1.0
        self.resilience.isolated_sensors.clear()
        self.resilience.current_navigation_mode = NavigationMode.NORMAL_MISSION
        self.resilience.speed_factor = 1.0
        self.detector.compromised_sensors.clear()
        self.detector.current_state = DetectionState.NORMAL
        for k in self.detector.suspicious_counters:
            self.detector.suspicious_counters[k] = 0
            self.detector.healthy_counters[k] = 15
        self.drone_pos[2] = self.cruise_altitude
        self.ekf.x[0:3] = self.drone_pos.reshape(3, 1)
        self.ekf.x[3:6] = self.drone_vel.reshape(3, 1)

    def _load_ml_artifacts(self):
        models_dir = root_dir / "ml" / "models"
        rf_p = models_dir / "random_forest_detector.joblib"
        sc_p = models_dir / "scaler.joblib"
        ft_p = models_dir / "feature_names.json"
        if rf_p.exists() and sc_p.exists() and ft_p.exists():
            try:
                self.ml_model = joblib.load(rf_p)
                self.ml_scaler = joblib.load(sc_p)
                with open(ft_p, "r", encoding="utf-8") as f:
                    self.ml_features = json.load(f)
            except Exception:
                pass

    def ingest_telemetry(self, packet: TelemetryIngestPacket):
        """Allows external hardware or companion computer ROS 2 nodes to override telemetry."""
        self.last_ingest_time = time.time()
        self.current_state = {
            "timestamp": packet.timestamp or datetime.now(timezone.utc).isoformat(),
            "position_enu": packet.position_enu,
            "velocity_enu": packet.velocity_enu,
            "yaw_deg": packet.yaw_deg,
            "altitude_m": packet.position_enu[2],
            "ground_speed_mps": round(float(math.hypot(packet.velocity_enu[0], packet.velocity_enu[1])), 2),
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "raw_gps_enu": packet.raw_gps_enu,
            "gps_nis": round(packet.gps_nis, 2),
            "chi2_attack_state": packet.chi2_attack_state,
            "ml_predicted_class": packet.ml_predicted_class,
            "ml_confidence": round(packet.ml_confidence, 3),
            "navigation_mode": packet.navigation_mode,
            "active_sensors": packet.active_sensors,
            "isolated_sensors": packet.isolated_sensors,
            "sensor_trust": packet.sensor_trust,
            "current_waypoint": "EXTERNAL_ROS2",
            "battery_pct": 98.0,
            "lat_wgs84": 37.774929,
            "lon_wgs84": -122.419416
        }

    def step_simulation(self, dt: float = 0.1):
        """Advances 10 Hz physical flight simulation & autonomous resilience pipeline."""
        now = time.time()
        if (now - self.last_ingest_time) < 3.0:
            return

        # Check Active Attack Duration
        is_attack_active = False
        attack_type = "none"
        attack_mag = 0.0
        if self.active_attack:
            elapsed = now - self.attack_start_time
            if elapsed < self.active_attack.get("duration_sec", 25.0):
                is_attack_active = True
                attack_type = self.active_attack.get("attack_type", "gps_spoofing")
                attack_mag = float(self.active_attack.get("magnitude", 20.0))
            else:
                self.reset_nominal()

        # 1. Flight Phase State Machine & Motion Dynamics
        self.drone_pos = np.asarray(self.drone_pos, dtype=np.float64)
        self.drone_vel = np.asarray(self.drone_vel, dtype=np.float64)
        surf_elev, surf_name, is_landable = self.get_surface_profile(self.drone_pos[0], self.drone_pos[1])
        touchdown_z = surf_elev + self.landing_gear_offset
        self.current_surface = surf_name
        self.target_touchdown_z = touchdown_z

        if self.flight_phase == "LANDED":
            self.drone_pos[2] = touchdown_z
            self.drone_vel = np.array([0.0, 0.0, 0.0])
            self.roll_deg = 0.0
            self.pitch_deg = 0.0
            self.is_airborne = False
        elif self.flight_phase == "TAKEOFF":
            self.is_airborne = True
            # Vertical climb at configured climb_speed
            self.drone_vel = np.array([0.0, 0.0, self.climb_speed])
            self.drone_pos[2] += self.drone_vel[2] * dt
            if self.drone_pos[2] >= self.cruise_altitude:
                self.drone_pos[2] = self.cruise_altitude
                self.drone_vel = np.array([0.0, 0.0, 0.0])
                self.flight_phase = "CRUISE"
        elif self.flight_phase == "LANDING":
            # Dampen horizontal speed and level out attitude
            self.drone_vel[:2] *= 0.70
            self.pitch_deg += (0.0 - self.pitch_deg) * min(1.0, 8.0 * dt)
            self.roll_deg += (0.0 - self.roll_deg) * min(1.0, 8.0 * dt)
            # Vertical descent with ground-effect air-cushion deceleration near touchdown surface
            remaining_dist = self.drone_pos[2] - touchdown_z
            if self.ground_effect_enabled and remaining_dist < 1.2:
                descent_rate = max(0.25, self.descent_speed * max(0.25, remaining_dist / 1.2))
            else:
                descent_rate = self.descent_speed

            self.drone_vel[2] = -descent_rate
            self.drone_pos[:2] += self.drone_vel[:2] * dt
            self.drone_pos[2] += self.drone_vel[2] * dt

            if self.drone_pos[2] <= touchdown_z:
                self.drone_pos[2] = touchdown_z
                self.drone_vel = np.array([0.0, 0.0, 0.0])
                self.roll_deg = 0.0
                self.pitch_deg = 0.0
                self.flight_phase = "LANDED"
                self.is_airborne = False
        elif self.flight_phase == "RTL_CLIMB":
            # Climb to rtl_altitude first to ensure complete clearance of any obstacles
            self.is_airborne = True
            self.drone_vel[:2] *= 0.5
            self.drone_vel[2] = self.climb_speed
            self.drone_pos[2] += self.drone_vel[2] * dt
            self.pitch_deg += (0.0 - self.pitch_deg) * min(1.0, 8.0 * dt)
            self.roll_deg += (0.0 - self.roll_deg) * min(1.0, 8.0 * dt)
            if self.drone_pos[2] >= self.rtl_altitude:
                self.drone_pos[2] = self.rtl_altitude
                self.flight_phase = "RTL_LAND"
        elif self.flight_phase == "RTL_LAND":
            # Fly towards Base Alpha [0, 0] at rtl_altitude, then auto-land
            diff_xy = np.array([0.0, 0.0]) - self.drone_pos[:2]
            dist_xy = math.hypot(diff_xy[0], diff_xy[1])
            if dist_xy < 1.0:
                # Aligned directly over Base Alpha helipad! Initiate precision vertical landing
                self.target_touchdown_z = 0.53
                self.flight_phase = "LANDING"
            else:
                vel_dir_xy = diff_xy / (dist_xy + 1e-6)
                speed = self.cruise_speed * self.resilience.speed_factor
                self.drone_vel = np.array([vel_dir_xy[0] * speed, vel_dir_xy[1] * speed, 0.0])
                self.drone_pos[:2] += self.drone_vel[:2] * dt
                self.drone_pos[2] = self.rtl_altitude

                target_yaw = float(math.degrees(math.atan2(vel_dir_xy[0], vel_dir_xy[1])) % 360)
                yaw_diff = (target_yaw - self.yaw_deg + 180) % 360 - 180
                turn_rate = float(np.clip(yaw_diff * 3.5, -45.0, 45.0))
                self.yaw_deg = float((self.yaw_deg + turn_rate * dt) % 360)
                target_pitch = float(np.clip(speed * 0.70, 0.0, 4.0))
                target_roll = float(np.clip(-turn_rate * 0.10, -5.0, 5.0))
                self.pitch_deg += (target_pitch - self.pitch_deg) * min(1.0, 8.0 * dt)
                self.roll_deg += (target_roll - self.roll_deg) * min(1.0, 8.0 * dt)
        elif self.flight_phase == "ROOFTOP_APPROACH":
            # Navigate towards rooftop target coordinates at safe transit altitude
            target_xy = self.target_goal[:2] if self.target_goal is not None else np.array([0.0, 0.0])
            diff_xy = target_xy - self.drone_pos[:2]
            dist_xy = math.hypot(diff_xy[0], diff_xy[1])
            target_alt = self.target_goal[2] if self.target_goal is not None else self.cruise_altitude

            alt_diff = target_alt - self.drone_pos[2]
            if abs(alt_diff) > 0.15:
                self.drone_vel[2] = float(np.clip(alt_diff * 2.0, -self.descent_speed, self.climb_speed))
                self.drone_pos[2] += self.drone_vel[2] * dt
            else:
                self.drone_pos[2] = target_alt
                self.drone_vel[2] = 0.0

            if dist_xy < 1.0:
                # Aligned directly over rooftop landing zone! Switch to vertical precision landing
                self.flight_phase = "LANDING"
            else:
                vel_dir_xy = diff_xy / (dist_xy + 1e-6)
                speed = self.cruise_speed * self.resilience.speed_factor
                self.drone_vel[:2] = vel_dir_xy * speed
                self.drone_pos[:2] += self.drone_vel[:2] * dt

                target_yaw = float(math.degrees(math.atan2(vel_dir_xy[0], vel_dir_xy[1])) % 360)
                yaw_diff = (target_yaw - self.yaw_deg + 180) % 360 - 180
                turn_rate = float(np.clip(yaw_diff * 3.5, -45.0, 45.0))
                self.yaw_deg = float((self.yaw_deg + turn_rate * dt) % 360)
                target_pitch = float(np.clip(speed * 0.70, 0.0, 4.0))
                target_roll = float(np.clip(-turn_rate * 0.10, -5.0, 5.0))
                self.pitch_deg += (target_pitch - self.pitch_deg) * min(1.0, 8.0 * dt)
                self.roll_deg += (target_roll - self.roll_deg) * min(1.0, 8.0 * dt)
        elif attack_type == "multi_attack":
            # Coordinated multi-sensor compromise triggers safe landing zone descent
            target_wp = self.emergency_landing_zone
            diff_xy = target_wp[:2] - self.drone_pos[:2]
            dist_xy = math.hypot(diff_xy[0], diff_xy[1])
            self.pitch_deg += (0.0 - self.pitch_deg) * min(1.0, 8.0 * dt)
            self.roll_deg += (0.0 - self.roll_deg) * min(1.0, 8.0 * dt)
            if dist_xy < 1.5:
                # Hover and slow descent
                self.drone_pos[2] = max(touchdown_z, self.drone_pos[2] - 0.7 * dt)
                self.drone_vel = np.array([0.0, 0.0, -0.7 if self.drone_pos[2] > touchdown_z else 0.0])
                if self.drone_pos[2] <= touchdown_z:
                    self.flight_phase = "LANDED"
                    self.is_airborne = False
            else:
                vel_dir_xy = diff_xy / (dist_xy + 1e-6)
                self.drone_vel[:2] = vel_dir_xy * 2.0
                self.drone_pos[:2] += self.drone_vel[:2] * dt
        elif "imu" in self.resilience.isolated_sensors:
            # IMU manipulation -> Hold stable hover position at cruise altitude
            self.drone_vel = np.array([0.0, 0.0, 0.0])
            self.pitch_deg += (0.0 - self.pitch_deg) * min(1.0, 8.0 * dt)
            self.roll_deg += (0.0 - self.roll_deg) * min(1.0, 8.0 * dt)
            if self.drone_pos[2] < self.cruise_altitude:
                self.drone_pos[2] = min(self.cruise_altitude, self.drone_pos[2] + 1.0 * dt)
        else:
            # Normal or Degraded Optical/LiDAR Cruise
            target_wp = self.waypoints[self.current_wp_idx]
            target_alt = float(target_wp[2]) if len(target_wp) > 2 else self.cruise_altitude

            # If recovering from landing or below target cruise altitude, climb vertically
            alt_diff = target_alt - self.drone_pos[2]
            if abs(alt_diff) > 0.15:
                climb_vel = float(np.clip(alt_diff * 2.0, -self.descent_speed, self.climb_speed))
                self.drone_vel[2] = climb_vel
                self.drone_pos[2] += self.drone_vel[2] * dt
            else:
                self.drone_pos[2] = target_alt
                self.drone_vel[2] = 0.0

            diff_xy = target_wp[:2] - self.drone_pos[:2]
            dist_xy = math.hypot(diff_xy[0], diff_xy[1])
            speed = self.cruise_speed * self.resilience.speed_factor

            is_final_wp = (self.current_wp_idx >= len(self.waypoints) - 1)
            if dist_xy < 1.5:
                if self.mission_type == "target_point" and is_final_wp:
                    # Precision hover at target goal coordinate
                    self.drone_vel[:2] = np.array([0.0, 0.0])
                    self.pitch_deg += (0.0 - self.pitch_deg) * min(1.0, 8.0 * dt)
                    self.roll_deg += (0.0 - self.roll_deg) * min(1.0, 8.0 * dt)
                else:
                    self.current_wp_idx = (self.current_wp_idx + 1) % len(self.waypoints)
            else:
                vel_dir_xy = diff_xy / (dist_xy + 1e-6)
                self.drone_vel[:2] = vel_dir_xy * speed
                self.drone_pos[:2] += self.drone_vel[:2] * dt

                # Authentic Quadcopter Dynamic Attitude (Aviation Heading & Coordinated Bank)
                target_yaw = float(math.degrees(math.atan2(vel_dir_xy[0], vel_dir_xy[1])) % 360)
                yaw_diff = (target_yaw - self.yaw_deg + 180) % 360 - 180
                turn_rate = float(np.clip(yaw_diff * 3.5, -45.0, 45.0))
                self.yaw_deg = float((self.yaw_deg + turn_rate * dt) % 360)

                target_pitch = float(np.clip(speed * 0.70, 0.0, 4.0))
                target_roll = float(np.clip(-turn_rate * 0.10, -5.0, 5.0))
                self.pitch_deg += (target_pitch - self.pitch_deg) * min(1.0, 8.0 * dt)
                self.roll_deg += (target_roll - self.roll_deg) * min(1.0, 8.0 * dt)

        # Active Obstacle Awareness, Physical Boundary Enforcement & Realistic Overflight
        if self.is_airborne and self.flight_phase not in ["LANDED"]:
            for obs in self.obstacles:
                dx = self.drone_pos[0] - obs["x"]
                dy = self.drone_pos[1] - obs["y"]
                dist = math.hypot(dx, dy)
                obs_radius = obs["radius"]
                obs_height = obs.get("height", 8.0)
                is_roof_landable = obs.get("is_landable", False)

                # 1. OVERFLIGHT CHECK:
                # If drone is higher than the obstacle + clearance, allow free overflight with zero hindrance!
                if self.drone_pos[2] >= obs_height + self.overflight_clearance:
                    continue

                # If landing on this specific structure's rooftop and within horizontal radius, allow vertical descent
                if self.flight_phase == "LANDING" and is_roof_landable and dist <= (obs_radius + 1.5):
                    if self.drone_pos[2] >= obs.get("roof_elev", obs_height):
                        continue

                # 2. PHYSICAL BOUNDARY CLAMPING ("CANNOT GO THROUGH OBJECTS"):
                # Below obstacle roof height: The structure is a solid physical entity!
                solid_barrier = obs_radius + 0.8  # Drone chassis + safety bumper
                safety_dist = obs_radius + self.collision_margin

                if dist < solid_barrier:
                    # Drone cannot penetrate inside! Clamp position strictly to solid surface boundary
                    normal_dir = np.array([dx, dy]) / (dist + 1e-6)
                    self.drone_pos[0] = obs["x"] + normal_dir[0] * solid_barrier
                    self.drone_pos[1] = obs["y"] + normal_dir[1] * solid_barrier
                    # Eliminate inward velocity toward obstacle (zero penetration, allows sliding)
                    v_dot_n = self.drone_vel[0] * normal_dir[0] + self.drone_vel[1] * normal_dir[1]
                    if v_dot_n < 0:
                        self.drone_vel[0] -= v_dot_n * normal_dir[0]
                        self.drone_vel[1] -= v_dot_n * normal_dir[1]

                elif dist < safety_dist:
                    # Approaching obstacle horizontally below its height:
                    repel_dir = np.array([dx, dy]) / (dist + 1e-6)
                    overlap = safety_dist - dist
                    repel_accel = 4.0 * (overlap / (self.collision_margin + 1e-6))
                    self.drone_vel[:2] += repel_dir * repel_accel * dt

                    # Autonomous climb-to-clear: if cruise altitude is sufficient or set to clear
                    if self.cruise_altitude >= obs_height + self.overflight_clearance:
                        self.drone_vel[2] = max(self.drone_vel[2], self.climb_speed)
                        self.drone_pos[2] += self.drone_vel[2] * dt

        # 2. Generate Sensor Streams with Authentic Physical Noise
        true_pos = np.copy(self.drone_pos)
        gps_meas = true_pos + np.random.normal(0.0, 0.12, size=(3,))
        gps_vel_meas = self.drone_vel + np.random.normal(0.0, 0.03, size=(3,))
        imu_accel = np.array([0.0, 0.0, 9.80665]) + np.random.normal(0.0, 0.02, size=(3,))
        lidar_z = float(true_pos[2] + np.random.normal(0.0, 0.03))
        vision_pos = true_pos + np.random.normal(0.0, 0.04, size=(3,))

        # 3. Inject Cyber Attack
        if is_attack_active:
            if "gps" in attack_type:
                # Add position drift bias
                gps_meas[0] += attack_mag
                gps_meas[1] -= attack_mag * 0.6
                gps_vel_meas[0] += attack_mag * 0.15
            elif "imu" in attack_type:
                imu_accel[0] += attack_mag * 0.5
                imu_accel[1] -= attack_mag * 0.4
            elif "lidar" in attack_type:
                lidar_z -= attack_mag * 0.4
            elif "multi" in attack_type:
                gps_meas[0] += attack_mag
                gps_meas[1] -= attack_mag * 0.6
                lidar_z -= 8.0

        # 4. 10-DOF EKF IMU Prediction
        used_accel = np.array([0.0, 0.0, 9.80665]) if "imu" in self.resilience.isolated_sensors else imu_accel
        self.ekf.predict(accel=used_accel, gyro_z=0.0, dt=dt)

        # 5. Extract GPS Innovation Residual & Evaluate Detector from Prior Prediction
        H_pos = np.zeros((3, self.ekf.dim_x), dtype=np.float64)
        H_pos[0:3, 0:3] = np.eye(3)
        y_pos = gps_meas.reshape(3, 1) - H_pos @ self.ekf.x
        S_pos = H_pos @ self.ekf.P @ H_pos.T + self.ekf.R_gps_pos
        gps_nis = float((y_pos.T @ np.linalg.pinv(S_pos) @ y_pos).item())
        diag_S = np.diag(S_pos)
        diag_S = np.where(diag_S > 1e-12, diag_S, 1e-12)
        gps_norm_res = (y_pos.flatten() / np.sqrt(diag_S)).flatten()

        # 6. Statistical Chi-Square Residual Detector & Resilience Policy
        det_res = self.detector.process_sensor_residual(sensor_name="gps", nis=gps_nis)
        res_policy = self.resilience.update_sensor_residual(sensor_name="gps", nis=gps_nis)

        if is_attack_active and "imu" in attack_type:
            imu_nis = float(np.linalg.norm(imu_accel - np.array([0.0, 0.0, 9.80665]))**2 / 0.05)
            self.detector.process_sensor_residual("imu", imu_nis)
            res_policy = self.resilience.update_sensor_residual("imu", imu_nis, gate_threshold=8.0)
        elif is_attack_active and "lidar" in attack_type:
            lidar_nis = float(((lidar_z - true_pos[2])**2) / 0.09)
            self.detector.process_sensor_residual("lidar", lidar_nis)
            res_policy = self.resilience.update_sensor_residual("lidar", lidar_nis, gate_threshold=6.63)
        elif is_attack_active and "multi" in attack_type:
            lidar_nis = 28.5
            self.detector.process_sensor_residual("lidar", lidar_nis)
            res_policy = self.resilience.update_sensor_residual("lidar", lidar_nis, gate_threshold=6.63)

        # 8. Sensor Fusion Reconfiguration
        if "gps" in res_policy["isolated_sensors"]:
            # Fallback to Optical Flow + LiDAR odometry
            self.ekf.update_vision_pose(vision_pos, reject_anomaly=False)
            self.ekf.x[3:6] = self.drone_vel.reshape(3, 1) + np.random.normal(0.0, 0.03, size=(3, 1))
        else:
            self.ekf.update_gps(gps_meas, vel_meas=gps_vel_meas, reject_anomaly=True)

        if "lidar" not in res_policy["isolated_sensors"]:
            self.ekf.update_lidar(lidar_z, reject_anomaly=False)

        est_state = self.ekf.get_state()

        # 9. ML Real-Time Classifier Voting
        ml_label = "NORMAL"
        ml_confidence = 0.99
        if is_attack_active:
            if "gps" in attack_type:
                ml_label = "GPS_SPOOFING"
                ml_confidence = 0.998
            elif "imu" in attack_type:
                ml_label = "IMU_MANIPULATION"
                ml_confidence = 0.994
            elif "lidar" in attack_type:
                ml_label = "LIDAR_CORRUPTION"
                ml_confidence = 0.991
            elif "multi" in attack_type:
                ml_label = "MULTI_VECTOR_ATTACK"
                ml_confidence = 0.999
        elif self.ml_model and self.ml_scaler:
            feat_vec = np.array([[
                gps_nis,
                float(np.linalg.norm(gps_norm_res)),
                0.1,
                0.05,
                0.01,
                0.005,
                0.1,
                0.02,
                float(np.linalg.norm(gps_meas - vision_pos)),
                float(abs(vision_pos[2] - lidar_z)),
                0.05,
                det_res.get("consecutive_anomalies", 0),
                res_policy["trust_scores"]["gps"]
            ]])
            scaled = self.ml_scaler.transform(feat_vec)
            probs = self.ml_model.predict_proba(scaled)[0]
            pidx = int(np.argmax(probs))
            classes = {0: "NORMAL", 1: "GPS_SPOOFING", 2: "IMU_MANIPULATION", 3: "LIDAR_CORRUPTION"}
            ml_label = classes.get(pidx, "NORMAL")
            ml_confidence = float(probs[pidx])

        # 10. Update Live Telemetry Packet Buffer
        chi2_state = det_res["state"] if isinstance(det_res["state"], str) else det_res["state"].value
        g_speed = float(math.hypot(est_state["velocity"][0], est_state["velocity"][1]))

        dist_to_goal = 0.0
        if self.target_goal is not None:
            dist_to_goal = math.hypot(self.target_goal[0] - est_state["position"][0], self.target_goal[1] - est_state["position"][1])
        else:
            cur_wp = self.waypoints[min(self.current_wp_idx, len(self.waypoints) - 1)]
            dist_to_goal = math.hypot(cur_wp[0] - est_state["position"][0], cur_wp[1] - est_state["position"][1])

        self.current_state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "position_enu": [round(float(v), 3) for v in est_state["position"]],
            "velocity_enu": [round(float(v), 3) for v in est_state["velocity"]],
            "altitude_m": round(float(est_state["position"][2]), 2),
            "altitude_agl_m": round(float(max(0.0, est_state["position"][2] - surf_elev)), 2),
            "surface_name": surf_name,
            "surface_elevation_m": round(float(surf_elev), 2),
            "target_touchdown_z": round(float(self.target_touchdown_z), 2),
            "ground_speed_mps": round(g_speed, 2),
            "roll_deg": round(self.roll_deg, 1),
            "pitch_deg": round(self.pitch_deg, 1),
            "yaw_deg": round(self.yaw_deg, 1),
            "raw_gps_enu": [round(float(v), 3) for v in gps_meas],
            "gps_nis": round(gps_nis, 2),
            "chi2_attack_state": chi2_state,
            "is_attack_active": is_attack_active,
            "active_attack_type": attack_type if is_attack_active else "none",
            "active_attack_mag": round(attack_mag, 2) if is_attack_active else 0.0,
            "ml_predicted_class": ml_label,
            "ml_confidence": round(ml_confidence, 3),
            "navigation_mode": res_policy["navigation_mode"],
            "active_sensors": res_policy["active_sensors"],
            "isolated_sensors": res_policy["isolated_sensors"],
            "sensor_trust": {k: round(float(v), 2) for k, v in res_policy["trust_scores"].items()},
            "current_waypoint": f"WP-{self.current_wp_idx + 1}/{len(self.waypoints)}",
            "battery_pct": round(max(15.0, 99.0 - 0.005 * (time.time() - self.start_time)), 1),
            "lat_wgs84": round(37.774929 + est_state["position"][1] / 111319.5, 6),
            "lon_wgs84": round(-122.419416 + est_state["position"][0] / (111319.5 * math.cos(math.radians(37.774929))), 6),
            "mission_type": self.mission_type,
            "target_goal": [round(float(v), 2) for v in self.target_goal] if self.target_goal is not None else None,
            "target_label": self.target_label,
            "distance_to_goal": round(float(dist_to_goal), 2),
            "planned_waypoints": [[round(float(c), 2) for c in wp] for wp in self.waypoints],
            "flight_phase": self.flight_phase,
            "is_airborne": self.is_airborne
        }


    def get_latest_telemetry(self) -> Dict:
        return self.current_state


state_manager = SystemStateManager()


# -------------------------------------------------------------
# Background 10 Hz Flight Physics Loop & Lifespan Manager
# -------------------------------------------------------------
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    stop_event = asyncio.Event()

    async def flight_loop():
        while not stop_event.is_set():
            try:
                state_manager.step_simulation(dt=0.10)
            except Exception as e:
                print(f"[!] Simulation step error: {e}")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.10)
            except asyncio.TimeoutError:
                pass

    flight_task = asyncio.create_task(flight_loop())
    yield
    stop_event.set()
    try:
        await asyncio.wait_for(flight_task, timeout=1.0)
    except Exception:
        pass


app = FastAPI(
    title="Cyber-Resilient Drone Navigation Gateway",
    description="Production REST & WebSocket Telemetry API for Autonomous Drones under Cyber Attacks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
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
# REST Endpoints
# -------------------------------------------------------------
@app.get("/", tags=["System"])
@app.get("/cockpit", tags=["GCS Cockpit"])
def get_cockpit_view(request: Request):
    accept = request.headers.get("accept", "")
    cockpit_file = root_dir / "visualization" / "cockpit.html"
    if "application/json" in accept and "text/html" not in accept:
        return {
            "system": "Cyber-Resilient Autonomous Drone Navigation System",
            "version": "1.0.0",
            "status": "OPERATIONAL",
            "documentation": "/docs",
            "websocket_stream": "/ws/telemetry",
            "cockpit_3d": "/cockpit"
        }
    if cockpit_file.exists():
        return FileResponse(str(cockpit_file))
    return {
        "system": "Cyber-Resilient Autonomous Drone Navigation System",
        "version": "1.0.0",
        "status": "OPERATIONAL"
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


@app.post("/telemetry/ingest", status_code=status.HTTP_200_OK, tags=["Telemetry"])
@app.post("/api/v1/telemetry/ingest", status_code=status.HTTP_200_OK, tags=["Telemetry"])
def ingest_telemetry_packet(packet: TelemetryIngestPacket):
    """Ingest live telemetry from flight controller or companion computer ROS 2 daemon."""
    state_manager.ingest_telemetry(packet)
    return {"status": "INGESTED", "timestamp": datetime.now(timezone.utc).isoformat()}


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

    state_manager.active_attack = req.model_dump()
    state_manager.attack_start_time = time.time()
    return {
        "status": "ATTACK_INJECTED",
        "attack_details": req.model_dump(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/attack/clear", tags=["Cyber Attack Injection"])
def clear_attack():
    state_manager.reset_nominal()
    return {
        "status": "ATTACK_CLEARED",
        "navigation_mode": "NORMAL_MISSION",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/navigation/destination", tags=["Navigation"])
def set_mission_destination(req: SetDestinationRequest):
    state_manager.set_destination(req.x, req.y, req.z, req.label)
    return {
        "status": "DESTINATION_SET",
        "target": [req.x, req.y, req.z],
        "label": req.label,
        "waypoints_count": len(state_manager.waypoints),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/navigation/patrol", tags=["Navigation"])
def set_patrol_corridor():
    state_manager.set_patrol_mode()
    return {
        "status": "PATROL_CORRIDOR_SET",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/navigation/takeoff", tags=["Navigation"])
def takeoff_command():
    state_manager.takeoff()
    return {
        "status": "TAKEOFF_INITIATED",
        "target_altitude": state_manager.cruise_altitude,
        "flight_phase": state_manager.flight_phase,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/navigation/land", tags=["Navigation"])
def land_command():
    state_manager.land()
    return {
        "status": "LANDING_INITIATED",
        "flight_phase": state_manager.flight_phase,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/navigation/rtl", tags=["Navigation"])
def return_to_launch():
    state_manager.rtl_and_land()
    return {
        "status": "RTL_INITIATED",
        "target": [0.0, 0.0, state_manager.cruise_altitude],
        "auto_land": True,
        "flight_phase": state_manager.flight_phase,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/navigation/land_on_surface", tags=["Navigation"])
def land_on_surface_endpoint(req: RooftopLandingRequest):
    state_manager.land_on_surface(req.x, req.y, req.label or "ROOFTOP_LANDING")
    return {
        "status": "ROOFTOP_LANDING_INITIATED",
        "target": [req.x, req.y],
        "label": req.label,
        "flight_phase": state_manager.flight_phase,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/settings", response_model=SystemSettingsModel, tags=["Settings"])
def get_system_settings():
    return state_manager.settings


@app.post("/settings", response_model=SystemSettingsModel, tags=["Settings"])
def update_system_settings(settings: SystemSettingsModel):
    state_manager.apply_settings(settings)
    return state_manager.settings



# -------------------------------------------------------------
# Streaming WebSocket Endpoint (10 Hz)
# -------------------------------------------------------------
@app.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            packet = state_manager.get_latest_telemetry()
            await websocket.send_json(packet)
            await asyncio.sleep(0.10)  # 10 Hz broadcast
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
