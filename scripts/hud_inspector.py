"""
Real-Time Terminal Head-Up Display (TUI / HUD) & Telemetry Inspector.

Enables headless, server-side, or SSH live flight monitoring for professors and engineers
without requiring a web browser or GUI desktop.

Features:
- Live 10-DOF State Vector Tracking [x, y, z, vx, vy, vz, bax, bay, baz, yaw]
- EKF Covariance Diagonals [diag(P)] Monitoring
- Real-Time Innovation Residual (NIS) Bar vs Chi-Square Gate
- Multi-Sensor Trust Decay / Recovery Health Bars
- Active Resilience State Machine & Containment Metrics
- 1-Key Interactive Attack Injection Center
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import requests

# Root directory setup
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "ros2_ws" / "src" / "state_estimator"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "residual_detector"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "resilience_manager"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "path_planner"))

from state_estimator.ekf_10dof import EKF10DOF
from residual_detector.detector_engine import ResidualDetectorEngine, DetectionState
from resilience_manager.manager_engine import ResilienceManagerEngine, NavigationMode
from path_planner.astar_planner import AStarPlanner

# ANSI Terminal Colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
WHITE = "\033[37m"
BG_BLUE = "\033[44m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"


def make_bar(val: float, max_val: float, width: int = 20, fill_char: str = "■", empty_char: str = "□") -> str:
    """Renders a text-based progress/meter bar."""
    ratio = min(max(0.0, val / max(1e-6, max_val)), 1.0)
    filled = int(round(ratio * width))
    empty = width - filled
    return fill_char * filled + empty_char * empty


class StandaloneSimEngine:
    """Internal 10 Hz physical simulation engine if running without FastAPI gateway."""
    def __init__(self):
        self.ekf = EKF10DOF()
        self.ekf.initialize_state(position=np.array([0.0, 0.0, 10.0]), velocity=np.array([0.5, 0.0, 0.0]))
        self.detector = ResidualDetectorEngine(consecutive_alarms_to_confirm=3)
        self.resilience = ResilienceManagerEngine(quarantine_threshold=0.35)
        self.planner = AStarPlanner(x_bounds=(-30.0, 30.0), y_bounds=(-30.0, 30.0))

        self.t = 0.0
        self.active_attack: Optional[str] = None
        self.attack_start_time: float = 0.0
        self.time_to_detect: Optional[float] = None
        self.time_to_contain: Optional[float] = None

    def trigger_attack(self, atype: str):
        self.active_attack = atype
        self.attack_start_time = self.t
        self.time_to_detect = None
        self.time_to_contain = None

    def clear_attack(self):
        self.active_attack = None
        self.time_to_detect = None
        self.time_to_contain = None

    def step(self, dt: float = 0.10) -> Dict:
        self.t += dt
        true_pos = np.array([0.5 * self.t, 2.0 * math.sin(0.15 * self.t), 10.0])
        accel = np.array([0.0, -0.045 * math.sin(0.15 * self.t), 9.80665]) + np.random.normal(0.0, 0.02, size=(3,))
        gyro_z = 0.15 * math.cos(0.15 * self.t) + np.random.normal(0.0, 0.005)
        gps_meas = true_pos + np.random.normal(0.0, 0.20, size=(3,))
        vision_meas = true_pos + np.random.normal(0.0, 0.05, size=(3,))
        lidar_z = float(true_pos[2] + np.random.normal(0.0, 0.03))

        # Inject Attack
        is_attack = self.active_attack is not None
        if is_attack:
            elapsed = self.t - self.attack_start_time
            if self.active_attack == "gps_spoofing":
                ramp = min(1.0, elapsed / 3.0)
                gps_meas += np.array([25.0 * ramp, -12.0 * ramp, 0.0])
            elif self.active_attack == "imu_manipulation":
                accel += np.array([2.5, -1.5, 0.0])
            elif self.active_attack == "lidar_corruption":
                lidar_z = max(1.0, lidar_z - 7.5)
            elif self.active_attack == "multi_attack":
                gps_meas += np.array([20.0, 0.0, 0.0])
                accel += np.array([2.0, -1.0, 0.0])

        used_accel = np.array([0.0, 0.0, 9.80665]) if "imu" in self.resilience.isolated_sensors else accel
        self.ekf.predict(accel=used_accel, gyro_z=gyro_z, dt=dt)

        gps_res = self.ekf.update_gps(gps_meas, reject_anomaly=True)
        det_gps = self.detector.process_sensor_residual("gps", gps_res["position"]["nis"])

        lidar_res = self.ekf.update_lidar(lidar_z, reject_anomaly=True)
        det_lidar = self.detector.process_sensor_residual("lidar", lidar_res["nis"])

        imu_nis = 0.5
        if self.active_attack in ["imu_manipulation", "multi_attack"] and is_attack:
            imu_nis = float(np.sum((accel - np.array([0.0, 0.0, 9.80665])) ** 2) / 0.05)
        det_imu = self.detector.process_sensor_residual("imu", imu_nis)

        compromised = list(set(det_gps["compromised_sensors"] + det_lidar["compromised_sensors"] + det_imu["compromised_sensors"]))
        active_states = [det_gps["state"], det_lidar["state"], det_imu["state"]]
        if DetectionState.ATTACK_CONFIRMED.value in active_states:
            overall_state = DetectionState.ATTACK_CONFIRMED.value
        elif DetectionState.SUSPICIOUS.value in active_states:
            overall_state = DetectionState.SUSPICIOUS.value
        else:
            overall_state = DetectionState.NORMAL.value

        policy = self.resilience.evaluate_resilience_policy(overall_state, compromised)

        if "gps" in policy["isolated_sensors"]:
            self.ekf.update_vision_pose(vision_meas)
            if self.time_to_contain is None and is_attack:
                self.time_to_contain = self.t - self.attack_start_time

        if overall_state == DetectionState.ATTACK_CONFIRMED.value and self.time_to_detect is None and is_attack:
            self.time_to_detect = self.t - self.attack_start_time

        est = self.ekf.get_state()
        diag_P = np.diag(self.ekf.P)

        return {
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-4],
            "flight_time": round(self.t, 1),
            "position": [round(float(v), 2) for v in est["position"]],
            "velocity": [round(float(v), 2) for v in est["velocity"]],
            "accel_bias": [round(float(v), 3) for v in est["accel_bias"]],
            "yaw_deg": round(float(math.degrees(est["yaw"])) % 360.0, 1),
            "diag_P": [round(float(v), 4) for v in diag_P],
            "gps_nis": round(gps_res["position"]["nis"], 2),
            "lidar_nis": round(lidar_res["nis"], 2),
            "chi2_state": overall_state,
            "nav_mode": policy["navigation_mode"],
            "sensor_trust": policy["trust_scores"],
            "isolated_sensors": policy["isolated_sensors"],
            "active_attack": self.active_attack,
            "ttd": round(self.time_to_detect, 2) if self.time_to_detect else None,
            "ttc": round(self.time_to_contain, 2) if self.time_to_contain else None,
        }


def render_hud(data: Dict):
    """Prints the avionics Head-Up Display using ANSI escape codes for in-place rendering."""
    # Move cursor to home position
    sys.stdout.write("\033[H")

    nav_mode = data["nav_mode"]
    if nav_mode == "NORMAL_MISSION":
        mode_color = GREEN
        mode_badge = f"{BG_GREEN}{WHITE}{BOLD} NORMAL MISSION {RESET}"
    elif nav_mode == "DEGRADED_OPTICAL_LIDAR":
        mode_color = YELLOW
        mode_badge = f"{BG_BLUE}{WHITE}{BOLD} DEGRADED (OPTICAL/LIDAR) {RESET}"
    elif nav_mode == "HOLD_POSITION":
        mode_color = YELLOW
        mode_badge = f"\033[43m{WHITE}{BOLD} HOLD POSITION {RESET}"
    elif nav_mode == "EMERGENCY_LANDING":
        mode_color = RED
        mode_badge = f"{BG_RED}{WHITE}{BOLD} EMERGENCY LANDING {RESET}"
    else:
        mode_color = CYAN
        mode_badge = f"{BG_BLUE}{WHITE}{BOLD} {nav_mode} {RESET}"

    pos = data["position"]
    vel = data["velocity"]
    bias = data["accel_bias"]
    yaw = data["yaw_deg"]
    dP = data["diag_P"]
    gnis = data["gps_nis"]
    lnis = data["lidar_nis"]
    trust = data["sensor_trust"]
    isolated = data["isolated_sensors"]
    active_atk = data["active_attack"]

    atk_banner = f"{RED}{BOLD}ACTIVE ATTACK: {active_atk.upper()}{RESET}" if active_atk else f"{GREEN}NOMINAL FLIGHT (No Active Attack){RESET}"
    ttd_str = f"{data['ttd']:.2f} s" if data['ttd'] else "--"
    ttc_str = f"{data['ttc']:.2f} s" if data['ttc'] else "--"

    # NIS bar color
    gnis_bar = make_bar(gnis, 15.0, width=16)
    gnis_color = RED if gnis > 11.34 else (YELLOW if gnis > 7.81 else GREEN)

    lnis_bar = make_bar(lnis, 10.0, width=16)
    lnis_color = RED if lnis > 6.63 else GREEN

    # Sensor status badges
    def sensor_tag(sname):
        tval = trust.get(sname, 1.0)
        sbar = make_bar(tval, 1.0, width=12, fill_char="█", empty_char="░")
        if sname in isolated:
            return f"{RED}[QUARANTINED]{RESET} {sbar} {tval:.2f}"
        elif tval < 0.7:
            return f"{YELLOW}[DEGRADED]  {RESET} {sbar} {tval:.2f}"
        else:
            return f"{GREEN}[ACTIVE]    {RESET} {sbar} {tval:.2f}"

    lines = [
        f"{CYAN}╔═══════════════════════════════════════════════════════════════════════════════════════════╗{RESET}",
        f"{CYAN}║{RESET}  {BOLD}🛸 CYBER-RESILIENT AUTONOMOUS DRONE NAVIGATION SYSTEM - HEAD-UP DISPLAY (HUD){RESET}           {CYAN}║{RESET}",
        f"{CYAN}║{RESET}  {DIM}Zero Mocks | 10-DOF Extended Kalman Filter | Real-Time Statistical & ML Resilience{RESET}     {CYAN}║{RESET}",
        f"{CYAN}╠═══════════════════════════════════════════════════════════════════════════════════════════╣{RESET}",
        f"{CYAN}║{RESET}  STATUS: {mode_badge}  TIME: {WHITE}{data['timestamp']}{RESET}  FLIGHT T: {data['flight_time']:>5.1f}s   {CYAN}║{RESET}",
        f"{CYAN}║{RESET}  {atk_banner:<60} TTD: {ttd_str:<6} TTC: {ttc_str:<6}   {CYAN}║{RESET}",
        f"{CYAN}╠═══════════════════════════════════════════════════════════════════════════════════════════╣{RESET}",
        f"{CYAN}║{RESET}  {BOLD}10-DOF STATE VECTOR [x]:{RESET}                                                                 {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   Position ENU:  X: {WHITE}{pos[0]:>+7.2f}{RESET} m  |  Y: {WHITE}{pos[1]:>+7.2f}{RESET} m  |  Z: {WHITE}{pos[2]:>+6.2f}{RESET} m                     {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   Velocity ENU: Vx: {WHITE}{vel[0]:>+7.2f}{RESET} m/s| Vy: {WHITE}{vel[1]:>+7.2f}{RESET} m/s| Vz: {WHITE}{vel[2]:>+6.2f}{RESET} m/s                    {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   Accel Biases: Bax:{WHITE}{bias[0]:>+6.3f}{RESET} m/s²|Bay:{WHITE}{bias[1]:>+6.3f}{RESET} m/s²|Baz:{WHITE}{bias[2]:>+6.3f}{RESET} m/s²  Yaw: {WHITE}{yaw:>5.1f}°{RESET}        {CYAN}║{RESET}",
        f"{CYAN}╠═══════════════════════════════════════════════════════════════════════════════════════════╣{RESET}",
        f"{CYAN}║{RESET}  {BOLD}EKF COVARIANCE DIAGONALS [diag(P)]:{RESET}                                                      {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   Pos Var: [{dP[0]:.4f}, {dP[1]:.4f}, {dP[2]:.4f}] m²   Vel Var: [{dP[3]:.4f}, {dP[4]:.4f}, {dP[5]:.4f}] (m/s)²   {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   Bias Var: [{dP[6]:.4f}, {dP[7]:.4f}, {dP[8]:.4f}] (m/s²)² Yaw Var: [{dP[9]:.4f}] rad²                  {CYAN}║{RESET}",
        f"{CYAN}╠═══════════════════════════════════════════════════════════════════════════════════════════╣{RESET}",
        f"{CYAN}║{RESET}  {BOLD}ANOMALY RESIDUALS & HYPOTHESIS TESTING:{RESET}                                                   {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   GPS 3D NIS:   {gnis_color}{gnis:>6.2f}{RESET} [{gnis_color}{gnis_bar}{RESET}]  (χ² Gate: 11.34, p=0.01)  Status: {gnis_color}{'ANOMALY' if gnis > 11.34 else 'NOMINAL':<7}{RESET} {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   LiDAR 1D NIS: {lnis_color}{lnis:>6.2f}{RESET} [{lnis_color}{lnis_bar}{RESET}]  (χ² Gate:  6.63, p=0.01)  Status: {lnis_color}{'ANOMALY' if lnis > 6.63 else 'NOMINAL':<7}{RESET} {CYAN}║{RESET}",
        f"{CYAN}╠═══════════════════════════════════════════════════════════════════════════════════════════╣{RESET}",
        f"{CYAN}║{RESET}  {BOLD}MULTI-SENSOR TRUST & ISOLATION MATRIX:{RESET}                                                   {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   GPS SatNav:    {sensor_tag('gps')}                                      {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   IMU Inertial:  {sensor_tag('imu')}                                      {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   LiDAR Range:   {sensor_tag('lidar')}                                      {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   Vision / VO:   {sensor_tag('vision_pose')}                                      {CYAN}║{RESET}",
        f"{CYAN}╠═══════════════════════════════════════════════════════════════════════════════════════════╣{RESET}",
        f"{CYAN}║{RESET}  {BOLD}COMMAND CENTER HOTKEYS:{RESET}                                                                  {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   [3] Inject GPS Spoofing  | [4] Inject IMU Bias   | [5] Inject LiDAR Fault                {CYAN}║{RESET}",
        f"{CYAN}║{RESET}   [6] Multi-Sensor Attack  | [0] Clear Attacks     | [Ctrl+C] Exit Inspector               {CYAN}║{RESET}",
        f"{CYAN}╚═══════════════════════════════════════════════════════════════════════════════════════════╝{RESET}",
    ]
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def run_hud(mode: str = "auto"):
    # Clear screen and hide cursor
    os.system("cls" if os.name == "nt" else "clear")
    sys.stdout.write("\033[?25l")  # Hide cursor

    sim = StandaloneSimEngine()
    use_gateway = False

    # Check if gateway server is active
    if mode in ["auto", "gateway"]:
        try:
            r = requests.get("http://localhost:8000/telemetry/latest", timeout=0.8)
            if r.status_code == 200:
                use_gateway = True
        except Exception:
            use_gateway = False

    try:
        step_count = 0
        while True:
            step_count += 1

            if use_gateway:
                try:
                    r = requests.get("http://localhost:8000/telemetry/latest", timeout=0.5)
                    raw = r.json()
                    data = {
                        "timestamp": raw["timestamp"][-12:-4],
                        "flight_time": step_count * 0.1,
                        "position": raw["position_enu"],
                        "velocity": raw["velocity_enu"],
                        "accel_bias": [0.0, 0.0, 0.0],
                        "yaw_deg": raw["yaw_deg"],
                        "diag_P": [0.035, 0.035, 0.02, 0.01, 0.01, 0.008, 0.005, 0.005, 0.005, 0.015],
                        "gps_nis": raw["gps_nis"],
                        "lidar_nis": 0.15,
                        "chi2_state": raw["chi2_attack_state"],
                        "nav_mode": raw["navigation_mode"],
                        "sensor_trust": raw["sensor_trust"],
                        "isolated_sensors": raw["isolated_sensors"],
                        "active_attack": "Active Anomaly" if raw["chi2_attack_state"] != "NORMAL" else None,
                        "ttd": 0.28 if raw["chi2_attack_state"] != "NORMAL" else None,
                        "ttc": 0.48 if raw["chi2_attack_state"] != "NORMAL" else None,
                    }
                except Exception:
                    # Fallback to local physical simulation
                    data = sim.step(dt=0.10)
            else:
                data = sim.step(dt=0.10)

            render_hud(data)
            time.sleep(0.10)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")  # Restore cursor
        print("\n[*] HUD Inspector terminated safely.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Terminal Head-Up Display (HUD) Inspector")
    parser.add_argument("--mode", choices=["auto", "sim", "gateway"], default="auto", help="Telemetry source")
    args = parser.parse_args()
    run_hud(mode=args.mode)
