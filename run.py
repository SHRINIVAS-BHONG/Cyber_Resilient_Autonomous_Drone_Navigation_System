"""
Cyber-Resilient Autonomous Drone Navigation System - Master 1-Click Launcher.

Runs the complete system with one simple command:
- Starts the FastAPI Telemetry Gateway (port 8000)
- Starts the Live Telemetry Bridge (feeds real flight telemetry & EKF state)
- Starts the Streamlit Mission Dashboard (port 8501)
- Automatically opens the Dashboard in your default web browser
- Provides an interactive terminal command center for 1-key attack injections
"""

import atexit
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import webbrowser
import requests

root_dir = Path(__file__).resolve().parent

# Ensure packages are discoverable
for sub in ["sensor_bridge", "state_estimator", "residual_detector", "resilience_manager", "path_planner", "ml_detector", "px4_controller"]:
    pkg_path = str(root_dir / "ros2_ws" / "src" / sub)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Managed subprocesses
managed_processes = []


def is_port_in_use(port: int) -> bool:
    """Check if a local network port is already open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def cleanup():
    """Terminate all background worker processes on exit."""
    print("\n[*] Shutting down Cyber-Resilient Drone System services...")
    for p in managed_processes:
        try:
            p.terminate()
        except Exception:
            pass


atexit.register(cleanup)


def start_fastapi_gateway():
    """Starts FastAPI REST/WebSocket server on port 8000."""
    if is_port_in_use(8000):
        print("[+] FastAPI Gateway is already running on port 8000.")
        return None

    print("[*] Starting FastAPI Telemetry Gateway on port 8000...")
    cmd = [
        sys.executable, "-m", "uvicorn", "api.server:app",
        "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"
    ]
    proc = subprocess.Popen(cmd, cwd=str(root_dir))
    managed_processes.append(proc)

    # Wait for gateway to become ready
    for _ in range(20):
        if is_port_in_use(8000):
            print("[+] FastAPI Gateway active: http://localhost:8000/docs")
            return proc
        time.sleep(0.3)
    return proc


def start_streamlit_dashboard():
    """Starts Streamlit web dashboard on port 8501."""
    if is_port_in_use(8501):
        print("[+] Streamlit Dashboard is already running on port 8501.")
        return None

    print("[*] Starting Streamlit Mission Dashboard on port 8501...")
    cmd = [
        sys.executable, "-m", "streamlit", "run", "visualization/dashboard.py",
        "--server.port", "8501", "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ]
    proc = subprocess.Popen(cmd, cwd=str(root_dir))
    managed_processes.append(proc)

    # Wait for dashboard to become ready
    for _ in range(25):
        if is_port_in_use(8501):
            print("[+] Streamlit Dashboard active: http://localhost:8501")
            return proc
        time.sleep(0.3)
    return proc


def trigger_attack(attack_type: str, magnitude: float = 20.0):
    """Sends 1-click cyber-attack injection request to the gateway."""
    url = "http://localhost:8000/attack/inject"
    payload = {
        "attack_type": attack_type,
        "magnitude": magnitude,
        "duration_sec": 30.0,
        "target_sensor": "gps" if "gps" in attack_type else "imu"
    }
    try:
        r = requests.post(url, json=payload, timeout=2.0)
        if r.status_code in [200, 202]:
            print(f"\n[!] CYBER-ATTACK INJECTED: {attack_type.upper()} (Magnitude: {magnitude})")
        else:
            print(f"\n[-] Attack injection failed: HTTP {r.status_code}")
    except Exception as e:
        print(f"\n[-] Failed to reach gateway: {e}")


def clear_attacks():
    """Clears all active cyber-attacks on the gateway."""
    url = "http://localhost:8000/attack/clear"
    try:
        r = requests.post(url, timeout=2.0)
        if r.status_code == 200:
            print("\n[+] ALL ATTACKS CLEARED: Returning to Normal Mission State.")
    except Exception as e:
        print(f"\n[-] Failed to reach gateway: {e}")


def print_menu():
    print("\n" + "=" * 75)
    print("  [*] CYBER-RESILIENT DRONE SYSTEM - COMMAND CENTER")
    print("=" * 75)
    print("  [1] Open Mission Dashboard in Browser  (http://localhost:8501)")
    print("  [2] Open Swagger API Documentation     (http://localhost:8000/docs)")
    print("  [3] Inject GPS Spoofing Attack         (Drift + Step Jump)")
    print("  [4] Inject IMU Bias Attack             (Gyro + Accel Manipulation)")
    print("  [5] Inject LiDAR Corruption Attack     (Altimeter Range Fault)")
    print("  [6] Inject Multi-Sensor Attack         (Coordinated GPS + IMU)")
    print("  [0] Clear Attacks / Return to Normal")
    print("  [t] Run All Automated System Tests     (pytest)")
    print("  [b] Run Quantitative Benchmarks")
    print("  [q] Stop All Services and Exit")
    print("-" * 75)


def main():
    os.system("cls" if os.name == "nt" else "clear")
    print("\n" + "=" * 75)
    print("  🛸 CYBER-RESILIENT AUTONOMOUS DRONE NAVIGATION SYSTEM")
    print("  Zero Mocks | Authentic Flight Telemetry | Dual Statistical & ML Engine")
    print("=" * 75)

    # 1. Start Services
    start_fastapi_gateway()
    start_streamlit_dashboard()

    # 2. Auto-open browser
    time.sleep(1.0)
    print("\n[*] Opening Mission Dashboard in your default browser...")
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass

    # 3. Interactive Menu Loop
    while True:
        print_menu()
        try:
            choice = input("Select an option: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if choice == "1":
            webbrowser.open("http://localhost:8501")
        elif choice == "2":
            webbrowser.open("http://localhost:8000/docs")
        elif choice == "3":
            trigger_attack("gps_spoofing", magnitude=25.0)
        elif choice == "4":
            trigger_attack("imu_manipulation", magnitude=2.5)
        elif choice == "5":
            trigger_attack("lidar_corruption", magnitude=8.0)
        elif choice == "6":
            trigger_attack("multi_attack", magnitude=20.0)
        elif choice == "0":
            clear_attacks()
        elif choice == "t":
            print("\n[*] Running 35 automated unit and integration tests...")
            subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=str(root_dir))
        elif choice == "b":
            print("\n[*] Running quantitative benchmark scenarios...")
            subprocess.run([sys.executable, "scripts/run_benchmarks.py"], cwd=str(root_dir))
        elif choice == "q":
            print("\nExiting...")
            break
        else:
            print("Invalid option. Please choose from the menu.")

    cleanup()


if __name__ == "__main__":
    main()
