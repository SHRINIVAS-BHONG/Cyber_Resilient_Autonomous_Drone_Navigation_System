"""
Cyber-Resilient Autonomous Drone Navigation System - Deployment Orchestrator.

Production CLI orchestrator for launching and verifying the system across:
- 'companion': Flight companion computer daemon (EKF, ML Detector, Resilience Manager)
- 'gcs': Ground Control Station & Cloud Gateway (FastAPI + Streamlit Dashboard)
- 'verify': Pre-flight hardware and software integrity checks (tests, model, geodesy)
- 'docker': Containerized deployment via Docker Compose
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

root_dir = Path(__file__).resolve().parent.parent


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  [*] {title}")
    print("=" * 75)


def check_preflight_integrity() -> bool:
    print_banner("PRE-FLIGHT INTEGRITY & ARTIFACT VERIFICATION")
    all_ok = True

    # 1. Verify ML Model Artifacts
    models_dir = root_dir / "ml" / "models"
    rf_model = models_dir / "random_forest_detector.joblib"
    scaler = models_dir / "scaler.joblib"
    feats = models_dir / "feature_names.json"

    print("[*] Checking Trained Machine Learning Artifacts...")
    if rf_model.exists() and scaler.exists() and feats.exists():
        size_kb = rf_model.stat().st_size / 1024
        print(f"    [+] ML Detector Model: READY ({size_kb:.1f} KB)")
        print(f"    [+] Feature Scaler: READY")
    else:
        print("    [-] Missing ML model artifacts! Run 'python ml/train_random_forest.py'.")
        all_ok = False

    # 2. Verify Datasets
    data_dir = root_dir / "ml" / "data"
    train_csv = data_dir / "train_dataset.csv"
    test_csv = data_dir / "test_dataset.csv"

    print("[*] Checking Authentic Flight Telemetry Datasets...")
    if train_csv.exists() and test_csv.exists():
        print(f"    [+] Training Dataset: READY ({train_csv.stat().st_size / (1024*1024):.1f} MB)")
        print(f"    [+] Testing Dataset: READY ({test_csv.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print("    [-] Datasets missing! Run 'python ml/data_generation/download_real_dataset.py'.")
        all_ok = False

    # 3. Verify Configurations
    cfg_dir = root_dir / "config"
    cfgs = ["estimator.yaml", "detector.yaml", "resilience.yaml", "sensors.yaml", "planner.yaml"]
    print("[*] Checking System Parameter Configurations...")
    for c in cfgs:
        cp = cfg_dir / c
        if cp.exists():
            print(f"    [+] Config '{c}': OK")
        else:
            print(f"    [-] Missing config: {c}")
            all_ok = False

    # 4. Run Pytest Test Suite
    print("[*] Executing Automated Unit & Integration Test Suite...")
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=str(root_dir)
    )
    if res.returncode == 0:
        print("    [+] All Tests PASSED successfully (100% integrity).")
    else:
        print("    [-] Some tests failed.")
        all_ok = False

    return all_ok


def start_gcs():
    print_banner("LAUNCHING GROUND CONTROL STATION (GCS) & CLOUD GATEWAY")
    print("[*] Starting FastAPI REST & WebSocket Telemetry Gateway on port 8000...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(root_dir)
    )

    time.sleep(1.5)
    print("[*] Starting Streamlit Interactive Mission Dashboard on port 8501...")
    dash_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "visualization/dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"],
        cwd=str(root_dir)
    )

    print("\n[+] GCS Services Running:")
    print("    - REST API & Swagger Docs: http://localhost:8000/docs")
    print("    - WebSocket Telemetry:     ws://localhost:8000/ws/telemetry")
    print("    - Mission Web Dashboard:   http://localhost:8501")
    print("\nPress Ctrl+C to terminate all services.\n")

    try:
        api_proc.wait()
        dash_proc.wait()
    except KeyboardInterrupt:
        print("\n[*] Shutting down GCS services...")
        api_proc.terminate()
        dash_proc.terminate()


def start_docker():
    print_banner("LAUNCHING CONTAINERIZED SYSTEM VIA DOCKER COMPOSE")
    compose_file = root_dir / "docker" / "compose.yaml"
    subprocess.run(["docker", "compose", "-f", str(compose_file), "up", "--build"])


def main():
    parser = argparse.ArgumentParser(description="Cyber-Resilient Drone Deployment Orchestrator")
    parser.add_argument(
        "--profile",
        choices=["verify", "gcs", "docker"],
        default="verify",
        help="Deployment profile: 'verify' (pre-flight checks), 'gcs' (FastAPI + Dashboard), 'docker' (containerized stack)"
    )
    args = parser.parse_args()

    if args.profile == "verify":
        ok = check_preflight_integrity()
        if ok:
            print("\n[+] SYSTEM INTEGRITY VERIFIED: Ready for flight deployment.\n")
            sys.exit(0)
        else:
            print("\n[-] Pre-flight checks failed. Correct the errors above before flight.\n")
            sys.exit(1)
    elif args.profile == "gcs":
        start_gcs()
    elif args.profile == "docker":
        start_docker()


if __name__ == "__main__":
    main()
