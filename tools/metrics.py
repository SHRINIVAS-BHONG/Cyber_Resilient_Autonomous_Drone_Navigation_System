#!/usr/bin/env python3
"""
CLI and Utility module to compute and inspect metrics from run folders.
Usage:
    python tools/metrics.py --run_dir runs/s1_gps_drift_defense_on_seed42
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyber_drone.metrics import MetricsEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate / Display Metrics from Run Directory")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory")
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    if not run_path.exists():
        print(f"Error: {run_path} does not exist.")
        sys.exit(1)

    metrics_file = run_path / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        print(json.dumps(metrics, indent=2))
    else:
        print(f"metrics.json not found in {run_path}")


if __name__ == "__main__":
    main()
