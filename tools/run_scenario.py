#!/usr/bin/env python3
"""
CLI Tool to execute a single cyber resilience scenario.
Usage:
    python tools/run_scenario.py --scenario s1 --defense on --seed 42
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path

# Add src to pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyber_drone.simulation_engine import SimulationEngine


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run Cyber-Resilient Drone Scenario")
    parser.add_argument("--scenario", type=str, default="s1_gps_drift", help="Scenario ID or filename (e.g. s0, s1, s1_gps_drift)")
    parser.add_argument("--defense", type=str, choices=["on", "off", "true", "false"], default="on", help="Defense system toggle (on/off)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output directory for logs and artifacts")
    args = parser.parse_args()

    defense_on = args.defense.lower() in ["on", "true"]

    # Resolve scenario config
    scenarios_dir = REPO_ROOT / "config" / "scenarios"
    scenario_name = args.scenario
    if not scenario_name.endswith(".yaml"):
        # Match prefix if abbreviated (e.g. s0 -> s0_normal.yaml)
        matching = [f for f in os.listdir(scenarios_dir) if f.startswith(scenario_name)]
        if matching:
            scenario_name = matching[0]
        else:
            scenario_name = f"{scenario_name}.yaml"

    scenario_path = scenarios_dir / scenario_name
    if not scenario_path.exists():
        print(f"Error: Scenario file {scenario_path} not found.")
        sys.exit(1)

    scenario_cfg = load_yaml(scenario_path)
    sensors_cfg = load_yaml(REPO_ROOT / "config" / "sensors.yaml")
    estimator_cfg = load_yaml(REPO_ROOT / "config" / "estimator.yaml")
    detector_cfg = load_yaml(REPO_ROOT / "config" / "detector.yaml")
    resilience_cfg = load_yaml(REPO_ROOT / "config" / "resilience.yaml")
    planner_cfg = load_yaml(REPO_ROOT / "config" / "planner.yaml")

    print(f"==================================================")
    print(f"Running Scenario: {scenario_cfg.get('name', scenario_cfg.get('id'))}")
    print(f"Defense Mode    : {'ON (Resilient)' if defense_on else 'OFF (Baseline Derailment)'}")
    print(f"Random Seed     : {args.seed}")
    print(f"==================================================")

    engine = SimulationEngine(
        scenario_config=scenario_cfg,
        sensors_config=sensors_cfg,
        estimator_config=estimator_cfg,
        detector_config=detector_cfg,
        resilience_config=resilience_cfg,
        planner_config=planner_cfg,
        defense_on=defense_on,
        seed=args.seed
    )

    results = engine.run()
    metrics = results["metrics"]

    # Output directory: runs/<scenario>_<defense>_<seed>/
    defense_str = "defense_on" if defense_on else "defense_off"
    run_folder_name = f"{scenario_cfg.get('id')}_{defense_str}_seed{args.seed}"
    out_dir = REPO_ROOT / args.output_dir / run_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save telemetry.csv
    results["telemetry"].to_csv(out_dir / "telemetry.csv", index=False)

    # 2. Save trust.csv
    results["trust"].to_csv(out_dir / "trust.csv", index=False)

    # 3. Save events.json
    with open(out_dir / "events.json", "w", encoding="utf-8") as f:
        json.dump(results["events"], f, indent=2)

    # 4. Save metrics.json
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 5. Save config_snapshot.yaml
    snapshot = {
        "scenario": scenario_cfg,
        "sensors": sensors_cfg,
        "estimator": estimator_cfg,
        "detector": detector_cfg,
        "resilience": resilience_cfg,
        "planner": planner_cfg,
        "defense_on": defense_on,
        "seed": args.seed
    }
    with open(out_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f)

    print("\n--- Execution Finished ---")
    print(f"Mission Status : {metrics['mission']['status']}")
    print(f"Position RMSE  : {metrics['navigation']['position_rmse_m']} m")
    print(f"Final Goal Err : {metrics['navigation']['final_goal_error_m']} m")
    print(f"Collisions     : {metrics['mission']['collisions']}")
    if defense_on and scenario_cfg.get("attack", {}).get("enabled"):
        print(f"Time to Detect : {metrics['detection']['time_to_detect_s']} s")
        print(f"Containment    : {metrics['resilience']['time_to_contain_s']} s")
    print(f"Results saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
