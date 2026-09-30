#!/usr/bin/env python3
"""
Batch Execution Runner
Runs 7 scenarios x 2 modes (defense OFF vs ON) across multiple seeds.
Generates comprehensive comparative results table and leaderboard.
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyber_drone.simulation_engine import SimulationEngine


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run Batch Cyber Resilience Evaluation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404], help="List of random seeds")
    parser.add_argument("--scenarios", type=str, nargs="+", default=None, help="Subset of scenarios to run (default all S0-S6)")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output directory")
    args = parser.parse_args()

    scenarios_dir = REPO_ROOT / "config" / "scenarios"
    if args.scenarios:
        scenario_files = [f"{s}.yaml" if not s.endswith(".yaml") else s for s in args.scenarios]
    else:
        scenario_files = sorted([f for f in os.listdir(scenarios_dir) if f.endswith(".yaml")])

    sensors_cfg = load_yaml(REPO_ROOT / "config" / "sensors.yaml")
    estimator_cfg = load_yaml(REPO_ROOT / "config" / "estimator.yaml")
    detector_cfg = load_yaml(REPO_ROOT / "config" / "detector.yaml")
    resilience_cfg = load_yaml(REPO_ROOT / "config" / "resilience.yaml")
    planner_cfg = load_yaml(REPO_ROOT / "config" / "planner.yaml")

    summary_rows = []
    total_runs = len(scenario_files) * 2 * len(args.seeds)
    print(f"Starting Batch Execution: {len(scenario_files)} scenarios x 2 modes x {len(args.seeds)} seeds = {total_runs} total runs\n")

    current_idx = 0
    for sc_file in scenario_files:
        scenario_cfg = load_yaml(scenarios_dir / sc_file)
        sc_id = scenario_cfg.get("id")

        for defense_on in [False, True]:
            defense_label = "DEFENSE_ON" if defense_on else "DEFENSE_OFF"

            for seed in args.seeds:
                current_idx += 1
                print(f"[{current_idx}/{total_runs}] Running {sc_id} | {defense_label} | Seed {seed}...", end="", flush=True)

                engine = SimulationEngine(
                    scenario_config=scenario_cfg,
                    sensors_config=sensors_cfg,
                    estimator_config=estimator_cfg,
                    detector_config=detector_cfg,
                    resilience_config=resilience_cfg,
                    planner_config=planner_cfg,
                    defense_on=defense_on,
                    seed=seed
                )

                results = engine.run()
                metrics = results["metrics"]

                # Save run folder
                defense_str = "defense_on" if defense_on else "defense_off"
                run_folder = REPO_ROOT / args.output_dir / f"{sc_id}_{defense_str}_seed{seed}"
                run_folder.mkdir(parents=True, exist_ok=True)
                results["telemetry"].to_csv(run_folder / "telemetry.csv", index=False)
                results["trust"].to_csv(run_folder / "trust.csv", index=False)
                with open(run_folder / "events.json", "w", encoding="utf-8") as f:
                    json.dump(results["events"], f, indent=2)
                with open(run_folder / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                summary_rows.append({
                    "scenario": sc_id,
                    "defense": "ON" if defense_on else "OFF",
                    "seed": seed,
                    "status": metrics["mission"]["status"],
                    "ttd_sec": metrics["detection"]["time_to_detect_s"],
                    "contain_sec": metrics["resilience"]["time_to_contain_s"],
                    "rmse_m": metrics["navigation"]["position_rmse_m"],
                    "max_err_m": metrics["navigation"]["max_position_error_m"],
                    "final_err_m": metrics["navigation"]["final_goal_error_m"],
                    "collisions": metrics["mission"]["collisions"]
                })
                print(f" Done ({metrics['mission']['status']}, FinalErr: {metrics['navigation']['final_goal_error_m']}m)")

    df_summary = pd.DataFrame(summary_rows)
    reports_dir = REPO_ROOT / args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "batch_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    print(f"\n=======================================================")
    print(f"Batch Execution Completed! Report saved to {summary_path}")
    print(f"=======================================================\n")

    # Print comparative aggregation
    agg_table = df_summary.groupby(["scenario", "defense"]).agg({
        "rmse_m": "mean",
        "final_err_m": "mean",
        "collisions": "sum",
        "ttd_sec": "mean"
    }).round(2)
    print("Aggregate Results (Defense OFF vs Defense ON):")
    print(agg_table)


if __name__ == "__main__":
    main()
