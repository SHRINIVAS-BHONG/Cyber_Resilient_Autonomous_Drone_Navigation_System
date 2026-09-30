import pytest
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyber_drone.simulation_engine import SimulationEngine


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_scenario_s0_baseline_clean_run():
    scenario_cfg = load_yaml(REPO_ROOT / "config" / "scenarios" / "s0_normal.yaml")
    sensors_cfg = load_yaml(REPO_ROOT / "config" / "sensors.yaml")
    estimator_cfg = load_yaml(REPO_ROOT / "config" / "estimator.yaml")
    detector_cfg = load_yaml(REPO_ROOT / "config" / "detector.yaml")
    resilience_cfg = load_yaml(REPO_ROOT / "config" / "resilience.yaml")
    planner_cfg = load_yaml(REPO_ROOT / "config" / "planner.yaml")

    engine = SimulationEngine(
        scenario_config=scenario_cfg,
        sensors_config=sensors_cfg,
        estimator_config=estimator_cfg,
        detector_config=detector_cfg,
        resilience_config=resilience_cfg,
        planner_config=planner_cfg,
        defense_on=True,
        seed=42
    )

    results = engine.run()
    metrics = results["metrics"]

    # Rule: Zero false alarms across clean runs
    assert metrics["detection"]["false_alarms"] == 0
    assert metrics["mission"]["completed"] is True
    assert metrics["mission"]["collisions"] == 0
    assert metrics["navigation"]["final_goal_error_m"] < 2.5


def test_scenario_s1_contrast_defense_off_vs_on():
    scenario_cfg = load_yaml(REPO_ROOT / "config" / "scenarios" / "s1_gps_drift.yaml")
    sensors_cfg = load_yaml(REPO_ROOT / "config" / "sensors.yaml")
    estimator_cfg = load_yaml(REPO_ROOT / "config" / "estimator.yaml")
    detector_cfg = load_yaml(REPO_ROOT / "config" / "detector.yaml")
    resilience_cfg = load_yaml(REPO_ROOT / "config" / "resilience.yaml")
    planner_cfg = load_yaml(REPO_ROOT / "config" / "planner.yaml")

    # Defense OFF: drone is derailed by GPS spoofing
    engine_off = SimulationEngine(
        scenario_config=scenario_cfg,
        sensors_config=sensors_cfg,
        estimator_config=estimator_cfg,
        detector_config=detector_cfg,
        resilience_config=resilience_cfg,
        planner_config=planner_cfg,
        defense_on=False,
        seed=42
    )
    res_off = engine_off.run()
    metrics_off = res_off["metrics"]
    # Target: Defense OFF derails off course (max error > 10m)
    assert metrics_off["navigation"]["max_position_error_m"] >= 10.0

    # Defense ON: drone detects, isolates, navigates on trusted, stays safe
    engine_on = SimulationEngine(
        scenario_config=scenario_cfg,
        sensors_config=sensors_cfg,
        estimator_config=estimator_cfg,
        detector_config=detector_cfg,
        resilience_config=resilience_cfg,
        planner_config=planner_cfg,
        defense_on=True,
        seed=42
    )
    res_on = engine_on.run()
    metrics_on = res_on["metrics"]

    # Target: Detects within 6s, zero collisions, final goal error under 2.5m
    assert metrics_on["detection"]["time_to_detect_s"] is not None
    assert metrics_on["detection"]["time_to_detect_s"] <= 6.0
    assert metrics_on["mission"]["collisions"] == 0
    assert metrics_on["navigation"]["final_goal_error_m"] <= 2.5
