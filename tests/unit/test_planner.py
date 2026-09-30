import numpy as np
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyber_drone.astar_planner import AStarPlanner


def test_astar_planner_finds_valid_path():
    planner = AStarPlanner()
    start = (0.0, 0.0)
    goal = (60.0, 40.0)

    res = planner.plan(start, goal, under_attack=False)
    assert not res["emergency"]
    assert len(res["waypoints"]) >= 2
    # Verify first and last waypoint are near start and goal
    first_wp = res["waypoints"][0]
    last_wp = res["waypoints"][-1]
    assert np.hypot(first_wp[0] - start[0], first_wp[1] - start[1]) <= 2.0
    assert np.hypot(last_wp[0] - goal[0], last_wp[1] - goal[1]) <= 2.0


def test_astar_obstacle_inflation_during_attack():
    planner = AStarPlanner()
    cost_normal = planner.build_cost_map(under_attack=False)
    cost_attack = planner.build_cost_map(under_attack=True, cyber_risk_score=0.8)

    # Obstacle inflation during attack (3.5m) must yield more blocked/infinite cells than normal (2.0m)
    inf_normal = np.isinf(cost_normal).sum()
    inf_attack = np.isinf(cost_attack).sum()
    assert inf_attack > inf_normal
