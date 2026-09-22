import sys
from pathlib import Path
import pytest

# Add path_planner package to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "ros2_ws" / "src" / "path_planner"))

from path_planner.astar_planner import AStarPlanner


def test_astar_straight_line_path():
    planner = AStarPlanner(grid_resolution=1.0, x_bounds=(-20.0, 20.0), y_bounds=(-20.0, 20.0))
    start = (-10.0, 0.0, 5.0)
    goal = (10.0, 0.0, 5.0)

    path = planner.plan(start, goal)
    assert path is not None
    assert len(path) > 0
    # First point near start, last point near goal
    assert pytest.approx(path[0][0], abs=1.0) == -10.0
    assert pytest.approx(path[-1][0], abs=1.0) == 10.0


def test_astar_obstacle_avoidance():
    planner = AStarPlanner(grid_resolution=0.5, x_bounds=(-15.0, 15.0), y_bounds=(-15.0, 15.0))
    start = (-8.0, 0.0, 5.0)
    goal = (8.0, 0.0, 5.0)

    # Place solid obstacle right on the direct path at (0, 0)
    planner.add_obstacle(0.0, 0.0, radius=2.0)

    path = planner.plan(start, goal)
    assert path is not None
    assert len(path) > 0

    # Ensure none of the generated path waypoints penetrate the obstacle
    for pt in path:
        dist_to_obs = ((pt[0] - 0.0) ** 2 + (pt[1] - 0.0) ** 2) ** 0.5
        assert dist_to_obs > 2.0  # Safe distance outside obstacle


def test_astar_cyber_risk_zone_routing():
    planner = AStarPlanner(grid_resolution=0.5, x_bounds=(-15.0, 15.0), y_bounds=(-15.0, 15.0))
    start = (-6.0, 0.0, 5.0)
    goal = (6.0, 0.0, 5.0)

    # Add high cyber-risk zone (e.g. GPS spoofing area) in the direct path
    planner.add_cyber_risk_zone(0.0, 0.0, radius=2.5, risk_level=2.0)

    path = planner.plan(start, goal)
    assert path is not None
    # Verify the path bends to avoid the cyber risk center
    y_coords = [abs(pt[1]) for pt in path]
    assert max(y_coords) > 1.5  # Detoured around the cyber-risk zone
