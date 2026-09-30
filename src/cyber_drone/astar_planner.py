"""
A* Path Planner with Dynamic Obstacle Inflation and Cyber-Risk Cost Weights
Supports 2.0m normal inflation, 3.5m attack inflation, restricted zones,
risk-aware cost function, and emergency hover/landing fallback.
"""

import heapq
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class AStarPlanner:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Grid bounds
        grid_cfg = self.config.get("grid", {})
        self.resolution = float(grid_cfg.get("resolution", 1.0))
        self.x_min = float(grid_cfg.get("x_min", -10.0))
        self.x_max = float(grid_cfg.get("x_max", 90.0))
        self.y_min = float(grid_cfg.get("y_min", -10.0))
        self.y_max = float(grid_cfg.get("y_max", 70.0))

        self.width = int(np.ceil((self.x_max - self.x_min) / self.resolution))
        self.height = int(np.ceil((self.y_max - self.y_min) / self.resolution))

        # Inflation radii
        infl_cfg = self.config.get("inflation", {})
        self.normal_inflation_m = float(infl_cfg.get("normal_m", 2.0))
        self.attack_inflation_m = float(infl_cfg.get("attack_m", 3.5))

        # Cost weights
        cw = self.config.get("cost_weights", {})
        self.w_dist = float(cw.get("distance", 1.0))
        self.w_obs = float(cw.get("obstacle_proximity", 3.0))
        self.w_unc = float(cw.get("uncertainty", 2.0))
        self.w_risk = float(cw.get("cyber_risk", 4.0))

        # Obstacles and restricted zones
        self.obstacles = self.config.get("default_obstacles", [
            {"x": 20.0, "y": 15.0, "radius": 3.0},
            {"x": 35.0, "y": 22.0, "radius": 4.0},
            {"x": 48.0, "y": 32.0, "radius": 3.5},
            {"x": 28.0, "y": 38.0, "radius": 3.0},
            {"x": 10.0, "y": 25.0, "radius": 2.5}
        ])
        self.restricted_zones = self.config.get("restricted_zones", [
            {"min_x": 25.0, "max_x": 35.0, "min_y": 20.0, "max_y": 35.0}
        ])

        # Nominal and degraded speed limits
        spd = self.config.get("speed_limits", {})
        self.nominal_speed = float(spd.get("nominal_mps", 3.0))
        self.degraded_speed = float(spd.get("degraded_mps", 1.5))

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(round((x - self.x_min) / self.resolution))
        gy = int(round((y - self.y_min) / self.resolution))
        gx = max(0, min(self.width - 1, gx))
        gy = max(0, min(self.height - 1, gy))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = self.x_min + gx * self.resolution
        y = self.y_min + gy * self.resolution
        return round(x, 2), round(y, 2)

    def is_in_restricted_zone(self, x: float, y: float) -> bool:
        for rz in self.restricted_zones:
            if rz["min_x"] <= x <= rz["max_x"] and rz["min_y"] <= y <= rz["max_y"]:
                return True
        return False

    def build_cost_map(self, under_attack: bool, cyber_risk_score: float = 0.0) -> np.ndarray:
        inflation = self.attack_inflation_m if under_attack else self.normal_inflation_m
        cost_map = np.ones((self.width, self.height), dtype=float)

        for gx in range(self.width):
            for gy in range(self.height):
                wx, wy = self.grid_to_world(gx, gy)

                # Check restricted zones (infinite cost)
                if self.is_in_restricted_zone(wx, wy):
                    cost_map[gx, gy] = np.inf
                    continue

                # Check obstacles
                min_dist_to_obs = np.inf
                for obs in self.obstacles:
                    ox, oy = obs["x"], obs["y"]
                    base_r = obs.get("radius", 2.0)
                    dist = np.hypot(wx - ox, wy - oy)
                    effective_radius = base_r + inflation

                    if dist <= effective_radius:
                        cost_map[gx, gy] = np.inf
                        break
                    
                    dist_to_edge = dist - effective_radius
                    if dist_to_edge < min_dist_to_obs:
                        min_dist_to_obs = dist_to_edge

                if cost_map[gx, gy] == np.inf:
                    continue

                # Proximity cost: higher cost when close to obstacle buffer
                proximity_cost = 0.0
                if min_dist_to_obs < 5.0:
                    proximity_cost = self.w_obs * (5.0 - min_dist_to_obs)

                # Cyber-risk & uncertainty cost
                risk_cost = 0.0
                if under_attack:
                    # Risk grows in narrow corridors and with cyber_risk_score
                    risk_cost = self.w_risk * cyber_risk_score + self.w_unc * (1.0 / (min_dist_to_obs + 0.1))

                cost_map[gx, gy] = 1.0 + proximity_cost + risk_cost

        return cost_map

    def plan(
        self,
        start_world: Tuple[float, float],
        goal_world: Tuple[float, float],
        under_attack: bool = False,
        cyber_risk_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Runs A* search from start_world to goal_world.
        Returns path waypoints, maximum velocity, and emergency flag.
        """
        cost_map = self.build_cost_map(under_attack, cyber_risk_score)
        start_g = self.world_to_grid(start_world[0], start_world[1])
        goal_g = self.world_to_grid(goal_world[0], goal_world[1])

        # If start or goal falls in obstacle, nudge to nearest free cell
        start_g = self._find_nearest_free_cell(start_g, cost_map)
        goal_g = self._find_nearest_free_cell(goal_g, cost_map)

        # 8-connectivity motion primitives
        motions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, -1, np.sqrt(2))
        ]

        open_set = []
        heapq.heappush(open_set, (0.0, start_g))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_g: 0.0}

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return np.hypot(a[0] - b[0], a[1] - b[1]) * self.resolution * self.w_dist

        path_found = False
        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == goal_g:
                path_found = True
                break

            for dx, dy, move_cost in motions:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if 0 <= nx < self.width and 0 <= ny < self.height:
                    step_cost = cost_map[nx, ny]
                    if np.isinf(step_cost):
                        continue

                    tentative_g = g_score[current] + move_cost * self.resolution * step_cost
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, goal_g)
                        heapq.heappush(open_set, (f_score, neighbor))

        if not path_found:
            # Emergency Hover / In-place Safe Landing Fallback
            return {
                "waypoints": [[start_world[0], start_world[1], 10.0], [start_world[0], start_world[1], 0.0]],
                "maximum_velocity": 0.5,
                "estimated_risk": 1.0,
                "planner": "emergency_hover_landing_fallback",
                "emergency": True
            }

        # Reconstruct path and smooth waypoints
        grid_path = []
        curr = goal_g
        while curr in came_from:
            grid_path.append(curr)
            curr = came_from[curr]
        grid_path.append(start_g)
        grid_path.reverse()

        # Convert to world coordinates and sample sparsely (every 5 meters)
        raw_waypoints = [self.grid_to_world(gx, gy) for gx, gy in grid_path]
        sparse_waypoints = self._simplify_path(raw_waypoints, altitude=10.0)

        max_vel = self.degraded_speed if under_attack else self.nominal_speed

        return {
            "waypoints": sparse_waypoints,
            "maximum_velocity": max_vel,
            "estimated_risk": round(cyber_risk_score, 2),
            "planner": "astar_risk_aware",
            "emergency": False
        }

    def _find_nearest_free_cell(self, cell: Tuple[int, int], cost_map: np.ndarray) -> Tuple[int, int]:
        if not np.isinf(cost_map[cell[0], cell[1]]):
            return cell
        for radius in range(1, 15):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = cell[0] + dx, cell[1] + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if not np.isinf(cost_map[nx, ny]):
                            return (nx, ny)
        return cell

    def _simplify_path(self, raw_points: List[Tuple[float, float]], altitude: float = 10.0) -> List[List[float]]:
        if len(raw_points) <= 2:
            return [[p[0], p[1], altitude] for p in raw_points]

        simplified = [raw_points[0]]
        accum_dist = 0.0
        for i in range(1, len(raw_points) - 1):
            p_prev = raw_points[i - 1]
            p_curr = raw_points[i]
            accum_dist += np.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            # Keep waypoints separated by ~5 meters or turns
            if accum_dist >= 5.0:
                simplified.append(p_curr)
                accum_dist = 0.0
        simplified.append(raw_points[-1])
        return [[round(p[0], 2), round(p[1], 2), altitude] for p in simplified]
