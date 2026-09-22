"""
Risk-Aware A* Path Planner for Cyber-Resilient Drone Navigation.

Generates optimal alternative trajectories navigating around physical obstacles
while dynamically avoiding high cyber-risk zones and sensor uncertainty areas.
"""

import heapq
import math
from typing import List, Optional, Set, Tuple
import numpy as np


class AStarPlanner:
    def __init__(
        self,
        grid_resolution: float = 0.5,     # meters per cell
        x_bounds: Tuple[float, float] = (-30.0, 30.0),
        y_bounds: Tuple[float, float] = (-30.0, 30.0),
        obstacle_inflation_radius: float = 1.5,
        cyber_risk_weight: float = 8.0,
        obstacle_proximity_weight: float = 3.5,
    ):
        self.res = grid_resolution
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.inflation_radius = obstacle_inflation_radius
        self.w_risk = cyber_risk_weight
        self.w_obs = obstacle_proximity_weight

        # Grid dimensions
        self.nx = int(math.ceil((self.x_max - self.x_min) / self.res))
        self.ny = int(math.ceil((self.y_max - self.y_min) / self.res))

        # Occupancy grid (0: free, 1: obstacle, 2: cyber-risk zone)
        self.grid = np.zeros((self.nx, self.ny), dtype=np.uint8)
        self.cost_map = np.zeros((self.nx, self.ny), dtype=np.float32)

    def world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convert world coordinates (meters) to discrete grid indices."""
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return None
        gx = int(round((x - self.x_min) / self.res))
        gy = int(round((y - self.y_min) / self.res))
        gx = min(max(0, gx), self.nx - 1)
        gy = min(max(0, gy), self.ny - 1)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (meters)."""
        x = self.x_min + gx * self.res
        y = self.y_min + gy * self.res
        return x, y

    def add_obstacle(self, x: float, y: float, radius: float = 1.0) -> None:
        """Adds a cylindrical obstacle and inflates safety margins."""
        center = self.world_to_grid(x, y)
        if not center:
            return

        total_radius = radius + self.inflation_radius
        cell_radius = int(math.ceil(total_radius / self.res))
        cx, cy = center

        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                dist = math.hypot(dx * self.res, dy * self.res)
                gx, gy = cx + dx, cy + dy
                if 0 <= gx < self.nx and 0 <= gy < self.ny:
                    if dist <= radius:
                        self.grid[gx, gy] = 1  # Solid obstacle
                    elif dist <= total_radius:
                        # Safety proximity penalty
                        proximity_cost = self.w_obs * (1.0 - (dist - radius) / self.inflation_radius)
                        self.cost_map[gx, gy] = max(self.cost_map[gx, gy], proximity_cost)

    def add_cyber_risk_zone(self, x: float, y: float, radius: float, risk_level: float = 1.0) -> None:
        """Marks a GPS-jammed or spoofed zone as high-cost to route around."""
        center = self.world_to_grid(x, y)
        if not center:
            return

        cell_radius = int(math.ceil(radius / self.res))
        cx, cy = center
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                dist = math.hypot(dx * self.res, dy * self.res)
                gx, gy = cx + dx, cy + dy
                if 0 <= gx < self.nx and 0 <= gy < self.ny:
                    if dist <= radius:
                        self.cost_map[gx, gy] += self.w_risk * risk_level

    def plan(
        self,
        start_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float]
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Executes A* search from start to goal in 2D/3D.
        Returns: list of 3D waypoints [(x, y, z), ...]
        """
        start_grid = self.world_to_grid(start_pos[0], start_pos[1])
        goal_grid = self.world_to_grid(goal_pos[0], goal_pos[1])

        if not start_grid or not goal_grid:
            return None

        if self.grid[start_grid[0], start_grid[1]] == 1 or self.grid[goal_grid[0], goal_grid[1]] == 1:
            return None

        # 8-connected grid motion: (dx, dy, step_cost)
        motions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)
        ]

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1]) * self.res

        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (heuristic(start_grid, goal_grid), 0.0, start_grid))

        came_from: dict = {}
        g_score: dict = {start_grid: 0.0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == goal_grid:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                # Convert grid cells to 3D world waypoints
                waypoints = []
                altitude = (start_pos[2] + goal_pos[2]) / 2.0
                for cell in path:
                    wx, wy = self.grid_to_world(cell[0], cell[1])
                    waypoints.append((wx, wy, altitude))

                return waypoints

            for dx, dy, step_cost in motions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.nx and 0 <= neighbor[1] < self.ny):
                    continue
                if self.grid[neighbor[0], neighbor[1]] == 1:
                    continue  # In solid obstacle

                # Path cost = distance + dynamic cyber risk / obstacle proximity
                cell_penalty = self.cost_map[neighbor[0], neighbor[1]]
                tentative_g = current_g + (step_cost * self.res) + cell_penalty

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return None
