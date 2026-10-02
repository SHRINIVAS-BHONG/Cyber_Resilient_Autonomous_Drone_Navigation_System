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
        self.obstacles: List[dict] = []
        self.cyber_risk_zones: List[dict] = []

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

    def add_obstacle(self, x: float, y: float, radius: float = 1.0, height: float = 100.0) -> None:
        """Adds a cylindrical obstacle with height and inflates safety margins."""
        self.obstacles.append({"x": float(x), "y": float(y), "radius": float(radius), "height": float(height)})
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
        self.cyber_risk_zones.append({
            "x": float(x),
            "y": float(y),
            "radius": float(radius),
            "risk_level": float(risk_level)
        })
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

    def is_cell_blocked(self, gx: int, gy: int, flight_alt: float) -> bool:
        """Checks if a grid cell is physically blocked at the commanded flight altitude."""
        if self.grid[gx, gy] == 0:
            return False
        if self.obstacles:
            wx, wy = self.grid_to_world(gx, gy)
            for obs in self.obstacles:
                if (obs["height"] + 1.2) > flight_alt:
                    if math.hypot(wx - obs["x"], wy - obs["y"]) <= obs["radius"]:
                        return True
            return False
        return bool(self.grid[gx, gy] == 1)

    def has_line_of_sight(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        flight_alt: float
    ) -> bool:
        """
        Ray-marching line-of-sight test between p1 and p2.
        Returns True if a direct straight-line connection is collision-free and cyber-safe.
        """
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 1e-3:
            return True

        step_size = max(0.2, self.res * 0.4)
        num_steps = max(2, int(math.ceil(dist / step_size)))

        for i in range(num_steps + 1):
            alpha = i / num_steps
            x = p1[0] + alpha * (p2[0] - p1[0])
            y = p1[1] + alpha * (p2[1] - p1[1])

            # Check physical obstacles
            for obs in self.obstacles:
                if (obs["height"] + 1.2) > flight_alt:
                    safe_r = obs["radius"] + self.inflation_radius
                    if math.hypot(x - obs["x"], y - obs["y"]) <= safe_r:
                        return False

            # Check cyber risk zones (e.g. GPS spoofing / jamming zones)
            for rz in self.cyber_risk_zones:
                if math.hypot(x - rz["x"], y - rz["y"]) <= rz["radius"]:
                    return False

            # Check discrete cost map
            g = self.world_to_grid(x, y)
            if g is not None:
                gx, gy = g
                if self.grid[gx, gy] == 1 and not self.obstacles:
                    return False
                if self.cost_map[gx, gy] >= (self.w_risk * 0.5):
                    return False
            else:
                return False

        return True

    def prune_path(
        self,
        raw_path: List[Tuple[float, float, float]],
        start_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
        flight_alt: float
    ) -> List[Tuple[float, float, float]]:
        """
        Greedy String-Pulling / Line-of-Sight Shortcut Algorithm.
        Reduces staircase zig-zag grid steps into minimal, clean, tangent flight segments.
        """
        if not raw_path or len(raw_path) <= 2:
            return [start_pos, goal_pos] if len(raw_path) == 2 else raw_path

        pruned = [start_pos]
        curr_idx = 0

        while curr_idx < len(raw_path) - 1:
            next_idx = len(raw_path) - 1
            while next_idx > curr_idx + 1:
                if self.has_line_of_sight(raw_path[curr_idx], raw_path[next_idx], flight_alt):
                    break
                next_idx -= 1
            pruned.append(raw_path[next_idx])
            curr_idx = next_idx

        pruned[0] = start_pos
        pruned[-1] = goal_pos
        return pruned

    def smooth_trajectory(
        self,
        key_points: List[Tuple[float, float, float]],
        point_spacing: float = 4.0
    ) -> List[Tuple[float, float, float]]:
        """
        Generates a smooth, flyable aerospace trajectory from pruned key waypoints.
        Applies corner fillets at turning vertices and uniform waypoint spacing.
        """
        if len(key_points) <= 1:
            return key_points

        if len(key_points) == 2:
            p1, p2 = key_points[0], key_points[1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist < 1e-3:
                return [p1, p2]
            num_segments = max(1, int(math.ceil(dist / point_spacing)))
            smooth = []
            for i in range(num_segments + 1):
                alpha = i / num_segments
                x = p1[0] + alpha * (p2[0] - p1[0])
                y = p1[1] + alpha * (p2[1] - p1[1])
                z = p1[2] + alpha * (p2[2] - p1[2])
                smooth.append((round(x, 2), round(y, 2), round(z, 2)))
            return smooth

        # Multi-segment trajectory with corner fillets
        smooth_path = [key_points[0]]
        corner_fillet_radius = 5.0  # meters turn radius for quadcopter banking

        for i in range(1, len(key_points) - 1):
            p_prev = np.array(key_points[i - 1], dtype=np.float64)
            p_curr = np.array(key_points[i], dtype=np.float64)
            p_next = np.array(key_points[i + 1], dtype=np.float64)

            d1 = np.linalg.norm(p_curr[:2] - p_prev[:2])
            d2 = np.linalg.norm(p_next[:2] - p_curr[:2])

            max_fillet = min(corner_fillet_radius, d1 * 0.45, d2 * 0.45)
            if max_fillet < 0.5:
                smooth_path.append(tuple(round(v, 2) for v in p_curr))
                continue

            u1 = (p_curr - p_prev) / (d1 + 1e-6)
            u2 = (p_next - p_curr) / (d2 + 1e-6)
            cut1 = p_curr - u1 * max_fillet
            cut2 = p_curr + u2 * max_fillet

            last_pt = np.array(smooth_path[-1], dtype=np.float64)
            leg_dist = np.linalg.norm(cut1[:2] - last_pt[:2])
            if leg_dist > point_spacing:
                leg_steps = int(math.ceil(leg_dist / point_spacing))
                for s in range(1, leg_steps):
                    interp = last_pt + (cut1 - last_pt) * (s / leg_steps)
                    smooth_path.append(tuple(round(v, 2) for v in interp))
            smooth_path.append(tuple(round(v, 2) for v in cut1))

            # Quadratic Bézier transition around vertex
            fillet_steps = 6
            for t_step in range(1, fillet_steps):
                t = t_step / fillet_steps
                b_pt = ((1 - t) ** 2) * cut1 + 2 * (1 - t) * t * p_curr + (t ** 2) * cut2
                smooth_path.append(tuple(round(v, 2) for v in b_pt))
            smooth_path.append(tuple(round(v, 2) for v in cut2))

        # Final leg to goal
        p_last = np.array(key_points[-1], dtype=np.float64)
        last_pt = np.array(smooth_path[-1], dtype=np.float64)
        leg_dist = np.linalg.norm(p_last[:2] - last_pt[:2])
        if leg_dist > point_spacing:
            leg_steps = int(math.ceil(leg_dist / point_spacing))
            for s in range(1, leg_steps):
                interp = last_pt + (p_last - last_pt) * (s / leg_steps)
                smooth_path.append(tuple(round(v, 2) for v in interp))
        smooth_path.append(tuple(round(v, 2) for v in p_last))

        return smooth_path

    def plan(
        self,
        start_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float]
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Executes Risk-Aware A* search with Line-of-Sight Shortcut Pruning
        and Continuous Curvature Smoothing for authentic flight trajectories.
        Returns: list of smooth 3D waypoints [(x, y, z), ...]
        """
        start_grid = self.world_to_grid(start_pos[0], start_pos[1])
        goal_grid = self.world_to_grid(goal_pos[0], goal_pos[1])

        if not start_grid or not goal_grid:
            return None

        flight_alt = min(start_pos[2], goal_pos[2])
        if self.is_cell_blocked(start_grid[0], start_grid[1], flight_alt) or self.is_cell_blocked(goal_grid[0], goal_grid[1], flight_alt):
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
                # Reconstruct raw grid path
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

                # Apply Line-of-Sight Shortcut Pruning & Curvature Smoothing
                pruned = self.prune_path(waypoints, start_pos, goal_pos, flight_alt)
                return self.smooth_trajectory(pruned, point_spacing=4.0)

            for dx, dy, step_cost in motions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.nx and 0 <= neighbor[1] < self.ny):
                    continue
                if self.is_cell_blocked(neighbor[0], neighbor[1], flight_alt):
                    continue  # In solid obstacle at this flight altitude

                # Path cost = distance + dynamic cyber risk / obstacle proximity
                cell_penalty = self.cost_map[neighbor[0], neighbor[1]]
                tentative_g = current_g + (step_cost * self.res) + cell_penalty

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return None
