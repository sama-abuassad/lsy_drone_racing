"""Gate-traversing state controller – fixed gate normal, subgoal logic, obstacle inflation."""
from __future__ import annotations
from typing import TYPE_CHECKING

import heapq
from lsy_drone_racing.envs.race_core import obs
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller

import sys
sys.stdout = open("output.log", "w")

if TYPE_CHECKING:
    from numpy.typing import NDArray

TAKEOFF_HEIGHT = 0.5
TAKEOFF_TIME   = 1.0

# ── Tuning ────────────────────────────────────────────────────────────────────
GATE_APPROACH_OFFSET  = 0.3   # m vor dem Gate-Zentrum
GATE_EXIT_OFFSET      = 0.3   # m hinter dem Gate-Zentrum
OBSTACLE_MARGIN       = 0.25  # physischer Radius der Hindernisse (m)
GRID_RESOLUTION       = 0.15  # A*-Auflösung (m)
APPROACH_THRESHOLD    = 0.15  # m – wann gilt Approach als erreicht?
GATE_THRESHOLD        = 0.12  # m – wann gilt Gate-Zentrum als erreicht?
GATE_HALF_WIDTH       = 0.45  # etwas größer als echte Gate-Öffnung
GATE_MARGIN           = 0.12  # kleinere Inflation für Gate-Rahmen
TIME_SCALE            = 3.0   # Zeitfaktor für Spline-Geschwindigkeit
SLOWNDOWN_SCALE       = 2.0   # Faktor um letzten Abschnitt zu verlangsamen
REPLAN_INTERVAL       = 0.05   # s – wie oft replanen (außer bei Nähe zum Gate)
GATE_PROXIMITY_THRESHOLD = 0.5  # m – wann gilt die Drohne als "nahe" am Gate (Spline einfrieren)
SENSOR_RANGE          = 0.7   # m – 100 for level 0,1, 0.7 für level 2
# ─────────────────────────────────────────────────────────────────────────────


class MyController(Controller):

    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq   = config.env.freq
        self._tick   = 0
        self._n_gates = len(obs["gates_pos"])
        self._gate_pos    = obs["gates_pos"].copy()
        self._gate_quat   = obs["gates_quat"].copy()
        self._gates_visited = obs["gates_visited"].copy()
        self._prev_target_gate = int(obs["target_gate"])

        self._spline = None
        self._spline_vel = None
        self._t_total = 0.0
        self._t_start_tick = 0

        self._pos_integral = np.zeros(3)
        self._current_goal  = None

        self._last_obstacles = np.empty((0, 8))

        print("\n========== INIT ==========")
        print(f"  n_gates      : {self._n_gates}")
        print(f"  start pos    : {obs['pos']}")
        print(f"  gate_pos     :\n{self._gate_pos}")
        print(f"  gates_visited: {self._gates_visited}")
        print(f"  target_gate  : {obs['target_gate']}")
        print(f"  freq         : {self._freq}")
        print("==========================\n")

        # print("\n===== GATE NORMALS DEBUG =====")
        # for i in range(self._n_gates):
        #     rot = R.from_quat(self._gate_quat[i])
        #     for axis, vec in [("X [1,0,0]", [1,0,0]), 
        #                       ("Y [0,1,0]", [0,1,0]), 
        #                       ("Z [0,0,1]", [0,0,1])]:
        #         n = rot.apply(vec)
        #         approach = self._gate_pos[i] - 0.4 * n
        #         exit_pt  = self._gate_pos[i] + 0.4 * n
        #         print(f"  Gate {i} | axis={axis} | normal={np.round(n,3)} | "
        #               f"approach={np.round(approach,3)} | exit={np.round(exit_pt,3)}")
        # print("==============================\n")

    # ── Hilfsmethode: Gate-Normalenvektor korrekt bestimmen ───────────────────
    def _gate_normal(self, gate_id: int) -> np.ndarray:
        rot = R.from_quat(self._gate_quat[gate_id])

        # Feste Definition: X-Achse = Durchflugrichtung
        normal = rot.apply([1.0, 0.0, 0.0])

        # Optional: nur XY
        normal[2] = 0.0

        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0])

        return normal / norm
    
    #── Debug-Hilfsmethode: Gate-Info ausgeben ─────────────────────────────────
    
    def print_gate_info(self, drone_pos=None):
        """Prints gate position, normal, approach and exit points."""
        gate_data = []

        print("\n===== GATE INFO =====")
        for i in range(self._n_gates):
            pos = self._gate_pos[i]

            normal = self._gate_normal(i)

            approach = pos - GATE_APPROACH_OFFSET * normal
            exit_pt  = pos + GATE_EXIT_OFFSET * normal

            print(f"Gate {i}:")
            print(f"  position : {np.round(pos, 3)}")
            print(f"  normal   : {np.round(normal, 3)}")
            print(f"  approach : {np.round(approach, 3)}")
            print(f"  exit     : {np.round(exit_pt, 3)}")

            gate_data.append({
                "id": i,
                "position": pos.copy(),
                "normal": normal.copy(),
                "approach": approach.copy(),
                "exit": exit_pt.copy()
            })

        print("=====================\n")
        return gate_data
    
    def _filter_trusted_obstacles(self, obs, pos):
        trusted = []
        pos_xy = pos[:2]
        for o in obs["obstacles_pos"]:
            if np.linalg.norm(o[:2] - pos_xy) < SENSOR_RANGE:  # sensor_range
                trusted.append(o)
        for g in obs["gates_pos"]:
            if np.linalg.norm(g[:2] - pos_xy) < SENSOR_RANGE:
                trusted.append(g)
        return np.array(trusted) if len(trusted) > 0 else np.empty((0,3))

    # ── Occupancy Grid ────────────────────────────────────────────────────────

    def _build_occupancy_grid(self, obs: dict, start_pos: np.ndarray, current_gate: int) -> np.ndarray:
        obstacles = obs["obstacles_pos"]
        print(f"  [build grid] raw obstacles:\n{np.round(obstacles,3)}")
        # obstacles = self._filter_trusted_obstacles(obs, start_pos)
        all_points = np.vstack([self._gate_pos[:, :2], obstacles[:, :2]])
        min_bounds = np.min(all_points, axis=0) - 3
        max_bounds = np.max(all_points, axis=0) + 3

        self._grid_origin = min_bounds
        self._grid_res    = GRID_RESOLUTION

        grid_size = np.ceil((max_bounds - min_bounds) / GRID_RESOLUTION).astype(int)
        grid = np.zeros(grid_size, dtype=bool)

        inflate_obs  = max(1, int(OBSTACLE_MARGIN / GRID_RESOLUTION))
        inflate_gate = max(1, int(GATE_MARGIN     / GRID_RESOLUTION))

        # ── Physische Hindernisse ─────────────────────────────────────────────
        for obs_pos in obstacles:
            self._mark_circle(grid, obs_pos[:2], inflate_obs, min_bounds)

        # ── Nicht-Ziel-Gates als Wand ─────────────────────────────────────────
        # Edit: Alle Gates als Wände markieren, damit A* nicht durch sie hindurch plant 
        # wir haben ja approach/exit points als Pflichtpunkte
        for i in range(self._n_gates):
            # if i == current_gate:
            #     continue

            gate_center = self._gate_pos[i][:2]
            normal = self._gate_normal(i)[:2]
            normal /= np.linalg.norm(normal) + 1e-9
            perp = np.array([-normal[1], normal[0]])  # entlang Gate-Breite

            # Punkte entlang der gesamten Gate-Breite (Wand)
            n_pts = int((2 * GATE_HALF_WIDTH) / GRID_RESOLUTION) + 1
            for j in range(n_pts):
                offset = -GATE_HALF_WIDTH + j * GRID_RESOLUTION
                pt = gate_center + offset * perp
                self._mark_circle(grid, pt, inflate_gate, min_bounds)

        return grid

    def _mark_circle(self, grid, xy, inflate, min_bounds):
        """Markiert einen Kreis im 2D-Grid."""
        idx = ((xy - min_bounds) / self._grid_res).astype(int)
        idx = np.clip(idx, 0, np.array(grid.shape) - 1)
        for dx in range(-inflate, inflate + 1):
            for dy in range(-inflate, inflate + 1):
                if dx*dx + dy*dy > inflate*inflate:
                    continue
                x, y = idx[0] + dx, idx[1] + dy
                if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                    grid[x, y] = True

    # # ── A* ────────────────────────────────────────────────────────────────────
    # def _astar(self, start: np.ndarray, goal: np.ndarray, grid: np.ndarray):
    #     def to_grid(p):
    #         return tuple(np.clip(
    #             ((p[:2] - self._grid_origin) / self._grid_res).astype(int),
    #             0, np.array(grid.shape) - 1
    #         ))

    #     def to_world_xy(idx):
    #         return self._grid_origin + self._grid_res * np.array(idx, dtype=float)

    #     start_idx = to_grid(start)
    #     goal_idx  = to_grid(goal)

    #     if grid[start_idx]: grid = grid.copy(); grid[start_idx] = False
    #     if grid[goal_idx]:  grid = grid.copy(); grid[goal_idx]  = False

    #     neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    #     open_set = []
    #     heapq.heappush(open_set, (0.0, start_idx))
    #     came_from = {}
    #     g_score   = {start_idx: 0.0}

    #     while open_set:
    #         _, current = heapq.heappop(open_set)
    #         if current == goal_idx:
    #             path_xy = []
    #             while current in came_from:
    #                 path_xy.append(to_world_xy(current))
    #                 current = came_from[current]
    #             path_xy.append(start[:2].copy())
    #             return path_xy[::-1]   # nur XY-Punkte!

    #         for d in neighbors:
    #             nb = (current[0]+d[0], current[1]+d[1])
    #             if not (0 <= nb[0] < grid.shape[0] and 0 <= nb[1] < grid.shape[1]):
    #                 continue
    #             if grid[nb]: continue
    #             tentative = g_score[current] + np.linalg.norm(d)
    #             if nb not in g_score or tentative < g_score[nb]:
    #                 came_from[nb] = current
    #                 g_score[nb]   = tentative
    #                 h = np.linalg.norm(np.array(nb) - np.array(goal_idx))
    #                 heapq.heappush(open_set, (tentative + h, nb))
    #     return None

    # We replace A* withg rrt- star so we can start the path independent of grid
    # we check still if the node placed is in an occupied cell,
    # but we are not limited to the grid connectivity
    # 

    # def _rrt_star(self, start, goal, grid, n_samples=3000, step_size=0.2, radius=0.4):
    #     nodes = [start[:2].copy()]
    #     parents = [-1]
    #     costs = [0.0]

    #     def in_collision(p):
    #         idx = ((p - self._grid_origin) / self._grid_res).astype(int)
    #         idx = np.clip(idx, 0, np.array(grid.shape) - 1)
    #         return grid[idx[0], idx[1]]

    #     def line_clear(a, b, n_checks=10):
    #         for i in range(n_checks + 1):
    #             p = a + (b - a) * i / n_checks
    #             if in_collision(p):
    #                 return False
    #         return True

    #     for _ in range(n_samples):
    #         # Sample: bias toward goal 20% of the time
    #         if np.random.rand() < 0.2:
    #             sample = goal[:2].copy()
    #         else:
    #             sample = self._grid_origin + np.random.rand(2) * (
    #                 np.array(grid.shape) * self._grid_res
    #             )

    #         # Find nearest node
    #         dists = np.linalg.norm(np.array(nodes) - sample, axis=1)
    #         nearest_idx = int(np.argmin(dists))
    #         nearest = nodes[nearest_idx]

    #         # Step toward sample
    #         direction = sample - nearest
    #         dist = np.linalg.norm(direction)
    #         if dist < 1e-6:
    #             continue
    #         new_pt = nearest + (direction / dist) * min(step_size, dist)

    #         if in_collision(new_pt):
    #             continue

    #         # RRT* rewiring
    #         new_cost = costs[nearest_idx] + np.linalg.norm(new_pt - nearest)
    #         best_parent = nearest_idx

    #         for i, node in enumerate(nodes):
    #             if np.linalg.norm(node - new_pt) < radius:
    #                 candidate_cost = costs[i] + np.linalg.norm(new_pt - node)
    #                 if candidate_cost < new_cost and line_clear(node, new_pt):
    #                     new_cost = candidate_cost
    #                     best_parent = i

    #         if not line_clear(nodes[best_parent], new_pt):
    #             continue

    #         nodes.append(new_pt)
    #         parents.append(best_parent)
    #         costs.append(new_cost)

    #         # Early exit if close to goal
    #         if np.linalg.norm(new_pt - goal[:2]) < step_size:
    #             break

    #     # Find best node near goal
    #     dists_to_goal = np.linalg.norm(np.array(nodes) - goal[:2], axis=1)
    #     best = int(np.argmin(dists_to_goal))

    #     # Reconstruct path
    #     path = []
    #     idx = best
    #     while idx != -1:
    #         path.append(nodes[idx])
    #         idx = parents[idx]
    #     return path[::-1]

    # def _theta_star(self, start: np.ndarray, goal: np.ndarray, grid: np.ndarray):

    #     def to_grid(p):
    #         return tuple(np.clip(
    #             ((np.array(p[:2]) - self._grid_origin) / self._grid_res).astype(int),
    #             0, np.array(grid.shape) - 1
    #         ))

    #     def to_world(idx):
    #         return self._grid_origin + self._grid_res * np.array(idx, dtype=float)

    #     def line_of_sight(a_idx, b_idx):
    #         x0, y0 = a_idx
    #         x1, y1 = b_idx
    #         dx, dy = abs(x1 - x0), abs(y1 - y0)
    #         sx = 1 if x0 < x1 else -1
    #         sy = 1 if y0 < y1 else -1
    #         err = dx - dy
    #         x, y = x0, y0
    #         while True:
    #             if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
    #                 return False
    #             if grid[x, y]:
    #                 return False
    #             if x == x1 and y == y1:
    #                 return True
    #             e2 = 2 * err
    #             if e2 > -dy: err -= dy; x += sx
    #             if e2 <  dx: err += dx; y += sy

    #     start_idx = to_grid(start)
    #     goal_idx  = to_grid(goal)

    #     g = grid.copy()
    #     if g[start_idx]: g[start_idx] = False
    #     if g[goal_idx]:  g[goal_idx]  = False

    #     neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    #     open_set = []
    #     heapq.heappush(open_set, (0.0, start_idx))

    #     # ── KEY CHANGE: store world-space parent positions ──────────────────────
    #     # came_from maps grid_idx → (parent_grid_idx, parent_world_xy)
    #     # For start, parent is itself at exact drone position (not snapped)
    #     came_from = {start_idx: (start_idx, start[:2].copy())}
    #     g_score   = {start_idx: 0.0}

    #     while open_set:
    #         _, current = heapq.heappop(open_set)

    #         if current == goal_idx:
    #             # Reconstruct using stored world positions
    #             path = []
    #             node = current
    #             while True:
    #                 parent_idx, parent_world = came_from[node]
    #                 if node == start_idx:
    #                     path.append(start[:2].copy())  # exact start, not snapped
    #                     break
    #                 path.append(to_world(node))
    #                 node = parent_idx
    #             path.reverse()
    #             path.append(goal[:2].copy())  # exact goal, not snapped
    #             return np.array(path)

    #         parent_idx, parent_world = came_from[current]

    #         for d in neighbors:
    #             nb = (current[0] + d[0], current[1] + d[1])
    #             if not (0 <= nb[0] < g.shape[0] and 0 <= nb[1] < g.shape[1]):
    #                 continue
    #             if g[nb]:
    #                 continue

    #             nb_world = to_world(nb)

    #             if line_of_sight(parent_idx, nb):
    #                 # Path 2: grandparent → nb using exact parent world pos
    #                 tentative = g_score[parent_idx] + np.linalg.norm(nb_world - parent_world)
    #                 if nb not in g_score or tentative < g_score[nb]:
    #                     came_from[nb] = (parent_idx, parent_world)
    #                     g_score[nb]   = tentative
    #                     h = np.linalg.norm(nb_world - to_world(goal_idx))
    #                     heapq.heappush(open_set, (tentative + h, nb))
    #             else:
    #                 # Path 1: current → nb
    #                 current_world = to_world(current)
    #                 tentative = g_score[current] + np.linalg.norm(nb_world - current_world)
    #                 if nb not in g_score or tentative < g_score[nb]:
    #                     came_from[nb] = (current, current_world)
    #                     g_score[nb]   = tentative
    #                     h = np.linalg.norm(nb_world - to_world(goal_idx))
    #                     heapq.heappush(open_set, (tentative + h, nb))

    #     print("  [Theta*] WARNING: no path found, using direct line")
    #     return np.array([start[:2].copy(), goal[:2].copy()])


    # we need to unite the attributes of the search alorithms to find a good path
    # without many strict curves whise respecting the occupancy grid
    def _custom_star(self, grid, start, goal, step_length= 0.2, n_samples=8):

        start_xy = start[:2].copy()
        goal_xy  = goal[:2].copy()

        def is_occupied(grid, point_xy: np.ndarray) -> bool:
            """Returns True if the world-space XY point falls on an occupied grid cell."""
            if not hasattr(self, '_grid_origin') or not hasattr(self, '_grid'):
                return False
            # 2. Welt-Koordinaten in Grid-Indizes umrechnen
            # Formel: (Punkt - Ursprung) / Auflösung
            idx = ((point_xy[:2] - self._grid_origin) / self._grid_res).astype(int)
            
            return bool(grid[idx[0], idx[1]])
        
        def line_clear(grid,a: np.ndarray, b: np.ndarray, n_checks: int = 12) -> bool:
            """True if the straight line from a to b has no occupied cells."""
            for t in np.linspace(0.0, 1.0, n_checks):
                if is_occupied(grid, a + t * (b - a)):
                    return False
            return True

        def is_valid_waypoint(grid, p, prev):
            """Point must be free AND reachable in a straight line from prev."""
            return not is_occupied(grid, p) and line_clear(grid, prev, p)
        
        def heuristic(p: np.ndarray) -> float:
            """Euclidean distance to goal — admissible, never overestimates."""
            return float(np.linalg.norm(p - goal_xy))
        
        def circle_candidates(center: np.ndarray) -> np.ndarray:
            """Generate n_samples evenly spaced on a circle around center."""
            angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
            # Bias first candidate toward goal direction
            goal_angle = np.arctan2(goal_xy[1] - center[1], goal_xy[0] - center[0])
            angles = angles + goal_angle  # rotate ring so one point aims at goal
            return np.column_stack([
                center[0] + step_length * np.cos(angles),
                center[1] + step_length * np.sin(angles)
            ])
        
        # ── Node storage ──────────────────────────────────────────────────────────
        # Each node: [x, y]
        nodes   = [start_xy.copy()]
        parents = [-1]
        g_costs = [0.0]                          # cost from start

        # Open set: (f_cost, node_index)
        open_set = []
        heapq.heappush(open_set, (heuristic(start_xy), 0))
        visited  = set()

        best_goal_idx  = -1
        best_goal_cost = np.inf

        # ── Main loop ─────────────────────────────────────────────────────────────
        # iterate for maximum double the path length to goal in steps, to avoid infinite loops in complex scenarios
        max_iterations = 2 * int(np.linalg.norm(start_xy - goal_xy) / step_length)
        for iteration in range(max_iterations):
            if not open_set:
                break

            f, current_idx = heapq.heappop(open_set)

            # Skip if already expanded
            if current_idx in visited:
                continue
            visited.add(current_idx)

            current_pos  = nodes[current_idx]
            current_g    = g_costs[current_idx]

            # ── Early exit if we reached the goal ────────────────────────────────
            if np.linalg.norm(current_pos - goal_xy) < step_length:
                if line_clear(grid, current_pos, goal_xy):
                    # Connect directly to goal
                    nodes.append(goal_xy.copy())
                    parents.append(current_idx)
                    g_costs.append(current_g + np.linalg.norm(goal_xy - current_pos))
                    best_goal_idx = len(nodes) - 1
                    break

            # ── Generate circle candidates ────────────────────────────────────────
            candidates = circle_candidates(current_pos)

            # Evaluate and filter each candidate
            valid_candidates = []
            for pt in candidates:
                if not is_valid_waypoint(grid, pt, current_pos):
                    continue  # line blocked or cell occupied — eliminate

                g_new = current_g + step_length
                h_new = heuristic(pt)
                f_new = g_new + h_new

                valid_candidates.append((f_new, g_new, pt))

            if not valid_candidates:
                continue

            # Sort by f_cost — best candidates first
            valid_candidates.sort(key=lambda x: x[0])

            # Add all valid candidates as nodes (not just the best —
            # lets the open set explore alternatives if best hits a dead end)
            for f_new, g_new, pt in valid_candidates:
                # Snap-check: don't add a point too close to an existing node
                too_close = any(
                    np.linalg.norm(pt - nodes[i]) < step_length * 0.4
                    for i in range(max(0, len(nodes) - 30), len(nodes))  # check recent nodes only
                )
                if too_close:
                    continue

                new_idx = len(nodes)
                nodes.append(pt.copy())
                parents.append(current_idx)
                g_costs.append(g_new)
                heapq.heappush(open_set, (f_new, new_idx))

                # Track best node that can see the goal
                if f_new < best_goal_cost and np.linalg.norm(pt - goal_xy) < step_length * 2:
                    if line_clear(grid, pt, goal_xy):
                        best_goal_cost = f_new
                        best_goal_idx  = new_idx

        # ── Reconstruct path ──────────────────────────────────────────────────────
        if best_goal_idx < 0:
            # No path reached goal — return closest node to goal
            dists = [np.linalg.norm(n - goal_xy) for n in nodes]
            best_goal_idx = int(np.argmin(dists))
            print(f"  [custom_star] WARNING: goal not reached, "
                  f"closest dist={dists[best_goal_idx]:.3f}m")

        path = []
        idx  = best_goal_idx
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()

        # Always end exactly at goal
        if np.linalg.norm(np.array(path[-1]) - goal_xy) > 0.02:
            path.append(goal_xy.copy())

        path = np.array(path)
        print(f"  [custom_star] {len(nodes)} nodes expanded, "
              f"path={len(path)} pts, "
              f"iters={iteration+1}")
        return path


        
        

    # ── Spline bauen ──────────────────────────────────────────────────────────

    def _insert_gate_waypoints(self, path_xy, approach_xy, center_xy, exit_xy):
        """Ersetzt den Teil des Pfades nahe dem Gate durch approach → center."""
        # Finde den Punkt im Pfad der approach am nächsten ist
        dists = np.linalg.norm(path_xy - approach_xy, axis=1)
        idx   = int(np.argmin(dists))
        # Schneide Pfad bei diesem Index ab und füge Pflichtpunkte ein
        before  = path_xy[:idx]
        after   = path_xy[idx+1:]  # alles hinter approach
        # Finde auch center im verbleibenden Pfad
        if len(after) > 0:
            dists2 = np.linalg.norm(after - center_xy, axis=1)
            idx2   = int(np.argmin(dists2))
            after  = after[idx2+1:]
        return np.vstack([before, [approach_xy], [center_xy], [exit_xy]]) if len(after) > 0 \
               else np.vstack([before, [approach_xy], [center_xy]])

    def _gate_waypoints(self, gate_id: int, prev_pos: np.ndarray):
        # center = self._gate_pos[gate_id].copy()
        gate = self._gate_pos[gate_id]
        dist_to_gate = np.linalg.norm(prev_pos - gate)

        if dist_to_gate < SENSOR_RANGE:
            normal = self._gate_normal(gate_id)[:2]
            center = gate
            center_xy = center[:2]
        else:
            normal = self._gate_normal(gate_id)[:2]
            center = gate
            center_xy = center[:2]

        

        # Zwei Seiten des Gates
        approach_xy = center_xy - normal * GATE_APPROACH_OFFSET
        exit_xy = center_xy + normal * GATE_APPROACH_OFFSET

        approach = np.array([approach_xy[0], approach_xy[1], center[2]])
        exit_pt  = np.array([exit_xy[0],     exit_xy[1],     center[2]])

        return approach, center, exit_pt
    
    def _obstacles_changed(self, obs, pos):
        current = self._filter_trusted_obstacles(obs, pos)

        # print(f"  [obstacle check] current trusted obstacles:\n{np.round(current,3)}")

        # First call → initialize
        if not hasattr(self, "_last_trusted"):
            self._last_trusted = current.copy()
            return False

        # Case 1: number changed (NEW obstacle appeared)
        if len(current) != len(self._last_trusted):
            print("  [REPLAN] new obstacle entered sensor range")
            self._last_trusted = current.copy()
            self._update_gates(obs)  # auch Gates updaten, falls sie jetzt sichtbar sind
            return True

        # Case 2: same number but positions changed significantly
        for c in current:
            dists = np.linalg.norm(self._last_trusted - c, axis=1)
            if len(dists) == 0 or np.min(dists) > 0.1:
                print("  [REPLAN] obstacle position refined")
                self._last_trusted = current.copy()
                self._update_gates(obs)  # auch Gates updaten, falls sie jetzt sichtbar sind
                return True

        return False

    def _build_spline(self, start_pos, gate_id, obs, label=""):

        # Debug: Gate-Info ausgeben
        self.print_gate_info(drone_pos=start_pos)
        self._pos_integral[:] = 0.0

        from_pos = start_pos


        # DEBUG
        print(f"  [waypoint debug] gate_id={gate_id} from_pos={np.round(from_pos[:2],3)}")
        
        #DEFINE GATE WAYPOINTS
        approach, center, exit_pt = self._gate_waypoints(gate_id, from_pos)
        # Baue Occupancy Grid und finde Pfad von aktueller Position zum approach point
        grid = self._build_occupancy_grid(obs, start_pos, gate_id)
        path_xy_1 = self._custom_star(grid, from_pos[:2], approach[:2], step_length=0.2)
        if path_xy_1 is None:
            print("  [WARNING] No path found to approach point! Using direct line.")
            path_xy_1 = np.array([from_pos[:2], approach[:2]])
        path_xy_2 = np.array([approach[:2], center[:2]])
        path_xy_3 = np.array([center[:2], exit_pt[:2]])

        self._current_exit = exit_pt.copy()

        xy = np.vstack([
        np.array(path_xy_1),
        approach[:2],          # 🔴 FORCE
        np.array(path_xy_2)[1:],
        center[:2],            # 🔴 FORCE
        np.array(path_xy_3)[1:],
        exit_pt[:2]            # 🔴 FORCE
        ])

        # Doppelte Punkte entfernen (können durch approach/center/exit Einfügung entstehen)
        def remove_duplicate_points(xy, eps=1e-3):
            filtered = [xy[0]]
            for p in xy[1:]:
                if np.linalg.norm(p - filtered[-1]) > eps:
                    filtered.append(p)
            return np.array(filtered)

        xy = remove_duplicate_points(xy)
        
        gate_z   = self._gate_pos[gate_id][2]

        # Drohen erst bei exit_pt auf Gate höhe - zu spät !!!
        # z         = np.linspace(start_pos[2], gate_z, len(xy))
        # waypoints = np.column_stack([xy, z])

        # Stattdessen: sofort auf Gate-Höhe gehen, damit wir nicht zu spät sind
        z         = np.full(len(xy), gate_z)
        waypoints = np.column_stack([xy, z])

        waypoints[0] = start_pos.copy()

        # we want to only ake a certain amount of waitpoints, so we recalculate more often
        # waypoints = waypoints[:20]

        # Zeitparametrisierung: gleichmäßig nach Distanz, mit Slow-down am Ende
        dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        dists = np.maximum(dists, 0.01)

        # Letzten Abschnitt verlangsamen
        t_scale = np.ones(len(dists))
        t_scale[-1] = SLOWNDOWN_SCALE

        t_knots = np.concatenate([[0.0], np.cumsum(dists * t_scale * TIME_SCALE)])
        self._t_total = t_knots[-1]

        t_elapsed = (self._tick - self._t_start_tick) / self._freq
        t_sp = min(t_elapsed, self._t_total)

        current_vel = self._spline_vel(t_sp) if self._spline else np.zeros(3)

        self._spline = CubicSpline(t_knots, waypoints, bc_type="not-a-knot")
        # self._spline     = CubicSpline(t_knots, waypoints, bc_type="natural")
        self._spline_vel = self._spline.derivative()
        self._t_start_tick = self._tick

        print(f"  exit_pt   : {np.round(exit_pt, 3)}")
        print(f"  approach  : {np.round(approach, 3)}")
        print(f"  center    : {np.round(center, 3)}")
        print(f"  waypoints :\n{np.round(waypoints, 3)}")
        print(f"  t_total   : {self._t_total:.3f}s")

    # ── Gate-Updates ──────────────────────────────────────────────────────────
    def _update_gates(self, obs: dict) -> bool:
        new_visited = obs["gates_visited"]
        rebuild = False
        for i in range(self._n_gates):
            if new_visited[i] and not self._gates_visited[i]:
                self._gate_pos[i]    = obs["gates_pos"][i]
                self._gate_quat[i]   = obs["gates_quat"][i]
                self._gates_visited[i] = True
                rebuild = True
                print(f"  [gate update] gate {i} now visited")
        return rebuild

    # ── Hauptschleife ─────────────────────────────────────────────────────────
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:

        pos = obs["pos"]
        vel = obs["vel"]
        t   = self._tick / self._freq
        dt  = 1.0 / self._freq
        current_gate = int(obs["target_gate"])

        # ── Phase 1: Takeoff ──────────────────────────────────────────────────
        if t < TAKEOFF_TIME:
            frac      = t / TAKEOFF_TIME
            target_z  = TAKEOFF_HEIGHT * frac
            des_vel_z = TAKEOFF_HEIGHT / TAKEOFF_TIME
            acc_z     = 3.0 * (target_z - pos[2]) + 1.5 * (des_vel_z - vel[2])
            return np.array([
                pos[0], pos[1], target_z,
                0.0, 0.0, des_vel_z,
                0.0, 0.0, float(np.clip(acc_z, -10, 10)),
                0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)

        # ── Phase 2: Spline initialisieren ────────────────────────────────────
        if self._spline is None:
            print(f"\n>>> Entering phase 2 at tick={self._tick}, t={t:.3f}")
            self._build_spline(pos, gate_id=current_gate, obs=obs, label="phase2_init")

        # # This works with levels 0 and 1 but for level 2 we need more replanning
        # if hasattr(self, "_current_exit") and self._current_exit is not None:
        #     dist_to_exit = np.linalg.norm(pos - self._current_exit)
        #     if dist_to_exit < 0.15:
        #         print(f"  Approaching exit point, dist_to_exit={dist_to_exit:.2f}")
        #         self._current_exit = None
        #         self._build_spline(pos, current_gate, obs, label="approach_exit")

        gate_center = self._gate_pos[current_gate]

        # we want to replan every time we detect something new 
        # but only if we are not already very close to the gate

        dist_to_gate = np.linalg.norm(pos[:2] - gate_center[:2])

        if self._obstacles_changed(obs, pos):
            if dist_to_gate > GATE_PROXIMITY_THRESHOLD:
                print(f"  Regular replan, dist_to_gate={dist_to_gate:.2f}")
                self._build_spline(pos, current_gate, obs, label="regular_replan")
            else:
                print(f"  Near gate (dist={dist_to_gate:.2f}), skipping replan to avoid late reaction")
      
        if dist_to_gate < SENSOR_RANGE:
            new_pos = obs["gates_pos"][current_gate]
            if np.linalg.norm(new_pos - self._gate_pos[current_gate]) > 0.05:
                print("  [REPLAN] gate pose refined")
                self._gate_pos[current_gate] = new_pos
                self._build_spline(pos, current_gate, obs, label="gate_update")
        # ── Spline auswerten ──────────────────────────────────────────────────
        t_elapsed = (self._tick - self._t_start_tick) / self._freq

        if t_elapsed > self._t_total + 0.05:
            # Nur replan wenn noch nicht in der Nähe des Gates/dahinter
            if dist_to_gate > GATE_PROXIMITY_THRESHOLD:
                print(f"  [timeout] replan, dist_to_gate={dist_to_gate:.2f}")
                self._build_spline(pos, current_gate, obs, label="timeout_replan")
            else:
                # Spline eingefroren lassen – Drone ist schon nah genug
                self._t_start_tick = self._tick  # Timer zurücksetzen damit kein Loop

        t_sp    = min(t_elapsed, self._t_total)
        des_pos = self._spline(t_sp)
        des_vel = self._spline_vel(t_sp)
        des_vel = np.clip(des_vel, -1.0, 1.0)

        # if self._tick % 20 == 0:
        #     future_t = min(t_sp + 0.5, self._t_total)
        #     future_pos = self._spline(future_t)

        #     print(f"[PATH DEBUG]")
        #     print(f"  current pos : {np.round(pos, 3)}")
        #     print(f"  des_pos     : {np.round(des_pos, 3)}")
        #     print(f"  next (0.5s) : {np.round(future_pos, 3)}")

        # PID
        if dist_to_gate < GATE_PROXIMITY_THRESHOLD:
            print(f"  Near gate (dist={dist_to_gate:.2f}), using more conservative PID gains")
            # Near gate: tighter proportional, moderate integral
            kp = np.array([3.0, 3.0, 2.0])   # restore XY tracking strength
            kd = np.array([3.0, 3.0, 1.5])
            ki = np.array([1.0, 1.0, 0.5])   # reduce integral near gate to avoid windup
        else:
            kp = np.array([4.0, 4.0, 3.0])
            kd = np.array([2.0, 2.0, 1.5])
            ki = np.array([2.0, 2.0, 1.0])

        scale = 3.0 / TIME_SCALE  # reference is your old stable value, 3.0 was stable

        # kp_new = kp * scale**2
        # kd_new = kd * scale**2
        # ki_new = ki * scale**2

        kp_new = kp
        kd_new = kd
        ki_new = ki

        pos_error = des_pos - pos
        vel_error = des_vel - vel

        # print("vz:", vel[2])

        self._pos_integral += pos_error * dt
        self._pos_integral  = np.clip(self._pos_integral, -0.5, 0.5)

        acc = kp_new * pos_error + kd_new * vel_error + ki_new * self._pos_integral
        # print(f"  [PID debug] pos_error={np.round(pos_error,3)} vel_error={np.round(vel_error,3)} acc={np.round(acc,3)}")
        acc[:2] = np.clip(acc[:2], -2.0, 2.0)
        acc[2]  = np.clip(acc[2], -2.0, 2.0)

        # print(f"des_z={des_pos[2]:.2f}, actual_z={pos[2]:.2f}")
        # print(f"pos_error={np.round(pos_error,3)}, vel_error={np.round(vel_error,3)}, acc={np.round(acc,3)}")

        target_pos = self._gate_pos[min(current_gate, self._n_gates - 1)]
        delta      = target_pos - pos
        des_yaw    = float(np.arctan2(delta[1], delta[0]))

        if self._tick % 20 == 0:
            print(f"  [t={t:.2f}] gate={current_gate} "
            f"pos={np.round(pos,3)} des={np.round(des_pos,3)} "
            f"err={np.round(pos-des_pos,3)}")

        return np.array([
            des_pos[0], des_pos[1], des_pos[2],
            des_vel[0], des_vel[1], des_vel[2],
            acc[0],     acc[1],     acc[2],
            des_yaw,
            0.0, 0.0, 0.0
        ], dtype=np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        if truncated:
            print("\n ===== TIMEOUT DETECTED =====")

            print(f"tick        : {self._tick}")
            print(f"position    : {np.round(obs['pos'], 3)}")
            print(f"velocity    : {np.round(obs['vel'], 3)}")

            current_gate = int(obs["target_gate"])
            print(f"target_gate : {current_gate}")

            if current_gate < self._n_gates:
                gate_pos = self._gate_pos[current_gate]
                dist = np.linalg.norm(obs["pos"] - gate_pos)

                print(f"gate_pos    : {np.round(gate_pos, 3)}")
                print(f"dist_to_gate: {dist:.3f}")

            if hasattr(self, "_current_exit") and self._current_exit is not None:
                dist_exit = np.linalg.norm(obs["pos"] - self._current_exit)
                print(f"dist_to_exit: {dist_exit:.3f}")

            print("============================\n")


        if terminated:
            print("\n💥 ===== CRASH DETECTED =====")

            print(f"tick        : {self._tick}")
            print(f"position    : {np.round(obs['pos'], 3)}")
            print(f"velocity    : {np.round(obs['vel'], 3)}")

            current_gate = int(obs["target_gate"])
            print(f"target_gate : {current_gate}")

            if current_gate < self._n_gates:
                gate_pos = self._gate_pos[current_gate]
                dist = np.linalg.norm(obs["pos"] - gate_pos)

                print(f"gate_pos    : {np.round(gate_pos, 3)}")
                print(f"dist_to_gate: {dist:.3f}")

            if hasattr(self, "_current_exit") and self._current_exit is not None:
                dist_exit = np.linalg.norm(obs["pos"] - self._current_exit)
                print(f"dist_to_exit: {dist_exit:.3f}")

            trusted_obsactles = self._filter_trusted_obstacles(obs, obs["pos"])
            print(f"trusted obstacles (in sensor range {SENSOR_RANGE}m):\n{np.round(trusted_obsactles,3)}")

            print("============================\n")

        return terminated or truncated

    def episode_callback(self):
        self._tick           = 0
        self._spline         = None
        self._spline_vel     = None
        self._t_total        = 0.0
        self._t_start_tick   = 0
        self._pos_integral   = np.zeros(3)
        self._current_goal   = None
        self._prev_target_gate = 0