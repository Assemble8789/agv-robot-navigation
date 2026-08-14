"""
Factory-floor humanoid navigation with AGV spatio-temporal avoidance.

Loads a factory map and a pre-computed AGV schedule (plan_*.json), then
plans a collision-free spatio-temporal path for the humanoid robot from
start to goal, and executes it using the RL walking policy.

Architecture:
  Layer 1 — load map + AGV plan
  Layer 2 — spatio-temporal A* (avoid static obstacles + AGV spacetime occupancy)
  Layer 3 — execute via waypoint Navigator + safety repulsion fallback

Usage:
  python navigation/navigate_factory.py maps/map_*.json maps/plan_*.json \
      --start LM000 --goal LM001
  python navigation/navigate_factory.py ... --plan-only   # print path, don't run
"""

from __future__ import annotations

import sys
import os
import json
import math
import heapq
from typing import List, Tuple, Dict, Set, Optional
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# Lazy imports — only needed for execution (not --plan-only)
_Navigator = None
_get_robot_pose = None
_EnvConfig = None
_SimpleEnv = None


def _ensure_mujoco_imports():
    """Import mujoco-dependent modules (only when executing)."""
    global _Navigator, _get_robot_pose, _EnvConfig, _SimpleEnv
    if _Navigator is None:
        from navigation.navigate import Navigator as _Nav, get_robot_pose as _grp
        from simple_env import EnvConfig as _EC, SimpleEnv as _SE
        _Navigator = _Nav
        _get_robot_pose = _grp
        _EnvConfig = _EC
        _SimpleEnv = _SE

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

HUMAN_STEP_TIME = 4     # AGV ticks (~seconds) to move 1 grid cell (≈0.25 m/s)
HUMAN_DIAG_TIME  = 6    # diagonal step cost (4 × √2 ≈ 5.7, round to 6)
WAIT_TIME = 1            # cost to wait 1 tick in place
AGV_SAFETY = 1           # extra cells around each AGV position
HUMAN_SAFETY_TIME = 0    # ±time margin (0 = A* timing handles separation)

# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 — Load map & AGV plan
# ═══════════════════════════════════════════════════════════════════════════

def load_factory_map(map_path: str) -> dict:
    """Load a factory map JSON, return width, height, obstacles, landmarks."""
    with open(map_path, "r") as f:
        data = json.load(f)
    width = data["width"]
    height = data["height"]
    obstacles = set()
    for o in data.get("obstacles", []):
        if isinstance(o, dict):
            obstacles.add((o["x"], o["y"]))
        else:
            obstacles.add((o[0], o[1]))
    landmarks = {}
    for lm in data.get("landmarks", []):
        landmarks[lm["name"]] = (lm["x"], lm["y"])
    return {
        "width": width, "height": height,
        "obstacles": obstacles,
        "landmarks": landmarks,
        "x_min": 0, "y_min": 0,  # simplified — assumes (0,0) origin
        "x_max": width, "y_max": height,
    }


def load_agv_plan(plan_path: str) -> list:
    """Load AGV schedule, return list of car trajectory dicts."""
    with open(plan_path, "r") as f:
        data = json.load(f)
    cars = []
    for car in data.get("cars", []):
        traj = [(p["t"], p["x"], p["y"]) for p in car["trajectory"]]
        cars.append({
            "car_id": car.get("car_id", car.get("id", "?")),
            "start": car.get("start_landmark", "?"),
            "goal": car.get("goal_landmark", "?"),
            "trajectory": traj,
        })
    return cars


def build_agv_reservations(car_trajs: list,
                           safety: int = AGV_SAFETY,
                           time_margin: int = HUMAN_SAFETY_TIME) -> Set[Tuple[int, int, int]]:
    """
    Build a set of (x, y, t) cells occupied by AGVs.

    Expands each AGV position by `safety` cells in x/y and `time_margin`
    in t to give the humanoid a comfortable buffer.
    """
    res = set()
    for car in car_trajs:
        for t, cx, cy in car["trajectory"]:
            for dx in range(-safety, safety + 1):
                for dy in range(-safety, safety + 1):
                    for dt in range(-time_margin, time_margin + 1):
                        res.add((cx + dx, cy + dy, t + dt))
    return res


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 — Spatio-temporal A* for the humanoid
# ═══════════════════════════════════════════════════════════════════════════

def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Octile distance (allows diagonals)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


_MOVE_DIRS = [
    (1, 0, HUMAN_STEP_TIME), (-1, 0, HUMAN_STEP_TIME),
    (0, 1, HUMAN_STEP_TIME), (0, -1, HUMAN_STEP_TIME),
    (1, 1, HUMAN_DIAG_TIME), (1, -1, HUMAN_DIAG_TIME),
    (-1, 1, HUMAN_DIAG_TIME), (-1, -1, HUMAN_DIAG_TIME),
]


def _cell_safe(x: int, y: int, t: int,
               obs_set: Set[Tuple[int, int]],
               agv_res: Set[Tuple[int, int, int]],
               x_min: int, y_min: int, x_max: int, y_max: int) -> bool:
    """True if (x, y) at time t is free of static obstacles and AGVs."""
    if not (x_min <= x < x_max and y_min <= y < y_max):
        return False
    if (x, y) in obs_set:
        return False
    if (x, y, t) in agv_res:
        return False
    return True


def inflate_obstacles(obs_set: Set[Tuple[int, int]],
                      margin: int = 1,
                      width: int = 999, height: int = 999
                      ) -> Set[Tuple[int, int]]:
    """Expand each obstacle cell by `margin` in all 8 directions."""
    inflated = set(obs_set)
    for (ox, oy) in obs_set:
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                nx, ny = ox + dx, oy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    inflated.add((nx, ny))
    return inflated


def plan_humanoid_path(start_xy: Tuple[int, int],
                       goal_xy: Tuple[int, int],
                       obs_set: Set[Tuple[int, int]],
                       agv_res: Set[Tuple[int, int, int]],
                       width: int, height: int,
                       start_time: int = 0,
                       max_time: int = 500,
                       max_iter: int = 200000) -> Optional[List[Tuple[int, int, int]]]:
    """
    Spatio-temporal A* for the humanoid.

    Returns list of (t, x, y) waypoints, or None if no path found.
    The path INCLUDES a leading (start_time, start_xy) and the goal.

    State: (x, y, t)
    Actions: 8-directional moves + wait
    """
    x_min, y_min = 0, 0
    x_max, y_max = width, height

    start = (start_xy[0], start_xy[1])
    goal = (goal_xy[0], goal_xy[1])

    # Check start / goal are feasible
    if start in obs_set:
        return None
    if goal in obs_set:
        return None
    # If start cell is occupied by AGVs, advance start_time past the window
    while (start[0], start[1], start_time) in agv_res and start_time < max_time:
        start_time += 1

    # Priority queue: (f_score, g_score, t, x, y, state_id)
    # state_id disambiguates ties when f/g are equal
    counter = 0
    open_set = []
    start_h = _heuristic(start, goal)
    heapq.heappush(open_set, (start_h, 0.0, start_time, start[0], start[1], counter))
    counter += 1

    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    g_score: Dict[Tuple[int, int, int], float] = {(start[0], start[1], start_time): 0.0}
    closed: Set[Tuple[int, int, int]] = set()

    while open_set and counter < max_iter:
        _, g, t, cx, cy, _ = heapq.heappop(open_set)
        state = (cx, cy, t)

        if state in closed:
            continue
        closed.add(state)

        # Goal check
        if (cx, cy) == goal:
            # Reconstruct path — state is (x, y, t), return [(t, x, y), ...]
            path_xy = []
            cur = state
            while cur in came_from:
                path_xy.append(cur)
                cur = came_from[cur]
            path_xy.append((start[0], start[1], start_time))
            path_xy.reverse()
            return [(t, x, y) for (x, y, t) in path_xy]

        if t >= max_time:
            continue

        # 1. Wait action
        nt = t + WAIT_TIME
        if _cell_safe(cx, cy, nt, obs_set, agv_res, x_min, y_min, x_max, y_max):
            nstate = (cx, cy, nt)
            ng = g + WAIT_TIME
            if ng < g_score.get(nstate, float("inf")):
                g_score[nstate] = ng
                came_from[nstate] = state
                heapq.heappush(open_set, (ng + _heuristic((cx, cy), goal),
                                          ng, nt, cx, cy, counter))
                counter += 1

        # 2. Move actions
        for dx, dy, cost in _MOVE_DIRS:
            nx, ny = cx + dx, cy + dy
            nt = t + cost  # arrival time

            # Check destination is safe at arrival time
            if not _cell_safe(nx, ny, nt, obs_set, agv_res,
                              x_min, y_min, x_max, y_max):
                continue

            # Quick check: also verify the cell is safe at an intermediate time
            # (prevents moving through an AGV that's passing by mid-step)
            mid_t = t + cost // 2
            if not _cell_safe(nx, ny, mid_t, obs_set, agv_res,
                              x_min, y_min, x_max, y_max):
                continue
            if not _cell_safe(cx, cy, mid_t, obs_set, agv_res,
                              x_min, y_min, x_max, y_max):
                continue

            nstate = (nx, ny, nt)
            ng = g + cost
            if ng < g_score.get(nstate, float("inf")):
                g_score[nstate] = ng
                came_from[nstate] = state
                heapq.heappush(open_set, (ng + _heuristic((nx, ny), goal),
                                          ng, nt, nx, ny, counter))
                counter += 1

    return None


def _bresenham_cells(x0, y0, x1, y1):
    """Return all grid cells on the line from (x0,y0) to (x1,y1) inclusive."""
    cells = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cx, cy = x0, y0
    while True:
        cells.append((cx, cy))
        if cx == x1 and cy == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            if cx == x1:
                break
            err += dy
            cx += sx
        if e2 <= dx:
            if cy == y1:
                break
            err += dx
            cy += sy
    return cells


# ═══════════════════════════════════════════════════════════════════════════
# Fine-grid (0.5 m) planner — avoids body collision with shelves
# ═══════════════════════════════════════════════════════════════════════════

FINE_CELL = 0.25          # metres per fine-grid cell
FINE_STEP = 2             # ticks per straight move (0.25 m / 0.25 m/s = 2s)
FINE_DIAG = 3             # ticks per diagonal  (≈ 2 * √2)
FINE_SCALE = int(1 / FINE_CELL)  # 4 — coarse cells per fine cell


def _coarse_to_fine(cx: int, cy: int) -> List[Tuple[int, int]]:
    """Map a 1m coarse cell to its fine-grid sub-cells."""
    cells = []
    fx0, fy0 = cx * FINE_SCALE, cy * FINE_SCALE
    for dx in range(FINE_SCALE):
        for dy in range(FINE_SCALE):
            cells.append((fx0 + dx, fy0 + dy))
    return cells


def _fine_to_world(fx: int, fy: int) -> Tuple[float, float]:
    """Fine-grid cell centre → world coordinates (metres)."""
    return (fx + 0.5) * FINE_CELL, (fy + 0.5) * FINE_CELL


def _world_to_fine(wx: float, wy: float) -> Tuple[int, int]:
    """World coords → nearest fine-grid cell index."""
    return int(wx / FINE_CELL), int(wy / FINE_CELL)


def plan_fine_path(start_world: Tuple[float, float],
                   goal_world: Tuple[float, float],
                   coarse_obs: Set[Tuple[int, int]],
                   agv_cars: list,
                   coarse_w: int, coarse_h: int,
                   start_time: int = 0,
                   obstacle_margin: int = 1,   # fine cells around obstacle
                   agv_safety: int = 1,        # coarse cells around AGV
                   ) -> Optional[List[Tuple[int, int, int]]]:
    """
    Plan a collision-free spacetime path on a 0.5 m fine grid.

    Steps:
      1. Map coarse obstacles → fine grid, inflate by margin.
      2. Map AGV trajectories → fine-grid spacetime reservations.
      3. Run spatio-temporal A* on fine grid.
      4. Convert waypoints back to world coordinates.

    Returns [(t, world_x, world_y), ...] or None.
    """
    fw, fh = coarse_w * FINE_SCALE, coarse_h * FINE_SCALE

    # ── Obstacles on fine grid (inflated) ──
    fine_obs = set()
    for (ox, oy) in coarse_obs:
        for (fx, fy) in _coarse_to_fine(ox, oy):
            fine_obs.add((fx, fy))
    # Inflate
    fine_obs = inflate_obstacles(fine_obs, margin=obstacle_margin,
                                 width=fw, height=fh)

    # ── AGV reservations on fine grid ──
    # Each AGV reserves its EXACT trajectory times, AND its FINAL position
    # is reserved for all future times (it's parked there).
    fine_agv = set()
    MAX_T = 500  # planning horizon
    for car in agv_cars:
        traj = car["trajectory"]
        # Exact-trajectory reservations
        for t, cx, cy in traj:
            for dx in range(-agv_safety, agv_safety + 1):
                for dy in range(-agv_safety, agv_safety + 1):
                    for (fx, fy) in _coarse_to_fine(cx + dx, cy + dy):
                        fine_agv.add((fx, fy, t))
        # Parked-forever: only block the exact cell (sphere is small, r=0.2m)
        # Runtime repulsion handles fine-grained avoidance near parked AGVs.
        t_final, cx_final, cy_final = traj[-1]
        for park_t in range(t_final + 1, MAX_T):
            for (fx, fy) in _coarse_to_fine(cx_final, cy_final):
                fine_agv.add((fx, fy, park_t))

    # ── A* on fine grid ──
    sfx, sfy = _world_to_fine(*start_world)
    gfx, gfy = _world_to_fine(*goal_world)

    # Build move dirs with fine-grid costs
    _move_dirs_fine = [
        (1, 0, FINE_STEP), (-1, 0, FINE_STEP),
        (0, 1, FINE_STEP), (0, -1, FINE_STEP),
        (1, 1, FINE_DIAG), (1, -1, FINE_DIAG),
        (-1, 1, FINE_DIAG), (-1, -1, FINE_DIAG),
    ]

    # Inline A* (same algorithm as plan_humanoid_path, different constants)
    start = (sfx, sfy)
    goal = (gfx, gfy)
    x_min, y_min = 0, 0
    x_max, y_max = fw, fh

    if start in fine_obs or goal in fine_obs:
        return None

    while (start[0], start[1], start_time) in fine_agv:
        start_time += 1

    counter = 0
    open_set = []
    start_h = _heuristic(start, goal)
    heapq.heappush(open_set, (start_h, 0.0, start_time, start[0], start[1], counter))
    counter += 1

    came_from = {}
    g_score = {(start[0], start[1], start_time): 0.0}
    closed = set()

    while open_set and counter < 500000:
        _, g, t, cx, cy, _ = heapq.heappop(open_set)
        state = (cx, cy, t)
        if state in closed:
            continue
        closed.add(state)

        if (cx, cy) == goal:
            path_xy = []
            cur = state
            while cur in came_from:
                path_xy.append(cur)
                cur = came_from[cur]
            path_xy.append((start[0], start[1], start_time))
            path_xy.reverse()
            # Convert to world coords
            return [(t, *_fine_to_world(x, y)) for (x, y, t) in path_xy]

        if t >= 500:
            continue

        # Wait
        nt = t + WAIT_TIME
        if (cx, cy, nt) not in fine_obs and (cx, cy, nt) not in fine_agv:
            nstate = (cx, cy, nt)
            ng = g + WAIT_TIME
            if ng < g_score.get(nstate, float("inf")):
                g_score[nstate] = ng
                came_from[nstate] = state
                heapq.heappush(open_set, (ng + _heuristic((cx, cy), goal),
                                          ng, nt, cx, cy, counter))
                counter += 1

        # Move
        for dx, dy, cost in _move_dirs_fine:
            nx, ny = cx + dx, cy + dy
            nt = t + cost
            if not (x_min <= nx < x_max and y_min <= ny < y_max):
                continue
            if (nx, ny) in fine_obs:
                continue
            if (nx, ny, nt) in fine_agv:
                continue
            # Bidirectional edge check
            ef = (cx, cy, nt, nx, ny)
            er = (nx, ny, nt, cx, cy)
            if ef in fine_agv or er in fine_agv:
                continue
            nstate = (nx, ny, nt)
            ng = g + cost
            if ng < g_score.get(nstate, float("inf")):
                g_score[nstate] = ng
                came_from[nstate] = state
                heapq.heappush(open_set, (ng + _heuristic((nx, ny), goal),
                                          ng, nt, nx, ny, counter))
                counter += 1

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Pure spatial A* — no time dimension.  Robot walks at its own pace.
# ═══════════════════════════════════════════════════════════════════════════

def plan_spatial_path(start_world: Tuple[float, float],
                      goal_world: Tuple[float, float],
                      coarse_obs: Set[Tuple[int, int]],
                      agv_cars: list,
                      coarse_w: int, coarse_h: int,
                      obstacle_margin: int = 1,    # fine cells around obstacles
                      agv_block: bool = True,      # treat parked AGV cells as static
                      ) -> Optional[List[Tuple[float, float]]]:
    """
    Plan a collision-free path WITHOUT a time dimension.

    - Obstacles inflated on 0.5 m fine grid.
    - AGV FINAL positions treated as static obstacles (they park there).
      Moving AGVs are ignored — they're 4× faster than the robot, so by
      the time the robot arrives they're long gone.
    - Standard A* on the fine grid.

    Returns [(world_x, world_y), ...] or None.
    """
    fw, fh = coarse_w * FINE_SCALE, coarse_h * FINE_SCALE

    # ── Obstacles on fine grid (inflated) ──
    fine_obs = set()
    for (ox, oy) in coarse_obs:
        for (fx, fy) in _coarse_to_fine(ox, oy):
            fine_obs.add((fx, fy))
    fine_obs = inflate_obstacles(fine_obs, margin=obstacle_margin,
                                 width=fw, height=fh)

    # ── AGV dwell/park points → static obstacles ──
    # Only block cells where an AGV STOPS (dwells: consecutive identical
    # positions) or its final endpoint.  Moving cells are ignored — the
    # runtime dynamic avoidance handles those.  This keeps the path short
    # while preventing the robot from walking through parked AGVs.
    if agv_block:
        for car in agv_cars:
            traj = car["trajectory"]
            stop_cells = set()
            prev = None
            for _, tx, ty in traj:
                if prev is not None and (tx, ty) == prev:
                    stop_cells.add((tx, ty))  # dwell
                prev = (tx, ty)
            # Final endpoint is always a park position
            stop_cells.add((traj[-1][1], traj[-1][2]))
            for (tx, ty) in stop_cells:
                for (fx, fy) in _coarse_to_fine(tx, ty):
                    fine_obs.add((fx, fy))

    # ── Standard A* (spatial only) ──
    start = _world_to_fine(*start_world)
    goal = _world_to_fine(*goal_world)

    # If start/goal coincide with a parked AGV cell, unlock just those cells
    # (robot starts there / AGV parked where we're headed — still reachable)
    for cell in (start, goal):
        if cell in fine_obs:
            fine_obs.discard(cell)
            # Also unlock neighbours so the cell is actually reachable
            fx, fy = cell
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= fx + dx < fw and 0 <= fy + dy < fh:
                        fine_obs.discard((fx + dx, fy + dy))

    if start in fine_obs or goal in fine_obs:
        return None

    counter = 0
    open_set = []
    heapq.heappush(open_set, (_heuristic(start, goal), 0.0, start[0], start[1], counter))
    counter += 1

    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    _dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while open_set and counter < 500000:
        _, g, cx, cy, _ = heapq.heappop(open_set)
        state = (cx, cy)
        if state in closed:
            continue
        closed.add(state)

        if state == goal:
            path_xy = []
            cur = state
            while cur in came_from:
                path_xy.append(cur)
                cur = came_from[cur]
            path_xy.append(start)
            path_xy.reverse()
            return [_fine_to_world(fx, fy) for (fx, fy) in path_xy]

        for dx, dy in _dirs:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < fw and 0 <= ny < fh):
                continue
            if (nx, ny) in fine_obs:
                continue
            cost = 1.0 if (dx == 0 or dy == 0) else 1.414
            nstate = (nx, ny)
            ng = g + cost
            if ng < g_score.get(nstate, float("inf")):
                g_score[nstate] = ng
                came_from[nstate] = state
                heapq.heappush(open_set, (ng + _heuristic(nstate, goal),
                                          ng, nx, ny, counter))
                counter += 1

    return None


def replan_agv_car(current_xy: Tuple[float, float],
                   goal_xy: Tuple[float, float],
                   coarse_obs: Set[Tuple[int, int]],
                   dyn_cells: Set[Tuple[int, int]],
                   width: int, height: int,
                   obstacle_margin: int = 2) -> Optional[List[Tuple[float, float]]]:
    """Replan a SINGLE AGV from current_xy to goal_xy, treating dyn_cells
    (the robot's current cell + other AGVs' current cells) as temporary
    obstacles.

    Used by the runtime AGV-yield variant: after an AGV stops to let the
    robot pass, this recomputes its remaining path so it continues correctly.
    Returns a spatial [(x, y), ...] path on the 0.25 m fine grid, or None.
    """
    obs = set(coarse_obs) | set(dyn_cells)
    return plan_spatial_path((float(current_xy[0]), float(current_xy[1])),
                             (float(goal_xy[0]), float(goal_xy[1])),
                             obs, [], width, height,
                             obstacle_margin=obstacle_margin, agv_block=False)


# ═══════════════════════════════════════════════════════════════════════════
# Closed-loop collision detection & correction
# ═══════════════════════════════════════════════════════════════════════════

# Actual MuJoCo collision geometries (must match bxi_elf3_scene_factory.xml)
_SHELF_HALF = (1.5, 0.35)     # x, y half-sizes (metres)
_EQUIP_HALF = (1.0, 0.5)      # x, y for equipment block
_AGV_RADIUS = 0.25             # AGV sphere + safety margin
_ROBOT_RADIUS = 0.35           # humanoid body radius
_OBSTACLE_GROUPS = [           # (cx, cy, hx, hy) — centre + half-sizes
    (9.0, 8.0, *_SHELF_HALF),  # obs_0: right shelves y=8
    (9.0, 5.0, *_SHELF_HALF),  # obs_1: right shelves y=5
    (6.5, 1.0, *_EQUIP_HALF),  # obs_2: equipment bottom
    (4.0, 5.0, *_SHELF_HALF),  # obs_3: left shelves y=5
    (4.0, 8.0, *_SHELF_HALF),  # obs_4: left shelves y=8
    (6.5, 2.0, *_EQUIP_HALF),  # obs_5: equipment bottom
]
# Landmark zone — AGVs park here, extra margin
_LANDMARK_ZONES = {
    (9, 7): 0.5,  # LM002 — AGV parking
    (4, 4): 0.5,  # LM000 — AGV parking
    (9, 4): 0.5,  # LM001 — AGV parking
    (13, 8): 0.5, # LM004 — AGV parking
    (4, 7): 0.5,  # LM003
}


def _dist_to_obstacle(px: float, py: float) -> Tuple[float, float, float]:
    """
    Minimum distance from (px,py) to any obstacle surface.
    Returns (min_dist, push_dx, push_dy) — push direction away from nearest obstacle.
    """
    best_dist = 999.0
    best_dx, best_dy = 0.0, 0.0

    # Static obstacles
    for (cx, cy, hx, hy) in _OBSTACLE_GROUPS:
        dx = px - cx
        dy = py - cy
        # Signed distance to box surface (positive = outside)
        dx_signed = abs(dx) - hx - _ROBOT_RADIUS
        dy_signed = abs(dy) - hy - _ROBOT_RADIUS
        if dx_signed < 0 and dy_signed < 0:
            # Inside collision zone — push out along shortest axis
            pen_x = -dx_signed
            pen_y = -dy_signed
            if pen_x < pen_y:
                push = np.sign(dx) if dx != 0 else 1.0
                d = -dx_signed
                best_dist = min(best_dist, d)
                best_dx, best_dy = push * pen_x, 0.0
            else:
                push = np.sign(dy) if dy != 0 else 1.0
                d = -dy_signed
                best_dist = min(best_dist, d)
                best_dx, best_dy = 0.0, push * pen_y
        else:
            d = max(dx_signed, dy_signed)
            if d < best_dist:
                best_dist = d
                best_dx, best_dy = dx, dy

    # Landmark parking zones
    for (lx, ly), margin in _LANDMARK_ZONES.items():
        dx, dy = px - lx, py - ly
        d = np.hypot(dx, dy)
        safe_d = margin + _AGV_RADIUS + _ROBOT_RADIUS
        if d < safe_d and d > 0.01:
            penetration = safe_d - d
            if penetration > -best_dist:
                best_dist = -penetration
                best_dx, best_dy = (dx / d) * penetration, (dy / d) * penetration

    return best_dist, best_dx, best_dy


def _push_out_of_obstacles(px: float, py: float,
                           clearance: float = 0.05) -> Tuple[float, float]:
    """Return (dx, dy) that moves (px,py) out of any obstacle's inflated box.

    The inflated box is the real box half-size plus _ROBOT_RADIUS plus a
    clearance.  If the point is inside it (the Catmull-Rom re-smooth in
    correct_path dips waypoints into shelves), push it out along the SHALLOWER
    axis — the smaller correction, avoiding a big detour.
    """
    for (cx, cy, hx, hy) in _OBSTACLE_GROUPS:
        dx, dy = px - cx, py - cy
        ex = abs(dx) - (hx + _ROBOT_RADIUS + clearance)
        ey = abs(dy) - (hy + _ROBOT_RADIUS + clearance)
        if ex < 0 and ey < 0:   # inside the inflated box
            sx = 1.0 if dx >= 0 else -1.0
            sy = 1.0 if dy >= 0 else -1.0
            if ex < ey:          # x penetration deeper → push out along y
                return 0.0, sy * (-ey)
            else:                # y penetration deeper (or equal) → push x
                return sx * (-ex), 0.0
    return 0.0, 0.0


def correct_path(path: List[Tuple[int, int, int]],
                 max_iter: int = 10) -> List[Tuple[int, int, int]]:
    """
    Closed-loop correction: push waypoints away from collision zones,
    re-smooth, repeat until clean.  Returns corrected path.
    """
    if len(path) < 2:
        return path

    wp_xy = np.array([(p[1], p[2]) for p in path], dtype=float)
    times = np.array([p[0] for p in path], dtype=float)

    for iteration in range(max_iter):
        n_fixed = 0
        for i in range(len(wp_xy)):
            dx, dy = _push_out_of_obstacles(wp_xy[i, 0], wp_xy[i, 1])
            if dx != 0.0 or dy != 0.0:
                wp_xy[i, 0] += dx
                wp_xy[i, 1] += dy
                n_fixed += 1
        if n_fixed == 0:
            break

    # Re-smooth the corrected waypoints with Catmull-Rom
    if len(wp_xy) >= 4:
        pts = wp_xy.copy()
        pa = np.vstack([pts[0]*2-pts[1], pts, pts[-1]*2-pts[-2]])

        smoothed = []
        for i in range(1, len(pa) - 2):
            p0, p1, p2, p3 = pa[i-1], pa[i], pa[i+1], pa[i+2]
            for alpha in np.linspace(0, 1, 3)[:3]:
                a2, a3 = alpha*alpha, alpha*alpha*alpha
                pt = 0.5 * ((2*p1) + (-p0+p2)*alpha +
                            (2*p0-5*p1+4*p2-p3)*a2 +
                            (-p0+3*p1-3*p2+p3)*a3)
                smoothed.append(pt)

        # Post-smooth safety: Catmull-Rom interpolation can dip INTO an
        # obstacle box when two neighbouring corrected waypoints straddle it
        # (this previously put a waypoint inside obs_0 at (9.9,7.9), wedging
        # the robot against the shelf).  Push any smoothed point out again.
        for _ in range(8):
            moved = False
            for j in range(len(smoothed)):
                dx, dy = _push_out_of_obstacles(smoothed[j][0], smoothed[j][1])
                if dx != 0.0 or dy != 0.0:
                    smoothed[j][0] += dx
                    smoothed[j][1] += dy
                    moved = True
            if not moved:
                break

        # Interpolate times to match new point count
        new_n = len(smoothed)
        if new_n > 2:
            new_times = np.linspace(times[0], times[-1], new_n)
            result = [(int(round(new_times[j])), float(smoothed[j][0]), float(smoothed[j][1]))
                      for j in range(new_n)]
            return result

    return path


def simplify_path(path: List[Tuple[int, int, int]],
                  obs_set: Set[Tuple[int, int]],
                  agv_res: Set[Tuple[int, int, int]],
                  width: int, height: int) -> List[Tuple[int, int, int]]:
    """Drop intermediate waypoints only if a direct jump is fully clear."""
    if len(path) <= 2:
        return path

    result = [path[0]]
    p_idx = 0
    while p_idx < len(path) - 1:
        pt, px, py = path[p_idx]
        # Try to jump as far as possible
        best_next = p_idx + 1
        for j in range(len(path) - 1, p_idx, -1):
            nt, nx, ny = path[j]
            # Check every cell along Bresenham line
            cells = _bresenham_cells(px, py, nx, ny)
            safe = True
            for (cx, cy) in cells:
                if ((cx, cy) in obs_set or
                    not (0 <= cx < width and 0 <= cy < height)):
                    safe = False
                    break
            if safe:
                # Also spot-check a midway time for AGV reservations
                mid_t = (pt + nt) // 2
                mid_cell = cells[len(cells) // 2]
                if (mid_cell[0], mid_cell[1], mid_t) in agv_res:
                    safe = False
            if safe:
                best_next = j
                break
        result.append(path[best_next])
        p_idx = best_next
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3 — Execution: SpacetimeNavigator
# ═══════════════════════════════════════════════════════════════════════════

class SpacetimeNavigator:
    """
    Navigates a spatio-temporal path.  At each control step:
      1. Find the waypoint whose scheduled time is closest to `now`
      2. Feed that waypoint to a standard Navigator for velocity commands
      3. If ahead of schedule — wait (cmd=0)
      4. If behind — go at max speed
    """

    def __init__(self, spacetime_path: List[Tuple[int, int, int]],
                 agv_trajs: Optional[list] = None,
                 fwd_speed: float = 0.7,
                 turn_speed: float = 1.0,
                 **kwargs):
        # spacetime_path: [(t, x, y), ...]
        self.path = spacetime_path
        self._path_idx = 0
        self._nav_kwargs = dict(fwd_speed=fwd_speed, turn_speed=turn_speed, **kwargs)
        self._last_wp_time = spacetime_path[0][0] if spacetime_path else 0
        # AGV data for safety repulsion
        self._agv_trajs = agv_trajs or []
        # Build fast lookup: {t: set of (x,y)}
        self._agv_map: Dict[int, Set[Tuple[int, int]]] = {}
        if agv_trajs:
            for car in agv_trajs:
                for t, cx, cy in car["trajectory"]:
                    self._agv_map.setdefault(t, set()).add((cx, cy))

        if spacetime_path:
            _, wx, wy = spacetime_path[0]
            _ensure_mujoco_imports()
            self._nav = _Navigator(wx, wy, **self._nav_kwargs)

    @property
    def current_waypoint(self) -> Optional[Tuple[int, int, int]]:
        if self._path_idx < len(self.path):
            return self.path[self._path_idx]
        return None

    @property
    def arrived(self) -> bool:
        return self._path_idx >= len(self.path)

    @property
    def heading_error(self) -> float:
        return getattr(self._nav, "heading_error", 0.0)

    @property
    def distance(self) -> float:
        return getattr(self._nav, "distance", float("inf"))

    def _agv_safety_repulsion(self, x: float, y: float, yaw: float,
                              now_t: float, cmd: np.ndarray) -> np.ndarray:
        """Soft repulsion from AGVs at current time (safety fallback)."""
        # Snap now_t to nearest integer for AGV lookup
        t_int = int(round(now_t))
        nearby = self._agv_map.get(t_int, set())
        if not nearby:
            return cmd

        rep = np.zeros(2, dtype=np.float32)
        for (ax, ay) in nearby:
            dx = x - ax
            dy = y - ay
            dist = math.hypot(dx, dy)
            if dist < 0.001:
                dist = 0.001
            if dist < AGV_SAFETY + 0.5:
                force = (AGV_SAFETY + 0.5 - dist) / (AGV_SAFETY + 0.5)
                rep[0] += force * dx / dist
                rep[1] += force * dy / dist

        if np.all(rep == 0):
            return cmd

        mag = math.hypot(rep[0], rep[1])
        if mag > 2.0:
            rep *= 2.0 / mag

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        fy = -sin_y * rep[0] + cos_y * rep[1]

        result = cmd.copy()
        result[1] += fy * 0.5
        result[0] *= max(0.3, 1.0 - abs(fy) * 0.5)
        result[1] = np.clip(result[1], -0.5, 0.5)
        result[0] = np.clip(result[0], 0.0, 0.7)
        return result

    def update(self, x: float, y: float, yaw: float,
               sim_time: float = 0.0) -> np.ndarray:
        """Get velocity command, advancing waypoints by spatial proximity."""
        if self.arrived:
            return np.zeros(3, dtype=np.float32)

        # Advance: if we're close enough to current waypoint, move to next
        wp = self.path[self._path_idx]
        _, wx, wy = wp
        dist_to_wp = math.hypot(x - wx, y - wy)
        if dist_to_wp < 0.35:
            self._path_idx += 1
            if self.arrived:
                return np.zeros(3, dtype=np.float32)
            wp = self.path[self._path_idx]
            _, wx, wy = wp
            self._nav = _Navigator(wx, wy, **self._nav_kwargs)

        # Schedule-awareness: if we're way ahead of schedule, wait
        wp_time = wp[0]
        time_ahead = wp_time - sim_time
        if time_ahead > HUMAN_STEP_TIME * 1.5:
            # We're too early — slow down / wait
            return np.zeros(3, dtype=np.float32)

        cmd = self._nav.update(x, y, yaw)
        cmd = self._agv_safety_repulsion(x, y, yaw, sim_time, cmd)
        return cmd


# ═══════════════════════════════════════════════════════════════════════════
# Spatial Navigator — pure waypoint follower, no time dimension
# ═══════════════════════════════════════════════════════════════════════════

class SpatialNavigator:
    """
    Walks a list of spatial waypoints [(x, y), ...] at the robot's own pace.
    Uses the standard Navigator for point-to-point control.
    """

    def __init__(self, waypoints: List[Tuple[float, float]],
                 fwd_speed: float = 0.7,
                 turn_speed: float = 1.0,
                 arrival: float = 0.35,
                 **kwargs):
        self.waypoints = waypoints
        self._idx = 0
        self._nav_kwargs = dict(fwd_speed=fwd_speed, turn_speed=turn_speed, **kwargs)
        self._arrival = arrival
        self._ensure_nav()

    def _ensure_nav(self):
        _ensure_mujoco_imports()
        if self._idx < len(self.waypoints):
            wx, wy = self.waypoints[self._idx]
            self._nav = _Navigator(wx, wy, **self._nav_kwargs)

    @property
    def current_waypoint(self) -> Optional[Tuple[float, float]]:
        if self._idx < len(self.waypoints):
            return self.waypoints[self._idx]
        return None

    @property
    def arrived(self) -> bool:
        return self._idx >= len(self.waypoints)

    @property
    def heading_error(self) -> float:
        return getattr(self._nav, "heading_error", 0.0)

    @property
    def distance(self) -> float:
        return getattr(self._nav, "distance", float("inf"))

    def update(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Velocity command toward current waypoint."""
        if self.arrived:
            return np.zeros(3, dtype=np.float32)

        wx, wy = self.waypoints[self._idx]
        dist = math.hypot(x - wx, y - wy)

        if dist < self._arrival and self._idx < len(self.waypoints) - 1:
            self._idx += 1
            wx, wy = self.waypoints[self._idx]
            print(f"  WP {self._idx}/{len(self.waypoints)}: ({wx:.2f},{wy:.2f})")
            self._ensure_nav()

        return self._nav.update(x, y, yaw)


# ═══════════════════════════════════════════════════════════════════════════
# Execution helpers (lazy mujoco imports)
# ═══════════════════════════════════════════════════════════════════════════

ELF3_SCENE = os.path.join(_PROJECT_ROOT, "model", "bxi_elf3", "bxi_elf3_scene.xml")


def _make_factory_env():
    """Create a SimpleEnv pointing to the standard elf3 scene."""
    _ensure_mujoco_imports()
    # Use a minimal config object
    class _Cfg:
        model_xml = ELF3_SCENE
        policy = os.path.join(_PROJECT_ROOT, "model", "bxi_elf3", "model_normal.onnx")
        simulation_dt = 0.005
        control_decimation = 4
        num_actions = 29
        num_obs = 96
        actor_obs_history_length = 1
        kp = _EnvConfig.kp
        kd = _EnvConfig.kd
        action_scale = _EnvConfig.action_scale
    return _SimpleEnv(_Cfg())


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def print_path(path: List[Tuple[int, int, int]],
               landmarks: Optional[dict] = None):
    """Pretty-print a spatio-temporal path."""
    inv_lm = {}
    if landmarks:
        inv_lm = {v: k for k, v in landmarks.items()}

    print(f"\n  Spatio-temporal path ({len(path)} points):")
    print(f"  {'Step':>4}  {'Time':>5}  {'Pos':>10}")
    print(f"  {'─'*4}  {'─'*5}  {'─'*10}")
    for i, (t, x, y) in enumerate(path):
        label = inv_lm.get((x, y), "")
        pos_str = f"({x:>3}, {y:>3})"
        if label:
            pos_str += f"  {label}"
        print(f"  {i:>4}  {t:>5}  {pos_str}")


def plot_path_ascii(path: List[Tuple[int, int, int]],
                    width: int, height: int,
                    obstacles: Set[Tuple[int, int]],
                    agv_trajs: Optional[list] = None):
    """Print an ASCII map of the path."""
    grid = [[" ." for _ in range(width)] for _ in range(height)]
    for (ox, oy) in obstacles:
        if 0 <= oy < height and 0 <= ox < width:
            grid[oy][ox] = "██"

    # Mark AGV trajectories
    if agv_trajs:
        for car in agv_trajs:
            for t, cx, cy in car["trajectory"]:
                if 0 <= cy < height and 0 <= cx < width:
                    if grid[cy][cx] == " .":
                        grid[cy][cx] = "··"

    # Mark humanoid path
    path_set = {(x, y) for _, x, y in path}
    for i, (t, x, y) in enumerate(path):
        if 0 <= y < height and 0 <= x < width:
            if i == 0:
                grid[y][x] = " S"
            elif i == len(path) - 1:
                grid[y][x] = " G"
            else:
                c = "·" + "0123456789abcdef"[min(i % 16, 15)]
                grid[y][x] = c

    print(f"\n  Map ({width}×{height})  S=start G=goal  █=obstacle  ·=AGV path  *=humanoid path")
    print(f"  {'─' * (width * 2 + 3)}")
    for yi in range(height - 1, -1, -1):
        row = " ".join(grid[yi])
        print(f"  {yi:>2}|{row}|")
    print(f"  {'─' * (width * 2 + 3)}")
    x_labels = " ".join(f"{xi//10}" if xi % 10 == 0 else " " for xi in range(width))
    print(f"     {x_labels}")
    x_labels2 = " ".join(str(xi % 10) for xi in range(width))
    print(f"     {x_labels2}")


# ═══════════════════════════════════════════════════════════════════════════
# Matplotlib visualisation — factory map + AGV paths + humanoid plan
# ═══════════════════════════════════════════════════════════════════════════

AGV_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


def visualize_factory_plan(factory: dict,
                           agv_cars: list,
                           humanoid_path: List[Tuple[int, int, int]],
                           start_lm: str = "",
                           goal_lm: str = "",
                           title: str = "Factory Navigation Plan",
                           block: bool = True):
    """Show a matplotlib plot of the factory map with the planned path."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    w, h = factory["width"], factory["height"]
    obs = factory["obstacles"]
    landmarks = factory["landmarks"]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-1, w)
    ax.set_ylim(-1, h)
    ax.set_aspect("equal")
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("x (grid cells)")
    ax.set_ylabel("y (grid cells)")

    # Obstacles
    if obs:
        ox, oy = zip(*obs)
        ax.scatter(ox, oy, marker="s", s=200, c="#555555",
                   edgecolors="#222", linewidths=0.5, zorder=3, label="Obstacles")

    # Landmarks
    for name, (lx, ly) in landmarks.items():
        marker = "D"
        color = "#0066cc"
        size = 80
        if name == start_lm:
            color, size, marker = "#27ae60", 150, "o"
        elif name == goal_lm:
            color, size, marker = "#e74c3c", 150, "s"
        ax.scatter(lx, ly, marker=marker, s=size, c=color,
                   edgecolors="black", linewidths=0.8, zorder=10)
        ax.annotate(name, (lx, ly + 0.25), fontsize=7, ha="center",
                   color=color, fontweight="bold")

    # AGV trajectories — with side offset when spatial paths overlap
    legend_handles = []
    # Detect: do any two cars share the exact same spatial path?
    car_paths_xy = []
    for car in agv_cars:
        traj = car["trajectory"]
        if traj:
            car_paths_xy.append(tuple((p[1], p[2]) for p in traj))
        else:
            car_paths_xy.append(())
    # Count occurrences of each unique spatial path
    path_counts = {}
    for p in car_paths_xy:
        if p:
            path_counts[p] = path_counts.get(p, 0) + 1

    path_offset_idx = {}
    for i, car in enumerate(agv_cars):
        traj = car["trajectory"]
        if not traj:
            continue
        color = AGV_COLORS[i % len(AGV_COLORS)]
        pts_xy = car_paths_xy[i]
        # If multiple cars share this path, offset them sideways
        n_shared = path_counts.get(pts_xy, 1)
        if n_shared > 1:
            idx_in_group = path_offset_idx.get(pts_xy, 0)
            path_offset_idx[pts_xy] = idx_in_group + 1
            offset = (idx_in_group - (n_shared - 1) / 2) * 0.12
        else:
            offset = 0.0

        pts = np.array([(p[1], p[2] + offset) for p in traj])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=2,
                alpha=0.7, zorder=4)
        ax.scatter(pts[0, 0], pts[0, 1], marker="o", s=60, c=color,
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], marker="X", s=80, c=color,
                   edgecolors="black", linewidths=0.5, zorder=5)
        # Direction arrows
        for j in range(len(pts) - 1):
            mx, my = (pts[j, 0] + pts[j + 1, 0]) / 2, (pts[j, 1] + pts[j + 1, 1]) / 2
            dx, dy = pts[j + 1, 0] - pts[j, 0], pts[j + 1, 1] - pts[j, 1]
            ax.arrow(mx - dx * 0.15, my - dy * 0.15, dx * 0.3, dy * 0.3,
                     head_width=0.15, head_length=0.2, fc=color, ec=color,
                     alpha=0.7, zorder=6)
        # Time range label
        t_start = traj[0][0]
        t_end = traj[-1][0]
        legend_handles.append(mpatches.Patch(
            color=color, alpha=0.6,
            label=f"AGV {car['car_id']} (t={t_start}→t={t_end}, {car['start']}→{car['goal']})"))

    # Humanoid path
    if humanoid_path:
        hp = np.array([(p[1], p[2]) for p in humanoid_path])
        ax.plot(hp[:, 0], hp[:, 1], "-", color="#e67e22", linewidth=3.5,
                alpha=0.9, zorder=7, label="Humanoid path")
        # Waypoints
        for i, (t, wx, wy) in enumerate(humanoid_path):
            if i == 0:
                ax.scatter(wx, wy, marker="o", s=140, c="#27ae60",
                           edgecolors="black", linewidths=1.5, zorder=9)
            elif i == len(humanoid_path) - 1:
                ax.scatter(wx, wy, marker="s", s=140, c="#e74c3c",
                           edgecolors="black", linewidths=1.5, zorder=9)
            else:
                ax.scatter(wx, wy, marker="D", s=60, c="#e67e22",
                           edgecolors="black", linewidths=0.8, zorder=8)
            ax.annotate(f"t={t}", (wx + 0.12, wy - 0.35), fontsize=6,
                       color="#c0392b", fontstyle="italic")
        legend_handles.append(mpatches.Patch(color="#e67e22",
                              label=f"Humanoid ({len(humanoid_path)} waypoints)"))

    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.9)

    plt.tight_layout()
    if block:
        plt.show()
    else:
        # Non-blocking — keeps window open alongside MuJoCo viewer
        try:
            plt.ion()
            plt.show(block=False)
            plt.pause(0.5)
        except Exception:
            pass


def animate_factory_plan(factory: dict,
                         agv_cars: list,
                         humanoid_path: List[Tuple[int, int, int]],
                         interval: int = 150,
                         title: str = "Factory Navigation"):
    """
    Animated timeline playback — like agv_visualizer, but with AGVs + humanoid.

    Shows Play/Pause + Step buttons, a time counter, and all entities
    moving along their trajectories in real-time.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button

    w, h = factory["width"], factory["height"]
    obs = factory["obstacles"]
    landmarks = factory["landmarks"]

    # Build time-indexed positions for fast lookup
    max_t = 0
    time_map: dict = {}  # t -> [(type, id, x, y, color), ...]

    for car in agv_cars:
        color = AGV_COLORS[car["car_id"] % len(AGV_COLORS)]
        for p in car["trajectory"]:
            # p can be dict {"t":...,"x":...,"y":...} or tuple (t,x,y)
            if isinstance(p, dict):
                t, px, py = p["t"], p["x"], p["y"]
            else:
                t, px, py = p
            max_t = max(max_t, t)
            time_map.setdefault(t, []).append(
                ("agv", car["car_id"], px, py, color))

    humanoid_color = "#e74c3c"
    for t, x, y in humanoid_path:
        max_t = max(max_t, t)
        time_map.setdefault(t, []).append(
            ("humanoid", "H", x, y, humanoid_color))

    # Fill humanoid at every tick: stays at current cell until next waypoint
    if len(humanoid_path) >= 2:
        for i in range(len(humanoid_path) - 1):
            t0, x0, y0 = humanoid_path[i]
            t1, x1, y1 = humanoid_path[i + 1]
            # Humanoid is at (x0,y0) for ticks t0 .. t1-1, then at (x1,y1) at t1
            for tt in range(t0 + 1, t1):
                if tt not in time_map or not any(
                        e[0] == "humanoid" for e in time_map[tt]):
                    time_map.setdefault(tt, []).append(
                        ("humanoid", "H", x0, y0, humanoid_color))

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(13, 8))
    plt.subplots_adjust(bottom=0.15)

    ax.set_xlim(-1, w)
    ax.set_ylim(-1, h)
    ax.set_aspect("equal")
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("x (grid cells)", fontsize=10)
    ax.set_ylabel("y (grid cells)", fontsize=10)

    # Static decorations
    if obs:
        ox, oy = zip(*obs)
        ax.scatter(ox, oy, marker="s", s=200, c="#555555",
                   edgecolors="#222", linewidths=0.5, zorder=2)

    for name, (lx, ly) in landmarks.items():
        ax.scatter(lx, ly, marker="o", s=40, c="#0066cc", alpha=0.5, zorder=3)
        ax.annotate(name, (lx + 0.2, ly + 0.2), fontsize=6, color="#0066cc")

    # Draw planned paths (faded background lines)
    # AGV paths
    for car in agv_cars:
        color = AGV_COLORS[car["car_id"] % len(AGV_COLORS)]
        traj = car["trajectory"]
        pts = np.array([(p[1], p[2]) if not isinstance(p, dict)
                        else (p["x"], p["y"]) for p in traj])
        ax.plot(pts[:, 0], pts[:, 1], "--", color=color, linewidth=1,
                alpha=0.25, zorder=4)
    # Humanoid path
    hp = np.array([(p[1], p[2]) for p in humanoid_path])
    ax.plot(hp[:, 0], hp[:, 1], "--", color=humanoid_color, linewidth=1.5,
            alpha=0.3, zorder=4)

    # Start/goal markers for humanoid
    ax.scatter(hp[0, 0], hp[0, 1], marker="o", s=120,
               facecolors="none", edgecolors=humanoid_color,
               linewidths=1.5, zorder=6)
    ax.scatter(hp[-1, 0], hp[-1, 1], marker="X", s=120,
               c=humanoid_color, edgecolors="black", linewidths=0.8, zorder=6)

    # Dynamic artists — use lists so mutable closures work
    agv_artists: dict = {}       # car_id -> (scatter, label)
    humanoid_scat = None          # scatter artist
    humanoid_label = None         # text label

    # --- Animation state ---
    class Ctrl:
        frame = 0
        playing = True
        max_frames = max_t

    ctrl = Ctrl()

    def update_frame():
        """Render all entities at ctrl.frame."""
        nonlocal humanoid_scat, humanoid_label
        ax.set_title(f"{title}  |  Time: {ctrl.frame}/{ctrl.max_frames}",
                     fontsize=12, fontweight="bold", color="#333")
        ax.set_xlabel(f"x (grid cells)    Tick: {ctrl.frame}", fontsize=10)

        entries = time_map.get(ctrl.frame, [])
        seen_agv = set()
        hx, hy = None, None

        for entry in entries:
            etype, eid, ex, ey, color = entry
            if etype == "agv":
                seen_agv.add(eid)
                # Simple arrow character for direction
                prev_t = ctrl.frame - 1
                marker = ">"
                if prev_t in time_map:
                    for pe in time_map[prev_t]:
                        if pe[0] == "agv" and pe[1] == eid:
                            dx, dy = ex - pe[2], ey - pe[3]
                            if dx > 0: marker = ">"
                            elif dx < 0: marker = "<"
                            elif dy > 0: marker = "^"
                            elif dy < 0: marker = "v"
                            break

                if eid not in agv_artists:
                    s = ax.text(ex, ey, marker, fontsize=18, color=color,
                                ha="center", va="center", fontweight="bold",
                                zorder=20)
                    l = ax.text(ex, ey - 0.5, f"AGV{eid}", fontsize=7,
                                ha="center", color="#333", zorder=21)
                    agv_artists[eid] = (s, l, marker)
                else:
                    s, l, old_marker = agv_artists[eid]
                    if old_marker != marker:
                        s.set_text(marker)
                        agv_artists[eid] = (s, l, marker)
                    s.set_position((ex, ey))
                    l.set_position((ex, ey - 0.5))

            elif etype == "humanoid":
                hx, hy = ex, ey

        # Remove AGVs that vanished
        for eid in list(agv_artists):
            if eid not in seen_agv:
                s, l, _ = agv_artists.pop(eid)
                s.remove()
                l.remove()

        # Humanoid — square marker + "H" label
        if hx is not None:
            if humanoid_scat is None:
                humanoid_scat = ax.scatter(
                    [hx], [hy], s=500, c="#e74c3c", marker="s",
                    edgecolors="black", linewidths=2.5, zorder=30)
                humanoid_label = ax.text(
                    hx, hy, "H", fontsize=10, ha="center", va="center",
                    color="white", fontweight="bold", zorder=31)
            else:
                humanoid_scat.set_offsets([[hx, hy]])
                humanoid_label.set_position((hx, hy))

    def next_frame():
        if ctrl.frame < ctrl.max_frames:
            ctrl.frame += 1
        else:
            ctrl.frame = 0
        update_frame()

    # --- Buttons ---
    ax_play = plt.axes([0.35, 0.03, 0.10, 0.06])
    ax_step = plt.axes([0.48, 0.03, 0.10, 0.06])
    ax_reset = plt.axes([0.61, 0.03, 0.10, 0.06])

    btn_play = Button(ax_play, "▶/⏸")
    btn_step = Button(ax_step, "Step ▸")
    btn_reset = Button(ax_reset, "⟳ Reset")

    def toggle(_):
        ctrl.playing = not ctrl.playing

    def step(_):
        ctrl.playing = False
        next_frame()
        fig.canvas.draw_idle()

    def reset(_):
        ctrl.playing = False
        ctrl.frame = 0
        update_frame()
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle)
    btn_step.on_clicked(step)
    btn_reset.on_clicked(reset)

    def anim_step(_i):
        if ctrl.playing:
            next_frame()
        return []

    update_frame()
    ani = FuncAnimation(fig, anim_step, interval=interval,
                        cache_frame_data=False, blit=False)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Main entry points
# ═══════════════════════════════════════════════════════════════════════════

def run_factory_navigation(map_path: str, plan_path: str,
                           start_lm: str = "LM006",
                           goal_lm: str = "LM002",
                           via_lm: str = "",
                           plan_only: bool = False,
                           animate: bool = False):
    """Full pipeline: load → plan → (optionally) execute."""
    # Layer 1 — Load
    factory = load_factory_map(map_path)
    agv_cars = load_agv_plan(plan_path)
    agv_res = build_agv_reservations(agv_cars)

    start_xy = factory["landmarks"].get(start_lm)
    goal_xy = factory["landmarks"].get(goal_lm)
    if not start_xy:
        print(f"Error: start landmark '{start_lm}' not found. "
              f"Available: {list(factory['landmarks'].keys())}")
        return
    if not goal_xy:
        print(f"Error: goal landmark '{goal_lm}' not found.")
        return

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  Factory Navigation — Spacetime A*          ║")
    print(f"║  Map:  {os.path.basename(map_path)}                  ║")
    print(f"║  Plan: {os.path.basename(plan_path)}                  ║")
    print(f"║  Start: {start_lm} ({start_xy[0]},{start_xy[1]})    →  Goal: {goal_lm} ({goal_xy[0]},{goal_xy[1]})  ║")
    print(f"║  AGV cars: {len(agv_cars)}  Reservations: {len(agv_res)}                 ║")
    print(f"╚══════════════════════════════════════════════╝")

    # Layer 2 — Plan (single or via)
    via_xy = factory["landmarks"].get(via_lm) if via_lm else None
    goals = []
    if via_xy:
        goals = [(start_xy, start_lm), (via_xy, via_lm), (goal_xy, goal_lm)]
        print(f"\n  Route: {start_lm} → {via_lm} → {goal_lm}")
    else:
        goals = [(start_xy, start_lm), (goal_xy, goal_lm)]

    all_paths = []       # spatial waypoints per leg
    current_start = start_xy

    for i in range(len(goals) - 1):
        s_xy, s_name = goals[i]
        g_xy, g_name = goals[i + 1]
        label = f"{s_name} → {g_name}"
        sw = (float(s_xy[0]), float(s_xy[1]))
        gw = (float(g_xy[0]), float(g_xy[1]))

        print(f"  Planning leg {i+1}: {label} (spatial A*) ...", end=" ", flush=True)
        leg = plan_spatial_path(
            sw, gw, factory["obstacles"], agv_cars,
            factory["width"], factory["height"],
            obstacle_margin=1, agv_block=False,
        )
        if not leg:
            print("FAILED")
            return
        print(f"OK ({len(leg)} pts)")
        all_paths.append(leg)
        current_start = g_xy

    # Combine legs
    path = all_paths[0]
    for leg in all_paths[1:]:
        if leg:
            path = path + leg[1:]

    print(f"\n  Combined: {len(path)} spatial waypoints")

    # ── Closed-loop correction: push waypoints out of collision zones ──
    path_with_t = [(i, x, y) for i, (x, y) in enumerate(path)]
    path_corrected = correct_path(path_with_t)
    path = [(x, y) for _, x, y in path_corrected]
    print(f"  After correction: {len(path)} pts")

    for i, (x, y) in enumerate(path):
        name = next((n for n, (lx, ly) in factory["landmarks"].items()
                     if abs(lx - x) < 0.6 and abs(ly - y) < 0.6), "")
        print(f"  {i:>3} ({x:.2f},{y:.2f}) {name}")

    # Visualisation
    route_label = f"{start_lm} → {via_lm} → {goal_lm}" if via_lm else f"{start_lm} → {goal_lm}"
    if animate:
        animate_factory_plan(
            factory, agv_cars, [(i, x, y) for i, (x, y) in enumerate(path)],
            title=f"Factory: {route_label}",
            interval=150,
        )
        return path

    visualize_factory_plan(
        factory, agv_cars, [(i, x, y) for i, (x, y) in enumerate(path)],
        start_lm=start_lm, goal_lm=goal_lm,
        title=f"Factory Navigation: {route_label}",
        block=plan_only,
    )

    if plan_only:
        return path

    # Layer 3 — Execute via MuJoCo + spatial Navigator
    _ensure_mujoco_imports()
    import mujoco
    import mujoco.viewer

    env = _make_factory_env()
    env.reset()

    # Use spatial path — Navigator walks at its own pace
    nav = SpatialNavigator(path)

    print(f"\n  Starting MuJoCo simulation ...")
    print(f"  Robot starts at ~(0,0) in MuJoCo → maps to factory coords.")
    print(f"  ESC to quit.\n")

    with mujoco.viewer.launch_passive(
        env.model, env.data,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 8.0
        viewer.cam.lookat[:] = (start_xy[0], start_xy[1], 1.0)
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 160

        print_tick = 0
        arrived_printed = False
        sim_time = 0.0

        while viewer.is_running():
            x, y, yaw = _get_robot_pose(env)
            cmd = nav.update(x, y, yaw)

            env.step()
            sim_time += env.simulation_dt

            if env.counter % env.control_decimation == 0:
                env.cmd = cmd
                env.calc_obs()
                env.policy_inference()

            viewer.sync()
            env.rate.sleep()

            print_tick += 1
            if print_tick % 200 == 0:
                wp = nav.current_waypoint
                wp_str = f"wp({wp[1]},{wp[2]}) t={wp[0]}" if wp else "—"
                print(f"\rpos=({x:.2f},{y:.2f})  sim_t={sim_time:.1f}s  "
                      f"dist={nav.distance:.2f}m  {wp_str}  arrived={nav.arrived}  ",
                      end="")

            if nav.arrived and not arrived_printed:
                print(f"\n✓ Arrived at {goal_lm}! Final pos=({x:.2f},{y:.2f})  standing by...")
                arrived_printed = True

    return path


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Factory-floor humanoid navigation with AGV avoidance")
    parser.add_argument("map_file", help="Factory map JSON (map_*.json)")
    parser.add_argument("plan_file", nargs="?", default=None,
                        help="AGV schedule JSON (plan_*.json)")
    parser.add_argument("--start", default="LM006", help="Start landmark")
    parser.add_argument("--goal", default="LM000", help="Goal landmark")
    parser.add_argument("--via", default="", help="Intermediate via landmark")
    parser.add_argument("--plan-only", action="store_true",
                        help="Static map plot, no simulation")
    parser.add_argument("--animate", action="store_true",
                        help="Animated timeline playback (Play/Pause/Step)")
    args = parser.parse_args()

    # Auto-detect plan file if not given
    plan_file = args.plan_file
    if not plan_file:
        maps_dir = os.path.dirname(args.map_file) or "."
        plans = sorted(
            [f for f in os.listdir(maps_dir)
             if f.startswith("plan_") and f.endswith(".json")],
            reverse=True,
        )
        if plans:
            plan_file = os.path.join(maps_dir, plans[0])
            print(f"Auto-detected plan: {plan_file}")
        else:
            print("Error: no plan file found. Specify one explicitly.")
            sys.exit(1)

    run_factory_navigation(
        args.map_file,
        plan_file,
        start_lm=args.start,
        goal_lm=args.goal,
        via_lm=args.via,
        plan_only=args.plan_only,
        animate=args.animate,
    )
