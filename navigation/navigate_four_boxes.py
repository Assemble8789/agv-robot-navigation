"""
Robot navigation in four-box scene — path planning + waypoint navigation.

Uses a visibility graph over pre-defined passage nodes to find the
shortest collision-free path from start to target, then navigates
through the waypoints sequentially.  Repulsive potential fields are
kept only as a safety fallback for close encounters.

Usage:
  python navigation/navigate_four_boxes.py                         # default target
  python navigation/navigate_four_boxes.py --tx 3.0 --ty 0.0       # custom target
  python navigation/navigate_four_boxes.py --tx 3.0 --ty 0.0 --headless
  python navigation/navigate_four_boxes.py --test                   # test suite
"""

from __future__ import annotations

import sys
import math
import os
from typing import List, Tuple, Optional

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_env import EnvConfig, SimpleEnv
from navigation.navigate import (
    Navigator,
    get_robot_pose,
    normalize_angle,
    add_target_marker,
    _add_geom_to_scene,
)

# ═══════════════════════════════════════════════════════════════════════════
# Box obstacle definitions — must match bxi_elf3_scene_four_boxes.xml
# ═══════════════════════════════════════════════════════════════════════════

BOX_HALF = 0.4                     # half-width of each box (square base)
BOX_SAFE = BOX_HALF * 1.414 + 0.20 # safe radius ≈ 0.766 m (half-diag + margin)

BOX_CENTERS = np.array([           # (x, y) from the XML
    [1.5,  1.5],  # box_1  front-left
    [1.5, -1.5],  # box_2  front-right
    [4.5,  1.5],  # box_3  back-left
    [4.5, -1.5],  # box_4  back-right
], dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════════
# Passage nodes — key positions that form the visibility graph
# ═══════════════════════════════════════════════════════════════════════════

PASSAGE_NODES = np.array([
    # Center corridor (y=0)
    [1.5,  0.0],   #  0 · between B1-B2
    [3.0,  0.0],   #  1 · centre of grid
    [4.5,  0.0],   #  2 · between B3-B4
    # Outer — left side
    [0.8,  2.5],   #  3 · above B1
    [0.8, -2.5],   #  4 · below B2
    # Outer — middle gap
    [3.0,  2.5],   #  5 · above the whole grid
    [3.0, -2.5],   #  6 · below the whole grid
    # Outer — right side
    [5.5,  2.5],   #  7 · above B3
    [5.5, -2.5],   #  8 · below B4
    # Extra outer ring
    [0.8,  0.0],   #  9 · far left centre
    [5.5,  0.0],   # 10 · far right centre
], dtype=np.float32)

# Arrival threshold for intermediate waypoints (larger than final)
WP_ARRIVAL = 0.30

# ═══════════════════════════════════════════════════════════════════════════
# Visibility-graph path planner
# ═══════════════════════════════════════════════════════════════════════════

def _dist_point_to_segment(px: float, py: float,
                           ax: float, ay: float,
                           bx: float, by: float) -> float:
    """Minimum distance from point P to segment AB."""
    dx = bx - ax
    dy = by - ay
    seg2 = dx * dx + dy * dy
    if seg2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _segment_clear(ax: float, ay: float,
                   bx: float, by: float,
                   clearance: float = BOX_SAFE) -> bool:
    """True if segment AB stays at least `clearance` away from every box."""
    for (cx, cy) in BOX_CENTERS:
        if _dist_point_to_segment(cx, cy, ax, ay, bx, by) < clearance:
            return False
    return True


def plan_path(start_xy: Tuple[float, float],
              target_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Return collision-free waypoints from start to target.

    Builds a visibility graph (passage nodes + start + target),
    runs Dijkstra, returns the shortest path as a list of (x, y).
    """
    # Collect all nodes
    nodes = [(start_xy[0], start_xy[1])]     # index 0 = start
    nodes += [(p[0], p[1]) for p in PASSAGE_NODES]
    target_idx = len(nodes)
    nodes.append((target_xy[0], target_xy[1]))   # last = target

    n = len(nodes)

    # Build adjacency — check visibility for every pair
    adj = [[] for _ in range(n)]
    for i in range(n):
        xi, yi = nodes[i]
        for j in range(i + 1, n):
            xj, yj = nodes[j]
            if _segment_clear(xi, yi, xj, yj):
                d = math.hypot(xj - xi, yj - yi)
                adj[i].append((j, d))
                adj[j].append((i, d))

    # Dijkstra
    INF = float('inf')
    dist = [INF] * n
    prev = [-1] * n
    visited = [False] * n
    dist[0] = 0.0

    for _ in range(n):
        u = -1
        best = INF
        for i in range(n):
            if not visited[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u == -1 or u == target_idx:
            break
        visited[u] = True
        for v, w in adj[u]:
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u

    if dist[target_idx] == INF:
        # Fallback: direct line (might go through boxes, repulsion will save us)
        return [start_xy, target_xy]

    # Reconstruct path
    path = []
    cur = target_idx
    while cur != -1:
        path.append(nodes[cur])
        cur = prev[cur]
    path.reverse()

    # Simplify: remove collinear / very close waypoints
    if len(path) > 2:
        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            px, py = simplified[-1]
            cx, cy = path[i]
            nx, ny = path[i + 1]
            # Drop middle if direct p→n is clear
            if not _segment_clear(px, py, nx, ny):
                simplified.append(path[i])
        simplified.append(path[-1])
        path = simplified

    return path


# ═══════════════════════════════════════════════════════════════════════════
# WaypointNavigator — serial waypoint navigation
# ═══════════════════════════════════════════════════════════════════════════

class WaypointNavigator:
    """
    Navigates through a sequence of waypoints using the existing Navigator
    for point-to-point control.  Switches to the next waypoint when the
    current one is reached.
    """

    def __init__(self, waypoints: List[Tuple[float, float]],
                 fwd_speed: float = 0.7,
                 turn_speed: float = 1.0,
                 **kwargs):
        self.waypoints = waypoints
        self._wp_idx = 0
        self._nav_kwargs = dict(fwd_speed=fwd_speed, turn_speed=turn_speed, **kwargs)
        # Create navigator for first waypoint
        wx, wy = waypoints[0] if waypoints else (0.0, 0.0)
        self._current_nav = Navigator(wx, wy, **self._nav_kwargs)

    @property
    def current_waypoint(self) -> Optional[Tuple[float, float]]:
        if self._wp_idx < len(self.waypoints):
            return self.waypoints[self._wp_idx]
        return None

    @property
    def arrived(self) -> bool:
        return self._wp_idx >= len(self.waypoints)

    @property
    def heading_error(self) -> float:
        return self._current_nav.heading_error

    @property
    def distance(self) -> float:
        return self._current_nav.distance

    def update(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Get velocity command, auto-advancing waypoints."""
        if self.arrived:
            return np.zeros(3, dtype=np.float32)

        cmd = self._current_nav.update(x, y, yaw)

        # Advance to next waypoint when close enough
        if self._current_nav.arrived:
            self._wp_idx += 1
            if not self.arrived:
                wx, wy = self.waypoints[self._wp_idx]
                self._current_nav = Navigator(wx, wy, **self._nav_kwargs)
                # Run one update immediately so heading_error / distance are fresh
                cmd = self._current_nav.update(x, y, yaw)

        return cmd


# ═══════════════════════════════════════════════════════════════════════════
# Repulsive safety fallback — only fires when robot is too close to a box
# ═══════════════════════════════════════════════════════════════════════════

def apply_safety_repulsion(x: float, y: float, yaw: float,
                           cmd: np.ndarray,
                           hard_radius: float = BOX_SAFE,
                           influence: float = 1.2) -> np.ndarray:
    """
    Modify `cmd` to push robot away from nearby boxes (safety fallback).

    Only activates within `influence` metres.  The hard zone (< hard_radius)
    applies strong lateral push + zero forward speed.
    """
    repulsive = np.zeros(2, dtype=np.float32)
    near_any = False

    for (cx, cy) in BOX_CENTERS:
        dx = x - cx
        dy = y - cy
        dist = float(np.sqrt(dx * dx + dy * dy))
        if dist < 0.001:
            dist = 0.001

        if dist < hard_radius:
            near_any = True
            force = 1.5
        elif dist < influence:
            t = (dist - hard_radius) / (influence - hard_radius)
            force = (1.0 - t) ** 2
        else:
            continue

        repulsive[0] += force * dx / dist
        repulsive[1] += force * dy / dist

    if np.all(repulsive == 0):
        return cmd  # nothing to do

    # Clamp
    mag = float(np.sqrt(repulsive[0] ** 2 + repulsive[1] ** 2))
    if mag > 2.0:
        repulsive *= 2.0 / mag

    # Robot-local frame
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    fy = -sin_yaw * repulsive[0] + cos_yaw * repulsive[1]  # lateral

    result = cmd.copy()
    # Lateral push
    result[1] += fy * 0.8
    # Slow down if near any box
    if near_any:
        result[0] *= 0.3
    # Clamp
    result[0] = np.clip(result[0], 0.0, 0.7)
    result[1] = np.clip(result[1], -0.5, 0.5)
    result[2] = np.clip(result[2], -1.0, 1.0)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Scene config
# ═══════════════════════════════════════════════════════════════════════════

FOUR_BOX_XML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "bxi_elf3", "bxi_elf3_scene_four_boxes.xml",
)


class BoxEnvConfig(EnvConfig):
    """EnvConfig pointing to the four-box scene."""
    model_xml: str = FOUR_BOX_XML_PATH


# ═══════════════════════════════════════════════════════════════════════════
# Viewer helpers
# ═══════════════════════════════════════════════════════════════════════════

def add_box_outline(scene, cx: float, cy: float,
                    half: float, height: float, rgba):
    """Wireframe outline of one box."""
    z_bot = -0.16
    z_top = z_bot + height
    corners = [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]
    for (px, py) in corners:
        _add_geom_to_scene(scene, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           size=[0.015, height / 2, 0],
                           pos=[px, py, (z_bot + z_top) / 2], rgba=rgba)
    n_c = len(corners)
    for i in range(n_c):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % n_c]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        seg = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) / 2
        _add_geom_to_scene(scene, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           size=[0.012, seg, 0],
                           pos=[mx, my, z_top], rgba=rgba)
        _add_geom_to_scene(scene, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           size=[0.012, seg, 0],
                           pos=[mx, my, z_bot + 0.02], rgba=rgba)


def add_safe_zone(scene, cx: float, cy: float, radius: float, rgba):
    """Thin ring at safe-radius boundary."""
    _add_geom_to_scene(scene, mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[radius, 0.005, 0],
                       pos=[cx, cy, -0.15], rgba=rgba)


def add_waypoint_marker(scene, px: float, py: float, rgba, radius=0.04):
    """Small sphere marking a waypoint."""
    _add_geom_to_scene(scene, mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[radius, 0, 0],
                       pos=[px, py, 0.1], rgba=rgba)


def add_path_segment(scene, x1, y1, x2, y2, rgba, z=0.05):
    """Thin capsule connecting two waypoints."""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    seg = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) / 2
    _add_geom_to_scene(scene, mujoco.mjtGeom.mjGEOM_CAPSULE,
                       size=[0.012, seg, 0],
                       pos=[mx, my, z], rgba=rgba)


# ═══════════════════════════════════════════════════════════════════════════
# Render & run
# ═══════════════════════════════════════════════════════════════════════════

BOX_COLORS = [
    [1.0, 0.5, 0.3, 0.25],
    [0.3, 0.8, 0.4, 0.25],
    [0.3, 0.5, 1.0, 0.25],
    [0.9, 0.8, 0.2, 0.25],
]


def render_navigation_four_boxes(target_x: float, target_y: float):
    """Launch viewer with path-planning + waypoint navigation."""
    config = BoxEnvConfig()
    env = SimpleEnv(config)
    env.reset()

    # Plan path from robot's starting position
    sx, sy, _ = get_robot_pose(env)
    waypoints = plan_path((sx, sy), (target_x, target_y))
    nav = WaypointNavigator(waypoints)

    print("╔══════════════════════════════════════════════╗")
    print("║  四箱场景避障导航 (路径规划 + waypoint)      ║")
    print(f"║  目标: ({target_x:.2f}, {target_y:.2f})                      ║")
    print(f"║  waypoints: {len(waypoints)}                                ║")
    for i, (wx, wy) in enumerate(waypoints):
        tag = "START" if i == 0 else ("GOAL" if i == len(waypoints) - 1 else f"WP{i}")
        print(f"║    {tag}: ({wx:.2f}, {wy:.2f})                          ║")
    print("║  ESC : 退出                                  ║")
    print("╚══════════════════════════════════════════════╝")

    with mujoco.viewer.launch_passive(
        env.model, env.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5.5
        viewer.cam.lookat[:] = (2.5, 0.0, 0.8)
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 160

        def draw_all():
            viewer.user_scn.ngeom = 0
            # Box outlines + safe zones
            for i, (cx, cy) in enumerate(BOX_CENTERS):
                add_box_outline(viewer.user_scn, cx, cy, BOX_HALF, 1.0, BOX_COLORS[i])
                add_safe_zone(viewer.user_scn, cx, cy, BOX_SAFE,
                              [0.8, 0.4, 0.4, 0.2])
            # Path segments
            wp_rgba = [1.0, 0.5, 0.0, 0.7]
            seg_rgba = [1.0, 0.7, 0.3, 0.35]
            for i in range(len(waypoints) - 1):
                x1, y1 = waypoints[i]
                x2, y2 = waypoints[i + 1]
                add_path_segment(viewer.user_scn, x1, y1, x2, y2, seg_rgba)
            for i, (wx, wy) in enumerate(waypoints):
                r = 0.06 if i == len(waypoints) - 1 else 0.04
                add_waypoint_marker(viewer.user_scn, wx, wy, wp_rgba, radius=r)
            # Target marker
            add_target_marker(viewer.user_scn, target_x, target_y)

        draw_all()

        print_tick = 0
        arrived_printed = False
        while viewer.is_running():
            x, y, yaw = get_robot_pose(env)
            wp = nav.current_waypoint
            cmd = nav.update(x, y, yaw)

            # Safety repulsion fallback
            cmd = apply_safety_repulsion(x, y, yaw, cmd)

            env.step()
            if env.counter % env.control_decimation == 0:
                env.cmd = cmd
                env.calc_obs()
                env.policy_inference()

            if viewer.user_scn.ngeom == 0:
                draw_all()

            viewer.sync()
            env.rate.sleep()

            print_tick += 1
            if print_tick % 200 == 0:
                deg = np.degrees(nav.heading_error)
                wp_str = f"({wp[0]:.2f},{wp[1]:.2f})" if wp else "—"
                print(f"\rpos=({x:.3f},{y:.3f})  yaw={np.degrees(yaw):.1f}°  "
                      f"dist={nav.distance:.3f}m  h_err={deg:+.1f}°  "
                      f"wp={wp_str}  arrived={nav.arrived}  ", end="")

            if nav.arrived and not arrived_printed:
                print(f"\n✓ Arrived at target! Final pos=({x:.3f},{y:.3f})  standing by...")
                arrived_printed = True


# ═══════════════════════════════════════════════════════════════════════════
# Headless runner
# ═══════════════════════════════════════════════════════════════════════════

def run_headless_four_boxes(target_x: float, target_y: float,
                            max_time: float = 30.0,
                            **nav_kwargs) -> dict:
    """Run one episode headless, with path planning + safety repulsion."""
    config = BoxEnvConfig()
    env = SimpleEnv(config)
    env.reset()

    sx, sy, _ = get_robot_pose(env)
    waypoints = plan_path((sx, sy), (target_x, target_y))
    nav = WaypointNavigator(waypoints, **nav_kwargs)

    traj = [(0.0, sx, sy)]
    t = 0.0
    dt = config.simulation_dt
    stuck_counter = 0

    while t < max_time:
        x, y, yaw = get_robot_pose(env)
        cmd = nav.update(x, y, yaw)
        cmd = apply_safety_repulsion(x, y, yaw, cmd)
        traj.append((t, x, y, yaw))

        env.step()
        if env.counter % env.control_decimation == 0:
            env.cmd = cmd
            env.calc_obs()
            env.policy_inference()

        t += dt

        if len(traj) >= 3:
            prev = traj[-50] if len(traj) >= 50 else traj[0]
            moved = math.hypot(x - prev[1], y - prev[2])
            if moved < 0.005 and nav.distance > 0.10:
                stuck_counter += 1
            else:
                stuck_counter = 0

        if stuck_counter > 400:
            break
        if nav.arrived:
            break

    final_x, final_y, _ = get_robot_pose(env)
    min_box_dist = float('inf')
    for (cx, cy) in BOX_CENTERS:
        d = math.hypot(final_x - cx, final_y - cy)
        if d < min_box_dist:
            min_box_dist = d

    # Also check minimum box distance over entire trajectory
    traj_min_box = float('inf')
    for _, px, py, _ in traj:
        for (cx, cy) in BOX_CENTERS:
            d = math.hypot(px - cx, py - cy)
            if d < traj_min_box:
                traj_min_box = d

    return {
        'arrived': nav.arrived,
        'stuck': stuck_counter > 400,
        'final_dist': math.hypot(target_x - final_x, target_y - final_y),
        'elapsed': t,
        'start_x': sx, 'start_y': sy,
        'final_x': final_x, 'final_y': final_y,
        'min_box_dist': min_box_dist,
        'traj_min_box_dist': traj_min_box,
        'num_waypoints': len(waypoints),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test suite
# ═══════════════════════════════════════════════════════════════════════════

FOUR_BOX_TARGETS = [
    ("between front boxes",      3.0,  0.0),
    ("right of box 2",           2.0, -2.5),
    ("left of box 1",            2.0,  2.5),
    ("between back boxes",       4.5,  0.0),
    ("past all boxes",           5.5,  0.0),
    ("diag past boxes",          5.0,  2.0),
    ("far right past box 4",     5.0, -2.5),
    ("behind box 3",             4.0,  2.5),
    ("far beyond centre",        6.0,  0.0),
    ("through centre gap",       3.0,  1.5),
    ("sharp left turn past B1",  3.0,  3.0),
    ("sharp right turn past B2", 3.0, -3.0),
]


def run_test_four_boxes(max_time: float = 30.0):
    """Run all four-box test targets headless, print summary."""
    print(f"\n{'='*85}")
    print(f"  FOUR-BOX NAVIGATION TEST  ({len(FOUR_BOX_TARGETS)} targets, max {max_time}s, path-planning)")
    print(f"{'='*85}\n")
    hdr = (f"  {'#':>2}  {'target':>24}  {'(x,y)':>16}  {'arr':>5}  "
           f"{'final_d':>8}  {'min_box_d':>9}  {'traj_min':>8}  {'wp':>3}  {'time':>6}")
    print(hdr)
    print(f"  {'─'*2}  {'─'*24}  {'─'*16}  {'─'*5}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*3}  {'─'*6}")

    results = []
    for i, (label, tx, ty) in enumerate(FOUR_BOX_TARGETS, 1):
        m = run_headless_four_boxes(tx, ty, max_time=max_time)
        results.append((label, tx, ty, m))
        print(f"  {i:>2}  {label:>24}  ({tx:>5.1f},{ty:>5.1f})   "
              f"{'YES' if m['arrived'] else 'NO':>5}  "
              f"{m['final_dist']:>8.4f}  "
              f"{m['min_box_dist']:>9.3f}  "
              f"{m['traj_min_box_dist']:>8.3f}  "
              f"{m['num_waypoints']:>3}  "
              f"{m['elapsed']:>5.1f}s")

    arrived = [r for r in results if r[3]['arrived']]
    dists = [r[3]['final_dist'] for r in results]
    box_dists = [r[3]['min_box_dist'] for r in results]
    traj_box = [r[3]['traj_min_box_dist'] for r in results]
    times = [r[3]['elapsed'] for r in results]

    print(f"\n{'─'*85}")
    print(f"  SUMMARY")
    print(f"  ───────")
    print(f"  Arrived: {len(arrived)}/{len(results)} ({100*len(arrived)/len(results):.0f}%)")
    print(f"  Min box dist (final):  mean={np.mean(box_dists):.3f}m  worst={np.min(box_dists):.3f}m")
    print(f"  Min box dist (traj):   mean={np.mean(traj_box):.3f}m  worst={np.min(traj_box):.3f}m")
    if np.min(traj_box) < BOX_SAFE * 0.8:
        print(f"  ⚠ WARNING: robot violated safe zone ({(BOX_SAFE * 0.8):.2f}m)!")
    print(f"  Final distance: mean={np.mean(dists):.4f}m  median={np.median(dists):.4f}m")
    print(f"  Elapsed time:   mean={np.mean(times):.1f}s  max={np.max(times):.1f}s")
    print(f"{'='*85}\n")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    target_x = 3.0
    target_y = 0.0

    if "--tx" in sys.argv:
        idx = sys.argv.index("--tx")
        target_x = float(sys.argv[idx + 1])
    if "--ty" in sys.argv:
        idx = sys.argv.index("--ty")
        target_y = float(sys.argv[idx + 1])

    if "--test" in sys.argv:
        run_test_four_boxes()
    elif "--headless" in sys.argv:
        m = run_headless_four_boxes(target_x, target_y, max_time=30.0)
        print(f"\nHeadless result: arrived={m['arrived']}  final_dist={m['final_dist']:.4f}m  "
              f"min_box_dist={m['min_box_dist']:.3f}m  traj_min={m['traj_min_box_dist']:.3f}m  "
              f"elapsed={m['elapsed']:.1f}s  waypoints={m['num_waypoints']}")
    else:
        render_navigation_four_boxes(target_x, target_y)
