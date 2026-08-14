"""
Full factory simulation — AGV-YIELD variant.

ROLE REVERSAL vs run_factory_full.py: the ROBOT has highest priority and never
stops (unless an AGV has reached its goal and blocks the path).  AGVs yield to
the robot, primarily by STOPPING.  When a stopped AGV's path is clear again it
REPLANS its remaining route at runtime and writes the updated plan to
maps/current_plan.json.

Usage:
  python navigation/run_factory_full_agv_yield.py [--headless]
"""

import json, os, sys, argparse, contextlib, heapq
import numpy as np
import mujoco, mujoco.viewer

_PARSER = argparse.ArgumentParser()
_PARSER.add_argument("--headless", action="store_true",
                     help="no viewer; print diagnostics and exit")
_PARSER.add_argument("--full", action="store_true",
                     help="headless: run the full window even after the robot arrives")
_PARSER.add_argument("--no-robot-stop", action="store_true",
                     help="AGVs still stop/yield for the robot, but the robot "
                          "itself never stops at blocked waypoints (A/B check)")
_PARSER.add_argument("--replan-timeout", type=float, default=15.0,
                     help="seconds an AGV waits for the robot before it "
                          "replans (safety net; lower forces replans for tests)")
_PARSER.add_argument("--no-active-replan", action="store_true",
                     help="disable the active conflict-scan replan layer "
                          "(A/B: forces then handle all AGV-AGV avoidance)")
_PARSER.add_argument("--no-agv-force", action="store_true",
                     help="disable the AGV-AGV repulsion entirely — count-only "
                          "(A/B: pure C_F replan must separate AGVs itself)")
_ARGS = _PARSER.parse_args()
ACTIVE_REPLAN = not _ARGS.no_active_replan
ENABLE_AGV_FORCE = not _ARGS.no_agv_force
_HEADLESS = _ARGS.headless
# Robot stops at its waypoint when an AGV is parked / queuing there.  This is
# only safe because robot_blocks() now stops AGVs for a STATIONARY robot too —
# without that, the stopped robot is invisible to the AGV layer and queuing
# AGVs drive straight through it (the 3000+ collision pile-up).
ROBOT_STOP = not _ARGS.no_robot_stop

# ── AGV-yield / replan parameters ──
# BLOCK_DIST: a MOVING robot approaching an AGV's near-future path within this
#   distance → the AGV stops.  Was 2.0 m — too large: an AGV sitting at a busy
#   station yielded for a robot that was still 2 m away, and kept re-yielding
#   while the robot crawled past (the car3-stuck-at-LM002 loop).  1.2 m is still
#   ~2 s of reaction (robot ~0.6 m/s, AGV clock freezes instantly).
# STATIONARY_BLOCK_DIST: a STOPPED robot on the AGV's path within this distance
#   → the AGV stops.  The robot can't move out of the way, so the AGV only needs
#   to avoid overlapping its body (~0.5 m), not keep a 2 m bubble.
BLOCK_DIST = 1.2              # moving-robot yield radius (m)
STATIONARY_BLOCK_DIST = 0.8   # stopped-robot yield radius (m)
REPLAN_TIMEOUT = _ARGS.replan_timeout  # safety net only: replan after this
                              # long even if the robot is still nearby.  Kept
                              # LONG so a busy station does NOT get a degenerate
                              # detour path — the AGV waits for the robot to
                              # pass and replans when the corridor is clear.
STOP_WAIT = 1.5      # robot waits at most this long at a NON-work blocked waypoint
AGV_SPEED = 0.7      # AGV speed factor (1.0 = 1 m/s as planned; 0.7 = 0.7 m/s)

# ── Work zones ──
# AGVs DO WORK (dock / load-unload) at these points and must not be interrupted
# by the robot: a 'working' AGV does NOT yield to the robot (robot_blocks → False),
# and the ROBOT waits up to WORK_WAIT s for it instead.  centre = LM coords,
# work_r = docked (working) radius, queue_r = approaching / queuing radius.
WORK_ZONES = {
    "LM002": {"center": (9.0, 7.0), "work_r": 0.6, "queue_r": 2.0},
    # add more load / unload points as needed
}
WORK_WAIT = 5.0        # robot waits this long (s) for a WORKING / QUEUING AGV

# ── Active replan layer ──
# Keeps the plan mutually avoiding at RUNTIME so the AGV-AGV repulsion never
# needs to engage: every RE_SCAN_INT s, any AGV whose near-future path is
# predicted (time-aligned) to cross a MOVING sibling is rerouted via the
# spatiotemporal replan BEFORE it reaches force range.  The runtime forces then
# stay dormant as a pure last resort.
RE_SCAN_INT = 0.5           # conflict-scan period (s)
# Replan only for a REAL near-miss (just above body contact ~0.4 m).  Was 0.8 m
# — too sensitive: AGVs passing 0.8 m apart (safe) got rerouted, and the
# time-aware A* responded with whole-map detours (green/yellow balls wandered
# on 133/93-point paths while the others finished).
AGV_CONFLICT_DIST = 0.5
AGV_REPLAN_COOLDOWN = 2.0   # min s between replans of the same AGV (avoid churn)

# Moving-moving repulsion radius — PURE LAST RESORT (m), just above the 0.4 m
# body-contact radius.  The replan (C_F) is the PRIMARY avoidance and keeps AGVs
# apart down to ~0.5-0.6 m (B' side-shift / wait); a force firing at 0.7 m just
# re-filled the space the replan had already cleared, inflating path length
# (measured: replan+force 0.7m -> 40-42 s arrival vs 37.3 s pure-force).  At 0.45
# the force only engages on a genuine prediction error and merely nudges.
AGV_AVOID_DIST = 0.45
# Body contact radius: two 0.2 m AGV spheres touch at 0.4 m centre distance.
AGV_BODY_R = 0.40
# Bounded steady-state lateral offset the force may hold (mirrors ROBOT_AVOID_AMP).
# The per-frame amplitude is pre-divided by the ×0.94/frame decay gain (≈1/0.06)
# so the steady-state offset is bounded at AGV_AVOID_AMP, not the naive 16.7×.
# The smoothstep is normalized over [AGV_BODY_R, AGV_AVOID_DIST] — the only 0.05 m
# of range a last-resort force actually has — so it ramps to full strength right
# where contact would happen instead of being flat near the boundary (a plain
# smoothstep over [0, 0.45] was ~0 at 0.44 and AGVs passed through each other).
AGV_AVOID_AMP = 0.60

# Robot-repulsion force (AGV-side, last-resort) — smooth and bounded.  The
# naive version (hard 0.7 m cutoff + radial from the live robot position +
# unbounded accumulation into lateral, whose ×0.94/frame decay gives a 1/0.06 ≈
# 16.7× steady-state gain) makes the push jitter: forces appear/disappear at
# the cutoff, and the push direction flips as the robot walks.  Instead:
#   - smooth smoothstep falloff over ROBOT_AVOID_R (acts early, no step at the
#     edge; stronger in the mid-range than a quadratic — a too-weak force lets
#     AGVs pass too close and the robot's own 0.6 m repulsion shoves it into
#     the shelves)
#   - repulse from the robot's PREDICTED position (velocity lookahead) so the
#     direction stays stable while the robot moves
#   - push amplitude pre-divided by the 0.06 decay gain so the steady-state
#     lateral offset is bounded at ROBOT_AVOID_AMP (no runaway accumulation)
ROBOT_AVOID_R = 1.2         # activation radius (m)
ROBOT_AVOID_AMP = 0.40      # max steady-state lateral offset held (m)
ROBOT_AVOID_LOOKAHEAD = 0.4 # s — repulse from predicted robot position

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(ROOT, "agv_simulation-main", "maps")
CURRENT_PLAN = os.path.join(MAPS_DIR, "current_plan.json")

# ── Lazy imports for robot control (need mujoco already imported) ──
sys.path.insert(0, ROOT)
from simple_env import EnvConfig
import onnxruntime as ort

# ── Load plan ───────────────────────────────────────────────────────────────
_plans = sorted([f for f in os.listdir(MAPS_DIR)
                 if f.startswith("plan_v2_") and f.endswith(".json")], reverse=True)
if not _plans:
    _plans = sorted([f for f in os.listdir(MAPS_DIR)
                     if f.startswith("plan_") and f.endswith(".json")], reverse=True)
if not _plans:
    raise SystemExit("No plan file")

with open(os.path.join(MAPS_DIR, _plans[0])) as f:
    plan = json.load(f)
agv_trajs = {car["car_id"]: [(p["t"], p["x"], p["y"]) for p in car["trajectory"]]
             for car in plan["cars"]}
max_t = max(p[-1][0] for p in agv_trajs.values())

# ── Precompute per-AGV stationary (dwell) time windows ──
#  An AGV is stationary at time t if it holds the same cell as at t-1
#  (dwell stops in the middle of its chain), or before departure / after arrival.
agv_stationary_times = {}  # car_id -> set of t where AGV is parked
for cid, traj in agv_trajs.items():
    parked = set()
    pos_prev = None
    for i, (t, x, y) in enumerate(traj):
        if i == 0:
            # before departure: parked at start for all t < first t
            pos_prev = (x, y)
            continue
        if (x, y) == pos_prev:
            parked.add(t)          # dwell: same cell as previous tick
        pos_prev = (x, y)
    agv_stationary_times[cid] = parked

# ── Load humanoid plan ──────────────────────────────────────────────────────
with open(os.path.join(MAPS_DIR, "humanoid_plan.json")) as f:
    hplan = json.load(f)
# Spatial waypoints [(x, y), ...]
humanoid_path = [(p["x"], p["y"]) for p in hplan["waypoints"]]
print(f"AGVs: {len(agv_trajs)}  Humanoid: {hplan['start']}->{hplan['via']}->{hplan['goal']}"
      f"  ({len(humanoid_path)} spatial waypoints)")

# ── Factory map + per-car remaining stops (for runtime replanning) ──
from navigation import navigate_factory as _nf
from navigation import cf_replan as _cf
_map_file = sorted([f for f in os.listdir(MAPS_DIR) if f.startswith("map_")])[0]
_factory = _nf.load_factory_map(os.path.join(MAPS_DIR, _map_file))
MAP_W, MAP_H = _factory["width"], _factory["height"]
COARSE_OBS = _factory["obstacles"]            # coarse grid obstacle cells
LM_COORDS = _factory["landmarks"]

# Remaining destination coords per car, in order (from the plan's stops chain)
car_goals = {}
for car in plan["cars"]:
    cid = car["car_id"]
    car_goals[cid] = [LM_COORDS[n] for n in car.get("stops", []) if n in LM_COORDS]
car_index = {car["car_id"]: i for i, car in enumerate(plan["cars"])}

# ── Constants ───────────────────────────────────────────────────────────────
GROUND_Z, AGV_R = -0.16, 0.2
AGV_Z = GROUND_Z + AGV_R
SAFE_DIST = 0.8

STATIC_OBS = {
    (3,5),(3,8),(4,5),(4,8),(5,5),(5,8),
    (6,1),(6,2),(7,1),(7,2),
    (8,5),(8,8),(9,5),(9,8),(10,5),(10,8),
}

# Real static-obstacle geometry (cx, cy, hx, hy) — must match the XML.
# Using these instead of the coarse grid cells is essential: the grid cell
# repulsion pushed the robot TOWARD a shelf (cell (7,8) is at the west edge
# of the box centred on (9,8), so "push away from the cell" pushed it east,
# into the shelf body).
OBS_BOXES = [
    (9.0, 8.0, 1.5, 0.35),   # obs_0 right shelf y=8
    (9.0, 5.0, 1.5, 0.35),   # obs_1 right shelf y=5
    (6.5, 1.0, 1.0, 0.5),    # obs_2 equipment bottom
    (4.0, 5.0, 1.5, 0.35),   # obs_3 left shelf y=5
    (4.0, 8.0, 1.5, 0.35),   # obs_4 left shelf y=8
    (6.5, 2.0, 1.0, 0.5),    # obs_5 equipment bottom
]
ROBOT_R = 0.30   # robot body radius for clearance


def lateral_free(px, py, heading, step=0.1, max_d=3.0):
    """Free lateral distance (m) the robot can move LEFT then RIGHT before it
    would touch an obstacle box (inflated by ROBOT_R + margin).  Used to pick
    the dodge direction with more room and to cap the dodge amplitude — in the
    narrow 2.3 m corridor a blind 0.8 m/s × 2 s dodge (1.6 m) plows into a
    shelf.
    """
    left_dir = np.array([-heading[1], heading[0]])
    right_dir = np.array([heading[1], -heading[0]])
    free = [max_d, max_d]
    for side, dr in enumerate((left_dir, right_dir)):
        for s in np.arange(step, max_d, step):
            qx, qy = px + dr[0] * s, py + dr[1] * s
            hit = any(abs(qx - cx) < hx + ROBOT_R + 0.05 and
                      abs(qy - cy) < hy + ROBOT_R + 0.05
                      for (cx, cy, hx, hy) in OBS_BOXES)
            if hit:
                free[side] = s
                break
    return free  # (free_left, free_right)


def free_along(px, py, direction, step=0.1, max_d=3.0):
    """Distance (m) the robot can travel from (px,py) along `direction`
    (a unit vector) before touching an obstacle box (inflated by ROBOT_R).
    Used to cap the dodge amplitude along the robot's actual diagonal motion,
    not the pure lateral line.
    """
    for s in np.arange(step, max_d, step):
        qx, qy = px + direction[0] * s, py + direction[1] * s
        if any(abs(qx - cx) < hx + ROBOT_R + 0.05 and
               abs(qy - cy) < hy + ROBOT_R + 0.05
               for (cx, cy, hx, hy) in OBS_BOXES):
            return s
    return max_d


CLEAR_MARGIN = 0.3   # static repulsion activates this far beyond the body


def static_repulsion(px, py):
    """World-frame push-away force from the real obstacle boxes.

    Returns (repel_x, repel_y).  Activates when the robot is within
    ROBOT_R + CLEAR_MARGIN of a box surface (so it starts resisting a vy
    dodge early), growing to a strong push at contact.
    """
    rep = np.zeros(2)
    act = ROBOT_R + CLEAR_MARGIN
    for (cx, cy, hx, hy) in OBS_BOXES:
        dx, dy = px - cx, py - cy
        ax, ay = abs(dx), abs(dy)
        sx, sy = np.sign(dx) if dx else 1.0, np.sign(dy) if dy else 1.0
        # penetration beyond each box axis
        ox, oy = ax - hx, ay - hy
        if ox < 0 and oy < 0:
            # centre inside the box footprint — push out along shallower axis
            pen_x, pen_y = -ox, -oy
            if pen_x < pen_y:
                rep[0] += sx * (act + pen_x)
            else:
                rep[1] += sy * (act + pen_y)
        else:
            # distance from robot centre to nearest box surface / corner
            nx, ny = min(ax, hx), min(ay, hy)
            vx, vy = ax - nx, ay - ny
            d = np.hypot(vx, vy)
            if d < act and d > 1e-6:
                force = (act - d) / act
                rep[0] += sx * (vx / d) * force
                rep[1] += sy * (vy / d) * force
    return rep

# ── Smooth AGV trajectories ─────────────────────────────────────────────────
def smooth_traj(waypoints, num_samples=200):
    pts = np.array([(p[1], p[2]) for p in waypoints], dtype=float)
    times = np.array([p[0] for p in waypoints], dtype=float)
    if len(pts) < 2:
        return times, pts[:,0], pts[:,1]
    pa = np.vstack([pts[0]*2-pts[1], pts, pts[-1]*2-pts[-2]])
    ta = np.concatenate([[times[0]-1], times, [times[-1]+1]])
    t_s = np.linspace(times[0], times[-1], num_samples)
    xs, ys, seg = [], [], 0
    for t in t_s:
        while seg < len(times)-1 and t > times[seg+1]: seg += 1
        seg = min(seg, len(times)-2); i = seg+1
        p0,p1,p2,p3 = pa[i-1],pa[i],pa[i+1],pa[i+2]
        t0,t2 = ta[i],ta[i+1]
        a = max(0.0, min(1.0, (t-t0)/(t2-t0) if t2>t0 else 0))
        a2,a3 = a*a, a*a*a
        r = 0.5*((2*p1)+(-p0+p2)*a+(2*p0-5*p1+4*p2-p3)*a2+(-p0+3*p1-3*p2+p3)*a3)
        xs.append(r[0]); ys.append(r[1])
    xa,ya = np.array(xs), np.array(ys)
    for i in range(len(xa)):
        for (ox,oy) in STATIC_OBS:
            dx,dy = xa[i]-ox, ya[i]-oy
            d = np.hypot(dx,dy)
            if d < SAFE_DIST and d > 0.01:
                xa[i] += (dx/d)*(SAFE_DIST-d); ya[i] += (dy/d)*(SAFE_DIST-d)
    return t_s, xa, ya

agv_smooth = {cid: smooth_traj(traj) for cid, traj in agv_trajs.items()}

def get_agv_pos(cid, t):
    ts, xs, ys = agv_smooth[cid]
    if t <= ts[0]: return float(xs[0]), float(ys[0])
    if t >= ts[-1]: return float(xs[-1]), float(ys[-1])
    idx = max(1, min(np.searchsorted(ts, t), len(ts)-1))
    f = (t-ts[idx-1])/(ts[idx]-ts[idx-1]) if ts[idx]>ts[idx-1] else 0
    return float(xs[idx-1]+f*(xs[idx]-xs[idx-1])), float(ys[idx-1]+f*(ys[idx]-ys[idx-1]))


def agv_heading(cid, clocks):
    """Unit heading of AGV `cid` along its plan at the current clock — the
    direction the replan intends it to travel (next non-degenerate segment;
    dwell/duplicate points are skipped).  Used to keep the repulsion force
    lateral-only, i.e. consistent with the replan's B' side-shift direction."""
    traj = agv_trajs[cid]
    t = clocks[cid]
    prev = None
    for wp in traj:
        if wp[0] <= t:
            prev = (wp[1], wp[2])
        else:
            if prev is not None:
                dx = wp[1] - prev[0]
                dy = wp[2] - prev[1]
                m = np.hypot(dx, dy)
                if m > 1e-6:
                    return (dx / m, dy / m)
            break
    return (1.0, 0.0)


def agv_is_working(cid, pos, clocks):
    """Classify AGV `cid` relative to the work zones (WORK_ZONES).

    Returns:
      'working' — the AGV is DOCKED at a work-zone centre (within work_r) AND
        stationary per its plan (dwelling / parked at the stop).  Its work must
        not be interrupted: robot_blocks() returns False for it, and the ROBOT
        waits (WORK_WAIT) instead.
      'queuing' — the AGV's NEXT goal IS this work zone and it is within
        queue_r, approaching to pull in: the robot lets it dock first.
      None      — ordinary AGV (yields to the robot as usual).
    """
    x, y = pos
    t = clocks[cid]
    parked = (t <= agv_trajs[cid][0][0] or t >= agv_trajs[cid][-1][0]
              or int(round(t)) in agv_stationary_times.get(cid, set()))
    for name, zone in WORK_ZONES.items():
        cx, cy = zone["center"]
        if np.hypot(x - cx, y - cy) < zone["work_r"] and parked:
            return "working"
    for name, zone in WORK_ZONES.items():
        cx, cy = zone["center"]
        if car_goals[cid] and \
           np.hypot(car_goals[cid][0][0] - cx, car_goals[cid][0][1] - cy) < 0.3 and \
           np.hypot(x - cx, y - cy) < zone["queue_r"]:
            return "queuing"
    return None


# ── AGV-yield helpers ────────────────────────────────────────────────────────
def robot_blocks(cid, rx, ry, rvx, rvy, clocks):
    """True if the AGV must stop for the robot (robot on its near-future path).

    Two stop reasons:
      - robot APPROACHING the AGV's path (velocity toward it) — normal yield;
      - robot essentially STATIONARY on the path — it cannot move out of the
        way, so the AGV stops too and replans around it after REPLAN_TIMEOUT.
        (Without this, a stopped robot vanishes from the AGV's obstacle world
        and queuing AGVs drive straight through it → the pile-up.)

    The stationary case only fires when the AGV is heading TOWARD the robot
    (plan direction dot "robot is ahead" > 0) — otherwise a stopped robot would
    freeze AGVs that have already passed it.  The plan direction comes from the
    next waypoint so it stays correct while the AGV dwells at a station."""
    pos0 = get_agv_pos(cid, clocks[cid])
    # A WORKING AGV (docked at a load/unload zone) must not be interrupted — it
    # does NOT yield to the robot; the ROBOT waits for it instead (r_blocked).
    if agv_is_working(cid, pos0, clocks) == "working":
        return False
    # An AGV flagged to MOVE OUT of the robot's waypoint (挪开) must not freeze
    # for the robot — its already-replanned path takes it away.  Without this it
    # replans, then robot_blocks() re-freezes it at its start (still within
    # BLOCK_DIST of the robot) → the deadlock persists.
    if cid == _move_agv:
        return False
    # plan heading at the current clock (robust to dwell: use next waypoint)
    dirv = np.zeros(2)
    traj = agv_trajs[cid]
    t = clocks[cid]
    for i in range(len(traj)):
        if traj[i][0] > t:
            if i > 0:
                dirv[0] = traj[i][1] - traj[i - 1][1]
                dirv[1] = traj[i][2] - traj[i - 1][2]
            break
    dlen = np.hypot(*dirv)
    if dlen > 1e-6:
        dirv /= dlen
    robot_speed2 = rvx * rvx + rvy * rvy
    for ta in (0.0, 0.5, 1.0, 1.5):
        ax, ay = get_agv_pos(cid, clocks[cid] + ta * AGV_SPEED)
        d = np.hypot(ax - rx, ay - ry)
        if d < BLOCK_DIST and rvx * (ax - rx) + rvy * (ay - ry) > 0:
            return True   # robot moving toward this AGV's path
        if robot_speed2 < 0.04 and ta > 0.0 and \
           d < STATIONARY_BLOCK_DIST and \
           dirv[0] * (rx - pos0[0]) + dirv[1] * (ry - pos0[1]) > 0:
            return True   # robot stopped ahead on the path
    return False


def write_current_plan():
    """Dump the in-memory plan (with any replanned trajectories) to
    maps/current_plan.json so the real-time plan is inspectable."""
    out = {"map_file": plan.get("map_file", ""),
           "cars": [{"car_id": c["car_id"],
                     "start_landmark": c.get("start_landmark", ""),
                     "goal_landmark": c.get("goal_landmark", ""),
                     "stops": c.get("stops", []),
                     "trajectory": [{"t": t, "x": x, "y": y}
                                    for (t, x, y) in agv_trajs[c["car_id"]]]}
                    for c in plan["cars"]]}
    with open(CURRENT_PLAN, "w") as f:
        json.dump(out, f, indent=1)


def replan_agv_car_time(cur, goal_xy, coarse_obs, width, height, cid, base_t,
                        clocks, obstacle_margin=2, clearance=0.5):
    """Spatiotemporal A* for a single AGV — the new path's (cell, time) must not
    conflict with the OTHER AGVs' PREDICTED trajectories (QoS-style).

    The original plan is time-coordinated (agv_planner_v2 reserves (x,y,t)
    vertices + bidirectional edges), but the runtime replan was pure spatial A*
    — it only blocked the other AGVs' CURRENT cells, so a replanned path could
    cross another AGV at the same moment.  Here each candidate fine cell is
    checked at the TIME the new path would reach it: other AGV `ocid` is
    predicted at clock = clocks[ocid] + (t - base_t) (all clocks advance at
    AGV_SPEED, so the same sim-time maps to a fixed clock offset).  The runtime
    AGV-AGV repulsion stays as the last resort for prediction error (an AGV
    paused mid-yield, etc.).

    Returns [(world_x, world_y), ...] or None.
    """
    fw, fh = width * _nf.FINE_SCALE, height * _nf.FINE_SCALE
    fine_obs = set()
    for (ox, oy) in coarse_obs:
        for (fx, fy) in _nf._coarse_to_fine(ox, oy):
            fine_obs.add((fx, fy))
    fine_obs = _nf.inflate_obstacles(fine_obs, margin=obstacle_margin,
                                     width=fw, height=fh)
    start = _nf._world_to_fine(*cur)
    goal = _nf._world_to_fine(*goal_xy)
    # unlock start/goal — the AGV may start beside another AGV, or head to a
    # stop another AGV already occupies (the runtime wait handles that)
    for cell in (start, goal):
        if cell in fine_obs:
            fine_obs.discard(cell)
            fx, fy = cell
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= fx + dx < fw and 0 <= fy + dy < fh:
                        fine_obs.discard((fx + dx, fy + dy))
    if start in fine_obs or goal in fine_obs:
        return None

    def _moving_conflict(fx, fy, t):
        """True if another AGV is predicted within `clearance` m of this fine
        cell's centre at clock-time t (on this AGV's new-path clock)."""
        wx, wy = _nf._fine_to_world(fx, fy)
        for ocid in agv_trajs:
            if ocid == cid:
                continue
            ox, oy = get_agv_pos(ocid, clocks[ocid] + (t - base_t))
            if np.hypot(ox - wx, oy - wy) < clearance:
                return True
        return False

    counter = 0
    open_set = []
    heapq.heappush(open_set,
                   (_nf._heuristic(start, goal), 0.0, start[0], start[1], counter))
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
            cur_st = state
            while cur_st in came_from:
                path_xy.append(cur_st)
                cur_st = came_from[cur_st]
            path_xy.append(start)
            path_xy.reverse()
            return [_nf._fine_to_world(fx, fy) for (fx, fy) in path_xy]

        for dx, dy in _dirs:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < fw and 0 <= ny < fh):
                continue
            if (nx, ny) in fine_obs:
                continue
            ng = g + (1.0 if (dx == 0 or dy == 0) else 1.414)
            # clock time when the new path would reach this neighbour
            t_n = base_t + ng * _nf.FINE_CELL / AGV_SPEED
            if (nx, ny) != goal and _moving_conflict(nx, ny, t_n):
                continue
            nstate = (nx, ny)
            if ng < g_score.get(nstate, float("inf")):
                g_score[nstate] = ng
                came_from[nstate] = state
                heapq.heappush(open_set, (ng + _nf._heuristic(nstate, goal),
                                          ng, nx, ny, counter))
                counter += 1
    return None


def replan_car(cid, rx, ry, clocks, robot_avoid=True):
    """Replan AGV `cid` from its current position through its remaining stops.
    robot_avoid=True (the 15 s safety net): the robot's current cell is a static
    obstacle.  robot_avoid=False (the ACTIVE AGV-AGV scan): skip the robot — the
    yield/stop layer handles the robot, so AGV-AGV replans don't detour around a
    2 m robot blob every time.  Updates agv_trajs + agv_smooth and writes
    current_plan.json.  Returns True on success."""
    if not car_goals[cid]:
        return False
    cur = list(get_agv_pos(cid, clocks[cid]))
    # the ROBOT's cell is a static obstacle (safety net only); the OTHER AGVs
    # are handled by the C_F runtime walk (cf_replan) — wait/detour per conflict
    dyn = {(int(round(rx)), int(round(ry)))} if robot_avoid else set()
    # C_F 式重规划: 干净空间 A* 基础路径(不绕全图) + 运行时 wait/detour/侧移
    # 决策, 自动经过所有剩余目标 (cf_replan.plan_conflict_free)
    wps = _cf.plan_conflict_free(cur, car_goals[cid], COARSE_OBS | dyn,
                                 MAP_W, MAP_H, cid, clocks[cid], clocks,
                                 agv_trajs, get_agv_pos, AGV_SPEED, _nf)
    if not wps:
        return False
    # time-parameterize the new path (AGV_SPEED m/s); each hold_s is a duplicate
    # point at a later time, which smooth_traj's Catmull-Rom keeps constant
    t0 = clocks[cid]
    new_traj = []
    prev = None
    acc = t0
    for (x, y, hold) in wps:
        if prev is not None:
            acc += np.hypot(x - prev[0], y - prev[1]) / AGV_SPEED
        new_traj.append((round(acc, 3), float(x), float(y)))
        if hold > 1e-4:
            acc += hold
            new_traj.append((round(acc, 3), float(x), float(y)))
        prev = (x, y)
    agv_trajs[cid] = new_traj
    agv_smooth[cid] = smooth_traj(new_traj)
    write_current_plan()
    print(f"\n  [REPLAN] car {cid}: {len(wps)} wps -> {len(new_traj)} pts", flush=True)
    return True


def predict_conflict(cid, clocks, horizon=3.0, step=0.3):
    """True if AGV `cid`'s near-future path (time-aligned) will come within
    AGV_CONFLICT_DIST of another MOVING AGV's near-future path.  The active
    replan scan uses this to reroute AGVs BEFORE they reach the repulsion force
    range.

    Near a goal stop, only a STATIONARY sibling is exempt (shared parking — the
    runtime _agv_block handles a parked sibling there).  A MOVING sibling
    crossing near the goal must still be flagged: the old blanket 1.5 m
    exemption hid exactly the AGV0(橙,east@y7.38)<->AGV2(绿,west@y7.12) head-on
    crossing at LM003 (plans converging to 0.32 m) and forced the repulsion to
    saturate at 0.70 m for ~1.5 s."""
    goal = car_goals[cid][0] if car_goals[cid] else None
    for ta in np.arange(step, horizon + 1e-6, step):
        ax, ay = get_agv_pos(cid, clocks[cid] + ta * AGV_SPEED)
        near_goal = goal is not None and np.hypot(ax - goal[0], ay - goal[1]) < 1.5
        for ocid in agv_trajs:
            if ocid == cid:
                continue
            if clocks[ocid] >= agv_trajs[ocid][-1][0]:
                continue   # other AGV parked — runtime _agv_block handles it
            ox, oy = get_agv_pos(ocid, clocks[ocid] + ta * AGV_SPEED)
            if np.hypot(ax - ox, ay - oy) < AGV_CONFLICT_DIST:
                if near_goal:
                    # shared parking only: exempt a STATIONARY sibling here
                    # (parked / mid-yield — _agv_block handles it); a MOVING
                    # crossing sibling is a real conflict and still flagged.
                    ox2, oy2 = get_agv_pos(ocid, clocks[ocid] + (ta + 0.2) * AGV_SPEED)
                    if np.hypot(ox2 - ox, oy2 - oy) < 0.05:
                        continue
                return True
    return False


# ── MuJoCo setup ────────────────────────────────────────────────────────────
SCENE_PATH = os.path.join(ROOT, "model", "bxi_elf3", "bxi_elf3_scene_factory.xml")
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)
# Reset to keyframe
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
else:
    mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# AGV mocap addresses (mocap bodies — position set via data.mocap_pos)
agv_adr = {}
for cid in agv_trajs:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"agv_{cid}")
    agv_adr[cid] = model.body_mocapid[bid]

# ── Robot control setup ─────────────────────────────────────────────────────
POLICY_PATH = os.path.join(ROOT, "model", "bxi_elf3", "model_normal.onnx")
session = ort.InferenceSession(POLICY_PATH, providers=["CPUExecutionProvider"])

kp = np.array(EnvConfig.kp, dtype=np.float32)
kd = np.array(EnvConfig.kd, dtype=np.float32)
action_scale = np.array(EnvConfig.action_scale, dtype=np.float32)
num_actions = EnvConfig.num_actions
num_obs = EnvConfig.num_obs
control_decimation = EnvConfig.control_decimation

default_angles = data.qpos[7:7+num_actions].copy()
obs_history = np.zeros((num_obs,), dtype=np.float32)
last_action = np.zeros(num_actions, dtype=np.float32)
target_dof_pos = default_angles.copy()


def quat_rotate_inverse(q, v):
    q_w, q_vec = q[-1], q[:3]
    a = v * (2.0*q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def get_robot_pose():
    x, y = data.qpos[0], data.qpos[1]
    qw, qx, qy, qz = data.qpos[3:7]
    siny = 2.0*(qw*qz + qx*qy)
    cosy = 1.0 - 2.0*(qy*qy + qz*qz)
    yaw = np.arctan2(siny, cosy)
    return float(x), float(y), float(yaw)


def robot_step(cmd_vx, cmd_vy, cmd_omega):
    """One control step: PD + policy, using Navigator-style cmd."""
    global obs_history, last_action, target_dof_pos
    # PD control to hold target positions
    tau = (target_dof_pos - data.qpos[7:7+num_actions]) * kp \
          + (np.zeros_like(kp) - data.qvel[6:6+num_actions]) * kd
    data.ctrl[:num_actions] = tau

    # Physics step
    mujoco.mj_step(model, data)

    # Policy inference every control_decimation steps
    counter = getattr(robot_step, "counter", 0)
    robot_step.counter = counter + 1
    if counter % control_decimation == 0:
        qj = data.qpos[7:7+num_actions] - default_angles
        dqj = data.qvel[6:6+num_actions]
        omega = data.qvel[3:6].astype(np.float64)
        grav = quat_rotate_inverse(
            data.sensor("Body_Quat").data[[1,2,3,0]].astype(np.float64),
            np.array([0, 0, -1]))
        obs = np.concatenate([omega, grav, qj, dqj, last_action,
                              np.array([cmd_vx, cmd_vy, cmd_omega])]).astype(np.float32)
        out = session.run(None, {"obs": obs.reshape(1, num_obs)})
        last_action = out[-1].reshape(-1)
        target_dof_pos = last_action * action_scale + default_angles


# ── Spatial waypoint navigator — reuses navigate.py's proven Navigator ───
from navigation.navigate import Navigator as _PtPNav

class SpatialNav:
    """Walks [(x, y), ...] waypoints using the standard Navigator."""

    def __init__(self, path, fwd_speed=0.7, turn_speed=1.2, arrival=0.4):
        self.path = path  # [(x, y), ...]
        self.idx = 0
        self.fwd_speed = fwd_speed
        self.turn_speed = turn_speed
        self.arrival = arrival
        self.arrived = False
        self._nav = None
        self.offset = np.zeros(2)  # world-frame dodge shift on the active target
        self._r_wait = -9.0        # robot stop-wait arming (see robot-stop block)
        self._make_nav()

    def _make_nav(self):
        if self.idx < len(self.path):
            wx, wy = self.path[self.idx]
            self._nav = _PtPNav(wx, wy, fwd_speed=self.fwd_speed,
                                turn_speed=self.turn_speed)
        else:
            self._nav = None

    def update(self, x, y, yaw):
        if self.idx >= len(self.path):
            self.arrived = True
            return np.zeros(3, dtype=np.float32)

        wx, wy = self.path[self.idx]
        dist = np.hypot(wx - x, wy - y)

        # Advance waypoint by spatial proximity (measured against the REAL
        # waypoint, so a dodge offset never makes the robot advance early)
        if dist < self.arrival:
            if self.idx < len(self.path) - 1:
                self.idx += 1
                self._make_nav()
                wx, wy = self.path[self.idx]
                print(f"  WP {self.idx}/{len(self.path)}: ({wx:.2f},{wy:.2f})")
            else:
                # Reached final waypoint — done
                self.arrived = True
                return np.zeros(3, dtype=np.float32)

        # Apply dodge offset by shifting the navigator's active target.
        # The navigator then steers around the AGV using its proven omega
        # control (the walking policy barely tracks vy, but tracks steering).
        if self._nav is not None:
            self._nav.target[0] = wx + self.offset[0]
            self._nav.target[1] = wy + self.offset[1]
        return self._nav.update(x, y, yaw)


def arc_cost(collisions, shelf_coll, falls, arrived, sim_time, min_dist):
    """Scalar cost for one ARC run — minimize this (used by tune_arc.py).

    Time is the PRIMARY objective: among runs that pass, the fastest wins.
    The hard terms (falls / collisions / shelf collisions / not arriving) are
    absolute rejects — no amount of speed justifies a contact or a missed goal.
    """
    c = 1000.0 * falls + 500.0 * collisions + 300.0 * shelf_coll
    if not arrived:
        c += 1000.0
    c += 3.0 * sim_time                     # time — the main thing to minimize
    c += 100.0 * max(0.0, 0.45 - min_dist)  # soft margin fine below contact dist
    return c


# ── Run ─────────────────────────────────────────────────────────────────────
print("Launching...  ESC to quit\n")
# Remove duplicate consecutive waypoints
dedup = []
for wp in humanoid_path:
    if not dedup or abs(wp[0] - dedup[-1][0]) > 0.01 or abs(wp[1] - dedup[-1][1]) > 0.01:
        dedup.append(wp)
print(f"  Spatial waypoints: {len(dedup)}")
nav = SpatialNav(dedup)

_viewer_ctx = (contextlib.nullcontext() if _HEADLESS else
               mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                            show_right_ui=False))
with _viewer_ctx as viewer:
    if not _HEADLESS:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (7.5, 5.0, 1.0)
        viewer.cam.distance = 18.0
        viewer.cam.elevation = -35
        viewer.cam.azimuth = 180

    sim_time = 0.0
    dt = model.opt.timestep
    SPEED = 10     # simulation speed multiplier
    clocks = {cid: 0.0 for cid in agv_trajs}
    lateral = {cid: np.zeros(2) for cid in agv_trajs}

    # ── AGV-yield state ──
    stopped_since = {cid: None for cid in agv_trajs}  # time an AGV began waiting
    _st_last = -9.0   # last station-table print (throttle to ~2 s)
    write_current_plan()   # initial current_plan.json
    # Teleport robot to start position (LM005 = 1,2 — clear of AGVs)
    data.qpos[0] = 1.0
    data.qpos[1] = 2.0
    data.qpos[2] = 1.1  # standing height
    mujoco.mj_forward(model, data)
    default_angles = data.qpos[7:7+num_actions].copy()
    target_dof_pos = default_angles.copy()
    robot_step.counter = 0
    print(f"Robot teleported to ({data.qpos[0]:.1f}, {data.qpos[1]:.1f})")

    n_yields = 0       # number of AGV-stop-for-robot events
    n_replans = 0
    n_agv_force = 0    # times the moving-moving AGV-AGV repulsion engaged
    force_events = []  # (sim_time, a, b, dist, push) — every AGV-AGV force frame
    n_robot_avoid = 0  # times the AGV-side robot-repulsion safety net engaged
    _scan_last = -9.0  # active-replan scan throttle (~0.5 s)
    _last_replan = {}  # cid -> sim_time of its last replan (cooldown)
    _move_agv = None   # a NORMAL stopped AGV parked on the robot's waypoint —
                       # it is rerouted to MOVE OUT (挪开) and unblock the robot
    n_collisions = 0
    n_static_col = 0
    n_falls = 0
    min_agv_dist = 999.0
    # robot collision geoms (group 3) and static obstacle geoms (obs_*_geom)
    robot_geoms = set(g for g in range(model.ngeom) if model.geom_group[g] == 3)
    static_geom_ids = set()
    for _i in range(6):
        _gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{_i}_geom")
        if _gid >= 0:
            static_geom_ids.add(_gid)
    while True:
        if not _HEADLESS and not viewer.is_running():
            break
        # headless: run until robot arrives (or --full window) / hard timeout
        if _HEADLESS and (not _ARGS.full and nav.arrived
                          or sim_time > (max_t + 45) / AGV_SPEED):
            break
        # AGV plan is scaled by AGV_SPEED (clocks advance at 0.7 m/s), so the
        # reset window must be too — previously robot was teleported mid-walk.
        if sim_time > (max_t + 45) / AGV_SPEED:
            sim_time = 0.0
            clocks = {c: 0.0 for c in agv_trajs}
            lateral = {c: np.zeros(2) for c in agv_trajs}
            stopped_since = {c: None for c in agv_trajs}
            _last_replan.clear()
            _scan_last = -9.0
            _move_agv = None
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[0] = 1.0; data.qpos[1] = 2.0  # back to LM005
            mujoco.mj_forward(model, data)
            robot_step.counter = 0
            target_dof_pos[:] = default_angles
            last_action[:] = 0.0
            obs_history[:] = 0.0

        # ── AGV positions ──
        pp = {cid: np.array(get_agv_pos(cid, clocks[cid])) for cid in agv_trajs}
        stationary = {}
        for cid, traj in agv_trajs.items():
            t = clocks[cid]
            # Stationary: before departure, after arrival, OR dwelling mid-chain
            stationary[cid] = (t <= traj[0][0] or t >= traj[-1][0]
                               or int(round(t)) in agv_stationary_times[cid])

        dyn_obs = set(STATIC_OBS)          # coarse cells (static + parked AGVs)
        dyn_agv = set()                    # only parked-AGV cells (for AGV-AGV)
        for cid in agv_trajs:
            if stationary[cid]:
                p = pp[cid] + lateral[cid]
                cell = (int(round(p[0])), int(round(p[1])))
                dyn_obs.add(cell)
                dyn_agv.add(cell)

        # robot pose + velocity (robot has priority — AGVs yield to it)
        rx_now, ry_now, _ = get_robot_pose()
        _rvx, _rvy = float(data.qvel[0]), float(data.qvel[1])

        for cid in agv_trajs:
            traj = agv_trajs[cid]
            t = clocks[cid]
            if t >= traj[-1][0]:
                continue
            next_wp = prev_wp = None
            for i in range(len(traj)):
                if traj[i][0] > t:
                    next_wp = np.array([traj[i][1], traj[i][2]], dtype=float)
                    if i > 0:
                        prev_wp = np.array([traj[i-1][1], traj[i-1][2]], dtype=float)
                    break
            if next_wp is None:
                continue
            # Static obstacle check uses the REAL box geometry (a replanned
            # fine-grid waypoint rounding to a coarse cell would otherwise be
            # mis-flagged as inside a static obstacle → AGV stuck forever).
            _in_box = any(abs(next_wp[0] - cx) < hx + 0.05 and
                          abs(next_wp[1] - cy) < hy + 0.05
                          for (cx, cy, hx, hy) in OBS_BOXES)
            nx, ny = int(round(next_wp[0])), int(round(next_wp[1]))
            # AGV-AGV blocking: next cell held by a stationary AGV
            _agv_block = (nx, ny) in dyn_agv and not stationary[cid]
            blocked = (_in_box or _agv_block)

            if blocked and prev_wp is not None:
                move_dir = next_wp - prev_wp
                perp = np.array([-move_dir[1], move_dir[0]])
                m = np.linalg.norm(perp)
                if m > 0.01:
                    perp /= m
                    for oid in agv_trajs:
                        if oid != cid and stationary[oid]:
                            opos = pp[oid] + lateral[oid]
                            if np.linalg.norm(opos - next_wp) < 0.5:
                                if np.dot(perp, opos - (pp[cid]+lateral[cid])) < 0:
                                    perp = -perp
                                break
                    lateral[cid] += perp * 0.08
                stopped_since[cid] = None   # AGV-AGV block, not robot wait
            else:
                # ── AGV-YIELD: the ROBOT has priority.  If the robot sits near
                # this AGV's near-future path, the AGV STOPS (clock paused).
                rblock = robot_blocks(cid, rx_now, ry_now, _rvx, _rvy, clocks)
                if rblock:
                    if stopped_since[cid] is None:
                        stopped_since[cid] = sim_time
                        n_yields += 1
                        print(f"\n  [YIELD] car {cid} stops for robot at "
                              f"({rx_now:.1f},{ry_now:.1f})", flush=True)
                    elif sim_time - stopped_since[cid] > REPLAN_TIMEOUT:
                        # waited long enough — replan around the robot
                        if replan_car(cid, rx_now, ry_now, clocks):
                            n_replans += 1
                        stopped_since[cid] = None
                else:
                    if stopped_since[cid] is not None:
                        # robot has moved clear — RESUME the original plan instead
                        # of replanning.  Replanning while the robot is still only
                        # ~1-2 m away treats its cell as an obstacle and forces a
                        # detour (e.g. the 66-pt south-dip to the map bottom).  The
                        # long REPLAN_TIMEOUT above remains the safety net for a
                        # genuinely stuck AGV (robot parked on its path).
                        stopped_since[cid] = None
                    clocks[cid] += dt * SPEED * AGV_SPEED

            # reached the next goal stop? advance the remaining-goals list
            if car_goals[cid]:
                gx, gy = car_goals[cid][0]
                if np.hypot(gx - pp[cid][0], gy - pp[cid][1]) < 0.6:
                    car_goals[cid].pop(0)

        # NOTE: no separate "stationary AGVs always advance" loop here —
        # loop 1 above already advances the clock for every non-blocked AGV
        # (incl. dwelling ones), so a second loop would DOUBLE-advance the
        # clock and break the plan's time scale (AGVs end up 2× fast).

        # ── Active replan layer: keep the plan mutually avoiding at runtime so
        # the AGV-AGV repulsion below never needs to engage.  Every RE_SCAN_INT
        # s, replan any AGV whose near-future path (time-aligned) is predicted
        # to cross a MOVING sibling — reroute it BEFORE it reaches force range.
        # (A wait-in-place variant was tried: freezing the AGV while a sibling
        # crosses makes it an INVISIBLE obstacle to the robot — the robot walks
        # into it (23 robot-AGV collisions) — so we detour instead.)
        if ACTIVE_REPLAN and sim_time - _scan_last > RE_SCAN_INT:
            _scan_last = sim_time
            for cid in agv_trajs:
                if clocks[cid] >= agv_trajs[cid][-1][0]:
                    continue   # parked / done
                lr = _last_replan.get(cid, -9.0)
                if sim_time - lr < AGV_REPLAN_COOLDOWN:
                    continue
                if predict_conflict(cid, clocks):
                    if replan_car(cid, rx_now, ry_now, clocks, robot_avoid=False):
                        n_replans += 1
                        _last_replan[cid] = sim_time

        for cid in agv_trajs:
            my = pp[cid] + lateral[cid]
            for (ox, oy) in STATIC_OBS:
                diff = my - np.array([ox, oy], dtype=float)
                d = np.linalg.norm(diff)
                if d < SAFE_DIST and d > 0.01:
                    lateral[cid] += (diff/d) * (SAFE_DIST - d) * 1.5

        # ── Moving-moving repulsion: PURE LAST RESORT — smooth + replan-consistent.
        # Radius 0.45 is just above the 0.4 m body contact, so the replan (C_F,
        # primary) keeps AGVs apart and this only nudges on a genuine prediction
        # error.  Two upgrades over the old linear-hard force:
        #   - smoothstep falloff (C1 at the boundary — no force step, no jitter);
        #   - direction consistency: each AGV is pushed LATERALLY (perpendicular
        #     to its planned heading) AWAY from the other — the same axis the
        #     replan's B' side-shift uses — so the force never fights the replan's
        #     chosen detour side nor its timing (no forward/back push).  Head-on
        #     (lateral ≈ 0) falls back to a radial nudge.
        cids = list(agv_trajs)
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = cids[i], cids[j]
                pa = pp[a] + lateral[a]
                pb = pp[b] + lateral[b]
                d = np.linalg.norm(pa - pb)
                if d < AGV_AVOID_DIST and d > 0.01:
                    n_agv_force += 1
                    # smoothstep normalized over the [contact, boundary] band:
                    # f=1 at body contact (max push), f=0 at 0.45 (slope 0 → C1,
                    # no force step at the activation radius).
                    x = np.clip((d - AGV_BODY_R) / (AGV_AVOID_DIST - AGV_BODY_R),
                                0.0, 1.0)
                    f = 1.0 - x * x * (3.0 - 2.0 * x)
                    push = AGV_AVOID_AMP * 0.06 * f     # steady-state ≤ AMP
                    force_events.append((sim_time, a, b, d, push,
                                         stationary[a], stationary[b],
                                         float(pa[0]), float(pa[1]),
                                         float(pb[0]), float(pb[1])))
                    # --no-agv-force: count-only (A/B) — the pure C_F replan must
                    # separate AGVs itself, so the force is never applied.
                    if not ENABLE_AGV_FORCE:
                        continue
                    for aid, bid in ((a, b), (b, a)):
                        _pa = pp[aid] + lateral[aid]
                        _pb = pp[bid] + lateral[bid]
                        _dx, _dy = _pa[0] - _pb[0], _pa[1] - _pb[1]
                        _hx, _hy = agv_heading(aid, clocks)
                        # signed lateral component of the away-from-bid vector in
                        # aid's frame (perp = (-hy, hx)); >0 => bid on aid's LEFT
                        _lat = -_hy * _dx + _hx * _dy
                        if abs(_lat) > 1e-3:
                            # push along the lateral direction that widens the gap:
                            # sign(lat) * perp,  where perp=(-hy,hx) is the left
                            # normal of the planned heading.  (The opposite sign
                            # pushed AGVs INWARD — measured min_d collapsed to
                            # 0.01 m, i.e. AGVs drove through each other.)
                            _sgn = 1.0 if _lat > 0 else -1.0
                            lateral[aid] += np.array([-_hy * _sgn, _hx * _sgn]) * push
                        else:
                            # head-on: lateral undefined -> radial nudge
                            _m = np.hypot(_dx, _dy)
                            if _m > 1e-6:
                                lateral[aid] += np.array([_dx, _dy]) / _m * (push * 0.5)

        # ── Robot-repulsion safety net: AGVs never overlap the robot body ──
        # (primary avoidance is robot_blocks stop + replan around; this is a
        # last-resort steer for a replanned path that grazes the robot before
        # the next frame's stop check engages).  Smooth C1 falloff over
        # ROBOT_AVOID_R, repulsing from the robot's PREDICTED position so the
        # push direction stays stable as it walks; amplitude pre-divided by the
        # ×0.94 decay gain (≈1/0.06) so the steady-state offset is bounded at
        # ROBOT_AVOID_AMP instead of the naive unbounded 16.7× accumulation.
        for cid in agv_trajs:
            pa = pp[cid] + lateral[cid]
            # repulse from where the robot will BE in LOOKAHEAD seconds
            px_ = rx_now + _rvx * ROBOT_AVOID_LOOKAHEAD
            py_ = ry_now + _rvy * ROBOT_AVOID_LOOKAHEAD
            dx, dy = pa[0] - px_, pa[1] - py_
            d = np.hypot(dx, dy)
            if d < ROBOT_AVOID_R and d > 0.01:
                n_robot_avoid += 1
                x = d / ROBOT_AVOID_R
                f = 1.0 - x * x * (3.0 - 2.0 * x)        # smoothstep: C1, f(0)=1, f(1)=0
                lateral[cid] += (np.array([dx, dy]) / d) * (ROBOT_AVOID_AMP * 0.06 * f)

        # ── AGVs follow their plan; the HUMAN does the yielding ──
        # (robot-side repulsion handles avoidance — see below)

        for cid in agv_trajs:
            lateral[cid] *= 0.94
            if np.linalg.norm(lateral[cid]) < 0.005:
                lateral[cid] = np.zeros(2)

        for cid in agv_trajs:
            x, y = pp[cid] + lateral[cid]
            a = agv_adr[cid]
            data.mocap_pos[a][0] = x
            data.mocap_pos[a][1] = y
            data.mocap_pos[a][2] = AGV_Z

        # ── Station occupancy / queue table (every 2 s) ──
        # parked  = AGV stopped within 0.6 m of the landmark
        # queue   = AGV whose NEXT goal is this landmark and is within 2 m
        if sim_time - _st_last > 2.0:
            _st_last = sim_time
            print(f"\n  [STATIONS] t={sim_time:6.1f}s", flush=True)
            for _nm, (_lx, _ly) in LM_COORDS.items():
                _parked, _queue = [], []
                for cid in agv_trajs:
                    _pp0 = pp[cid]
                    _ax, _ay = _pp0[0] + lateral[cid][0], _pp0[1] + lateral[cid][1]
                    _d = np.hypot(_ax - _lx, _ay - _ly)
                    _nxt = get_agv_pos(cid, clocks[cid] + 0.2 * AGV_SPEED)
                    _stpd = np.hypot(_nxt[0] - _pp0[0], _nxt[1] - _pp0[1]) < 0.1
                    if _d < 0.6 and _stpd:
                        _parked.append(cid)
                    elif car_goals[cid] and _d < 2.0 and \
                            np.hypot(car_goals[cid][0][0] - _lx,
                                     car_goals[cid][0][1] - _ly) < 0.3:
                        _queue.append(cid)
                if _parked or _queue:
                    print(f"    {_nm}: parked={_parked}  queue={_queue}", flush=True)

        # ── Humanoid robot ──
        # ROBOT HAS PRIORITY: AGVs yield to it.  The robot keeps moving, BUT:
        #   - an AGV stopped / queuing AT the robot's next waypoint → the robot
        #     STOPS and waits briefly;
        #   - an AGV stopped ON the robot's path (passable) → the robot
        #     DETOURS around it (shifts the nav target perpendicular away).
        rx, ry, ryaw = get_robot_pose()
        vx, vy, omega = nav.update(rx, ry, ryaw)
        # ROBOT STOPS when a stopped AGV occupies its current waypoint (the
        # "AGV stopped / queuing ahead" case the user wants).  The stop is
        # limited to STOP_WAIT seconds to break the yield-deadlock: the AGV
        # replans after its own timeout, then the robot proceeds.
        r_blocked = False
        if not nav.arrived and ROBOT_STOP:
            wx, wy = nav.path[nav.idx]
            wp_dist = np.hypot(wx - rx, wy - ry)
            # Work-zone logic: a WORKING or QUEUING AGV at the robot's waypoint
            # makes the robot WAIT (WORK_WAIT s) — it must not interrupt the
            # docked work / let it pull in first.  A NORMAL stopped AGV gets only
            # a brief STOP_WAIT pause (bounded, so no robot↔AGV deadlock): the
            # AGV-side yield resolves it and the robot then passes.
            work_state = None      # 'working' / 'queuing'
            normal_stop_cid = None
            for cid in agv_trajs:
                # Use the PLAN position (pp) for both stopped-ness and waypoint
                # proximity — the lateral shift is only runtime dodge, so an
                # AGV parked exactly ON the waypoint must still be detected.
                _pp = pp[cid]
                _d_wp = np.hypot(_pp[0] - wx, _pp[1] - wy)
                st = agv_is_working(cid, _pp, clocks)
                if st == "working" and _d_wp < 1.2:
                    work_state = "working"
                    break
                if st == "queuing" and _d_wp < 2.0:
                    work_state = "queuing"
                    break
                _nxt = get_agv_pos(cid, clocks[cid] + 0.2 * AGV_SPEED)
                is_stopped = (stopped_since[cid] is not None or
                              np.hypot(_nxt[0] - _pp[0], _nxt[1] - _pp[1]) < 0.1)
                if is_stopped and _d_wp < 0.45:
                    normal_stop_cid = cid
            if work_state and wp_dist < 1.5:
                _move_agv = None   # working/queuing AGV is NOT told to move — the robot waits
                if nav._r_wait < 0.0:          # first blocked frame — start the wait
                    nav._r_wait = sim_time
                if sim_time - nav._r_wait < WORK_WAIT:
                    r_blocked = True
            elif normal_stop_cid is not None and wp_dist < 1.0:
                # a NORMAL stopped AGV is parked on the robot's waypoint — tell it
                # to MOVE OUT (挪开): it gets the short replan timeout and reroutes
                # around the stopped robot, unblocking the waypoint.
                if _move_agv != normal_stop_cid:
                    # immediately reroute the blocking AGV so it can MOVE OUT —
                    # its new path avoids the robot, and robot_blocks() then lets
                    # it go (only flagged if the reroute SUCCEEDS — otherwise it
                    # stays frozen for safety and we retry next frame).  Uses the
                    # SANE spatial A* (plan_spatial_path), NOT the time-aware
                    # replan_agv_car_time — that one detours whole-map (the
                    # 133/113-pt wandering) and pushed car0 up to y=9.6 + 9 shelf
                    # hits.
                    _cid0 = normal_stop_cid
                    _cur0 = list(get_agv_pos(_cid0, clocks[_cid0]))
                    # route through ALL remaining goals (like replan_car) — a
                    # single-goal reroute TRUNCATES the chain and the AGV parks at
                    # the first stop forever (the orange-ball-stuck bug).
                    _dyn0 = {(int(round(rx)), int(round(ry)))}
                    _path0 = []
                    _ok0 = True
                    for _g0 in car_goals[_cid0]:
                        _leg0 = _nf.replan_agv_car(_cur0, _g0, COARSE_OBS,
                                                   _dyn0, MAP_W, MAP_H)
                        if not _leg0:
                            _ok0 = False
                            break
                        _path0 = _path0 + _leg0 if not _path0 else _path0 + _leg0[1:]
                        _cur0 = _g0
                    if _ok0 and _path0:
                            _t0 = clocks[_cid0]
                            _nt = []
                            _prev = None
                            _acc = _t0
                            for (px, py) in _path0:
                                if _prev is not None:
                                    _acc += np.hypot(px - _prev[0],
                                                     py - _prev[1]) / AGV_SPEED
                                _nt.append((round(_acc, 3), float(px), float(py)))
                                _prev = (px, py)
                            agv_trajs[_cid0] = _nt
                            agv_smooth[_cid0] = smooth_traj(_nt)
                            n_replans += 1
                            _move_agv = _cid0
                if nav._r_wait < 0.0:
                    nav._r_wait = sim_time
                if sim_time - nav._r_wait < STOP_WAIT:
                    r_blocked = True
            else:
                _move_agv = None
                nav._r_wait = -9.0             # disarm
        if r_blocked:
            vx, vy, omega = 0.0, 0.0, 0.0

        # static-obstacle repulsion (real box geometry) — mild safety net
        repel = static_repulsion(rx, ry) * 2.5
        _rmag = np.hypot(*repel)
        if _rmag > 3.0:
            repel *= 3.0 / _rmag

        # AGV close-range repulsion — push laterally into FREE space only.  A
        # blind radial push sums with the shelf repulsion and shoves the robot
        # INTO the shelf when an AGV passes on the shelf side (measured: 5
        # sub-mm shelf grazes at LM002).  So push sideways, away from the AGV,
        # capped by the free lateral room on that side (lateral_free) — never
        # into a shelf body.
        cl, sl = np.cos(ryaw), np.sin(ryaw)
        for cid in agv_trajs:
            ax_pos = pp[cid] + lateral[cid]
            dx, dy = rx - ax_pos[0], ry - ax_pos[1]
            dist = np.hypot(dx, dy)
            safe = 0.6
            if dist < safe and dist > 0.01:
                force = (safe - dist) / safe
                # signed lateral component (along the robot's left axis) of the
                # away-from-AGV push (dx,dy): + = push to the robot's LEFT.
                push_y = -sl * dx + cl * dy
                # cap it by the free room on the push side (never into a shelf)
                free_l, free_r = lateral_free(rx, ry, (cl, sl))
                cap = free_l if push_y > 0 else free_r
                push_y = push_y * min(1.0, cap * 2.0) * force
                repel[0] += (-sl * push_y) * 4.0
                repel[1] += (cl * push_y) * 4.0

        mag = np.hypot(*repel)
        if mag > 3.0:
            repel *= 3.0 / mag
        cos_y, sin_y = np.cos(ryaw), np.sin(ryaw)
        fy = -sin_y * repel[0] + cos_y * repel[1]
        fx = cos_y * repel[0] + sin_y * repel[1]
        vy += fy * 0.8
        omega += fy * 0.5
        if fx > 0.5:
            vx = max(0.0, vx - fx * 0.5)

        for _ in range(SPEED):
            robot_step(vx, vy, omega)

        # ── REAL collision + fall detection (right after mj_step) ──
        # MuJoCo already computed data.contact during robot_step's mj_step.
        # AGV sphere geoms have default group 0; robot collision geoms group 3.
        agv_geom_id_by_cid = {}
        for cid in agv_trajs:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"agv_{cid}_geom")
            agv_geom_id_by_cid[gid] = cid
        for c in data.contact:
            g1, g2 = c.geom1, c.geom2
            agv_cid = agv_geom_id_by_cid.get(g1)
            if agv_cid is None:
                agv_cid = agv_geom_id_by_cid.get(g2)
            if agv_cid is not None:
                other_g = g2 if agv_geom_id_by_cid.get(g1) is not None else g1
                if model.geom_group[other_g] == 3:  # robot collision geom
                    n_collisions += 1
                    ap = pp[agv_cid] + lateral[agv_cid]
                    print(f"\n  !! REAL COLLISION t={sim_time:.1f}s  robot <-> AGV{agv_cid}"
                          f"  pen={c.dist:.4f}m  at ({data.qpos[0]:.1f},{data.qpos[1]:.1f})"
                          f"  AGV at ({ap[0]:.2f},{ap[1]:.2f})  clock={clocks[agv_cid]:.2f}"
                          f"  lat=({lateral[agv_cid][0]:+.2f},{lateral[agv_cid][1]:+.2f})",
                          flush=True)

        # ── Static obstacle collisions (robot colliding with shelves/equip) ──
        for c in data.contact:
            g1, g2 = c.geom1, c.geom2
            rg = g1 if g1 in robot_geoms else (g2 if g2 in robot_geoms else None)
            sg = g2 if g1 in static_geom_ids else (g1 if g2 in static_geom_ids else None)
            if rg is not None and sg is not None:
                n_static_col += 1
                print(f"\n  !! SHELF COLLISION t={sim_time:.1f}s"
                      f"  pen={c.dist:.4f}m  at ({data.qpos[0]:.1f},{data.qpos[1]:.1f})",
                      flush=True)

        # min robot-AGV distance (for ARC on/off comparison)
        for cid in agv_trajs:
            ap = pp[cid] + lateral[cid]
            d = np.hypot(rx - ap[0], ry - ap[1])
            if d < min_agv_dist:
                min_agv_dist = d

        # Fall detection — require SIGNIFICANT height drop (not gait tilt).
        # Normal walking z ≈ 0.9-1.1; fallen ≈ 0.2-0.5. Use z < 0.6.
        torso_z = data.qpos[2]
        if torso_z < 0.6:
            n_falls += 1
            print(f"\n  !! FELL t={sim_time:.1f}s  z={torso_z:.2f}"
                  f"  pos=({data.qpos[0]:.2f},{data.qpos[1]:.2f})"
                  f"  wp={nav.idx}/{len(nav.path)}", flush=True)

        # Re-assert AGV positions (mocap — no physics drift)
        for cid in agv_trajs:
            x, y = pp[cid] + lateral[cid]
            a = agv_adr[cid]
            data.mocap_pos[a][0] = x
            data.mocap_pos[a][1] = y
            data.mocap_pos[a][2] = AGV_Z

        # ── Real-time speed print ──
        robot_speed = np.linalg.norm(data.qvel[0:3])  # base linear velocity
        if int(sim_time * 5) % 10 == 0:  # ~every 2 sim-seconds
            if not nav.arrived and nav.idx < len(nav.path):
                wp = nav.path[nav.idx]
                wp_str = f"({wp[0]:.1f},{wp[1]:.1f})"
            else:
                wp_str = "done"
            print(f"\r  t={sim_time:>5.1f}s  robot_speed={robot_speed:.3f} m/s  "
                  f"cmd=({vx:.2f},{vy:.2f},{omega:.2f})  "
                  f"pos=({rx:.2f},{ry:.2f})  wp={wp_str}  ", end="", flush=True)

        if not _HEADLESS:
            viewer.sync()
        sim_time += dt * SPEED

    if _HEADLESS:
        print("\n" + "=" * 56)
        print("HEADLESS SUMMARY")
        print("=" * 56)
        print(f"AGV yields   : {n_yields}")
        print(f"AGV replans  : {n_replans}")
        print(f"AGV-AGV force: {n_agv_force}  (want ~0 — active replan should")
        print(f"                 reroute before the repulsion ever engages)")
        print(f"robot-avoid  : {n_robot_avoid}")
        # ── Abnormal force analysis: group force frames into continuous
        # "episodes" per AGV pair (consecutive frames < 0.06 s apart) and flag
        # the abnormal ones (persistent >5 frames or a very close pass <0.5 m).
        if force_events:
            episodes = []
            seq = {}
            _stat_frames = 0   # force frames where >=1 AGV is stationary
            for (t, a, b, d, push, sa, sb, ax, ay, bx, by) in force_events:
                if sa or sb:
                    _stat_frames += 1
                key = (a, b) if a < b else (b, a)
                ei = seq.get(key)
                if ei is not None and t - episodes[ei][2] <= 0.06:
                    episodes[ei][2] = t
                    episodes[ei][3] += 1
                    if d < episodes[ei][4]:
                        episodes[ei][4] = d
                        episodes[ei][7] = (ax, ay, bx, by)
                    episodes[ei][5] = max(episodes[ei][5], push)
                    episodes[ei][6] |= (sa or sb)
                    continue
                episodes.append([key, t, t, 1, d, push, (sa or sb), (ax, ay, bx, by)])
                seq[key] = len(episodes) - 1
            print(f"  [FORCE-ANALYSIS] frames={len(force_events)} "
                  f"episodes={len(episodes)}  "
                  f"stationary-involved={_stat_frames}")
            for (key, t0, t1, frames, md, mp, _st, _pos) in episodes:
                flag = "  <-- abnormal" if (frames >= 8 or md < 0.50 or _st) else ""
                print(f"      AGV{key[0]}<->AGV{key[1]}  t={t0:.1f}~{t1:.1f}s  "
                      f"frames={frames:>3}  min_d={md:.3f}  max_push={mp:.3f}"
                      f"  stat={int(_st)}  at A=({_pos[0]:.1f},{_pos[1]:.1f})"
                      f" B=({_pos[2]:.1f},{_pos[3]:.1f}){flag}")
        print(f"collisions   : {n_collisions}")
        print(f"shelf_coll   : {n_static_col}")
        print(f"falls        : {n_falls}")
        print(f"min AGV dist : {min_agv_dist:.3f} m")
        print(f"arrived      : {nav.arrived}  wp={nav.idx}/{len(nav.path)}")
        print(f"final pos    : ({data.qpos[0]:.2f},{data.qpos[1]:.2f})  "
              f"t={sim_time:.1f}s")
        _cost = arc_cost(n_collisions, n_static_col, n_falls, nav.arrived,
                         sim_time, min_agv_dist)
        print(f"COST         : {_cost:.2f}")
        for cid in agv_trajs:
            traj = agv_trajs[cid]
            cx, cy = get_agv_pos(cid, clocks[cid])
            print(f"  AGV{cid}: clock={clocks[cid]:.1f}/{traj[-1][0]:.1f}"
                  f"  pos=({cx:.2f},{cy:.2f})  goals_left={car_goals[cid]}"
                  f"  parked={clocks[cid] >= traj[-1][0]}")
