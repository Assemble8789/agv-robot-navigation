"""
Full factory simulation: AGVs + humanoid robot navigation.

Usage:
  python navigation/run_factory_full.py
"""

import json, os, sys, argparse, contextlib
import numpy as np
import mujoco, mujoco.viewer

_PARSER = argparse.ArgumentParser()
_PARSER.add_argument("--headless", action="store_true",
                     help="no viewer; print ARC diagnostics and exit")
_PARSER.add_argument("--no-arc", action="store_true",
                     help="disable predictive soft path deformation (A/B test)")
_PARSER.add_argument("--arc", default="",
                     help="override ARC params, e.g. 'vy_dodge=0.7,hold=2.0'")
_ARGS = _PARSER.parse_args()
_HEADLESS = _ARGS.headless
NO_ARC = _ARGS.no_arc

# ── Tunable ARC parameters (auto-tuned by navigation/tune_arc.py) ──
ARC_PARAMS = dict(
    vy_dodge=0.5,      # mild lateral assist while yielding (m/s) — tuned
    vx_yield=0.2,      # creep forward speed while yielding (m/s) — tuned
    lat_thresh=0.5,    # min-future-distance trigger (m): only real contact
                       # threats (<0.5, contact=0.45) — avoids dodging safe
                       # 0.5-0.7 m pass-bys (the "don't move, it'll pass" case)
    lookahead=2.0,     # prediction horizon (s)
    hold=2.5,          # yield duration (s) — tuned
)
if _ARGS.arc:
    for _kv in _ARGS.arc.split(","):
        if "=" in _kv:
            _k, _v = _kv.split("=", 1)
            if _k in ARC_PARAMS:
                ARC_PARAMS[_k] = float(_v)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(ROOT, "agv_simulation-main", "maps")

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
    AGV_SPEED = 0.7  # AGV speed factor (1.0 = 1 m/s as planned; 0.7 = 0.7 m/s)
    clocks = {cid: 0.0 for cid in agv_trajs}
    lateral = {cid: np.zeros(2) for cid in agv_trajs}

    # ── Predictive soft-path-deformation state ──
    OFF_AMP = 0.25      # max lateral offset (m)
    OFF_SPEED = 0.35    # lateral offset velocity (m/s)
    vy_cmd = 0.0        # smoothed lateral-velocity command (vy dodge)
    # Teleport robot to start position (LM005 = 1,2 — clear of AGVs)
    data.qpos[0] = 1.0
    data.qpos[1] = 2.0
    data.qpos[2] = 1.1  # standing height
    mujoco.mj_forward(model, data)
    default_angles = data.qpos[7:7+num_actions].copy()
    target_dof_pos = default_angles.copy()
    robot_step.counter = 0
    print(f"Robot teleported to ({data.qpos[0]:.1f}, {data.qpos[1]:.1f})")

    arc_events = []   # (t, cid, close, lat, sign) of every ARC trigger
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
        # headless: run until robot arrives or hard timeout
        if _HEADLESS and (nav.arrived or sim_time > (max_t + 45) / AGV_SPEED):
            break
        # AGV plan is scaled by AGV_SPEED (clocks advance at 0.7 m/s), so the
        # reset window must be too — previously robot was teleported mid-walk.
        if sim_time > (max_t + 45) / AGV_SPEED:
            sim_time = 0.0
            clocks = {c: 0.0 for c in agv_trajs}
            lateral = {c: np.zeros(2) for c in agv_trajs}
            vy_cmd = 0.0
            nav.offset = np.zeros(2)
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

        dyn_obs = set(STATIC_OBS)
        for cid in agv_trajs:
            if stationary[cid]:
                p = pp[cid] + lateral[cid]
                dyn_obs.add((int(round(p[0])), int(round(p[1]))))

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
            nx, ny = int(round(next_wp[0])), int(round(next_wp[1]))
            blocked = (nx, ny) in dyn_obs and not stationary[cid]

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
            else:
                clocks[cid] += dt * SPEED * AGV_SPEED

        # NOTE: no separate "stationary AGVs always advance" loop here —
        # loop 1 above already advances the clock for every non-blocked AGV
        # (incl. dwelling ones), so a second loop would DOUBLE-advance the
        # clock and break the plan's time scale (AGVs end up 2× fast).

        for cid in agv_trajs:
            my = pp[cid] + lateral[cid]
            for (ox, oy) in STATIC_OBS:
                diff = my - np.array([ox, oy], dtype=float)
                d = np.linalg.norm(diff)
                if d < SAFE_DIST and d > 0.01:
                    lateral[cid] += (diff/d) * (SAFE_DIST - d) * 1.5

        # ── Moving-moving repulsion: any two AGVs too close push apart ──
        cids = list(agv_trajs)
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = cids[i], cids[j]
                pa = pp[a] + lateral[a]
                pb = pp[b] + lateral[b]
                d = np.linalg.norm(pa - pb)
                if d < 0.7 and d > 0.01:
                    push_dir = (pa - pb) / d
                    push = (0.7 - d) * 0.5
                    lateral[a] += push_dir * push
                    lateral[b] -= push_dir * push

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

        # ── Humanoid robot ──
        rx, ry, ryaw = get_robot_pose()
        vx, vy, omega = nav.update(rx, ry, ryaw)
        # Clear any previous-frame dodge offset; the ARC block below sets it
        # again while dodging.  This guarantees the offset never persists
        # after the dodge (e.g. during turn-in-place when vx <= 0.05).
        nav.offset[:] = 0.0

        # ── Level 2: Predictive Soft Path Deformation (smooth lateral dodge) ──
        # When an AGV will cross the robot's forward corridor in the next
        # ~1s, inject a SMOOTH lateral offset via vy (no vx braking, no
        # omega spikes).  Robot walks a gentle arc past the AGV, then
        # returns.  Small amplitude + hysteresis = robot stays stable.
        if not NO_ARC and vx > 0.05 and not nav.arrived:
            heading = np.array([np.cos(ryaw), np.sin(ryaw)])
            robot_pos = np.array([rx, ry])

            # ── Time-aligned threat detection (path-projected + approach) ──
            # Project the robot along its ACTUAL waypoint path (not a straight
            # line — the robot turns, so a straight-line projection misses real
            # threats) at its forward speed, and measure the min future
            # robot-AGV distance.  Require the AGV to be APPROACHING (distance
            # at the end of the lookahead < distance now), which filters
            # lane-mates that merely pass near the robot's line while pulling
            # AHEAD (e.g. AGV1 at t=6.2: along 2.1→3.5, never closes).
            best_lat = 999.0      # min future robot-AGV distance (m)
            best_sign = 0
            best_cid = None
            best_along = 0.0
            best_dist = 999.0
            now_t = sim_time
            _v_now = max(vx, 0.3)   # robot forward speed for path projection
            _path = nav.path
            _pi = nav.idx
            for cid in agv_trajs:
                min_d = 999.0
                along_at_min = 0.0
                d_first = None
                d_last = None
                for t_ahead in np.arange(0.0, ARC_PARAMS['lookahead'] + 0.01, 0.5):
                    ax, ay = get_agv_pos(cid, clocks[cid] + t_ahead * AGV_SPEED)
                    # robot position after traveling vx*t_ahead along the path
                    px, py = rx, ry
                    _rem = _v_now * t_ahead
                    _i = _pi
                    while _rem > 0 and _i < len(_path):
                        _tx, _ty = _path[_i]
                        _seg = np.hypot(_tx - px, _ty - py)
                        if _seg <= _rem:
                            _rem -= _seg
                            px, py = _tx, _ty
                            _i += 1
                        else:
                            if _seg > 0:
                                _f = _rem / _seg
                                px += (_tx - px) * _f
                                py += (_ty - py) * _f
                            _rem = 0
                    d = float(np.hypot(ax - px, ay - py))
                    if d_first is None:
                        d_first = d
                    d_last = d
                    if d < min_d:
                        min_d = d
                        along_at_min = float((ax-rx)*heading[0] + (ay-ry)*heading[1])
                # approach filter: reject AGVs pulling AHEAD (lane-mates far
                # ahead that never time-align close) — but KEEP one that is
                # already within contact range even if not approaching, because
                # a parallel course at ~0.3 m offset grazes (constant distance,
                # d_last ≈ d_first, so a strict "approaching" test wrongly
                # filters it and the robot rubs the AGV).
                CONTACT_D = 0.45   # robot half-width + AGV radius
                if d_first is not None and (d_last < d_first or min_d < CONTACT_D) \
                        and min_d < best_lat:
                    best_lat = min_d
                    best_cid = cid
                    best_along = along_at_min
                    best_dist = min_d

            # Dodge AWAY from the AGV's CURRENT side.  cross>0 → AGV on the
            # RIGHT → dodge LEFT (+vy, away from it).  (Dodging into the open
            # side was tried — the AGV usually comes from the open corridor
            # centre, so that dodged INTO it → AGV collisions.)
            if best_cid is not None:
                ax, ay = get_agv_pos(best_cid, clocks[best_cid])
                rel = np.array([ax, ay]) - robot_pos
                cross_now = float(rel[0]*heading[1] - rel[1]*heading[0])
                best_sign = np.sign(cross_now) if cross_now != 0 else best_sign

            # Hysteresis: once triggered, hold even if detection flickers
            if not hasattr(nav, '_arc_until'):
                nav._arc_until = -9.0
                nav._arc_sign = 0
            in_dodge = now_t < nav._arc_until

            if not in_dodge and best_cid is not None and best_lat < ARC_PARAMS['lat_thresh']:
                # NEW dodge only — commit direction & duration so the robot
                # makes ONE clean arc (re-evaluating every frame makes the
                # sign flip and the offset never builds up).
                nav._arc_sign = best_sign
                nav._arc_until = now_t + ARC_PARAMS['hold']
                if not hasattr(nav, '_arc_last_log') or now_t - nav._arc_last_log > 0.4:
                    nav._arc_last_log = now_t
                    arc_events.append((now_t, best_cid, best_lat, best_sign))
                    # AGV world positions at 0/1/2s + robot heading, to judge
                    # whether the dodge was really needed.
                    _agv_world = []
                    _rel_prof = []
                    for _ta in (0.0, 1.0, 2.0):
                        _ax, _ay = get_agv_pos(best_cid, clocks[best_cid] + _ta * AGV_SPEED)
                        _agv_world.append(f"({_ax:.1f},{_ay:.1f})")
                        _rel = np.array([_ax, _ay]) - robot_pos
                        _rel_prof.append((float(_rel[0]*heading[0] + _rel[1]*heading[1]),
                                          float(_rel[0]*heading[1] - _rel[1]*heading[0])))
                    _cp = [f"{a:.1f}/{c:+.2f}" for a, c in _rel_prof]
                    print(f"\n  [ARC] t={now_t:6.1f}s  AGV{best_cid}"
                          f"  lat={best_lat:.2f}m  along={best_along:.2f}m"
                          f"  dist={best_dist:.2f}m  sign={best_sign:+.0f}"
                          f"  hdg={np.degrees(ryaw):+.0f}deg"
                          f"  AGV_world={_agv_world[0]} {_agv_world[1]} {_agv_world[2]}"
                          f"  (along/cross)={_cp[0]} {_cp[1]} {_cp[2]}"
                          f"  robot=({rx:.2f},{ry:.2f})", flush=True)

            # ── Yield: creep + small vy dodge ──
            # NOTE: full stopping was tried (a "stop when crossing" branch) and
            # made it WORSE (0→33 collisions on the default): with 4 AGVs always
            # circulating, a stationary robot becomes a target for the OTHER
            # AGVs even when the triggering one passes safely.  Creeping at the
            # minimum walkable speed keeps the robot moving (not a sitting duck)
            # while the AGV crosses — this is the safe middle ground.
            if in_dodge:
                vx = min(vx, ARC_PARAMS['vx_yield'])
                vy_target = nav._arc_sign * ARC_PARAMS['vy_dodge']
            else:
                vy_target = 0.0
            vy_cmd += (vy_target - vy_cmd) * 0.4
            vy += vy_cmd

            if int(now_t * 5) % 10 == 0:
                if best_cid is not None:
                    print(f"\r  [ARC] vy={vy_cmd:+.2f} vy_tgt={vy_target:+.2f}"
                          f"  AGV{best_cid} lat={best_lat:.2f} along={best_along:.2f}"
                          f"  dist={best_dist:.2f}  dodging={'Y' if in_dodge else 'N'}"
                          f"  vx={vx:.2f}", end="")
                else:
                    print(f"\r  [ARC] vy={vy_cmd:+.2f}  no threat  vx={vx:.2f}", end="")

        # Repulsion from static obstacles (real box geometry) — a gentle
        # safety net only.  The primary shelf avoidance is the amplitude-capped
        # dodge below (dodges toward the open side, limited by free space), so
        # this repulsion must stay weak enough not to destabilise the gait.
        repel = static_repulsion(rx, ry) * 2.5
        _rmag = np.hypot(*repel)
        if _rmag > 3.0:
            repel *= 3.0 / _rmag

        # AGV close-range repulsion (emergency — < 0.5m)
        for cid in agv_trajs:
            ax_pos = pp[cid] + lateral[cid]
            dx, dy = rx - ax_pos[0], ry - ax_pos[1]
            dist = np.hypot(dx, dy)
            safe = 0.5
            if dist < safe and dist > 0.01:
                force = (safe - dist) / safe
                repel += (np.array([dx, dy]) / dist) * force * 5.0

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
        print(f"ARC triggers : {len(arc_events)}")
        for ev in arc_events[:20]:
            print(f"  t={ev[0]:6.1f}s  AGV{ev[1]}  lat={ev[2]:.2f}m  "
                  f"sign={ev[3]:+.0f}")
        print(f"collisions   : {n_collisions}")
        print(f"shelf_coll   : {n_static_col}")
        print(f"falls        : {n_falls}")
        print(f"min AGV dist : {min_agv_dist:.3f} m")
        print(f"arrived      : {nav.arrived}  wp={nav.idx}/{len(nav.path)}")
        print(f"final pos    : ({data.qpos[0]:.2f},{data.qpos[1]:.2f})  "
              f"t={sim_time:.1f}s")
        _cost = arc_cost(n_collisions, n_static_col, n_falls, nav.arrived,
                         sim_time, min_agv_dist)
        print(f"COST         : {_cost:.2f}   params={ARC_PARAMS}")
