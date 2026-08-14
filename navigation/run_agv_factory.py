"""
AGV factory runtime — smooth trajectories with obstacle avoidance.

- Catmull-Rom splines smooth the grid waypoints into curved paths.
- Post-processed to stay clear of static obstacles.
- Per-AGV clock pauses when blocked by stationary AGVs.
- Runtime obstacle + AGV repulsion as safety fallback.

Usage:
  python navigation/run_agv_factory.py
"""

import json, os
import numpy as np
import mujoco, mujoco.viewer

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(ROOT, "agv_simulation-main", "maps")
SCENE_PATH = os.path.join(ROOT, "model", "bxi_elf3", "bxi_elf3_scene_factory.xml")

_plans = sorted([f for f in os.listdir(MAPS_DIR)
                 if f.startswith("plan_v2_") and f.endswith(".json")], reverse=True)
if not _plans:
    _plans = sorted([f for f in os.listdir(MAPS_DIR)
                     if f.startswith("plan_") and f.endswith(".json")], reverse=True)
if not _plans:
    raise SystemExit("No plan file in maps/")

# ── Data ────────────────────────────────────────────────────────────────────
with open(os.path.join(MAPS_DIR, _plans[0])) as f:
    plan = json.load(f)

agv_trajs = {}
for car in plan["cars"]:
    cid = car["car_id"]
    agv_trajs[cid] = [(p["t"], p["x"], p["y"]) for p in car["trajectory"]]
max_t = max(p[-1][0] for p in agv_trajs.values())
print(f"Plan: {_plans[0]}  AGVs: {len(agv_trajs)}  max_t: {max_t}")

# Precompute dwell (stationary mid-chain) times per AGV
agv_stationary_times = {}
for cid, traj in agv_trajs.items():
    parked = set()
    pos_prev = None
    for i, (t, x, y) in enumerate(traj):
        if i == 0:
            pos_prev = (x, y)
            continue
        if (x, y) == pos_prev:
            parked.add(t)
        pos_prev = (x, y)
    agv_stationary_times[cid] = parked

# ── Constants ───────────────────────────────────────────────────────────────
GROUND_Z, AGV_R = -0.16, 0.2
AGV_Z = GROUND_Z + AGV_R
SAFE_DIST = 0.8  # shelf half-size(0.5) + AGV radius(0.2) + margin(0.1)

STATIC_OBS = {
    (3,5),(3,8),(4,5),(4,8),(5,5),(5,8),
    (6,1),(6,2),(7,1),(7,2),
    (8,5),(8,8),(9,5),(9,8),(10,5),(10,8),
}

# ── Smooth trajectories ─────────────────────────────────────────────────────
def catmull_rom_smooth(waypoints, num_samples=200):
    pts = np.array([(p[1], p[2]) for p in waypoints], dtype=float)
    times = np.array([p[0] for p in waypoints], dtype=float)
    if len(pts) < 2:
        return times, pts[:,0], pts[:,1]
    # Augment
    pa = np.vstack([pts[0]*2 - pts[1], pts, pts[-1]*2 - pts[-2]])
    ta = np.concatenate([[times[0]-1], times, [times[-1]+1]])
    t_s = np.linspace(times[0], times[-1], num_samples)
    xs, ys, seg = [], [], 0
    for t in t_s:
        while seg < len(times)-1 and t > times[seg+1]:
            seg += 1
        seg = min(seg, len(times)-2)
        i = seg + 1
        p0, p1, p2, p3 = pa[i-1], pa[i], pa[i+1], pa[i+2]
        t0, t2 = ta[i], ta[i+1]
        alpha = (t - t0) / (t2 - t0) if t2 > t0 else 0
        alpha = max(0.0, min(1.0, alpha))
        a, a2, a3 = alpha, alpha*alpha, alpha*alpha*alpha
        r = 0.5 * ((2*p1) + (-p0+p2)*a + (2*p0-5*p1+4*p2-p3)*a2 + (-p0+3*p1-3*p2+p3)*a3)
        xs.append(r[0]); ys.append(r[1])
    xa, ya = np.array(xs), np.array(ys)
    # Push away from obstacles
    for i in range(len(xa)):
        for (ox, oy) in STATIC_OBS:
            dx, dy = xa[i] - ox, ya[i] - oy
            d = np.hypot(dx, dy)
            if d < SAFE_DIST and d > 0.01:
                xa[i] += (dx/d)*(SAFE_DIST - d)
                ya[i] += (dy/d)*(SAFE_DIST - d)
    return t_s, xa, ya

agv_smooth = {}
for cid, traj in agv_trajs.items():
    agv_smooth[cid] = catmull_rom_smooth(traj)

def get_pos(cid, t):
    t_s, x_s, y_s = agv_smooth[cid]
    if t <= t_s[0]: return float(x_s[0]), float(y_s[0])
    if t >= t_s[-1]: return float(x_s[-1]), float(y_s[-1])
    idx = np.searchsorted(t_s, t)
    idx = max(1, min(idx, len(t_s)-1))
    f = (t - t_s[idx-1]) / (t_s[idx] - t_s[idx-1]) if t_s[idx] > t_s[idx-1] else 0
    return float(x_s[idx-1] + f*(x_s[idx] - x_s[idx-1])), float(y_s[idx-1] + f*(y_s[idx] - y_s[idx-1]))

# ── MuJoCo setup ────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)
agv_adr = {}
for cid in agv_trajs:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"agv_{cid}")
    agv_adr[cid] = model.body_mocapid[bid]

# ── Run ─────────────────────────────────────────────────────────────────────
print("Launching MuJoCo viewer...  ESC to quit\n")
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = (7.5, 5.0, 1.0)
    viewer.cam.distance = 16.0
    viewer.cam.elevation = -35
    viewer.cam.azimuth = 180

    sim_time = 0.0
    dt = model.opt.timestep
    SPEED = 10      # simulation speed multiplier
    AGV_SPEED = 0.7  # AGV speed factor (1.0 = 1 m/s as planned; 0.7 = 0.7 m/s)
    clocks = {cid: 0.0 for cid in agv_trajs}
    lateral = {cid: np.zeros(2) for cid in agv_trajs}

    while viewer.is_running():
        if sim_time > max_t + 5:
            sim_time = 0.0
            clocks = {c: 0.0 for c in agv_trajs}
            lateral = {c: np.zeros(2) for c in agv_trajs}

        pp = {cid: np.array(get_pos(cid, clocks[cid])) for cid in agv_trajs}
        stationary = {}
        for cid, traj in agv_trajs.items():
            t = clocks[cid]
            stationary[cid] = (t <= traj[0][0] or t >= traj[-1][0]
                               or int(round(t)) in agv_stationary_times[cid])

        # Dynamic obstacle set
        dyn_obs = set(STATIC_OBS)
        for cid in agv_trajs:
            if stationary[cid]:
                p = pp[cid] + lateral[cid]
                dyn_obs.add((int(round(p[0])), int(round(p[1]))))

        # Per-AGV: block if next cell occupied
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
            my = pp[cid] + lateral[cid]
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
                                if np.dot(perp, opos - my) < 0:
                                    perp = -perp
                                break
                    lateral[cid] += perp * 0.08
            else:
                clocks[cid] += dt * SPEED * AGV_SPEED

        # Stationary AGVs always advance clock
        for cid in agv_trajs:
            if stationary[cid] and clocks[cid] < agv_trajs[cid][-1][0]:
                clocks[cid] += dt * SPEED * AGV_SPEED

        # Obstacle repulsion (runtime safety)
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

        # Decay
        for cid in agv_trajs:
            lateral[cid] *= 0.94
            if np.linalg.norm(lateral[cid]) < 0.005:
                lateral[cid] = np.zeros(2)

        # Apply (mocap bodies)
        for cid in agv_trajs:
            x, y = pp[cid] + lateral[cid]
            a = agv_adr[cid]
            data.mocap_pos[a][0] = x
            data.mocap_pos[a][1] = y
            data.mocap_pos[a][2] = AGV_Z

        mujoco.mj_forward(model, data)
        viewer.sync()
        sim_time += dt * SPEED
