"""Headless: run full sim, detect REAL MuJoCo collisions (data.contact)."""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from simple_env import EnvConfig
import onnxruntime as ort
import mujoco

MAPS_DIR = os.path.join(ROOT, "agv_simulation-main", "maps")

# ── Load AGV plan ──
plans = sorted([f for f in os.listdir(MAPS_DIR)
                if f.startswith("plan_v2_") and f.endswith(".json")], reverse=True)
with open(os.path.join(MAPS_DIR, plans[0])) as f:
    plan = json.load(f)
agv_trajs = {car["car_id"]: [(p["t"], p["x"], p["y"]) for p in car["trajectory"]]
             for car in plan["cars"]}

# ── Load humanoid plan ──
with open(os.path.join(MAPS_DIR, "humanoid_plan.json")) as f:
    hplan = json.load(f)
humanoid_path = [(p["x"], p["y"]) for p in hplan["waypoints"]]
print(f"Plan: {plans[0]}  humanoid waypoints: {len(humanoid_path)}")

STATIC_OBS = {(3,5),(3,8),(4,5),(4,8),(5,5),(5,8),
              (6,1),(6,2),(7,1),(7,2),(8,5),(8,8),(9,5),(9,8),(10,5),(10,8)}

# ── AGV smoothing ──
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
    return t_s, np.array(xs), np.array(ys)

agv_smooth = {cid: smooth_traj(traj) for cid, traj in agv_trajs.items()}

def get_agv_pos(cid, t):
    ts, xs, ys = agv_smooth[cid]
    if t <= ts[0]: return float(xs[0]), float(ys[0])
    if t >= ts[-1]: return float(xs[-1]), float(ys[-1])
    idx = max(1, min(np.searchsorted(ts, t), len(ts)-1))
    f = (t-ts[idx-1])/(ts[idx]-ts[idx-1]) if ts[idx]>ts[idx-1] else 0
    return float(xs[idx-1]+f*(xs[idx]-xs[idx-1])), float(ys[idx-1]+f*(ys[idx]-ys[idx-1]))

# ── MuJoCo setup ──
SCENE_PATH = os.path.join(ROOT, "model/bxi_elf3/bxi_elf3_scene_factory.xml")
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
else:
    mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# Get AGV geom ids for collision check
agv_geom_ids = {}
for cid in agv_trajs:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"agv_{cid}_geom")
    agv_geom_ids[cid] = gid

# Robot torso geom (first collision geom on torso_link)
robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
robot_geom_ids = set()
for g in range(model.body_geomadr[robot_body_id],
               model.body_geomadr[robot_body_id] + model.body_geomnum[robot_body_id]):
    robot_geom_ids.add(g)
# Also include all robot collision geoms (group 3) — legs, arms
robot_geom_ids = set()
for g in range(model.ngeom):
    if model.geom_group[g] == 3:  # collision group
        robot_geom_ids.add(g)

# Teleport robot to LM005
data.qpos[0] = 1.0; data.qpos[1] = 2.0; data.qpos[2] = 1.1
mujoco.mj_forward(model, data)

# ── Robot control ──
POLICY_PATH = os.path.join(ROOT, "model/bxi_elf3/model_normal.onnx")
session = ort.InferenceSession(POLICY_PATH, providers=["CPUExecutionProvider"])
kp = np.array(EnvConfig.kp, dtype=np.float32)
kd = np.array(EnvConfig.kd, dtype=np.float32)
action_scale = np.array(EnvConfig.action_scale, dtype=np.float32)
num_actions = EnvConfig.num_actions
num_obs = EnvConfig.num_obs
control_decimation = EnvConfig.control_decimation

default_angles = data.qpos[7:7+num_actions].copy()
target_dof_pos = default_angles.copy()
last_action = np.zeros(num_actions, dtype=np.float32)

def quat_rotate_inverse(q, v):
    q_w, q_vec = q[-1], q[:3]
    a = v * (2.0*q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def robot_step(cmd):
    global target_dof_pos, last_action
    tau = (target_dof_pos - data.qpos[7:7+num_actions]) * kp \
          + (np.zeros_like(kp) - data.qvel[6:6+num_actions]) * kd
    data.ctrl[:num_actions] = tau
    mujoco.mj_step(model, data)
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
                              np.array(cmd)]).astype(np.float32)
        out = session.run(None, {"obs": obs.reshape(1, num_obs)})
        last_action = out[-1].reshape(-1)
        target_dof_pos = last_action * action_scale + default_angles

def get_robot_pose():
    x, y = data.qpos[0], data.qpos[1]
    qw, qx, qy, qz = data.qpos[3:7]
    siny = 2.0*(qw*qz + qx*qy)
    cosy = 1.0 - 2.0*(qy*qy + qz*qz)
    yaw = np.arctan2(siny, cosy)
    return float(x), float(y), float(yaw)

# ── Navigator ──
from navigation.navigate import Navigator as PtPNav
class SpatialNav:
    def __init__(self, path, fwd_speed=0.7):
        self.path = path
        self.idx = 0
        self.fwd_speed = fwd_speed
        self.arrived = False
        self._nav = None
        self._make()
    def _make(self):
        if self.idx < len(self.path):
            wx, wy = self.path[self.idx]
            self._nav = PtPNav(wx, wy, fwd_speed=self.fwd_speed)
        else:
            self._nav = None
    def update(self, x, y, yaw):
        if self.idx >= len(self.path):
            self.arrived = True
            return np.zeros(3, dtype=np.float32)
        wx, wy = self.path[self.idx]
        if np.hypot(wx-x, wy-y) < 0.4 and self.idx < len(self.path)-1:
            self.idx += 1
            self._make()
        return self._nav.update(x, y, yaw)

nav = SpatialNav(humanoid_path)

# ── Main loop with REAL collision detection ──
SPEED = 10
AGV_SPEED = 0.7
dt = model.opt.timestep
sim_time = 0.0
clocks = {cid: 0.0 for cid in agv_trajs}

collision_events = []  # (sim_time, robot_geom, agv_geom)
min_dist = 999.0
fell = False
fall_log = []

for step in range(4000):
    # AGV clocks advance
    for cid in agv_trajs:
        clocks[cid] += dt * SPEED * AGV_SPEED

    # Set AGV positions
    for cid in agv_trajs:
        ax, ay = get_agv_pos(cid, clocks[cid])
        # find qpos addr for AGV freejoint
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"agv_{cid}")
        jnt = model.body_jntadr[bid]
        adr = model.jnt_qposadr[jnt]
        data.qpos[adr+0] = ax; data.qpos[adr+1] = ay
        data.qpos[adr+2] = -0.16 + 0.2
        data.qpos[adr+3] = 1.0; data.qpos[adr+4:adr+7] = 0.0

    mujoco.mj_forward(model, data)

    # ── REAL COLLISION CHECK via data.contact ──
    for c in data.contact:
        g1, g2 = c.geom1, c.geom2
        # robot geom vs AGV geom?
        in_robot = g1 in robot_geom_ids or g2 in robot_geom_ids
        is_agv1 = g1 in agv_geom_ids.values()
        is_agv2 = g2 in agv_geom_ids.values()
        if in_robot and (is_agv1 or is_agv2):
            agv_cid = [cid for cid, gid in agv_geom_ids.items()
                       if gid == (g1 if is_agv1 else g2)][0]
            collision_events.append((sim_time, agv_cid, c.dist))
            if len(collision_events) <= 20:
                print(f"\n  ⚠ COLLISION t={sim_time:.1f}s robot-AGV{agv_cid}"
                      f"  penetration={c.dist:.4f}m", flush=True)

    # ── Robot control ──
    rx, ry, ryaw = get_robot_pose()
    # min dist to any AGV (position-based)
    for cid in agv_trajs:
        ax, ay = get_agv_pos(cid, clocks[cid])
        d = np.hypot(rx-ax, ry-ay)
        if d < min_dist: min_dist = d

    vx, vy, omega = nav.update(rx, ry, ryaw)

    # ── STOP-AND-WAIT avoidance (same as run_factory_full) ──
    if vx > 0.05 and not nav.arrived and nav.idx < len(nav.path):
        heading = np.array([np.cos(ryaw), np.sin(ryaw)])
        wx, wy = nav.path[nav.idx]
        wp_dist = np.hypot(wx - rx, wy - ry)
        wp_eta = wp_dist / max(vx, 0.1)
        block = False
        for cid in agv_trajs:
            for dt_off in (-1.0, 0.0, 1.0, 2.0):
                agv_t = clocks[cid] + wp_eta + dt_off
                if agv_t < 0:
                    continue
                ax, ay = get_agv_pos(cid, agv_t)
                if np.hypot(wx - ax, wy - ay) < 0.9:
                    block = True
                    break
            if block:
                break
        if block:
            vx *= 0.2

    if int(sim_time) % 10 == 0:
        print(f"\r  t={sim_time:.0f}s  pos=({rx:.2f},{ry:.2f})  vx={vx:.2f}"
              f"  wp={nav.idx}/{len(nav.path)}  block={block if 'block' in dir() else '-'}", end="")

    robot_step([vx, vy, omega])
    sim_time += dt * SPEED

    # ── FALL detection: significant height collapse (real fall) ──
    torso_z = data.qpos[2]
    if torso_z < 0.6:
        if not fell:
            fell = True
            fall_log.append((sim_time, rx, ry, torso_z, data.qpos[3], nav.idx))
            print(f"\n  !! FELL at t={sim_time:.1f}s pos=({rx:.2f},{ry:.2f})"
                  f" z={torso_z:.2f} qw={data.qpos[3]:.2f} wp={nav.idx}", flush=True)
            break

    if nav.arrived:
        print(f"\n  ✓ Robot arrived at {sim_time:.1f}s")
        break

print(f"\n=== COLLISION RESULT ===")
print(f"sim_time: {sim_time:.1f}s  arrived: {nav.arrived}  waypoints: {nav.idx}/{len(humanoid_path)}")
print(f"REAL collisions (MuJoCo contact): {len(collision_events)}")
for t, cid, pen in collision_events[:10]:
    print(f"  t={t:.1f}s  robot vs AGV{cid}  penetration={pen:.4f}")
print(f"Min robot-AGV center distance: {min_dist:.3f} m")
print(f"Robot final pos: ({rx:.2f}, {ry:.2f})")
if fell:
    print(f"\n!! ROBOT FELL: t={fall_log[0][0]:.1f}s pos=({fall_log[0][1]:.2f},{fall_log[0][2]:.2f})"
          f" z={fall_log[0][3]:.2f} qw={fall_log[0][4]:.2f} wp={fall_log[0][5]}")
