"""Headless test: does predictive avoidance detect AGVs approaching?"""
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

# ── AGV smoothing (same as run_factory_full) ──
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

# Teleport to LM005
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

def robot_step(cmd_vx, cmd_vy, cmd_omega):
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
                              np.array([cmd_vx, cmd_vy, cmd_omega])]).astype(np.float32)
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

# ── Simple waypoint follower ──
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

# ── Main loop ──
SPEED = 10
AGV_SPEED = 0.7
dt = model.opt.timestep
sim_time = 0.0
clocks = {cid: 0.0 for cid in agv_trajs}
OFF_AMP = 0.25
OFF_SPEED = 0.35
off_cur = 0.0

detect_count = 0
detect_log = []
min_agv_dist = 999.0
collisions = 0

for step in range(2000):
    # AGV clocks
    for cid in agv_trajs:
        clocks[cid] += dt * SPEED * AGV_SPEED

    rx, ry, ryaw = get_robot_pose()
    vx, vy, omega = nav.update(rx, ry, ryaw)

    # ── Predictive detection (CPA + hysteresis hold) ──
    if vx > 0.05 and not nav.arrived:
        heading = np.array([np.cos(ryaw), np.sin(ryaw)])
        robot_pos = np.array([rx, ry])

        best_lat = 999.0
        best_along = 999.0
        best_sign = 0
        best_cid = None
        for cid in agv_trajs:
            min_lat = 999.0
            min_along = 999.0
            cross_at_min = 0.0
            for t_ahead in np.arange(0.5, 3.1, 0.5):
                agv_x, agv_y = get_agv_pos(cid, clocks[cid] + t_ahead)
                rel = np.array([agv_x, agv_y]) - robot_pos
                along = float(np.dot(rel, heading))
                cross = float(rel[0]*heading[1] - rel[1]*heading[0])
                lat = abs(cross)
                if lat < min_lat:
                    min_lat = lat
                    min_along = along
                    cross_at_min = cross
            if min_along > 0.2 and min_lat < best_lat:
                best_lat = min_lat
                best_along = min_along
                best_sign = np.sign(cross_at_min)
                best_cid = cid

        # Hysteresis hold: once triggered, hold offset for 2.5s
        if not hasattr(nav, '_hold_until'):
            nav._hold_until = -9.0
            nav._held_sign = 0

        if best_cid is not None and best_lat < 0.7 and best_along < 4.0:
            nav._held_sign = best_sign
            nav._hold_until = sim_time + 2.5
            detect_count += 1
            if detect_count <= 10:
                detect_log.append((sim_time, best_cid, best_lat, best_along, best_sign))

        if sim_time < nav._hold_until:
            off_target = nav._held_sign * OFF_AMP
        else:
            off_target = 0.0

        err = off_target - off_cur
        vy_extra = np.clip(err * 2.0, -OFF_SPEED, OFF_SPEED)
        vy += vy_extra
        off_cur += vy_extra * dt * SPEED

        if int(sim_time) % 2 == 0:
            print(f"\r  t={sim_time:.1f}s  off={off_cur:+.3f}  tgt={off_target:+.2f}"
                  f"  vy={vy_extra:+.3f}  detect={best_cid}  lat={best_lat:.2f}"
                  f"  hold={nav._hold_until-sim_time:.1f}s", end="")

    robot_step(vx, vy, omega)
    sim_time += dt * SPEED

    # Track closest robot-AGV distance
    for cid in agv_trajs:
        ax, ay = get_agv_pos(cid, clocks[cid])
        d = np.hypot(rx - ax, ry - ay)
        if d < min_agv_dist:
            min_agv_dist = d
        if d < 0.6:  # collision / near miss
            collisions += 1

    if nav.arrived:
        break

print(f"\n=== RESULT ===")
print(f"sim_time: {sim_time:.1f}s  arrived: {nav.arrived}  waypoints done: {nav.idx}/{len(humanoid_path)}")
print(f"Detections: {detect_count}")
for t, cid, ta, along, cross in detect_log:
    print(f"  t={t:.1f}s  AGV{cid} lat={ta:.2f}  along={along:.2f} sign={cross:+.0f}")
print(f"final robot pos: ({rx:.2f}, {ry:.2f})")
print(f"final off_cur: {off_cur:.3f}")
print(f"MIN AGV-robot distance: {min_agv_dist:.3f} m")
print(f"Near-miss/collision frames (<0.6m): {collisions}")
