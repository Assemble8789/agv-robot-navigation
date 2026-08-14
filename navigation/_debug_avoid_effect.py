"""Quantify avoidance effectiveness: run sim with ARC on vs off, compare collisions."""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from simple_env import EnvConfig
import onnxruntime as ort
import mujoco

MAPS_DIR = os.path.join(ROOT, "agv_simulation-main", "maps")
plans = sorted([f for f in os.listdir(MAPS_DIR)
                if f.startswith("plan_v2_") and f.endswith(".json")], reverse=True)
with open(os.path.join(MAPS_DIR, plans[0])) as f:
    plan = json.load(f)
agv_trajs = {car["car_id"]: [(p["t"], p["x"], p["y"]) for p in car["trajectory"]]
             for car in plan["cars"]}
with open(os.path.join(MAPS_DIR, "humanoid_plan.json")) as f:
    hplan = json.load(f)
humanoid_path = [(p["x"], p["y"]) for p in hplan["waypoints"]]

STATIC_OBS = {(3,5),(3,8),(4,5),(4,8),(5,5),(5,8),
              (6,1),(6,2),(7,1),(7,2),(8,5),(8,8),(9,5),(9,8),(10,5),(10,8)}

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

# Precompute AGV stationary times
agv_stationary_times = {}
for cid, traj in agv_trajs.items():
    parked = set()
    pos_prev = None
    for i, (t, x, y) in enumerate(traj):
        if i == 0:
            pos_prev = (x, y); continue
        if (x, y) == pos_prev:
            parked.add(t)
        pos_prev = (x, y)
    agv_stationary_times[cid] = parked

def run_sim(with_arc):
    model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "model/bxi_elf3/bxi_elf3_scene_factory.xml"))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    data.qpos[0] = 1.0; data.qpos[1] = 2.0; data.qpos[2] = 1.1
    mujoco.mj_forward(model, data)

    # AGV geom ids
    agv_geom_ids = {}
    for cid in agv_trajs:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"agv_{cid}_geom")
        agv_geom_ids[cid] = gid
    robot_geoms = set(g for g in range(model.ngeom) if model.geom_group[g] == 3)

    session = ort.InferenceSession(os.path.join(ROOT, "model/bxi_elf3/model_normal.onnx"),
                                   providers=["CPUExecutionProvider"])
    kp=np.array(EnvConfig.kp,dtype=np.float32); kd=np.array(EnvConfig.kd,dtype=np.float32)
    action_scale=np.array(EnvConfig.action_scale,dtype=np.float32)
    num_actions=EnvConfig.num_actions; num_obs=EnvConfig.num_obs; cd_=EnvConfig.control_decimation
    default_angles=data.qpos[7:7+num_actions].copy()
    target=default_angles.copy(); last=np.zeros(num_actions,dtype=np.float32)

    def quat_rotate_inverse(q,v):
        q_w,q_vec=q[-1],q[:3]
        a=v*(2.0*q_w*q_w-1.0); b=np.cross(q_vec,v)*q_w*2.0; c=q_vec*np.dot(q_vec,v)*2.0
        return a-b+c

    def robot_step(cmd):
        nonlocal target, last
        tau=(target-data.qpos[7:7+num_actions])*kp + (np.zeros_like(kp)-data.qvel[6:6+num_actions])*kd
        data.ctrl[:num_actions]=tau
        mujoco.mj_step(model,data)
        cnt = getattr(robot_step,"counter",0)
        robot_step.counter = cnt+1
        if cnt%cd_==0:
            qj=data.qpos[7:7+num_actions]-default_angles
            dqj=data.qvel[6:6+num_actions]
            omega=data.qvel[3:6].astype(np.float64)
            grav=quat_rotate_inverse(data.sensor("Body_Quat").data[[1,2,3,0]].astype(np.float64),np.array([0,0,-1]))
            obs=np.concatenate([omega,grav,qj,dqj,last,np.array(cmd)]).astype(np.float32)
            out=session.run(None,{"obs":obs.reshape(1,num_obs)})
            last=out[-1].reshape(-1)
            target=last*action_scale+default_angles

    def get_pose():
        x,y=data.qpos[0],data.qpos[1]
        qw,qx,qy,qz=data.qpos[3:7]
        yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        return float(x),float(y),float(yaw)

    from navigation.navigate import Navigator as PtPNav
    class SpatialNav:
        def __init__(self,path,fwd_speed=0.7):
            self.path=path; self.idx=0; self.fwd_speed=fwd_speed; self.arrived=False
            self._nav=None; self._make()
        def _make(self):
            if self.idx<len(self.path):
                wx,wy=self.path[self.idx]
                self._nav=PtPNav(wx,wy,fwd_speed=self.fwd_speed)
            else: self._nav=None
        def update(self,x,y,yaw):
            if self.idx>=len(self.path):
                self.arrived=True; return np.zeros(3,dtype=np.float32)
            wx,wy=self.path[self.idx]
            if np.hypot(wx-x,wy-y)<0.4 and self.idx<len(self.path)-1:
                self.idx+=1; self._make()
            return self._nav.update(x,y,yaw)

    nav=SpatialNav(humanoid_path)
    SPEED=10; AGV_SPEED=0.7; dt=model.opt.timestep
    sim_time=0.0; clocks={cid:0.0 for cid in agv_trajs}
    collisions=0; min_dist=999.0; arc_triggers=0; fell=False
    off_cur=0.0; off_target=0.0
    lateral={cid:np.zeros(2) for cid in agv_trajs}

    for step in range(4000):
        for cid in agv_trajs:
            clocks[cid]+=dt*SPEED*AGV_SPEED
        # AGV mutual repulsion (same as run_factory_full)
        for cid in agv_trajs:
            my=np.array(get_agv_pos(cid,clocks[cid]))+lateral[cid]
            for (ox,oy) in STATIC_OBS:
                diff=my-np.array([ox,oy],dtype=float)
                d=np.linalg.norm(diff)
                if d<0.8 and d>0.01:
                    lateral[cid]+=(diff/d)*(0.8-d)*1.5
        cids=list(agv_trajs)
        for i in range(len(cids)):
            for j in range(i+1,len(cids)):
                a,b=cids[i],cids[j]
                pa=np.array(get_agv_pos(a,clocks[a]))+lateral[a]
                pb=np.array(get_agv_pos(b,clocks[b]))+lateral[b]
                d=np.linalg.norm(pa-pb)
                if d<0.7 and d>0.01:
                    push_dir=(pa-pb)/d
                    push=(0.7-d)*0.5
                    lateral[a]+=push_dir*push
                    lateral[b]-=push_dir*push
        for cid in agv_trajs:
            lateral[cid]*=0.94
        for cid in agv_trajs:
            ax,ay=get_agv_pos(cid,clocks[cid])
            lx,ly=lateral[cid]
            bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f"agv_{cid}")
            adr=model.body_mocapid[bid]
            data.mocap_pos[adr][0]=ax+lx; data.mocap_pos[adr][1]=ay+ly
            data.mocap_pos[adr][2]=-0.16+0.2
        mujoco.mj_forward(model,data)

        rx,ry,ryaw=get_pose()
        vx,vy,omega=nav.update(rx,ry,ryaw)

        # ARC avoidance
        if with_arc and vx>0.05 and not nav.arrived:
            heading=np.array([np.cos(ryaw),np.sin(ryaw)])
            robot_pos=np.array([rx,ry])
            best_close=999.0; best_sign=0; best_cid=None; best_lat=999.0
            for cid in agv_trajs:
                close=999.0; lat_min=999.0; cross_min=0.0
                if int(round(clocks[cid])) in agv_stationary_times[cid] or \
                   clocks[cid]>=agv_trajs[cid][-1][0] or clocks[cid]<=agv_trajs[cid][0][0]:
                    ax,ay=get_agv_pos(cid,clocks[cid])
                    rel=np.array([ax,ay])-robot_pos
                    d=float(np.linalg.norm(rel))
                    close=d
                    cross=float(rel[0]*heading[1]-rel[1]*heading[0])
                    if d<2.0:
                        lat_min=abs(cross); cross_min=cross
                else:
                    for ta in np.arange(0.5,2.6,0.5):
                        ax,ay=get_agv_pos(cid,clocks[cid]+ta)
                        rel=np.array([ax,ay])-robot_pos
                        d=float(np.linalg.norm(rel))
                        if d<close: close=d
                        cross=float(rel[0]*heading[1]-rel[1]*heading[0])
                        if d<2.0 and abs(cross)<lat_min:
                            lat_min=abs(cross); cross_min=cross
                if lat_min<best_lat and close<best_close:
                    best_lat=lat_min; best_sign=np.sign(cross_min)
                    best_cid=cid; best_close=close
            if best_cid is not None and best_close<0.9:
                arc_until=getattr(nav,"_arc_until",-9.0)
                if sim_time<arc_until:
                    off_target=-getattr(nav,"_arc_sign",0)*0.25
                else:
                    nav._arc_sign=best_sign; nav._arc_until=sim_time+2.0
                    off_target=-best_sign*0.25
                    arc_triggers+=1
            else:
                off_target=0.0
            err=off_target-off_cur
            vy_extra=np.clip(err*0.8,-0.25,0.25)
            vy+=vy_extra
            off_cur+=vy_extra*dt*SPEED

        robot_step([vx,vy,omega])
        sim_time+=dt*SPEED

        # collision check (contact is authoritative — uses actual qpos)
        for c in data.contact:
            g1,g2=c.geom1,c.geom2
            if (g1 in robot_geoms and g2 in agv_geom_ids.values()) or \
               (g2 in robot_geoms and g1 in agv_geom_ids.values()):
                collisions+=1
        for cid in agv_trajs:
            ax,ay=get_agv_pos(cid,clocks[cid])
            lx,ly=lateral[cid]
            d=np.hypot(rx-(ax+lx),ry-(ay+ly))
            if d<min_dist: min_dist=d
        if data.qpos[2]<0.6:
            fell=True
            break
        if nav.arrived:
            break

    return {"collisions":collisions,"min_dist":min_dist,"arc_triggers":arc_triggers,
            "fell":fell,"arrived":nav.arrived,"wp":nav.idx,"sim_time":sim_time,
            "final_pos":(rx,ry)}

print("="*60)
print(f"Plan: {plans[0]}  waypoints: {len(humanoid_path)}")
print("="*60)
for with_arc in (False, True):
    r = run_sim(with_arc)
    print(f"\n--- ARC {'ON' if with_arc else 'OFF'} ---")
    print(f"  collisions (MuJoCo contact): {r['collisions']}")
    print(f"  min robot-AGV distance:      {r['min_dist']:.3f} m")
    print(f"  arc triggers:                {r['arc_triggers']}")
    print(f"  fell:                        {r['fell']}")
    print(f"  arrived:                     {r['arrived']}  wp={r['wp']}/{len(humanoid_path)}")
    print(f"  sim_time:                    {r['sim_time']:.1f}s")
    print(f"  final pos:                   ({r['final_pos'][0]:.2f},{r['final_pos'][1]:.2f})")
