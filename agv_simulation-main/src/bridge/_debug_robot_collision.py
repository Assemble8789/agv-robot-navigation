"""复现 10/15 车带机器人场景的机器人碰撞, dump 每次碰撞现场。
用法:
  PY src/bridge/_debug_robot_collision.py --cars 10 --path 1 --steps 150
"""
import os
import sys
import math
import random
import argparse

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BRIDGE_DIR)
sys.path.insert(0, BRIDGE_DIR)

from topic_bus import Bus
import planner_node
from planner_node import PlannerNode
from mujoco_node import MujocoNode
import validate_agv
import mujoco

MAP = validate_agv.MAP
COLLS = []


def wrap_path_conflicts():
    """打印 _path_conflicts 的判定输入/输出 (每车每整秒一次)。"""
    last = {}
    orig = PlannerNode._path_conflicts
    def wrapped(self, car_id, **kw):
        wt = int(self.world.world_time)
        key = (car_id, wt)
        res = orig(self, car_id, **kw)
        if key != last.get(car_id):
            last[car_id] = key
            rp = self._robot_remaining_time_path()
            tr = self.world.cars.get(car_id, {}).get('trajectory', [])
            fut = [p for p in tr if p[0] >= self.world.world_time
                   and p[0] <= self.world.world_time + 6.0]
            robot = self._last_robot
            line = f"[PC] t={wt} car={car_id} robot=({robot.get('x') if robot else '?':.2f}," \
                   f"{robot.get('y') if robot else '?':.2f}) sp={robot.get('speed') if robot else '?':.2f} " \
                   f"robot_pts={len(rp)} → 冲突={res}"
            if car_id in ('car00', 'car04') and 15 <= wt <= 21:
                # 详细: 每个未来点 vs 最近机器人路径点的 (距离, 时间差)
                for (t, x, y, _d) in fut:
                    if rp:
                        dmin, trr = min((math.hypot(x - px, y - py), tr2)
                                        for px, py, tr2 in rp)
                        line += f"\n    未来点({x},{y}) t={t} → 最近路径点距离={dmin:.2f} " \
                                f"时间差|{t}-({wt}+{trr:.1f})|={abs(t - (wt + trr)):.2f}"
            print(line, flush=True)
        return res
    PlannerNode._path_conflicts = wrapped


def wrap_collisions():
    import mujoco_node
    orig = mujoco_node.MujocoNode._count_robot_collisions

    def wrapped(self):
        n0 = self.n_robot_collisions
        orig(self)
        if self.n_robot_collisions <= n0:
            return
        rx, ry, ryaw = self._walker.get_pose()
        nav = self._nav
        wp_i = nav.idx if hasattr(nav, 'idx') else 0
        n_wp = len(nav.wps)
        tx, ty = None, None
        if wp_i < n_wp:
            w = nav.wps[wp_i]
            tx, ty = (w['x'], w['y']) if isinstance(w, dict) else (w[0], w[1])
        rdx, rdy = math.cos(ryaw), math.sin(ryaw)
        # 碰撞 geom (AGV sphere 名)
        hit = []
        ncon = self.data.ncon
        for i in range(ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1 = self.model.geom_group[g1] == 3
            r2 = self.model.geom_group[g2] == 3
            if r1 == r2:
                continue
            other = g2 if r1 else g1
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
            if nm and nm.startswith("agv_"):
                hit.append((nm, list(c.pos)))
        sphere2car = {v: k for k, v in self._car_sphere.items()}
        hit_cars = [sphere2car.get(nm, nm) for nm, _ in hit]
        print(f"\n[ROBOT-COLLISION 现场] t={self.t_sim:.1f} total={self.n_robot_collisions}",
              flush=True)
        print(f"  机器人 @({rx:.2f},{ry:.2f}) yaw={ryaw:.2f}rad 朝向=({rdx:.2f},{rdy:.2f}) "
              f"wp={wp_i}/{n_wp} 目标=({tx},{ty}) arrived={nav.arrived}", flush=True)
        for nm, cpos in hit:
            print(f"  碰撞 AGV geom={nm} 接触点=({cpos[0]:.2f},{cpos[1]:.2f})", flush=True)
        cars = list(self._car_positions())
        for cid, x, y in cars:
            vx, vy, heading, speed = self._car_sample(cid, x, y)
            d = math.hypot(x - rx, y - ry)
            flag = "  <== 碰撞车" if cid in hit_cars else ""
            if d < 1.5 or flag:
                relx, rely = rx - x, ry - y
                print(f"  车 {cid} @({x:.2f},{y:.2f}) v=({vx:.2f},{vy:.2f}) "
                      f"speed={speed:.2f} 距机器人={d:.3f} "
                      f"车→机向量=({relx:.2f},{rely:.2f}){flag}", flush=True)
        rest = []
        for j in range(wp_i, min(wp_i + 5, n_wp)):
            w = nav.wps[j]
            rest.append((w['x'], w['y']) if isinstance(w, dict) else (w[0], w[1]))
        print(f"  机器人剩余路径前5点: {rest}", flush=True)
        COLLS.append({'t': self.t_sim, 'n': self.n_robot_collisions})

    mujoco_node.MujocoNode._count_robot_collisions = wrapped


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=10)
    ap.add_argument("--path", type=int, default=1)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dwell", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()
    planner_node.DWELL_STOP = max(0.0, args.dwell)

    wrap_collisions()
    mdata, lms, free = validate_agv.load_map(MAP)
    rng = random.Random(args.seed + args.cars * 1000 + args.path)
    spawns, chains = validate_agv.gen_chains(args.cars, lms, free, rng)

    import json as _json
    import demo_bridge
    default_plan = os.path.join(SRC_DIR, "..", "maps", "humanoid_plan.json")
    with open(default_plan, encoding="utf-8") as f:
        _rp_raw = _json.load(f)
    robot_plan = demo_bridge._normalize_robot_plan(_rp_raw, MAP)

    for rep in range(args.reps):
        print(f"\n########## rep {rep} ##########", flush=True)
        COLLS.clear()
        bus = Bus()
        planner = PlannerNode(bus, MAP, station_hold=planner_node.HOLD_SLOT)
        sim = MujocoNode(bus, headless=True, wall_speed=1.0, robot=True,
                         robot_plan=robot_plan)
        for (cell, name) in spawns:
            planner.world.lm_dict[name] = cell
        planner.start()
        cmds = []
        for i, (spawn_name, stops) in enumerate(chains):
            car_id = "car%02d" % i
            cmds.append(("add", (car_id, spawn_name, "MEDIUM")))
            goals = tuple(stops) + ((spawn_name,) if True else ())
            cmds.append(("route", (car_id, goals)))
        planner.queue_commands(cmds)
        sim.run(steps=args.steps)
        planner.stop()

        m = planner.metrics_summary()
        print(f"[RESULT rep{rep}] completion={m['completion']:.0%} "
              f"robot_collisions={sim.n_robot_collisions} robot_fell={sim.robot_fell_count} "
              f"agv_collisions={sim.n_agv_collisions} yield={m['yield_events']} "
              f"replan_fail={m['replan_fail']}", flush=True)
        print(f"[RESULT rep{rep}] 机器人碰撞 t = "
              f"{[round(c['t'],1) for c in COLLS]}", flush=True)


if __name__ == "__main__":
    main()
