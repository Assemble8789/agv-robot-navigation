"""跑一遍 headless, 实时记录 AGV + 机器人位置 (每整秒), 存 JSON 供回放。
用法:
  PY src/bridge/_record_trajectory.py --cars 5 --path 0 --steps 150
  输出 docs/traj_5_p0.json  (含地图 + 每整秒的车/机器人位置)
"""
import os
import sys
import json
import math
import random
import threading
import argparse
import time

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BRIDGE_DIR)
sys.path.insert(0, BRIDGE_DIR)

from topic_bus import Bus
import planner_node
from planner_node import PlannerNode
from mujoco_node import MujocoNode
import validate_agv
import demo_bridge

MAP = validate_agv.MAP


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=5)
    ap.add_argument("--path", type=int, default=0)
    ap.add_argument("--steps", type=float, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dwell", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    planner_node.DWELL_STOP = max(0.0, args.dwell)

    import json as _json
    default_plan = os.path.join(SRC_DIR, "..", "maps", "humanoid_plan.json")
    with open(default_plan, encoding="utf-8") as f:
        rp = _json.load(f)
    robot_plan = demo_bridge._normalize_robot_plan(rp, MAP)

    mdata, lms, free = validate_agv.load_map(MAP)
    rng = random.Random(args.seed + args.cars * 1000 + args.path)
    spawns, chains = validate_agv.gen_chains(args.cars, lms, free, rng)

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

    # 采样线程: 每整秒记录所有 AGV mocap 位置 + 机器人位置
    rec = {}
    stop = threading.Event()

    def sampler():
        last_t = -1
        while not stop.is_set():
            t = int(sim.t_sim)
            if t != last_t and t >= 0:
                last_t = t
                cars = {}
                try:
                    for cid, x, y in sim._car_positions():
                        cars[cid] = [round(x, 3), round(y, 3)]
                    rx, ry, _ = sim._walker.get_pose()
                    rec[t] = {"cars": cars, "robot": [round(float(rx), 3), round(float(ry), 3)]}
                except Exception:
                    pass
            time.sleep(0.005)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    t0 = time.time()
    sim.run(steps=args.steps)
    stop.set()
    th.join(timeout=1.0)
    planner.stop()

    # 机器人完整路径 (回放用)
    robot_path = [(p["x"], p["y"]) for p in robot_plan["waypoints"]]
    out = {
        "map": MAP,
        "width": mdata["width"], "height": mdata["height"],
        "obstacles": [[o[0], o[1]] if isinstance(o, list) else [o["x"], o["y"]]
                      for o in mdata["obstacles"]],
        "landmarks": {name: [x, y] for name, (x, y) in lms.items()},
        "spawns": {name: [cell[0], cell[1]] for (cell, name) in spawns},
        "chains": [[stops] for _, stops in chains],
        "robot_path": robot_path,
        "robot_start": robot_plan["start"],
        "traj": rec,
        "steps": args.steps,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = args.out or os.path.join(SRC_DIR, "..", "docs",
                                        f"traj_{args.cars}_p{args.path}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    m = planner.metrics_summary()
    print(f"[RECORD] cars={args.cars} path={args.path} steps={len(rec)}s "
          f"completion={m['completion']:.0%} robot_collisions={sim.n_robot_collisions}")
    print(f"[RECORD] saved to {os.path.normpath(out_path)}")


if __name__ == "__main__":
    main()
