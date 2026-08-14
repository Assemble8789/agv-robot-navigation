"""可视化运行带机器人场景 (MuJoCo 窗口): 任意 车数×路径 的随机路线。
用法:
  PY src/bridge/_debug_visual.py --cars 5 --path 1      # 5车 P1 (让行风暴/死锁场景)
  PY src/bridge/_debug_visual.py --cars 15 --path 0     # 15车 P0 (机器人碰撞场景)
  关窗口即结束; 终端实时打印 [YIELD]/[REPLAN]/[ROBOT-COLLISION]。
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
import demo_bridge

MAP = validate_agv.MAP


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="可视化带机器人场景")
    ap.add_argument("--cars", type=int, default=5)
    ap.add_argument("--path", type=int, default=0)
    ap.add_argument("--steps", type=float, default=0, help="仿真秒 (0=一直跑到关窗)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dwell", type=float, default=1.0)
    args = ap.parse_args()
    planner_node.DWELL_STOP = max(0.0, args.dwell)

    import json
    default_plan = os.path.join(SRC_DIR, "..", "maps", "humanoid_plan.json")
    with open(default_plan, encoding="utf-8") as f:
        rp = json.load(f)
    robot_plan = demo_bridge._normalize_robot_plan(rp, MAP)

    mdata, lms, free = validate_agv.load_map(MAP)
    rng = random.Random(args.seed + args.cars * 1000 + args.path)
    spawns, chains = validate_agv.gen_chains(args.cars, lms, free, rng)
    print(f"=== 可视化 {args.cars}车 path{args.path} 带机器人 ===")
    for i, (spawn_name, stops) in enumerate(chains):
        print(f"  car{i:02d}: {spawn_name} -> {stops}")

    bus = Bus()
    planner = PlannerNode(bus, MAP, station_hold=planner_node.HOLD_SLOT)
    sim = MujocoNode(bus, headless=False, wall_speed=1.0, robot=True,
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
    print(f"[RESULT] completion={m['completion']:.0%} "
          f"yield={m['yield_events']} replan_fail={m['replan_fail']} "
          f"robot_collisions={sim.n_robot_collisions} agv_collisions={sim.n_agv_collisions}")


if __name__ == "__main__":
    main()
