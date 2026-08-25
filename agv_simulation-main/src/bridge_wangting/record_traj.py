"""
record_traj.py — 记录 wangting bridge (AGV + 机器人) 每整秒位置 → JSON, 供 2D 回放
===============================================================================

headless 跑一遍在线 bridge (waitfree A* + 膨胀 + 可选机器人), 采样线程每整秒记录
所有 AGV mocap 位置 + 机器人位置, 坐标换算成 0.1m 整数格 (cell), 与
bridge/_playback.py 的格子画法对齐, 直接可回放 (流畅、不卡、看清每个动作)。

用法:
  PY src/bridge_wangting/record_traj.py --demo5 --robot --steps 150 --out docs/traj_wangting.json
  PY src/bridge/_playback.py docs/traj_wangting.json

回放交互: ←/→ 步进 1s, 空格 播放/暂停, 点击图 +1s, Home/End 首/末帧。
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import time
import threading

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_DIR = os.path.join(SRC_DIR, "bridge")
MAPS_DIR = os.path.join(SRC_DIR, "..", "maps")
for p in (SRC_DIR, BRIDGE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from topic_bus import Bus                                       # noqa: E402
import run_wangting as rw                                        # noqa: E402
import map_scene                                                 # noqa: E402
from wangting_nodes import make_planner, make_sim                # noqa: E402


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None, help="wangting 地图 (默认 map_wangting_10lm.json)")
    ap.add_argument("--demo5", action="store_true", help="5 车演示")
    ap.add_argument("--demo", action="store_true", help="3 车演示")
    ap.add_argument("--cars", type=int, default=0, help="N 车随机链")
    ap.add_argument("--seed", type=int, default=0, help="随机链种子")
    ap.add_argument("--robot", action="store_true", help="带人形机器人")
    ap.add_argument("--robot-wp", default="LM006,LM008,LM002,LM010")
    ap.add_argument("--steps", type=float, default=150, help="仿真秒数")
    ap.add_argument("--inflate", type=float, default=None, help="障碍膨胀 (默认 0.2)")
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    args = ap.parse_args()

    map_file = os.path.normpath(args.map) if args.map else os.path.normpath(rw.DEFAULT_MAP)
    md, frame, _ = map_scene.load_map_data(map_file)

    # 演示命令
    if args.cars:
        lm_names = [lm["name"] for lm in md["landmarks"]]
        cmds = rw.gen_random_commands(args.cars, lm_names, seed=args.seed)
    elif args.demo:
        cmds = rw.WANGTING_DEMO
    else:
        cmds = rw.WANGTING_DEMO_5

    robot_plan = None
    if args.robot:
        from robot_path import build_robot_plan
        boxes = map_scene.obstacle_boxes(md, frame)
        robot_plan = build_robot_plan(args.robot_wp, md, frame, boxes)

    bus = Bus()
    planner = make_planner(bus, map_file, inflate=args.inflate)
    sim = make_sim(bus, map_file, headless=True, wall_speed=1.0,
                   robot=args.robot, robot_plan=robot_plan)
    planner.start()
    planner.queue_commands(cmds)

    # 采样线程: 每整秒记录 AGV + 机器人位置 (米 → 0.1m 格)
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
                        cars[cid] = list(frame.to_cell(x, y))
                except Exception:
                    pass
                entry = {"cars": cars}
                if sim._walker is not None:
                    try:
                        rx, ry, _ = sim._walker.get_pose()
                        entry["robot"] = list(frame.to_cell(float(rx), float(ry)))
                    except Exception:
                        pass
                rec[t] = entry
            time.sleep(0.005)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    sim.run(steps=args.steps)
    stop.set()
    th.join(timeout=1.0)
    planner.stop()

    robot_path = [list(frame.to_cell(p["x"], p["y"]))
                  for p in robot_plan["waypoints"]] if robot_plan else []
    robot_start = list(frame.to_cell(*robot_plan["start"])) if robot_plan else []

    out = {
        "map": map_file,
        "width": md["width"], "height": md["height"],
        "obstacles": [list(o) for o in md["obstacles"]],
        "landmarks": {lm["name"]: [lm["x"], lm["y"]] for lm in md["landmarks"]},
        "robot_path": robot_path,
        "robot_start": robot_start,
        "traj": rec,
        "steps": args.steps,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = args.out or os.path.join(SRC_DIR, "..", "docs", "traj_wangting.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[RECORD] saved to {out_path}  ({len(rec)}s  robot={args.robot})")


if __name__ == "__main__":
    main()