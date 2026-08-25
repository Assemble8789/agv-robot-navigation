"""
run_wangting.py — 阶段1 (纯 AGV) bridge 入口: 小数版地图在 MuJoCo 桥里跑
======================================================================

把 0.1m 小数版地图 (wangting 源文件或原生 10lm 图) 直接跑进 AGV↔MuJoCo bridge:
  - A* 在 0.1m 格子里规划 (QoS 时空 A* + ALT)
  - 发布轨迹时 cell→meter 精确换算, MuJoCo 里球在十进制米上移动 (4.1m 就是 4.1m)
  - 障碍盒 / 车库 / 场景从地图生成 (不改原文件)

用法 (PY = D:/download/anaconda3/envs/tutorial_for_mujoco/python.exe):
  PY src/bridge_wangting/run_wangting.py --demo5 --headless --steps 120
  PY src/bridge_wangting/run_wangting.py --demo5 --speed 5          # 交互窗口
  PY src/bridge_wangting/run_wangting.py --cars 5 --headless --steps 120
  PY src/bridge_wangting/run_wangting.py --map maps/wangting.cmrn...json --demo5 --headless --steps 120
"""
from __future__ import annotations

import os
import sys
import argparse
import random

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_DIR = os.path.join(SRC_DIR, "bridge")
MAPS_DIR = os.path.join(SRC_DIR, "..", "maps")
for p in (SRC_DIR, BRIDGE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from topic_bus import Bus                                   # noqa: E402
from topic_monitor import TopicMonitor                       # noqa: E402
from tcp_bridge import TcpRelay                              # noqa: E402
import map_scene                                             # noqa: E402

try:
    from wangting_nodes import make_planner, make_sim            # noqa: E402
except ModuleNotFoundError as _e:
    if "mujoco" in str(_e).lower():
        print("=" * 60)
        print("缺 mujoco 依赖 — 请用 tutorial_for_mujoco 环境运行:")
        print("  PY src/bridge_wangting/run_wangting.py --demo5 --speed 5")
        print("  PY = D:/download/anaconda3/envs/tutorial_for_mujoco/python.exe")
        print("=" * 60)
    raise

DEFAULT_MAP = os.path.join(MAPS_DIR, "map_wangting_10lm.json")

QOS_CYCLE = ["HIGH", "MEDIUM", "LOW"]

# ── 演示路线 (LM001..LM010): 出生地标互不为他人经停点, 站台分散, 避免拥挤卡死 ──
WANGTING_DEMO = [
    ("add", ("car01", "LM001", "HIGH")),
    ("route", ("car01", ("LM006", "LM009"))),
    ("add", ("car02", "LM002", "MEDIUM")),
    ("route", ("car02", ("LM007", "LM010"))),
    ("add", ("car03", "LM003", "LOW")),
    ("route", ("car03", ("LM008", "LM006"))),
]

WANGTING_DEMO_4 = [
    ("add", ("car01", "LM001", "HIGH")),
    ("route", ("car01", ("LM006",))),
    ("add", ("car02", "LM002", "MEDIUM")),
    ("route", ("car02", ("LM007",))),
    ("add", ("car03", "LM003", "MEDIUM")),
    ("route", ("car03", ("LM008",))),
    ("add", ("car04", "LM004", "LOW")),
    ("route", ("car04", ("LM009",))),
]

WANGTING_DEMO_5 = [
    ("add", ("car01", "LM001", "HIGH")),
    ("route", ("car01", ("LM006", "LM009"))),
    ("add", ("car02", "LM002", "MEDIUM")),
    ("route", ("car02", ("LM007", "LM010"))),
    ("add", ("car03", "LM003", "MEDIUM")),
    ("route", ("car03", ("LM008", "LM006"))),
    ("add", ("car04", "LM004", "LOW")),
    ("route", ("car04", ("LM009", "LM007"))),
    ("add", ("car05", "LM005", "LOW")),
    ("route", ("car05", ("LM010", "LM008"))),
]


def gen_random_commands(n_cars, lm_names, seed=None):
    """n 辆车随机链: 出生地标互不相同, 每车 2~3 个经停。返回 demo 命令列表。"""
    rng = random.Random(seed)
    spawns = rng.sample(lm_names, min(n_cars, len(lm_names)))
    cmds = []
    for i, sp in enumerate(spawns):
        stops = [x for x in rng.sample(lm_names, 3) if x != sp][:rng.randint(2, 3)]
        qos = QOS_CYCLE[i % len(QOS_CYCLE)]
        cmds.append(("add", (f"car{i + 1:02d}", sp, qos)))
        cmds.append(("route", (f"car{i + 1:02d}", tuple(stops))))
    return cmds


def _commands_to_chains(cmds):
    """demo 命令 → 每车一串地标 [[出生,停1,...], ...] (纯 AGV 预规划用)。"""
    chains = []
    cur = None
    for action, a in cmds:
        if action == "add":
            cur = [a[1]]
        elif action == "route" and cur is not None:
            cur.extend(a[1])
            chains.append(cur)
            cur = None
    return chains


def main():
    parser = argparse.ArgumentParser(description="小数版地图 → AGV↔MuJoCo bridge (纯 AGV)")
    parser.add_argument("--map", default=None, help="地图 JSON (默认 map_wangting_10lm.json; 源文件也可)")
    parser.add_argument("--demo", action="store_true", help="3 车演示")
    parser.add_argument("--demo4", action="store_true", help="4 车演示")
    parser.add_argument("--demo5", action="store_true", help="5 车演示")
    parser.add_argument("--cars", type=int, default=0, help="N 辆车随机链")
    parser.add_argument("--seed", type=int, default=None, help="随机链种子")
    parser.add_argument("--headless", action="store_true", help="无界面运行")
    parser.add_argument("--steps", type=float, default=0, help="headless 运行秒数 (0=一直跑)")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度")
    parser.add_argument("--echo", action="store_true", help="打印总线消息")
    parser.add_argument("--port", type=int, default=0, help="TCP 中继端口")
    parser.add_argument("--obstacle", default=None, help='动态障碍 "x,y" 或 "x,y,r"')
    parser.add_argument("--inflate", type=float, default=None,
                        help="障碍膨胀半径 (m), 默认 AGV_RADIUS=0.2; 0=关闭")
    parser.add_argument("--robot", action="store_true", help="加载 elf3 人形机器人")
    parser.add_argument("--robot-wp", default="LM006,LM008,LM002,LM010",
                        help='机器人出生+路径地标, 逗号分隔 (默认 LM006,LM008,LM002,LM010)')
    parser.add_argument("--scene-out", default=None, help="场景 XML 输出路径")
    parser.add_argument("--pure", action="store_true",
                        help="纯 AGV 离线预规划: 全车 t=0 同时发车 (套在线短停车窗/跳站机制)")
    args = parser.parse_args()

    map_file = os.path.normpath(args.map) if args.map else os.path.normpath(DEFAULT_MAP)
    if not os.path.exists(map_file):
        print(f"Map file {map_file} not found.")
        return

    cmds = None
    if args.demo5:
        cmds = WANGTING_DEMO_5
    elif args.demo4:
        cmds = WANGTING_DEMO_4
    elif args.demo:
        cmds = WANGTING_DEMO
    elif args.cars:
        from agv_map_common import load_normalized
        lm_names = [lm["name"] for lm in load_normalized(map_file)["landmarks"]]
        cmds = gen_random_commands(args.cars, lm_names, seed=args.seed)
    if cmds:
        print("[DEMO] injecting commands:")
        for action, a in cmds:
            print(f"        {action} {a}")

    # ── 纯 AGV: 离线预规划 + t=0 同时发车 (不走在线 planner) ──
    if args.pure:
        import pure_agv
        chains = _commands_to_chains(cmds) if cmds else [["LM001", "LM006", "LM009"]]
        pure_agv.run(map_file, chains, headless=args.headless, steps=args.steps,
                     speed=args.speed, inflate=args.inflate, scene_out=args.scene_out)
        return

    bus = Bus()
    planner = make_planner(bus, map_file, inflate=args.inflate)

    robot_plan = None
    if args.robot:
        from robot_path import build_robot_plan
        md, frame, _ = map_scene.load_map_data(map_file)
        boxes = map_scene.obstacle_boxes(md, frame)
        robot_plan = build_robot_plan(args.robot_wp, md, frame, boxes)
    sim = make_sim(bus, map_file, headless=args.headless,
                   wall_speed=args.speed, scene_out=args.scene_out,
                   robot=args.robot, robot_plan=robot_plan)

    if args.echo:
        TopicMonitor(bus)
        print("[ECHO] topic monitor on")
    relay = None
    if args.port:
        relay = TcpRelay(bus, port=args.port)

    cmds = None
    if args.demo5:
        cmds = WANGTING_DEMO_5
    elif args.demo4:
        cmds = WANGTING_DEMO_4
    elif args.demo:
        cmds = WANGTING_DEMO
    elif args.cars:
        from agv_map_common import load_normalized
        lm_names = [lm["name"] for lm in load_normalized(map_file)["landmarks"]]
        cmds = gen_random_commands(args.cars, lm_names, seed=args.seed)

    if cmds:
        print("[DEMO] injecting commands:")
        for action, a in cmds:
            print(f"        {action} {a}")
        planner.queue_commands(cmds)

    if args.obstacle:
        parts = args.obstacle.split(",")
        x, y = float(parts[0]), float(parts[1])
        r = float(parts[2]) if len(parts) > 2 else 0.3
        planner.queue_commands([("obst", ("o1", x, y, r))])
    elif not args.headless and not cmds:
        planner.start_cli()

    planner.start()
    try:
        sim.run(steps=args.steps)
    finally:
        planner.stop()
        if relay is not None:
            relay.stop()
    print("[WANGTING-BRIDGE] done.")


if __name__ == "__main__":
    main()