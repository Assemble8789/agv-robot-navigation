"""
Planner ↔ MuJoCo ROS-topic 风格接口的演示
==========================================

建一个进程内 Bus, 把 PlannerNode (AGVWorld) 和 MujocoNode (MuJoCo 渲染)
接成两个解耦节点:
  - sim 是时间主, 发布 /clock; planner 订阅 /clock 同步 world_time
  - planner 发布 /planner/path (add/move/route 成功变更后); sim 订阅并渲染
  - sim 每帧发布 /sim/car_state + /sim/collision + /sim/obstacle_distance
    (碰撞 / 障碍距离反馈回到 planner, planner 打印)

用法:
  python demo_bridge.py                          # 交互 CLI + MuJoCo 窗口
  python demo_bridge.py --demo                   # 脚本演示 (3 车 + 多路径点)
  python demo_bridge.py --headless --demo --steps 150   # 无界面自动化验证
"""

import os
import sys
import argparse

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BRIDGE_DIR)     # src/
sys.path.insert(0, BRIDGE_DIR)

from topic_bus import Bus
from planner_node import PlannerNode
from mujoco_node import MujocoNode
from topic_monitor import TopicMonitor
from tcp_bridge import TcpRelay

MAP = os.path.join(SRC_DIR, "..", "maps", "map_20260730_162111.json")

# 与 agv_world_qos_mujoco.py 的 --demo 相同的路线组合 (已验证无碰撞):
# 车出生在 add 地标, 然后依次走多路径点 (到站自动派下一段)。
DEMO_COMMANDS = [
    ("add", ("car01", "LM000", "HIGH")),
    ("route", ("car01", ("LM001", "LM002"))),
    ("add", ("car02", "LM005", "MEDIUM")),
    ("route", ("car02", ("LM006", "LM004"))),
    ("add", ("car03", "LM003", "LOW")),
    ("route", ("car03", ("LM000", "LM005"))),
]

# 4 车场景 (--demo4): 各车终点互不相同 (LM001/LM004/LM002/LM003), 避免 QoS 目标冲突。
DEMO_COMMANDS_4 = [
    ("add", ("car01", "LM003", "HIGH")),
    ("route", ("car01", ("LM000", "LM001"))),
    ("add", ("car02", "LM005", "MEDIUM")),
    ("route", ("car02", ("LM006", "LM004"))),
    ("add", ("car03", "LM004", "MEDIUM")),
    ("route", ("car03", ("LM002",))),
    ("add", ("car04", "LM006", "LOW")),
    ("route", ("car04", ("LM005", "LM003"))),
]

# 5 车 + 复杂经停 (--demo5): 终点 LM001/LM004/LM002/LM003/LM005 互不相同;
# 多条路线穿过机器人路径 (LM002→LM000→LM001→LM005), 触发更多让行/重规划。
DEMO_COMMANDS_5 = [
    ("add", ("car01", "LM003", "HIGH")),
    ("route", ("car01", ("LM000", "LM002", "LM001"))),
    ("add", ("car02", "LM005", "MEDIUM")),
    ("route", ("car02", ("LM006", "LM000", "LM004"))),
    ("add", ("car03", "LM004", "MEDIUM")),
    ("route", ("car03", ("LM000", "LM002"))),
    ("add", ("car04", "LM006", "LOW")),
    ("route", ("car04", ("LM005", "LM003"))),
    ("add", ("car05", "LM001", "LOW")),
    ("route", ("car05", ("LM000", "LM006", "LM005"))),
]


def _robot_plan_from_wp(wp_str, map_file):
    """把地标路径字符串 'LM002,LM000,LM001,LM005' 解析成 robot_plan:
    start=首点, waypoints=其余。 用于机器人自定义出生点 + 路径。"""
    import json
    with open(map_file, encoding="utf-8") as f:
        mdata = json.load(f)
    lms = {lm["name"].upper(): (float(lm["x"]), float(lm["y"]))
           for lm in mdata.get("landmarks", [])}
    names = [s.strip().upper() for s in wp_str.split(",") if s.strip()]
    coords = []
    for n in names:
        if n not in lms:
            print(f"[ERR] landmark {n} not found")
            return None
        coords.append(lms[n])
    if len(coords) < 2:
        print("[ERR] --robot-wp 至少需要 起点,路径点1[,路径点2...]")
        return None
    return {"start": coords[0],
            "waypoints": [{"x": c[0], "y": c[1]} for c in coords[1:]]}


# 柜子物理盒 (与 mujoco_node.OBS_BOXES 一致, 机器人 A* 绕它们)
OBS_BOXES = [
    (9.0, 8.0, 1.5, 0.35), (9.0, 5.0, 1.5, 0.35), (6.5, 1.0, 1.0, 0.5),
    (4.0, 5.0, 1.5, 0.35), (4.0, 8.0, 1.5, 0.35), (6.5, 2.0, 1.0, 0.5),
]
FINE_CELL = 0.25      # 机器人 A* 用 0.25m 细格 (精确表示柜子几何)
ROBOT_CLEAR = 0.5     # 机器人到柜子净空 = 半径0.3 + 0.2 余量 (柜子实体, 不能穿)


def _fine_obs_cells():
    """把每个物理盒扩展 ROBOT_CLEAR 后, 覆盖到的 0.25m 细格都当障碍。
    用细格而不是粗格: 粗格膨胀会把机器人出生点(如 LM002=9,7, 柜子上方空档)
    整个盖住, 细格能精确留出空档。"""
    cells = set()
    for cx, cy, hx, hy in OBS_BOXES:
        x0, x1 = cx - hx - ROBOT_CLEAR, cx + hx + ROBOT_CLEAR
        y0, y1 = cy - hy - ROBOT_CLEAR, cy + hy + ROBOT_CLEAR
        ix0, ix1 = int(x0 / FINE_CELL), int(x1 / FINE_CELL)
        iy0, iy1 = int(y0 / FINE_CELL), int(y1 / FINE_CELL)
        for ix in range(ix0 - 1, ix1 + 2):
            for iy in range(iy0 - 1, iy1 + 2):
                gx, gy = (ix + 0.5) * FINE_CELL, (iy + 0.5) * FINE_CELL
                if x0 <= gx <= x1 and y0 <= gy <= y1:
                    cells.add((ix, iy))
    return cells


def _astar_fine(start_xy, goal_xy, obs_cells, width, height):
    """0.25m 细格空间 A* (8 连通), 路径中间避开柜子, 返回世界坐标点列。
    起点/终点允许在障碍格 (车/机器人就在那)。"""
    import heapq
    W, H = int(width / FINE_CELL), int(height / FINE_CELL)
    def cell(p):
        return (int(p[0] / FINE_CELL), int(p[1] / FINE_CELL))
    s, g = cell(start_xy), cell(goal_xy)
    if s == g:
        return [start_xy]
    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    open_set = [(h(s, g), s)]
    came_from = {}
    gcost = {s: 0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == g:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return [((cx + 0.5) * FINE_CELL, (cy + 0.5) * FINE_CELL)
                    for cx, cy in path]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if (nx, ny) in obs_cells and (nx, ny) != s and (nx, ny) != g:
                continue
            ng = gcost[cur] + (1.414 if dx and dy else 1.0)
            if (nx, ny) not in gcost or ng < gcost[(nx, ny)]:
                gcost[(nx, ny)] = ng
                came_from[(nx, ny)] = cur
                heapq.heappush(open_set, (ng + h((nx, ny), g), (nx, ny)))
    return None


def _normalize_robot_plan(robot_plan, map_file):
    """把 robot_plan 规范化: start 解析成坐标, 相邻路径点之间用细格 A* 扩展成
    避开柜子 (物理盒+净空) 的路径点序列。"""
    import json
    with open(map_file, encoding="utf-8") as f:
        mdata = json.load(f)
    lms = {lm["name"].upper(): (float(lm["x"]), float(lm["y"]))
           for lm in mdata.get("landmarks", [])}
    width = int(mdata.get("width", 15))
    height = int(mdata.get("height", 10))
    obs = _fine_obs_cells()

    start = robot_plan.get("start") if isinstance(robot_plan, dict) else None
    if isinstance(start, str) and start.upper() in lms:
        start = lms[start.upper()]
    if not isinstance(start, (list, tuple)) or len(start) < 2:
        start = (2.0, 2.5)
    start = (float(start[0]), float(start[1]))

    out = []
    prev = start
    for wp in robot_plan.get("waypoints", []):
        goal = (float(wp["x"]), float(wp["y"]))
        seg = _astar_fine(prev, goal, obs, width, height)
        if seg:
            for p in seg[1:]:
                out.append({"x": p[0], "y": p[1]})
        else:
            out.append({"x": goal[0], "y": goal[1]})     # 找不到就直线
        prev = goal
    return {"start": list(start), "waypoints": out}


def main():
    parser = argparse.ArgumentParser(
        description="Planner ↔ MuJoCo ROS-topic 风格接口演示")
    parser.add_argument("--map", default=None, help="地图 JSON 路径")
    parser.add_argument("--demo", action="store_true",
                        help="脚本演示 (3 车 + 多路径点)")
    parser.add_argument("--demo4", action="store_true",
                        help="脚本演示 (4 车同时走, 终点互不相同)")
    parser.add_argument("--demo5", action="store_true",
                        help="脚本演示 (5 车 + 复杂经停, 更多让行/重规划)")
    parser.add_argument("--headless", action="store_true",
                        help="无界面运行 (自动化验证)")
    parser.add_argument("--steps", type=float, default=0,
                        help="headless 运行多少仿真秒 (0=一直跑到窗口关/结束)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="播放速度: 仿真秒/墙上秒 (默认 1.0)")
    parser.add_argument("--echo", action="store_true",
                        help="本终端打印总线上每条消息 (类似 ros2 topic echo)")
    parser.add_argument("--port", type=int, default=0,
                        help="启动 TCP 中继端口 (默认 0=关); 另开终端用 "
                             "src/bridge/listen.py --port <port> 监听, 不刷屏")
    parser.add_argument("--obstacle", default=None,
                        help='开局放一个动态障碍 (红色球): "x,y" 或 "x,y,radius" '
                             "(默认半径 0.3), 如 --obstacle 5,5")
    parser.add_argument("--robot", action="store_true",
                        help="加载 elf3 人形机器人 (走 humanoid_plan 路径, 真实策略)")
    parser.add_argument("--robot-plan", default=None,
                        help="机器人路径 JSON (默认 maps/humanoid_plan.json)")
    parser.add_argument("--robot-wp", default=None,
                        help='机器人出生点+路径地标: "LM002,LM000,LM001,LM005" '
                             "(首点=出生地标, 其余=依次要走的点)")
    args = parser.parse_args()

    map_file = os.path.normpath(args.map) if args.map else os.path.normpath(MAP)
    if not os.path.exists(map_file):
        print(f"Map file {map_file} not found.")
        return

    # 机器人路径: --robot-wp 优先 (地标名解析成坐标); 否则用 --robot-plan 或默认
    robot_plan = args.robot_plan
    if args.robot_wp:
        robot_plan = _robot_plan_from_wp(args.robot_wp, map_file)
        if robot_plan is None:
            return
    elif robot_plan is None and args.robot:
        import json
        default_plan = os.path.join(SRC_DIR, "..", "maps", "humanoid_plan.json")
        with open(default_plan, encoding="utf-8") as f:
            robot_plan = json.load(f)
    # 用 A* 把机器人路径扩展成避开柜子的路径点 (柜子实体, 不能穿)
    if args.robot and robot_plan is not None:
        robot_plan = _normalize_robot_plan(robot_plan, map_file)

    bus = Bus()
    planner = PlannerNode(bus, map_file)
    sim = MujocoNode(bus, headless=args.headless, wall_speed=args.speed,
                     robot=args.robot, robot_plan=robot_plan)

    if args.echo:
        TopicMonitor(bus)
        print("[ECHO] topic monitor on: "
              "/clock /planner/path /planner/remove "
              "/sim/car_state /sim/collision /sim/obstacle_distance")
    relay = None
    if args.port:
        relay = TcpRelay(bus, port=args.port)

    planner.start()
    if args.demo or args.demo4 or args.demo5:
        cmds = (DEMO_COMMANDS_5 if args.demo5
                else DEMO_COMMANDS_4 if args.demo4 else DEMO_COMMANDS)
        print("[DEMO] injecting commands:")
        for action, a in cmds:
            print(f"        {action} {a}")
        planner.queue_commands(cmds)
    if args.obstacle:
        parts = args.obstacle.split(",")
        x, y = float(parts[0]), float(parts[1])
        r = float(parts[2]) if len(parts) > 2 else 0.3
        print(f"[DEMO] + dynamic obstacle at ({x},{y}) r={r}")
        planner.queue_commands([("obst", ("o1", x, y, r))])
    elif not args.headless:
        planner.start_cli()

    try:
        sim.run(steps=args.steps)          # 主线程 = MuJoCo 渲染循环 (时间主)
    finally:
        planner.stop()
        if relay is not None:
            relay.stop()
    print("[BRIDGE] done.")


if __name__ == "__main__":
    main()
