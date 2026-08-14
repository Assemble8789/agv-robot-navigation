"""
AGV 算法验证: 车数 × 路径集 × 重复 的实验矩阵 + markdown 报告
================================================================

指标 (类别 一/二/三/五):
  一 规划质量: replan 总数/原因(AGV停下导致)/失败/路径效率(曼哈顿vs实际)/完成率/makespan
  二 进站队列: 排队次数/超时跳站/停车总时长
  三 安全:     AGV-AGV 碰撞/全程最小间距/近距离事件
  五 稳定性:   同配置重复 mean±std

实验矩阵 (默认):
  车数 5/10/15 × 路径集 3(同场景不同随机路线) × 重复 3
  → 3×3×3 = 27 次 headless 纯 AGV 运行

用法:
  # 冒烟
  PY src/bridge/validate_agv.py --cars 5 --paths 1 --reps 1 --steps 60
  # 全矩阵 (结果写入 docs/agv_validation.md)
  PY src/bridge/validate_agv.py --cars 5,10,15 --paths 3 --reps 3 --steps 150
"""
import os
import sys
import json
import argparse
import time
import random

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BRIDGE_DIR)
sys.path.insert(0, BRIDGE_DIR)

from topic_bus import Bus
import planner_node
from planner_node import PlannerNode
from mujoco_node import MujocoNode
import demo_bridge          # 复用机器人路径解析/细格 A* 规范化

MAP = os.path.join(SRC_DIR, "..", "maps", "map_20260730_162111.json")

# (metrics key, 中文标签, 类型: int / float / ratio)
METRICS = [
    ("replan_total",    "replan总数",      "int"),
    ("replan_by_stop",  "AGV停下→replan",  "int"),
    ("replan_fail",     "replan失败",      "int"),
    ("yield_events",    "停车让行",        "int"),
    ("queue_events",    "排队",            "int"),
    ("queue_timeouts",  "超时跳站",        "int"),
    ("parked_time",     "停车时长(s)",     "float"),
    ("move_aside",      "挪开",            "int"),
    ("collisions",      "碰撞",            "int"),
    ("near_miss",       "近距离<0.6m",     "int"),
    ("min_agv_agv",     "最小间距(m)",     "float"),
    ("robot_collisions", "机器人碰撞",     "int"),
    ("robot_fell",      "机器人摔倒",      "int"),
    ("makespan",        "makespan(s)",     "float"),
    ("completion",      "完成率",          "ratio"),
    ("path_overhead",   "绕行开销",        "ratio"),
]


def load_map(map_file):
    with open(map_file, encoding="utf-8") as f:
        m = json.load(f)
    lms = {lm["name"]: (lm["x"], lm["y"]) for lm in m.get("landmarks", [])}
    obs = set()
    for o in m.get("obstacles", []):
        obs.add((o[0], o[1]) if isinstance(o, list) else (o["x"], o["y"]))
    free = [(x, y) for y in range(m["height"]) for x in range(m["width"])
            if (x, y) not in obs]
    return m, lms, free


def gen_chains(n_cars, lms, free, rng):
    """每车链 [出生格名, 停1, 停2].
    出生格 = 互不相同的自由格 (临时地标 _SP_i), 贪心取【相互最远】的格 →
    避免出生格相邻导致刚出发就撞 (实测 t=0.6 碰撞就是相邻出生格);
    经停用真实地标 (站)。"""
    fs = list(free)
    rng.shuffle(fs)
    spawn_cells = [fs[0]]
    for _ in range(n_cars - 1):
        best, best_d = None, -1
        for c in fs:
            if c in spawn_cells:
                continue
            d = min(abs(c[0] - s[0]) + abs(c[1] - s[1]) for s in spawn_cells)
            if d > best_d:
                best_d, best = d, c
        if best is None:
            break
        spawn_cells.append(best)
    spawns = [(c, "_SP_%d" % i) for i, c in enumerate(spawn_cells)]
    lm_names = list(lms.keys())
    chains = []
    for i in range(n_cars):
        stops = rng.sample(lm_names, min(2, len(lm_names)))
        chains.append((spawns[i][1], stops))       # (spawn_name, [stop1, stop2])
    return spawns, chains


def run_once(map_file, chains, spawns, steps, garage=True, fifo=True,
             robot=True, robot_wp=None, robot_plan=None):
    """一次 headless 运行, 返回指标 dict。
    garage=True: 每车在经停站结束后【回自己的出生格(车库)】, 释放站台,
    避免先到车永久占着终点站导致撞站后车排队超时。
    fifo=True : 站台短窗口预约 (先到先进), 后车能规划进前车离站后的空窗;
    fifo=False: 站台永久预约 (原行为, 先派发者锁死站台)。
    robot=True : 同时加载 elf3 人形机器人走固定路径 (默认 humanoid_plan.json),
    验证 AGV↔机器人避让 / 让行 / 站台等待 的完整流程。"""
    bus = Bus()
    planner = PlannerNode(bus, map_file,
                          station_hold=planner_node.HOLD_SLOT if fifo else None)
    plan_file = robot_plan            # 命令行传入的机器人路径 JSON (None=默认)
    rp = None
    if robot:
        import json
        if robot_wp:
            rp = demo_bridge._robot_plan_from_wp(robot_wp, map_file)
            if rp is None:
                rp = {}
        else:
            default_plan = os.path.normpath(plan_file) if plan_file else \
                os.path.join(SRC_DIR, "..", "maps", "humanoid_plan.json")
            with open(default_plan, encoding="utf-8") as f:
                rp = json.load(f)
        rp = demo_bridge._normalize_robot_plan(rp, map_file)
    sim = MujocoNode(bus, headless=True, wall_speed=1.0,
                     robot=robot, robot_plan=rp)
    for (cell, name) in spawns:
        planner.world.lm_dict[name] = cell
    planner.start()
    cmds = []
    for i, (spawn_name, stops) in enumerate(chains):
        car_id = "car%02d" % i
        cmds.append(("add", (car_id, spawn_name, "MEDIUM")))
        goals = tuple(stops) + ((spawn_name,) if garage else ())
        cmds.append(("route", (car_id, goals)))
    planner.queue_commands(cmds)
    sim.run(steps=steps)
    planner.stop()
    m = planner.metrics_summary()
    m['collisions'] = sim.n_agv_collisions
    m['min_agv_agv'] = sim.metrics['min_agv_agv']
    m['near_miss'] = sim.metrics['near_miss']
    # 机器人指标 (--robot 时)
    m['robot_collisions'] = sim.n_robot_collisions
    m['robot_fell'] = sim.robot_fell_count
    return m


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, var ** 0.5


def cell(vals, kind):
    """把一列数值格式化成 "mean±std"。"""
    mean, std = mean_std(vals)
    if mean != mean:                     # nan
        return "  -  "
    if kind == "int":
        return f"{mean:,.0f}±{std:.1f}"
    if kind == "ratio":
        return f"{mean*100:.1f}%±{std*100:.1f}"
    if mean == float("inf"):
        return "  -  "
    return f"{mean:.2f}±{std:.2f}"


def single(v, kind):
    """格式化单值。"""
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "ratio":
        return f"{v*100:.1f}%"
    if v == float("inf") or v != v:
        return "  -  "
    return f"{v:.2f}"


def _rows_from_runs(results, keys, kind_map):
    """把 {key: [values]} 转成一行文本 (按 METRICS 顺序)。"""
    out = []
    for key, _label, kind in keys:
        out.append(cell(results[key], kind))
    return "| " + " | ".join(out) + " |"


def main():
    # Windows 控制台默认 GBK, 强制 UTF-8 (否则 ✅/中文 打印报错)
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="AGV 算法验证 (车数×路径×重复)")
    parser.add_argument("--cars", default="5,10,15",
                        help="车数列表, 逗号分隔 (默认 5,10,15)")
    parser.add_argument("--paths", type=int, default=3,
                        help="同场景不同路径集数 (默认 3)")
    parser.add_argument("--reps", type=int, default=3,
                        help="同配置重复次数 (默认 3)")
    parser.add_argument("--steps", type=int, default=150,
                        help="每局仿真秒 (默认 150)")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子 (默认 0)")
    parser.add_argument("--dwell", type=float, default=1.0,
                        help="每站停留秒数 (0 = 不停站, 到站立刻派下一段)")
    parser.add_argument("--garage", type=int, default=1,
                        help="经停结束后是否回出生格(车库)释放站台 (1/0, 默认 1)")
    parser.add_argument("--fifo", type=int, default=1,
                        help="站台短窗口预约=先到先进 (1/0, 默认 1; 0=站台永久预约)")
    parser.add_argument("--timeout", type=float, default=planner_node.QUEUE_TIMEOUT,
                        help="进站排队超时 (s): 等待点等这么久还没进就跳站 "
                             "(默认随模块 QUEUE_TIMEOUT)")
    parser.add_argument("--robot", type=int, default=1,
                        help="是否加载人形机器人验证 AGV↔机器人流程 (1/0, 默认 1)")
    parser.add_argument("--robot-wp", default=None,
                        help="机器人地标路径 'LM000,LM001,...' (默认 humanoid_plan.json)")
    parser.add_argument("--robot-plan", default=None,
                        help="机器人路径 JSON 文件 (默认 humanoid_plan.json; "
                             "humanoid_plan_edge.json=外围避开站台)")
    parser.add_argument("--map", default=None, help="地图 JSON 路径")
    parser.add_argument("--out", default="docs/agv_validation.md",
                        help="markdown 报告输出路径 (默认 docs/agv_validation.md)")
    args = parser.parse_args()

    # 不停站模式 / 排队超时: 直接改模块常量 (派发时实时读取)
    planner_node.DWELL_STOP = max(0.0, args.dwell)
    planner_node.QUEUE_TIMEOUT = max(0.0, args.timeout)

    map_file = os.path.normpath(args.map) if args.map else os.path.normpath(MAP)
    cars_list = [int(c) for c in args.cars.split(",") if c.strip()]
    mdata, lms, free = load_map(map_file)

    results = {}                       # (nc, path) -> list[metric dict]
    t0 = time.time()
    for nc in cars_list:
        for p in range(args.paths):
            rng = random.Random(args.seed + nc * 1000 + p)
            spawns, chains = gen_chains(nc, lms, free, rng)
            runs = []
            for r in range(args.reps):
                m = run_once(map_file, chains, spawns, args.steps,
                             garage=bool(args.garage), fifo=bool(args.fifo),
                             robot=bool(args.robot), robot_wp=args.robot_wp,
                             robot_plan=args.robot_plan)
                runs.append(m)
                print(f"[run] cars={nc} path={p} rep={r} "
                      f"replan={m['replan_total']} collision={m['collisions']} "
                      f"completion={m['completion']:.0%} makespan={m['makespan']:.0f}")
            results[(nc, p)] = runs
    dt = time.time() - t0

    # ── 组报告 ──
    L = []
    A = L.append
    A(f"# AGV 算法验证报告\n")
    A(f"- 地图: `{os.path.basename(map_file)}`  ({mdata['width']}x{mdata['height']}, "
      f"地标 {len(lms)} 个)\n"
      f"- 车数: {cars_list} | 路径集: {args.paths} | 重复: {args.reps} | "
      f"仿真秒/局: {args.steps} | 随机种子: {args.seed}\n"
      f"- 总运行: {len(cars_list)*args.paths*args.reps} 局, 耗时 {dt:.0f}s "
      f"({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
    A("\n## 指标定义\n")
    A("| 指标 | 含义 |\n|---|---|")
    for key, label, _k in METRICS:
        A(f"| {label} | `{key}` |")
    A("")

    # 汇总趋势表: 行=指标, 列=车数
    A("## 汇总趋势 (按车数, 跨路径×重复的均值)\n")
    header = "| 指标 | " + " | ".join(f"{nc}车" for nc in cars_list) + " |"
    sep = "|" + "---|" * (len(cars_list) + 1)
    A(header)
    A(sep)
    for key, label, kind in METRICS:
        cells = []
        for nc in cars_list:
            vals = [m[key] for p in range(args.paths) for m in results[(nc, p)]]
            cells.append(cell(vals, kind))
        A(f"| {label} | " + " | ".join(cells) + " |")
    A("")

    # 每个车数的路径表: 行=路径, 列=指标
    for nc in cars_list:
        A(f"## 车数 {nc}: 路径敏感度 (mean±std over {args.reps} 次重复)\n")
        header = "| 路径 | " + " | ".join(label for _k, label, _t in METRICS) + " |"
        sep = "|" + "---|" * (len(METRICS) + 1)
        A(header)
        A(sep)
        for p in range(args.paths):
            runs = results[(nc, p)]
            cols = []
            for key, _label, kind in METRICS:
                cols.append(cell([m[key] for m in runs], kind))
            A(f"| P{p} | " + " | ".join(cols) + " |")
        A("")

    # 结论 / 自动标注
    A("## 结论\n")
    flags = []
    any_col = any(m['collisions'] > 0 for nc in cars_list
                  for p in range(args.paths) for m in results[(nc, p)])
    any_near = any(m['near_miss'] > 0 for nc in cars_list
                   for p in range(args.paths) for m in results[(nc, p)])
    any_incomp = any(m['completion'] < 0.999 for nc in cars_list
                     for p in range(args.paths) for m in results[(nc, p)])
    any_replan_stop = any(m['replan_by_stop'] > 0 for nc in cars_list
                          for p in range(args.paths) for m in results[(nc, p)])
    A(f"- AGV-AGV 碰撞: {'⚠ 有' if any_col else '✅ 全部 0'}")
    A(f"- 近距离事件 (<0.6m): {'⚠ 有' if any_near else '✅ 无'}")
    A(f"- 完成率: {'⚠ 有车未完成全部站' if any_incomp else '✅ 全部 100%'}")
    A(f"- AGV 停下导致的重规划: {'⚠ 存在 (需要后续 replan 绕开)' if any_replan_stop else '✅ 无'}")
    A("")
    for nc in cars_list:
        vals = [m for p in range(args.paths) for m in results[(nc, p)]]
        avg_replan = sum(m['replan_total'] for m in vals) / len(vals)
        avg_queue = sum(m['queue_events'] for m in vals) / len(vals)
        avg_timeout = sum(m['queue_timeouts'] for m in vals) / len(vals)
        avg_makespan = sum(m['makespan'] for m in vals) / len(vals)
        avg_comp = sum(m['completion'] for m in vals) / len(vals)
        A(f"- **{nc} 车**: replan≈{avg_replan:.0f}, 排队≈{avg_queue:.0f}, "
          f"超时跳站≈{avg_timeout:.0f}, makespan≈{avg_makespan:.0f}s, "
          f"完成率≈{avg_comp*100:.0f}%")

    text = "\n".join(L)
    print("\n" + text)
    out = os.path.normpath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[report] written to {out}")


if __name__ == "__main__":
    main()
