"""5 车 P0 场景 2D 路线图 (matplotlib PNG): 地图网格 + 障碍 + 站台 + 每车路线 + 机器人路径。
用法:
  PY src/bridge/_debug_route_map.py --cars 5 --path 0
  生成 docs/route_map_5_p0.png
"""
import os
import sys
import random
import argparse

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BRIDGE_DIR)
sys.path.insert(0, BRIDGE_DIR)
sys.path.insert(0, SRC_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import validate_agv
import demo_bridge

MAP = validate_agv.MAP

CAR_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
              "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080"]


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=5)
    ap.add_argument("--path", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="PNG 输出路径")
    args = ap.parse_args()

    mdata, lms, free = validate_agv.load_map(MAP)
    rng = random.Random(args.seed + args.cars * 1000 + args.path)
    spawns, chains = validate_agv.gen_chains(args.cars, lms, free, rng)
    lm_coord = {name: (x, y) for name, (x, y) in lms.items()}
    lm_coord.update({name: cell for (cell, name) in spawns})

    # 机器人路径 (normalize 细格)
    import json
    default_plan = os.path.join(SRC_DIR, "..", "maps", "humanoid_plan.json")
    with open(default_plan, encoding="utf-8") as f:
        rp = json.load(f)
    rp = demo_bridge._normalize_robot_plan(rp, MAP)
    robot_pts = [(p["x"], p["y"]) for p in rp["waypoints"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    W, H = mdata["width"], mdata["height"]
    x_max = W - 0.5
    y_max = H - 0.5
    ax.set_xlim(-0.5, x_max)
    ax.set_ylim(-0.5, y_max)
    ax.set_aspect("equal")
    ax.set_title(f"{args.cars} cars, path {args.path} (seed={args.seed}) - routes + robot path", fontsize=12, pad=12)
    ax.set_xticks(range(0, W))
    ax.set_yticks(range(0, H))
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

    # 障碍: 参考 agv_world_qos 的 imshow 网格画法
    import numpy as np
    grid_arr = np.zeros((H, W))
    for o in mdata["obstacles"]:
        x, y = (o[0], o[1]) if isinstance(o, list) else (o["x"], o["y"])
        grid_arr[y, x] = 1
    ax.imshow(grid_arr, extent=[-0.5, x_max, -0.5, y_max], origin="lower",
              cmap="Greys", alpha=0.35, interpolation="nearest", zorder=1)

    # 站台: 参考 agv_world_qos 蓝点 + 名字
    for name, (x, y) in lms.items():
        ax.plot(x, y, "o", color="#0066cc", markersize=8, alpha=0.85, zorder=4)
        ax.text(x, y + 0.28, name, fontsize=8, ha="center", color="#0066cc", zorder=5)

    # 机器人路径: 红色粗线 (区别于车)
    rx = [p[0] for p in robot_pts]
    ry = [p[1] for p in robot_pts]
    ax.plot(rx, ry, color="red", linewidth=3, alpha=0.5, zorder=2, label="Robot path")
    ax.plot(rx[0], ry[0], "r*", markersize=12, zorder=5)

    # 每车路线 (trail 风格)
    for i, ((spawn_cell, spawn_name), (_, stops)) in enumerate(zip(spawns, chains)):
        c = CAR_COLORS[i % len(CAR_COLORS)]
        pts = [spawn_cell] + [lm_coord[s] for s in stops] + [spawn_cell]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=c, linewidth=2, alpha=0.8, marker="o", markersize=5,
                zorder=3, label=f"car{i:02d}: {spawn_name}->{'>'.join(stops)}->garage")
        ax.text(spawn_cell[0] - 0.45, spawn_cell[1] + 0.45, f"car{i:02d}",
                color=c, fontsize=9, fontweight="bold", zorder=6)

    ax.legend(loc="upper left", fontsize=7, framealpha=0.6)
    fig.tight_layout()
    out = args.out or os.path.join(SRC_DIR, "..", "docs",
                                   f"route_map_{args.cars}_p{args.path}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[MAP] saved to {os.path.normpath(out)}")

    # ── ASCII 网格路线图 (终端可看) ──
    grid = [["  ." for _ in range(W)] for _ in range(H)]
    meta = [["" for _ in range(W)] for _ in range(H)]
    def put(x, y, ch, m=""):
        ix, iy = x, y
        if 0 <= ix < W and 0 <= iy < H:
            grid[iy][ix] = " %s" % ch
            meta[iy][ix] = m
    for o in mdata["obstacles"]:
        x, y = (o[0], o[1]) if isinstance(o, list) else (o["x"], o["y"])
        put(x, y, "#", "obstacle")
    station_idx = {}   # 站台名 -> 格, 站台格优先显示, 车/机器人不覆盖
    for idx, (name, (x, y)) in enumerate(lms.items()):
        station_idx[(x, y)] = name
        put(x, y, "S%d" % idx, f"station {name}")
    # 机器人路径 (格级, 不覆盖站台/障碍)
    for (px, py) in robot_pts:
        gx, gy = int(round(px)), int(round(py))
        if grid[gy][gx] in ("  .",):
            put(gx, gy, "*", "robot path")
    # 每车路线: 覆盖 出生→站→回库 的格 (不覆盖站台/障碍/机器人星号优先格)
    route_grid = {}
    for i, ((spawn_cell, _), (_, stops)) in enumerate(zip(spawns, chains)):
        pts = [spawn_cell] + [lm_coord[s] for s in stops] + [spawn_cell]
        for k in range(len(pts) - 1):
            x0, y0 = pts[k]
            x1, y1 = pts[k + 1]
            for t in range(0, 11):
                gx = round(x0 + (x1 - x0) * t / 10)
                gy = round(y0 + (y1 - y0) * t / 10)
                if (gx, gy) in station_idx or (gx, gy) in {tuple(o) for o in mdata["obstacles"]}:
                    continue
                route_grid[(gx, gy)] = route_grid.get((gx, gy), set()) | {i}
    for (gx, gy), cars in route_grid.items():
        if grid[gy][gx] == "*":
            continue                       # 机器人路径格优先 (相遇点)
        if len(cars) > 1:
            put(gx, gy, "X", f"shared {sorted(cars)}")
        else:
            put(gx, gy, str(min(cars)), f"car{min(cars):02d} route")
    # 出生格标 car 号 (若被站台/机器人占则跳过)
    for i, (cell, name) in enumerate(spawns):
        if grid[cell[1]][cell[0]] in ("  .",):
            put(cell[0], cell[1], str(i), f"car{i:02d} spawn")
    print("\n===== 5车 P0 ASCII 路线图 =====")
    print("     " + " ".join(f"{x:>3d}" for x in range(W)))
    for iy in range(H - 1, -1, -1):
        print(f"y={iy:<2d} " + "".join(grid[iy][ix] for ix in range(W)))
    print("\n图例: #=障碍  *=机器人路径  S0-S6=站台(LM000-LM006)  0-4=car00..04  X=多车共享")
    for i, ((cell, name), (_, stops)) in enumerate(zip(spawns, chains)):
        print(f"  car{i:02d}: 出生{name}@{cell} -> {' -> '.join(stops)} -> 回库{cell}")


if __name__ == "__main__":
    main()
