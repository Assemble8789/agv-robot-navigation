"""
robot_path.py — 在 wangting 工厂(米制)里给 elf3 机器人规划路径
================================================================

从地标序列生成机器人路径 (米制), 用 0.25m 细格 A* 绕开工厂障碍盒:
  - 障碍盒按 [宽-优先] 多级净空膨胀: 优先走 ≥1.4m 宽通道, 逐级放宽到 0.3m,
    实现"避开窄走廊" (能走宽的绝不挤窄的)。
  - 起点/终点 (地标) 允许在膨胀格内 (车/机器人就在那)。

输出 {"start":[mx,my], "waypoints":[{x,y}米制]} —— mujoco_node._setup_robot 直接吃。
"""
from __future__ import annotations

import heapq
import math

# 净空逐级候选 (m): 半径越大 = 要求通道越宽。机器人硬性下限 ROBOT_CLEAR=0.5
# (包络 0.3 + 0.2 余量) → 最窄走廊 1.0m, 决不钻 <1.0m 的窄缝。
ROBOT_CLEAR = 0.5
CLEAR_LEVELS = [0.7, 0.5]                 # 优先走 ≥1.4m, 退而求其次 ≥1.0m
FINE = 0.25                                # 机器人细格 (m), 与 demo_bridge 一致


def _bounds(md, frame):
    occ = set((o[0], o[1]) for o in md["obstacles"])
    xs = [o[0] for o in occ]
    ys = [o[1] for o in occ]
    x0 = frame.cx2m(min(xs)) - frame.res
    x1 = frame.cx2m(max(xs)) + frame.res
    y0 = frame.cy2m(min(ys)) - frame.res
    y1 = frame.cy2m(max(ys)) + frame.res
    return (x0, x1, y0, y1)


def _obstacle_cells(boxes, x0, y0, dilation):
    """每个障碍盒向外膨胀 dilation 后覆盖的细格集合。"""
    cells = set()
    for (cx, cy, hx, hy) in boxes:
        xa = cx - hx - dilation
        xb = cx + hx + dilation
        ya = cy - hy - dilation
        yb = cy + hy + dilation
        ix0 = int(math.floor((xa - x0) / FINE))
        ix1 = int(math.floor((xb - x0) / FINE))
        iy0 = int(math.floor((ya - y0) / FINE))
        iy1 = int(math.floor((yb - y0) / FINE))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                cells.add((ix, iy))
    return cells


def _fine_astar(start_xy, goal_xy, obs_cells, x0, y0, x1, y1):
    """细格 8 连通 A*, 起点/终点允许在障碍格, 返回米制点列 (含起点)。"""
    W = max(1, int(math.ceil((x1 - x0) / FINE)))
    H = max(1, int(math.ceil((y1 - y0) / FINE)))

    def cell(p):
        return (int(math.floor((p[0] - x0) / FINE)),
                int(math.floor((p[1] - y0) / FINE)))

    def xy(c):
        return ((c[0] + 0.5) * FINE + x0, (c[1] + 0.5) * FINE + y0)

    s, g = cell(start_xy), cell(goal_xy)
    if s == g:
        return [start_xy]

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(h(s, g), 0, s)]
    came_from = {}
    gcost = {s: 0}
    while open_set:
        _, _, cur = heapq.heappop(open_set)
        if cur == g:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return [xy(c) for c in path]
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
                heapq.heappush(open_set, (ng + h((nx, ny), g), ng, (nx, ny)))
    return None


def build_robot_plan(wp_str, md, frame, boxes):
    """地标序列 → 机器人路径 (米制, 细格 A*, 避开窄走廊)。
    每段优先走 ≥1.4m 宽通道 (clear=0.7), 退到 ≥1.0m (clear=0.5);
    若某地标只能经 <1.0m 窄缝到达 (全部候选失败) → 跳过该点, 绝不钻窄缝。"""
    names = [n.strip().upper() for n in wp_str.split(",") if n.strip()]
    lms = {lm["name"].upper(): frame.to_meters(lm["x"], lm["y"])
           for lm in md["landmarks"]}
    names = [n for n in names if n in lms]
    if len(names) < 2:
        raise ValueError(f"需要至少 2 个已知地标: {wp_str}")

    x0, x1, y0, y1 = _bounds(md, frame)
    out = []
    chain = [names[0]]
    prev = lms[names[0]]

    def _seg(a, b):
        for clear in CLEAR_LEVELS:
            obs = _obstacle_cells(boxes, x0, y0, clear)
            seg = _fine_astar(a, b, obs, x0, y0, x1, y1)
            if seg:
                return seg, clear
        return None, None

    for n in names[1:]:
        seg, clear = _seg(prev, lms[n])
        if seg is None:
            print(f"[robot_path] 跳过 {n}: 仅窄走廊(<1.0m)可达, 避开窄缝")
            continue
        out.extend(seg[1:])
        prev = lms[n]
        chain.append(n)

    if len(chain) < 2 or not out:
        raise ValueError("机器人路径为空: 所有目标都只能经窄走廊到达")
    print(f"[robot_path] {chain}: {len(out) + 1} waypoints "
          f"(最后一段净空 {clear}m → 走廊≥{2 * clear:.1f}m)")
    return {"start": list(lms[chain[0]]),
            "waypoints": [{"x": p[0], "y": p[1]} for p in out]}