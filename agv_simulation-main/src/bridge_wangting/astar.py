"""
astar.py — 修正版 ALT 运动学时空 A* (bridge_wangting)
====================================================

原 agv_world_qos_alt._astar_core 在大图 (0.1m) 上 (x,y,dir,t) 时间维爆掉:
同一 (x,y,dir) 会被以无数不同 t 反复 push → iterations 撞 max_iter=1e6 → 返回 None
(15 车小图状态空间小, 不触发; 大图必然触发)。

修复:
  1. best_cost_to_state[(x,y,dir)] -> min_time 剪枝: 同格同向更晚到达直接跳过
     (等待本就在原地生成, 等价路径已覆盖, 状态数从无限塌缩到 ~272k)
  2. 保留 ALT 差分启发式 (比 Manhattan 展开少)
  3. max_iter 放宽到 1e7

签名与 agv_world_qos.astar_with_time 兼容 (返回 (path, conflicts), conflicts 恒空)。
"""
from __future__ import annotations

import heapq
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agv_world_qos_alt import manhattan, build_differential, \
    DIR_MAP, REV_DIR_MAP, DX_DY            # noqa: E402

_MAX_ITER = 10000000

# 差分表缓存: 同一 (地图, obs_set) 只跑一次地标运动学 BFS (~4s), 之后所有段复用
_DIFF_CACHE = {}


def _get_diff(obs_set, width, height, n_landmarks, x_min, y_min):
    key = (width, height, x_min, y_min, n_landmarks, frozenset(obs_set))
    if key not in _DIFF_CACHE:
        _DIFF_CACHE[key] = build_differential(obs_set, width, height,
                                              n_landmarks, x_min, y_min)
    return _DIFF_CACHE[key]


def astar_alt_fixed(width, height, obs_set, start, goal, start_time, start_dir_str,
                    reservation_info, x_min=0, y_min=0, n_landmarks=6,
                    max_iter=_MAX_ITER):
    """修正版 ALT 运动学时空 A*。返回 (path, conflicts=[])。"""
    diff = _get_diff(obs_set, width, height, n_landmarks, x_min, y_min)

    def hfun(n, g):
        return max(manhattan(n, g), diff(n, g))

    start_dir = DIR_MAP.get(start_dir_str, 0)
    open_set = [(hfun(start, goal), start_time, start[0], start[1], start_dir)]
    came_from = {}
    g_score = {}               # (x,y,dir) -> min_time
    visited = set()
    x_max, y_max = x_min + width, y_min + height
    iterations = 0

    while open_set and iterations < max_iter:
        iterations += 1
        priority, t, cx, cy, cdir = heapq.heappop(open_set)

        if (cx, cy) == goal:
            path = []
            curr_state = (cx, cy, cdir, t)
            while curr_state in came_from:
                x, y, d, tm = curr_state
                path.append((tm, x, y, REV_DIR_MAP[d]))
                curr_state = came_from[curr_state]
            x, y, d, tm = curr_state
            path.append((tm, x, y, REV_DIR_MAP[d]))
            return (list(reversed(path)), [])

        # best_cost_to_state 剪枝: 同格同向更晚到达跳过 (等待已在原地覆盖)
        key = (cx, cy, cdir)
        if g_score.get(key) is not None and t > g_score[key]:
            continue
        g_score[key] = t

        state_key = (cx, cy, cdir, t)
        if state_key in visited:
            continue
        visited.add(state_key)

        next_actions = []
        next_actions.append(('wait', cx, cy, cdir))
        next_actions.append(('turn', cx, cy, (cdir + 1) % 4))
        next_actions.append(('turn', cx, cy, (cdir - 1) % 4))
        dx, dy = DX_DY[cdir]
        nx, ny = cx + dx, cy + dy
        if x_min <= nx < x_max and y_min <= ny < y_max and (nx, ny) not in obs_set:
            next_actions.append(('move', nx, ny, cdir))

        for act_type, nx, ny, nd in next_actions:
            nt = t + 1
            if (nx, ny, nt) in reservation_info:
                continue
            if act_type == 'move' and (nx, ny, nt, cx, cy) in reservation_info:
                continue
            h = hfun((nx, ny), goal)
            new_node_key = (nx, ny, nd, nt)
            heapq.heappush(open_set, (nt + h, nt, nx, ny, nd))
            if new_node_key not in came_from:
                came_from[new_node_key] = (cx, cy, cdir, t)

    return (None, [])


# ───────────────────── 等待直到空闲 (时间折叠 EAT) ─────────────────────

_MAX_WAIT = 1000     # 单次等待上限 (tick)


def _earliest_free(sx, sy, tx, ty, from_t, reservation_info):
    """最早时刻 s>=from_t+1: 车从 (sx,sy) 出发于 s 到达 (tx,ty)。
    车在 (sx,sy) 等待 [from_t+1, s-1] (需全空), 在 s 移动 (顶点+边需空)。
    返回 s 或 None (等待超 _MAX_WAIT)。"""
    s = from_t + 1
    while s - from_t <= _MAX_WAIT:
        ok = True
        for tt in range(from_t + 1, s):
            if (sx, sy, tt) in reservation_info:
                ok = False
                break
        if ok:
            if (tx, ty, s) in reservation_info:
                ok = False
            elif (sx, sy, s, tx, ty) in reservation_info:
                ok = False
            elif (tx, ty, s, sx, sy) in reservation_info:
                ok = False
        if ok:
            return s
        s += 1
    return None


def astar_waitfree(width, height, obs_set, start, goal, start_time, start_dir_str,
                   reservation_info, x_min=0, y_min=0, n_landmarks=6,
                   max_iter=_MAX_ITER):
    """等待直到空闲版运动学时空 A* (时间折叠):
      状态键 = (x,y,dir) 不含 t; g[(x,y,dir)] = 最早到达时刻。
      移动到邻居被占时 → 原地算最早空闲时刻直接跳过去 (等待隐式, 不生成等待态)。
      状态数从 (x,y,dir,t) 无限维塌缩到 (x,y,dir) 至多 ~272k。"""
    diff = _get_diff(obs_set, width, height, n_landmarks, x_min, y_min)

    def hfun(n, g):
        return max(manhattan(n, g), diff(n, g))

    start_dir = DIR_MAP.get(start_dir_str, 0)
    start_key = (start[0], start[1], start_dir)
    g = {start_key: start_time}
    came_from = {}
    open_set = [(hfun(start, goal), start_time, start[0], start[1], start_dir)]
    x_max, y_max = x_min + width, y_min + height
    iterations = 0

    while open_set and iterations < max_iter:
        iterations += 1
        priority, t, cx, cy, cdir = heapq.heappop(open_set)
        key = (cx, cy, cdir)
        if g.get(key) is not None and t > g[key]:
            continue                      # 已有更早到达, 晚到态丢弃

        if (cx, cy) == goal:
            path = []
            curr = key
            while curr in came_from:
                x, y, d = curr
                path.append((g[curr], x, y, REV_DIR_MAP[d]))
                curr = came_from[curr]
            x, y, d = curr
            path.append((g[curr], x, y, REV_DIR_MAP[d]))
            path.reverse()
            # 展开等待点: 相邻两点时间不连续则补原地等待
            expanded = []
            for i, pt in enumerate(path):
                expanded.append(pt)
                if i + 1 < len(path):
                    t1, x1, y1, d1 = pt
                    t2 = path[i + 1][0]
                    for tt in range(t1 + 1, t2):
                        expanded.append((tt, x1, y1, d1))
            return (expanded, [])

        # 转向 (同格, 需格在等待窗内全空, 含转向时刻)
        for nd in ((cdir + 1) % 4, (cdir - 1) % 4):
            s = from_t = t
            s = s + 1
            while s - from_t <= _MAX_WAIT:
                ok = True
                for tt in range(from_t + 1, s + 1):
                    if (cx, cy, tt) in reservation_info:
                        ok = False
                        break
                if ok:
                    nk = (cx, cy, nd)
                    if g.get(nk) is None or s < g[nk]:
                        g[nk] = s
                        came_from[nk] = key
                        heapq.heappush(open_set, (s + hfun((cx, cy), goal), s, cx, cy, nd))
                    break
                s += 1

        # 前进 (等待直到空闲)
        dx, dy = DX_DY[cdir]
        nx, ny = cx + dx, cy + dy
        if x_min <= nx < x_max and y_min <= ny < y_max and (nx, ny) not in obs_set:
            s = _earliest_free(cx, cy, nx, ny, t, reservation_info)
            if s is not None:
                nk = (nx, ny, cdir)
                if g.get(nk) is None or s < g[nk]:
                    g[nk] = s
                    came_from[nk] = key
                    heapq.heappush(open_set, (s + hfun((nx, ny), goal), s, nx, ny, cdir))

    return (None, [])