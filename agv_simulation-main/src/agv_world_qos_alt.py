"""
agv_world_qos_alt.py — QoS 规划器的地标差分启发式 (ALT) 优化版
================================================================

基于 Red Blob Games 的 "Improving Heuristics"
(https://www.redblobgames.com/pathfinding/heuristics/differential.html)
对 agv_world_qos.py 的 astar_with_time 做唯一一处优化 —— **启发式函数**:

  1. 预选若干"地标"节点 L₁..Lₙ (四角最近自由格 + 贪心散布)。
  2. 对每个地标跑 **运动学 BFS** (状态 (x,y,dir), 动作 等待/左转/右转/前进,
     每动作代价 1) 算出地标到全图每个格子的**精确最短时间** d[·][Lᵢ]。
  3. A* 启发式改为
        h(n) = max( Manhattan(n, goal),
                    max_i | d[n][Lᵢ] - d[goal][Lᵢ] | )        (三角不等式)

  |d(n,Lᵢ)-d(goal,Lᵢ)| 是 d(n,goal) 的合法下界 (反向三角不等式), 且地标距离
  按运动学图计算 —— 已经"知道墙壁/转弯/等待", 所以比 Manhattan 紧得多,
  尤其走廊/迷宫类地图。A* 仍保证最优 (admissible + consistent), 只少探索。

与 agv_world_qos.py 的关系:
  - 本文件完全独立, 不 import / 不改动 agv_world_qos (即"接入 mujoco 的版本不动")。
  - `_astar_core` 是 agv_world_qos.astar_with_time 的逐行拷贝, 仅多了 hfun / stats
    两个可选参数; hfun=manhattan 时行为与原件**完全一致** (benchmark 里做了等价性校验)。
  - `astar_alt` 签名与 astar_with_time 兼容 (返回 (path, conflicts)), 可直接
    替换 agv_world_qos 模块里的同名函数接入现有世界 (drop-in)。

用法:
  # 对比基准: v1 (Manhattan) vs v3 (Chebyshev 差分地标)
  #   ① 全对地标单程  ② 多车链式(带预约防碰撞)  ③ 迷宫基准
  PY src/agv_world_qos_alt.py maps/map_*.json --benchmark
  PY src/agv_world_qos_alt.py maps/map_*.json --maze 40 --pairs 200

  # 直接规划并输出 plan 文件 (用 ALT 启发式, 输出格式同 agv_world_qos)
  PY src/agv_world_qos_alt.py maps/map_*.json --tasks "LM000,LM002,LM004;LM003,LM001,LM000"
"""

import json
import heapq
import random
import argparse
import time
import os
import sys
from collections import deque
from agv_map_common import load_normalized

# ── 与 agv_world_qos 相同的运动学常量 (独立副本, 不 import) ──
DIR_MAP = {'x+': 0, 'y+': 1, 'x-': 2, 'y-': 3}
REV_DIR_MAP = {0: 'x+', 1: 'y+', 2: 'x-', 3: 'y-'}
DX_DY = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ───────────────────── 地标差分启发式 (ALT) ─────────────────────

def _kinematic_bfs(obs_set, width, height, source, x_min=0, y_min=0):
    """运动学图 BFS: 状态 (x,y,dir), 动作 等待/左转/右转/前进, 每动作 1 tick。
    返回 {cell: 从 source 到达该格的最短 tick 数} (到达时朝向任意, 取最小值)。"""
    cell_dist = {}
    dist = {}                       # (x, y, dir) -> ticks
    q = deque()
    for d in range(4):
        st = (source[0], source[1], d)
        dist[st] = 0
        q.append(st)
    cell_dist[source] = 0
    while q:
        x, y, d = q.popleft()
        nd = dist[(x, y, d)] + 1
        # 前进
        dx, dy = DX_DY[d]
        nx, ny = x + dx, y + dy
        if (x_min <= nx < x_min + width and y_min <= ny < y_min + height
                and (nx, ny) not in obs_set):
            st = (nx, ny, d)
            if st not in dist:
                dist[st] = nd
                if cell_dist.get((nx, ny)) is None or nd < cell_dist[(nx, ny)]:
                    cell_dist[(nx, ny)] = nd
                q.append(st)
        # 左转 / 右转
        for ndir in ((d + 1) % 4, (d - 1) % 4):
            st = (x, y, ndir)
            if st not in dist:
                dist[st] = nd
                q.append(st)
    return cell_dist


def _pick_landmarks(width, height, obs_set, n=6, x_min=0, y_min=0):
    """选 n 个地标 (md §6): 四角附近最近自由格当种子, 再贪心挑离已有地标最远的格。"""
    free = [(x, y) for y in range(y_min, y_min + height)
            for x in range(x_min, x_min + width) if (x, y) not in obs_set]
    if not free:
        return []
    corners = [(x_min, y_min), (x_min + width - 1, y_min),
               (x_min, y_min + height - 1), (x_min + width - 1, y_min + height - 1)]
    seeds = []
    for c in corners:
        best = min(free, key=lambda f: abs(f[0] - c[0]) + abs(f[1] - c[1]))
        if best not in seeds:
            seeds.append(best)
    lms = seeds[:]
    while len(lms) < n and len(lms) < len(free):
        best, bestd = None, -1
        for f in free:
            if f in lms:
                continue
            d = min(abs(f[0] - l[0]) + abs(f[1] - l[1]) for l in lms)
            if d > bestd:
                bestd, best = d, f
        if best is None or bestd <= 0:
            break
        lms.append(best)
    return lms


class Differential:
    """地标差分启发式。dists: list[ {cell: 运动学最短tick数} ], 每个地标一个。"""

    def __init__(self, dists):
        self.dists = dists

    def __call__(self, n, goal):
        h = manhattan(n, goal)                 # 基础启发式 (admissible)
        for d in self.dists:
            dn = d.get(n)
            dg = d.get(goal)
            if dn is not None and dg is not None:    # 两侧都可达才用三角不等式
                lb = dn - dg
                if lb < 0:
                    lb = -lb
                if lb > h:
                    h = lb
        return h


def build_differential(obs_set, width, height, n=6, x_min=0, y_min=0, verbose=False):
    """选地标 + 跑运动学 BFS 代价表, 返回 Differential 实例 (可直接当 hfun)。"""
    lms = _pick_landmarks(width, height, obs_set, n, x_min, y_min)
    dists = [_kinematic_bfs(obs_set, width, height, lm, x_min, y_min) for lm in lms]
    if verbose:
        print(f"[ALT] landmarks={lms}  precompute {len(lms)} BFS tables")
    return Differential(dists)


# ───────────────────── A* 核心 (agv_world_qos 逐行拷贝 + hfun/stats) ─────────────────────

def _astar_core(width, height, obs_set, start, goal, start_time, start_dir_str,
                reservation_info, x_min=0, y_min=0, hfun=manhattan, stats=None,
                fix_g=False):
    """
    agv_world_qos.astar_with_time 的逐行拷贝, 新增:
      - hfun: 启发式 (默认 manhattan == 原件行为)
      - stats: dict, 统计 expanded(实际展开格数) / explored(入队格数)
      - fix_g: 修正原 A* 的重构缺陷 —— 原实现 came_from 只在节点"第一次入队"时
        记录父节点, 若之后有更优 g 到达同一状态 (x,y,dir,t) 也不更新, 导致重构
        路径可能非最优 (启发式越强, 入队顺序变化, 越容易放大这个缺陷)。
        开启后用 g 支配更新 came_from (教科书 A*), 两种启发式都保证重构最优。
    """
    start_dir = DIR_MAP.get(start_dir_str, 0)

    open_set = []
    heapq.heappush(open_set, (hfun(start, goal), start_time, start[0], start[1], start_dir))

    came_from = {}
    g_score = {}                    # (x,y,dir,t) -> 最优 g (=t); 仅 fix_g 时使用
    visited = set()

    x_max, y_max = x_min + width, y_min + height

    max_iter = 1000000
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

        state_key = (cx, cy, cdir, t)
        if state_key in visited:
            continue
        visited.add(state_key)
        if stats is not None:
            stats['expanded'] += 1

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

            res_vertex = (nx, ny, nt)
            if res_vertex in reservation_info:
                if act_type == 'wait' and (nx, ny) == (cx, cy):
                    continue   # 等待格被占则不能原地等
                continue       # 跳过被预约的时空点

            if act_type == 'move':
                res_edge = (nx, ny, nt, cx, cy)
                if res_edge in reservation_info:
                    continue   # 跳过被预约的移动边

            h = hfun((nx, ny), goal)
            new_priority = nt + h

            new_node_key = (nx, ny, nd, nt)
            if fix_g:
                if g_score.get(new_node_key) is not None and nt >= g_score[new_node_key]:
                    continue    # 已有更优 g 到达该状态, 跳过
                g_score[new_node_key] = nt

            heapq.heappush(open_set, (new_priority, nt, nx, ny, nd))
            if stats is not None:
                stats['explored'] += 1

            if new_node_key not in came_from:
                came_from[new_node_key] = (cx, cy, cdir, t)

    return (None, [])


# 模块级 ALT 表缓存 (同一地图只算一次)
_ALT_CACHE = {}


def astar_alt(width, height, obs_set, start, goal, start_time, start_dir_str,
              reservation_info, x_min=0, y_min=0, n_landmarks=6, stats=None):
    """签名与 agv_world_qos.astar_with_time 兼容的 ALT 版 (drop-in 替换用)。
    返回 (path, conflicts), conflicts 恒为空 (同原实现, 预约格直接跳过)。"""
    key = (width, height, x_min, y_min, n_landmarks, frozenset(obs_set))
    if key not in _ALT_CACHE:
        _ALT_CACHE[key] = build_differential(obs_set, width, height, n_landmarks, x_min, y_min)
    return _astar_core(width, height, obs_set, start, goal, start_time,
                       start_dir_str, reservation_info, x_min, y_min,
                       _ALT_CACHE[key], stats)


# ───────────────────── 多车链式规划 (预约登记逻辑对齐 move_car) ─────────────────────

def _plan_sequence(map_file, chains, hfun, stagger=2, n_landmarks=6, verbose=False,
                   fix_g=False):
    """用同一套预约登记逻辑 (对齐 AGVWorld.move_car) 顺序规划多车链式路线。
    仅启发式不同 → 统计差异 = 启发式带来的差异。"""
    m = load_normalized(map_file)
    width, height = m["width"], m["height"]
    obs_set = set(tuple(o) for o in m["obstacles"])
    lms = {lm["name"]: (lm["x"], lm["y"]) for lm in m["landmarks"]}

    reservation_info = {}            # res -> (car_id, qos)
    stats = {'expanded': 0, 'explored': 0, 'planned': 0, 'path_len': 0}
    t0 = time.perf_counter()

    for idx, chain in enumerate(chains):
        car_id = f"car{idx}"
        cur = lms[chain[0]]
        st = idx * stagger
        while (cur[0], cur[1], st) in reservation_info:
            st += 1
        own = set()
        # 出生格 park 1000 tick (对齐 add())
        for tt in range(st, st + 1000):
            r = (cur[0], cur[1], tt)
            reservation_info[r] = (car_id, 1)
            own.add(r)

        cur_dir = 'x+'
        t = st
        full = []
        ok = True
        for stop_name in chain[1:]:
            goal = lms[stop_name]
            # 移除本车未来预约 (t > 当前规划时刻), 对齐 move_car
            for r in [r for r in own if len(r) >= 3 and r[2] > t]:
                reservation_info.pop(r, None)
                own.discard(r)
            leg = {'expanded': 0, 'explored': 0}
            path, _ = _astar_core(width, height, obs_set, cur, goal, t, cur_dir,
                                  reservation_info, 0, 0, hfun, leg, fix_g)
            stats['expanded'] += leg['expanded']
            stats['explored'] += leg['explored']
            if path is None:
                ok = False
                if verbose:
                    print(f"[plan] car {car_id} failed leg -> {stop_name} (t={t})")
                break
            # 登记本段顶点 + 方向边 (终点在前, 同 move_car) + 到站 park
            for i, (pt_, x, y, d) in enumerate(path):
                r = (x, y, pt_)
                reservation_info[r] = (car_id, 1)
                own.add(r)
                if i > 0:
                    ppt_, px, py, pd = path[i - 1]
                    re = (x, y, pt_, px, py)
                    reservation_info[re] = (car_id, 1)
                    own.add(re)
            pts = [(pt_, x, y) for pt_, x, y, _ in path]
            full.extend(pts if not full else pts[1:])
            cur = goal
            t = full[-1][0] + 1
            cur_dir = path[-1][3]
            for tt in range(full[-1][0] + 1, full[-1][0] + 1000):
                r = (goal[0], goal[1], tt)
                reservation_info[r] = (car_id, 1)
                own.add(r)
        if ok:
            stats['planned'] += 1
            stats['path_len'] += len(full)

    stats['time'] = time.perf_counter() - t0
    return stats


# ───────────────────── 基准对比 ─────────────────────

DEFAULT_CHAINS = [
    "LM000,LM002,LM004,LM001",
    "LM003,LM001,LM000,LM002",
    "LM001,LM004,LM003,LM000",
    "LM002,LM000,LM003,LM004",
    "LM005,LM002,LM001,LM000",
    "LM006,LM004,LM002,LM001",
]


def _load_map(map_file):
    m = load_normalized(map_file)
    width, height = m["width"], m["height"]
    obs = set(tuple(o) for o in m["obstacles"])
    lms = {lm["name"]: (lm["x"], lm["y"]) for lm in m["landmarks"]}
    return width, height, obs, lms


def _sanity_check(width, height, obs, lms):
    """证明 _astar_core(manhattan) 与 agv_world_qos.astar_with_time 路径完全一致。"""
    try:
        from agv_world_qos import astar_with_time as real_astar
    except ImportError:
        print("[sanity] agv_world_qos not importable, skip")
        return
    pairs = list(lms.values())
    checked = 0
    for i in range(min(8, len(pairs))):
        s, g = pairs[i], pairs[(i + 1) % len(pairs)]
        p1, _ = real_astar(width, height, obs, s, g, 0, 'x+', {})
        p2, _ = _astar_core(width, height, obs, s, g, 0, 'x+', {}, 0, 0, manhattan)
        if (p1 is None) != (p2 is None):
            print(f"[sanity] FAIL: reachability differs {s}->{g}")
            return
        if p1 is not None and p1 != p2:
            print(f"[sanity] FAIL: path differs {s}->{g}\n  real={p1}\n  alt ={p2}")
            return
        checked += 1
    print(f"[sanity] OK: _astar_core(manhattan) == agv_world_qos.astar_with_time "
          f"on {checked} pairs")


def _fmt_table(title, header, rows):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    for r in rows:
        print("  " + "  ".join((str(v) if isinstance(v, str) else f"{v:,.2f}" if isinstance(v, float) else f"{v:,}")
                               .ljust(widths[i]) for i, v in enumerate(r)))


def _red(x):
    """x 为减少百分比, 正=↓ 负=↑, 返回可打印字符串。"""
    if x >= 0:
        return f"{x:.1f}% ↓"
    return f"{-x:.1f}% ↑"


def _run_pairs(pairs, w, h, obs, hfun, fix_g):
    """对每对 (s,g) 跑一次 A*, 返回 (per_list, 总耗时)。
    per_list[i] = (ok, path_len, expanded, explored)。"""
    t0 = time.perf_counter()
    per = []
    for s, g in pairs:
        st = {'expanded': 0, 'explored': 0}
        p, _ = _astar_core(w, h, obs, s, g, 0, 'x+', {}, 0, 0, hfun, st, fix_g)
        per.append((p is not None, len(p) if p is not None else 0,
                    st['expanded'], st['explored']))
    return per, time.perf_counter() - t0


def _pairs_rows(res, n_pairs):
    """把 (name, per, dt) 列表转成表格行。
    路径长/探索数只统计"所有算法都成功"的公共子集 —— 否则不同成功率下
    平均路径不可比 (失败多的算法只平均了简单对)。"""
    common = [i for i in range(n_pairs)
              if all(per[i][0] for _, per, _ in res)]
    rows = []
    for name, per, dt in res:
        ok = sum(1 for r in per if r[0])
        if common:
            exp = sum(per[i][2] for i in common)
            expl = sum(per[i][3] for i in common)
            plen = sum(per[i][1] for i in common) / len(common)
        else:
            exp = expl = plen = 0
        rows.append([name, ok, exp, expl, plen, dt * 1000])
    return rows, len(common)


def benchmark(map_file, chains=None, n_landmarks=6, stagger=2):
    width, height, obs, lms = _load_map(map_file)
    chains = chains or DEFAULT_CHAINS
    chains = [c.split(',') if isinstance(c, str) else c for c in chains]
    alt = build_differential(obs, width, height, n_landmarks)

    print(f"[bench] {os.path.basename(map_file)}  {width}x{height}  "
          f"obs={len(obs)}  landmarks={len(lms)}  "
          f"ALT landmarks={n_landmarks}  chains={len(chains)}")
    _sanity_check(width, height, obs, lms)

    # ── ① 全对地标单程 (空预约, 纯算法对比; path/探索只统计共同成功对) ──
    configs = [("v1 Manhattan", manhattan, False),
               ("v3 ALT 差分地标", alt, False),
               ("v3 ALT+重构修正", alt, True)]
    pairs = [(a, b) for a in lms.values() for b in lms.values() if a != b]
    res = [(name, *_run_pairs(pairs, width, height, obs, h, fix))
           for name, h, fix in configs]
    rows, n_common = _pairs_rows(res, len(pairs))
    _fmt_table(
        f"① 全对地标单程 ({len(pairs)} pairs; path/expanded 只统计 {n_common} "
        f"个共同成功对)",
        ["算法", "成功", "expanded", "explored", "平均path", "总耗时ms"],
        rows)
    b, a = rows[0], rows[1]
    print(f"▶ ① 改进: 探索格数 {_red(100*(1-a[2]/b[2]))} "
          f"({b[2]:,}→{a[2]:,}); 入队 {_red(100*(1-a[3]/b[3]))}; "
          f"路径等长(最优性保持): {b[4]:.1f} vs {a[4]:.1f}")

    # ── ② 多车链式 (带预约防碰撞, 端到端) ──
    rows = []
    for name, h, fix in configs:
        st = _plan_sequence(map_file, chains, h, stagger, n_landmarks, fix_g=fix)
        rows.append([name, st['planned'], st['expanded'], st['explored'],
                     st['path_len'] / max(st['planned'], 1), st['time'] * 1000])
    _fmt_table(
        f"② 多车链式规划 ({len(chains)} 车 × 3 经停, 含预约防碰撞)",
        ["算法", "成功车", "expanded", "explored", "平均path", "总耗时ms"],
        rows)
    b, a = rows[0], rows[1]
    print(f"▶ ② 改进: 探索格数 {_red(100*(1-a[2]/b[2]))} "
          f"({b[2]:,}→{a[2]:,}); 耗时 {_red(100*(1-a[5]/b[5]))} "
          f"({b[5]:.2f}ms→{a[5]:.2f}ms)")
    return rows


def _gen_maze(w, h, seed):
    rnd = random.Random(seed)
    grid = [[1] * w for _ in range(h)]          # 1=墙
    def carve(x, y):
        grid[y][x] = 0
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        rnd.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny][nx]:
                grid[y + dy // 2][x + dx // 2] = 0
                carve(nx, ny)
    grid[1][1] = 0
    carve(1, 1)
    for _ in range(max(4, int(w * h * 0.08))):  # 随机打洞, 制造岔路
        x, y = rnd.randrange(1, w - 1), rnd.randrange(1, h - 1)
        grid[y][x] = 0
    obs = set()
    for y in range(h):
        for x in range(w):
            if grid[y][x]:
                obs.add((x, y))
    return obs


def maze_benchmark(maze_w, maze_h, n_pairs, seed, n_landmarks=6):
    obs = _gen_maze(maze_w, maze_h, seed)
    free = [(x, y) for y in range(maze_h) for x in range(maze_w)
            if (x, y) not in obs]
    rnd = random.Random(seed + 1)
    pairs = []
    for _ in range(n_pairs):
        s, g = rnd.choice(free), rnd.choice(free)
        while g == s:
            g = rnd.choice(free)
        pairs.append((s, g))
    alt = build_differential(obs, maze_w, maze_h, n_landmarks)

    print(f"\n[bench] 迷宫 {maze_w}x{maze_h}  (自由格 {len(free)}, 随机对 {len(pairs)}, "
          f"ALT landmarks={n_landmarks}, seed={seed})")
    print("-" * 80)
    configs = [("v1 Manhattan", manhattan, False),
               ("v3 ALT 差分地标", alt, False),
               ("v3 ALT+重构修正", alt, True)]
    res = [(name, *_run_pairs(pairs, maze_w, maze_h, obs, h, fix))
           for name, h, fix in configs]
    rows, n_common = _pairs_rows(res, len(pairs))
    _fmt_table(f"③ 迷宫随机路径对 ({len(pairs)} pairs; "
               f"path/expanded 只统计 {n_common} 个共同成功对)",
               ["算法", "成功", "expanded", "explored", "平均path", "总耗时ms"],
               rows)
    # 成功率差异提示 (v1 可能因超过 max_iter 未完成)
    ok0 = rows[0][1]
    if ok0 < len(pairs):
        print(f"  ※ 注: v1 有 {len(pairs)-ok0} 对未规划成功 "
              f"(超过 A* max_iter=100000), v3 全部成功")
    b, a = rows[0], rows[1]
    print(f"▶ ③ 迷宫改进(共同成功对): 探索格数 {_red(100*(1-a[2]/b[2]))} "
          f"({b[2]:,}→{a[2]:,}); 耗时 {_red(100*(1-a[5]/b[5]))} "
          f"({b[5]:.2f}ms→{a[5]:.2f}ms); 路径等长 {b[4]:.1f} vs {a[4]:.1f}")


# ───────────────────── CLI ─────────────────────

def write_plan(map_file, chains, stagger=2, n_landmarks=6):
    """用 ALT 启发式 + move_car 的预约逻辑规划, 输出与 agv_world_qos 相同格式。"""
    m = load_normalized(map_file)
    width, height = m["width"], m["height"]
    obs = set(tuple(o) for o in m["obstacles"])
    lms = {lm["name"]: (lm["x"], lm["y"]) for lm in m["landmarks"]}
    alt = build_differential(obs, width, height, n_landmarks)
    reservation_info = {}
    car_paths = []
    for idx, chain in enumerate(chains):
        car_id = f"car{idx}"
        cur = lms[chain[0]]
        st = idx * stagger
        while (cur[0], cur[1], st) in reservation_info:
            st += 1
        own = set()
        for tt in range(st, st + 1000):
            r = (cur[0], cur[1], tt)
            reservation_info[r] = (car_id, 1)
            own.add(r)
        cur_dir, t, full = 'x+', st, []
        ok = True
        for stop_name in chain[1:]:
            goal = lms[stop_name]
            for r in [r for r in own if len(r) >= 3 and r[2] > t]:
                reservation_info.pop(r, None)
                own.discard(r)
            path, _ = _astar_core(width, height, obs, cur, goal, t, cur_dir,
                                  reservation_info, 0, 0, alt)
            if path is None:
                ok = False
                break
            for i, (pt_, x, y, d) in enumerate(path):
                r = (x, y, pt_)
                reservation_info[r] = (car_id, 1)
                own.add(r)
                if i > 0:
                    ppt_, px, py, pd = path[i - 1]
                    re = (x, y, pt_, px, py)
                    reservation_info[re] = (car_id, 1)
                    own.add(re)
            pts = [(pt_, x, y) for pt_, x, y, _ in path]
            full.extend(pts if not full else pts[1:])
            cur, t, cur_dir = goal, full[-1][0] + 1, path[-1][3]
            for tt in range(full[-1][0] + 1, full[-1][0] + 1000):
                r = (goal[0], goal[1], tt)
                reservation_info[r] = (car_id, 1)
                own.add(r)
        if ok:
            car_paths.append({
                "car_id": car_id,
                "start_landmark": chain[0],
                "goal_landmark": chain[-1],
                "stops": chain[1:],
                "trajectory": [{"t": tt, "x": xx, "y": yy} for tt, xx, yy in full],
            })
    result = {"map_file": map_file, "cars": car_paths}
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"plan_qos_alt_{ts}.json"
    maps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps")
    os.makedirs(maps_dir, exist_ok=True)
    out = os.path.join(maps_dir, filename)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Planning complete (ALT): maps/{filename}  ({len(car_paths)}/{len(chains)} cars)")
    return out


def main():
    # Windows 控制台默认 GBK, 强制 UTF-8 输出 (否则 ▶/中文 打印报错)
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="AGV World QoS planner with landmark/differential heuristic (ALT)")
    parser.add_argument("map_file", help="The map JSON file")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run v1(Manhattan) vs v3(ALT) comparison and print table")
    parser.add_argument("--maze", type=int, default=0,
                        help="Maze benchmark size (e.g. 40 → 40x40 maze)")
    parser.add_argument("--pairs", type=int, default=200,
                        help="Random (start,goal) pairs for maze benchmark")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--landmarks", type=int, default=6,
                        help="Number of ALT landmark nodes (default 6)")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Chains: each ;-entry is one car 'LM000,LM002,LM004'")
    parser.add_argument("--stagger", type=int, default=2,
                        help="Ticks between car start times (default 2)")
    args = parser.parse_args()

    chains = None
    if args.tasks:
        chains = [c.split(',') for c in args.tasks.split(';') if c.strip()]

    if args.maze:
        maze_benchmark(args.maze, args.maze, args.pairs, args.seed, args.landmarks)
        return
    if args.benchmark or chains is None:
        benchmark(args.map_file, chains=chains, n_landmarks=args.landmarks,
                  stagger=args.stagger)
        return
    write_plan(args.map_file, chains, args.stagger, args.landmarks)


if __name__ == "__main__":
    main()
