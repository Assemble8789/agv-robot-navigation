"""
C_F (conflict-free) 式动态避碰重规划
====================================
借鉴论文《考虑变速运动的多AGV分拣系统人机动态避碰问题研究》的两阶段框架:

  第一阶段(基础路径):  干净的空间 A* (plan_spatial_path) —— 只避静态障碍,
                      不绕全图。这是对原"时间感知 A* 绕疯路"问题的根治。
  第二阶段(运行时消解): 沿路径按时间轴展开,逐点预测与"介入者"
                      (机器人 / 其他 AGV) 的时间对齐冲突,然后按代价决策:

                      决策规则 (论文式 3-16~3-19):
                        remain(阻塞者剩余停留) <= detour(绕行代价)
                            -> 等待   (在原路径当前位置插入停留 hold)
                        否则
                            -> 绕行   (局部 A* 绕开阻塞格, 不绕全图)

  复合干预:  多 AGV 冲突时按"已避让次数"排序 —— 避让多的获得优先权
            (avoid_count 由调用方维护, 见 track_avoidance)。

核心函数: plan_conflict_free(cur, goals, ...) -> [(x, y, hold_s), ...] | None
返回值与 run_factory_full_agv_yield.py 的 replan_agv_car_time 兼容:
  hold_s 秒的停留编码为轨迹中的重复坐标点 (平滑后的 Catmull-Rom 保持不变)。
"""

import numpy as np

# ── 参数 (可调) ──
CONFLICT_DIST = 0.5     # 时间对齐冲突半径 (m), 略高于 AGV 接触 ~0.4m
DETOUR_COST = 3.0       # 绕行代价估计 (s): 阻塞者停留超过它 -> 绕行更划算
WAIT_STEP = 0.3         # 等待分辨率 (s)
MAX_WAIT = 6.0          # 单点最大等待 (s) —— 超过则更倾向于绕行
HORIZON = 4.0           # remain 估算的采样视界 (s)


def _conflict(wx, wy, t, cid, clocks, agv_trajs, get_agv_pos, agv_speed,
              base_t, robot=None):
    """返回时刻 t 时占据 (wx,wy) 的介入者标识, 无冲突返回 None.

    标识:  'robot' 或其它 AGV 的 ocid。  t 为 AGV 自身的新路径时钟;
    其它 AGV 按 clocks[ocid] + (t - base_t) 对齐 (所有 clock 同速推进)。
    """
    for ocid in agv_trajs:
        if ocid == cid:
            continue
        ox, oy = get_agv_pos(ocid, clocks[ocid] + (t - base_t))
        if np.hypot(ox - wx, oy - wy) < CONFLICT_DIST:
            return ocid
    if robot is not None:
        rx, ry, rvx, rvy = robot
        dt = (t - base_t) / agv_speed       # clock 差 -> sim 秒
        rxp = rx + rvx * dt
        ryp = ry + rvy * dt
        if np.hypot(rxp - wx, ryp - wy) < CONFLICT_DIST:
            return "robot"
    return None


def _remain(blocker, wx, wy, cid, clocks, agv_trajs, get_agv_pos,
            agv_speed, base_t):
    """估算介入者在冲突点 (wx,wy) 的剩余停留时间 (s)。

    - 其它 AGV: 沿其轨迹采样, 看多久离开冲突半径 (计划驻留会给出大 remain)。
    - 机器人:   位置基本停住时 remain 大 -> 倾向绕行; 快速通过时小 -> 倾向等待。
    """
    if blocker == "robot":
        # 保守: 机器人行为不确定, 默认它停留足够久 (触发绕行决策)。
        # 调用方若知道机器人很快通过, 可覆盖此值。
        return HORIZON + 1.0
    remain = 0.0
    while remain < HORIZON:
        ox, oy = get_agv_pos(blocker, clocks[blocker] + remain)
        if np.hypot(ox - wx, oy - wy) >= CONFLICT_DIST:
            return remain
        remain += WAIT_STEP
    return HORIZON


def _block_cells(blocker, wx, wy, robot):
    """局部绕行要挡掉的粗格集合: 介入者当前格 + 机器人当前格。"""
    cells = set()
    if blocker != "robot":
        cells.add((int(round(wx)), int(round(wy))))   # 阻塞 AGV 的预测位置格
    if robot is not None:
        rx, ry, _, _ = robot
        cells.add((int(round(rx)), int(round(ry))))
    return cells


def _free_cell(qx, qy, coarse_obs, width, height):
    """世界坐标 (qx,qy) 对应的粗格是否可通行。"""
    cx, cy = int(round(qx)), int(round(qy))
    if not (0 <= cx < width and 0 <= cy < height):
        return False
    return (cx, cy) not in coarse_obs


def _shift_run(px, py, i, path, blocker, nxt_t, cid, clocks, agv_trajs,
               get_agv_pos, agv_speed, base_t, coarse_obs, width, height,
               robot):
    """论文 B' 侧移算子 (扩展为一整段): 冲突点在横向上离阻塞者太近时
    (等待改不了横向几何), 把从 i+1 起的一段连续 waypoint 垂直平移到远离
    阻塞者的一侧, 使并排经过获得 >0.7m 间隙; 阻塞者让出后回到原车道。

    返回 True 表示 path 被修改 (调用方 continue 重新走该段)。
    """
    nx, ny = path[i + 1]
    dirv = np.array([nx - px, ny - py])
    dl = np.hypot(*dirv)
    if dl < 1e-6:
        return False
    perp = np.array([-dirv[1], dirv[0]]) / dl
    if blocker == "robot":
        if robot is None:
            return False
        bx, by = robot[0], robot[1]
    else:
        bx, by = get_agv_pos(blocker, clocks[blocker] + (nxt_t - base_t))
    # 阻塞者在路径段横向上的偏移 —— 接近 0 表示正对 (对向), 交给 wait/detour
    lat = perp[0] * (bx - nx) + perp[1] * (by - ny)
    if abs(lat) < 0.05:
        return False
    away = -perp if lat > 0 else perp
    s_best = None
    for s in (0.25, 0.5, 0.75, 1.0):
        qx, qy = nx + away[0] * s, ny + away[1] * s
        if not _free_cell(qx, qy, coarse_obs, width, height):
            break
        if np.hypot(qx - bx, qy - by) >= 0.8:   # 侧移后离阻塞者 >0.8m (> 斥力 0.7)
            s_best = s
            break
    if s_best is None:
        return False
    off = away * s_best
    k = i + 1
    while k < len(path):
        ox_, oy_ = path[k]
        qx, qy = ox_ + off[0], oy_ + off[1]
        if not _free_cell(qx, qy, coarse_obs, width, height):
            break                       # 侧移道到头了, 退回原车道
        if blocker != "robot":
            # 到达该 waypoint 的大致 clock: nxt_t + 从 i+1 到 k 的段耗时
            tk = nxt_t
            for _j in range(i + 1, k):
                tk += np.hypot(path[_j + 1][0] - path[_j][0],
                               path[_j + 1][1] - path[_j][1]) / agv_speed
            bk = get_agv_pos(blocker, clocks[blocker] + (tk - base_t))
            if np.hypot(qx - bk[0], qy - bk[1]) >= 0.8:
                break                   # 阻塞者已让开 —— 停止平移, 回原车道
        path[k] = (round(qx, 3), round(qy, 3))
        k += 1
    return k > i + 1


def plan_conflict_free(cur, goals, coarse_obs, width, height, cid, base_t,
                       clocks, agv_trajs, get_agv_pos, agv_speed, nf,
                       robot=None):
    """C_F 式单 AGV 重规划: 干净空间 A* 基础路径 + 运行时 wait/detour/侧移决策。

    参数:
      cur        当前世界坐标 (x, y)
      goals      剩余目标点列表 [(x,y), ...]
      coarse_obs 粗网格静态障碍 (set)
      width,height 地图粗格数
      cid        本 AGV id
      base_t     新路径的起始 clock
      clocks     各 AGV 当前 clock
      agv_trajs  各 AGV 轨迹 {cid: [(t,x,y), ...]}
      get_agv_pos (ocid, t) -> (x, y)   预测函数
      agv_speed  时钟/速度系数
      nf         navigate_factory 模块 (提供 replan_agv_car / plan_spatial_path)
      robot      机器人 (rx, ry, rvx, rvy) 或 None (None 表示不避机器人)

    返回 [(x, y, hold_s), ...] 或 None。  hold_s>0 表示到达该点后停留。
    """
    # ── 1. 干净空间 A* 基础路径 (逐段拼接所有剩余目标) ──
    path = []
    cur_i = cur
    for g in goals:
        leg = nf.replan_agv_car(cur_i, g, coarse_obs, set(), width, height)
        if not leg:
            return None
        path = path + leg if not path else path + leg[1:]
        cur_i = g
    if not path:
        return None

    # ── 2. C_F 运行时消解: 沿路径按时间轴展开 ──
    wps = []
    acc = base_t          # 当前 waypoint 的到达 clock
    prev = None
    i = 0
    last_detour_i = -1    # 同一下标只绕行一次, 防止"绕行路径重新冲突"死循环
    safety = 0
    while i < len(path) and safety < 10000:
        safety += 1
        x, y = path[i]
        if prev is not None:
            acc += np.hypot(x - prev[0], y - prev[1]) / agv_speed
        hold = 0.0

        if i < len(path) - 1:
            nx, ny = path[i + 1]
            d_next = np.hypot(nx - x, ny - y) / agv_speed
            nxt_t = acc + d_next
            blocker = _conflict(nx, ny, nxt_t, cid, clocks, agv_trajs,
                                get_agv_pos, agv_speed, base_t, robot)

            if blocker is not None:
                # ── B' 侧移优先: 横向间隙不足 (并排/侧方阻塞) 先平移一段车道,
                # 比等待有效 (等待改不了横向几何), 比整段 A* 绕行更省 (论文算子)。
                if i != last_detour_i and _shift_run(
                        x, y, i, path, blocker, nxt_t, cid, clocks, agv_trajs,
                        get_agv_pos, agv_speed, base_t, coarse_obs, width,
                        height, robot):
                    last_detour_i = i
                    continue            # 重新走该段 (新路径上再次检查冲突)
                remain = _remain(blocker, nx, ny, cid, clocks, agv_trajs,
                                 get_agv_pos, agv_speed, base_t)
                if remain > DETOUR_COST and i != last_detour_i:
                    # ── 绕行: 局部 A* 绕开阻塞格 (不是全图) ──
                    alt = nf.replan_agv_car((x, y), path[-1], coarse_obs,
                                            _block_cells(blocker, nx, ny, robot),
                                            width, height)
                    if alt and len(alt) >= 2 and alt[1:] != path[i + 1:i + len(alt)]:
                        # 用绕行路径替换剩余段: 当前点 + alt[1:] (alt[0]=当前位置)
                        path = path[:i + 1] + alt[1:]
                        last_detour_i = i   # 该下标不再重复绕行
                        continue            # 重新检查新路径 (acc 已含到当前点的耗时)
                    remain = MAX_WAIT       # 绕行失败 (走廊堵死) -> 退化为等待
                # ── 等待: 原地停留直到阻塞清除 ──
                while hold < min(remain, MAX_WAIT) and \
                        _conflict(nx, ny, acc + d_next, cid, clocks, agv_trajs,
                                  get_agv_pos, agv_speed, base_t, robot):
                    hold += WAIT_STEP
                    acc += WAIT_STEP

        wps.append((round(x, 3), round(y, 3), round(hold, 3)))
        prev = (x, y)
        i += 1
    return wps


# ── 复合干预: 避让次数优先级 ──
# 调用方维护 avoid_count = {cid: int}, 每次避让 +1。
# 两个 AGV 冲突时, 让"已避让较多"的先走 (论文式 3-20 的优先级传递)。
def track_avoidance(cid, avoid_count):
    """避让次数 +1 并返回 (当前计数, 是否应获得优先权)。

    priority 越高说明它已经让过别人很多次, 现在该别人让它了。
    """
    avoid_count[cid] = avoid_count.get(cid, 0) + 1
    return avoid_count[cid]


def priority_between(a, b, avoid_count):
    """返回冲突 AGV a, b 中应优先的一方。 避让次数多者优先。"""
    return a if avoid_count.get(a, 0) >= avoid_count.get(b, 0) else b
