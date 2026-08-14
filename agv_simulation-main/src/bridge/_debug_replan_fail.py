"""复现 REPLAN-FAIL (规划失败) 并可视化现场。
用法:
  PY src/bridge/_debug_replan_fail.py --cars 10 --path 2 --robot 0 --steps 150
  复现 10 车某个路径集的规划失败, dump 失败时刻的世界状态 + ASCII 网格。
"""
import os
import sys
import json
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

MAP = validate_agv.MAP
DUMPS = []          # 收集所有失败现场
MOVE_ERRS = []      # 收集所有 move_car Error
_printed_err = set()   # 同一 (car,goal,err) 只打一次

# ── ASCII 网格 ─────────────────────────────────────────────
def grid_ascii(planner, highlight=(None, None), wt=None):
    w = planner.world
    gx0, gy0 = w.x_min, w.y_min
    W, H = w.width, w.height
    cells = [[None] * W for _ in range(H)]       # 字符
    meta = [[None] * W for _ in range(H)]        # 说明
    # 站台
    for name, (x, y) in w.lm_dict.items():
        iy, ix = y - gy0, x - gx0
        if 0 <= ix < W and 0 <= iy < H:
            cells[iy][ix] = name if not name.startswith('_') else 'Q'
            meta[iy][ix] = f"站台 {name}"
    # 障碍
    for (x, y) in w.obs_set:
        iy, ix = y - gy0, x - gx0
        if 0 <= ix < W and 0 <= iy < H:
            cells[iy][ix] = '#'
            meta[iy][ix] = '障碍'
    # 车
    for cid, car in w.cars.items():
        wtc = wt if wt is not None else w.world_time
        if wtc >= car.get('last_time', -1):
            cx, cy = car.get('last_pos', (None, None))
        else:
            cur = planner._car_pos(cid, wtc)
            cx, cy = (round(cur[0]), round(cur[1])) if cur else (None, None)
        if cx is None:
            continue
        iy, ix = cy - gy0, cx - gx0
        if not (0 <= ix < W and 0 <= iy < H):
            continue
        tag = cid[-2:]
        if cells[iy][ix] is None or str(cells[iy][ix]).startswith('_') or cells[iy][ix] == 'Q':
            cells[iy][ix] = tag
            meta[iy][ix] = (f"车 {cid} goal={car.get('current_goal')} "
                            f"停={wtc>=car.get('last_time',-1)}")
    # 失败车高亮
    hx, hy = highlight
    if hx is not None:
        iy, ix = hy - gy0, hx - gx0
        if 0 <= ix < W and 0 <= iy < H:
            cells[iy][ix] = '!!'
    lines = []
    lines.append("     " + "  ".join(f"{x%10}" for x in range(W)))
    for iy in range(H - 1, -1, -1):
        row = "  ".join(str(cells[iy][ix] or '.') for ix in range(W))
        lines.append(f"y={iy+gy0:<3d} {row}")
    return "\n".join(lines), meta


# ── 失败现场 dump ─────────────────────────────────────────
def dump_fail(planner, car_id, goal, reason, wt, err):
    w = planner.world
    car = w.cars.get(car_id)
    if not car:
        return
    cx, cy = car.get('last_pos', (None, None))
    if wt < car.get('last_time', -1):
        cur = planner._car_pos(car_id, wt)
        if cur:
            cx, cy = int(round(cur[0])), int(round(cur[1]))
    # 失败车当前格的 5 个时空点预约情况 (当前格+4邻格, 下一tick)
    neighbors = [(cx, cy), (cx, cy+1), (cx, cy-1), (cx+1, cy), (cx-1, cy)]
    labs = ["当前", "上", "下", "右", "左"]
    res_lines = []
    for (nx, ny), lab in zip(neighbors, labs):
        k = (nx, ny, int(wt) + 1)
        who = w.reservation_info.get(k)
        res_lines.append(f"    {lab}格({nx},{ny}) t+1 预约者 = {who}")
    grid, meta = grid_ascii(planner, highlight=(cx, cy), wt=wt)
    info = {
        't': wt, 'car': car_id, 'goal': goal, 'reason': reason, 'err': err,
        'pos': (cx, cy), 'last_pos': car.get('last_pos'),
        'last_time': car.get('last_time'),
        'goal_coord': w.lm_dict.get(goal),
        'res_neighbors': res_lines,
        'grid': grid,
        'station_queue': dict(planner._station_queue),
        'queue_spot': dict(planner._queue_spot),
        'queue_goal': dict(planner._queue_goal),
        'replan_failed': set(planner._replan_failed),
        'yield': set(planner._yield),
        'all_cars': {cid: (c.get('current_goal'), c.get('last_pos'),
                           round(c.get('last_time', 0), 1),
                           bool(w.world_time >= c.get('last_time', 0)))
                     for cid, c in w.cars.items()},
    }
    DUMPS.append(info)
    print(f"\n========== [REPLAN-FAIL 现场] t={wt:.1f} car={car_id} "
          f"goal={goal} err={err} ==========", flush=True)
    print(f"失败车 pos={info['pos']} last_pos={car.get('last_pos')} "
          f"last_time={car.get('last_time'):.1f} "
          f"目标 {goal}@{w.lm_dict.get(goal)}", flush=True)
    for line in res_lines:
        print(line, flush=True)
    print(grid, flush=True)
    print(f"  _station_queue = {dict(planner._station_queue)}", flush=True)
    print(f"  _queue_goal    = {dict(planner._queue_goal)}", flush=True)
    print(f"  _queue_spot    = {dict(planner._queue_spot)}", flush=True)
    print(f"  _replan_failed = {set(planner._replan_failed)}", flush=True)
    print(f"  _yield         = {set(planner._yield)}", flush=True)
    print(f"  全部车 = {info['all_cars']}", flush=True)


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=10)
    ap.add_argument("--path", type=int, default=2)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--robot", type=int, default=0)
    ap.add_argument("--dwell", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    planner_node.DWELL_STOP = max(0.0, args.dwell)

    from agv_world_qos import AGVWorld
    planner_ref = {}
    orig_move = AGVWorld.move_car
    def wrapped_move(self, car_id, goal_name, verbose=False, qos_override=None):
        wt0 = self.world_time
        res = orig_move(self, car_id, goal_name, verbose=verbose, qos_override=qos_override)
        if str(res).startswith("Error"):
            g = self.lm_dict.get(goal_name)
            if g is None:
                return res
            # 目标格物理占用者 + 目标格未来预约者
            occ = [oid for oid, oc in self.cars.items()
                   if oc.get('last_pos') == g and oc.get('last_time', -1) <= wt0]
            rt = self.reservation_info.get((g[0], g[1], int(wt0) + 1))
            key = (car_id, goal_name, res)
            MOVE_ERRS.append({
                't': wt0, 'car': car_id, 'goal': goal_name, 'goal_coord': g,
                'err': res, 'occupant': occ, 'res_t1': rt,
                'last_pos': self.cars[car_id].get('last_pos') if car_id in self.cars else None,
            })
            if key not in _printed_err:
                _printed_err.add(key)
                print(f"[MOVE-ERR][{car_id}->{goal_name}@{g}] t={wt0:.1f} {res} "
                      f"| 目标格已停车={occ} 目标格t+1预约={rt}", flush=True)
                if "could not find path" in res and not str(goal_name).startswith('_'):
                    # 进站无路: 画现场网格 (只对真实站台画)
                    pl = planner_ref.get('p')
                    if pl is not None:
                        print(grid_ascii(pl, highlight=(None, None), wt=wt0)[0], flush=True)
        return res
    AGVWorld.move_car = wrapped_move

    orig_dispatch = PlannerNode._dispatch_next_route
    _disp_printed = set()
    def wrapped_dispatch(self, car_id, **kw):
        q_before = list(self._route_queue.get(car_id, [])
                        or [self.world.cars.get(car_id, {}).get('current_goal')])
        cg_before = self.world.cars.get(car_id, {}).get('current_goal')
        res = orig_dispatch(self, car_id, **kw)
        q_after = list(self._route_queue.get(car_id, []))
        cur_goal = self.world.cars.get(car_id, {}).get('current_goal')
        car = self.world.cars.get(car_id)
        moving = bool(car and car.get('trajectory')
                      and car['trajectory'][-1][0] > self.world.world_time + 0.001)
        is_busy = str(res) == "busy"
        key = (car_id, str(res)[:30])
        if q_before != q_after or (is_busy and moving and key not in _disp_printed):
            if is_busy:
                _disp_printed.add(key)
            popped = [g for g in q_before if g not in q_after]
            print(f"[DISPATCH] car {car_id} t={self.world.world_time:.1f} "
                  f"q={q_before}->{q_after} 弹={popped} "
                  f"cg_前={cg_before} cg_后={cur_goal} 在途={moving} "
                  f"queue_goal={self._queue_goal.get(car_id)} "
                  f"station_queue_队首={[sq[0] for S, sq in self._station_queue.items() if sq]}",
                  flush=True)
        return res
    PlannerNode._dispatch_next_route = wrapped_dispatch

    orig_replan = PlannerNode.replan_car
    def wrapped(self, car_id, goal=None, extra_obs=None, reason=""):
        wt0 = self.world.world_time
        res = orig_replan(self, car_id, goal=goal, extra_obs=extra_obs, reason=reason)
        if car_id in self._replan_failed:
            dump_fail(self, car_id, goal, reason, wt0, res)
        return res
    PlannerNode.replan_car = wrapped

    mdata, lms, free = validate_agv.load_map(MAP)
    rng = random.Random(args.seed + args.cars * 1000 + args.path)
    spawns, chains = validate_agv.gen_chains(args.cars, lms, free, rng)

    bus = Bus()
    planner = PlannerNode(bus, MAP, station_hold=planner_node.HOLD_SLOT)
    planner_ref['p'] = planner
    sim = MujocoNode(bus, headless=True, wall_speed=1.0,
                     robot=bool(args.robot))
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
    print(f"\n[RESULT] cars={args.cars} path={args.path} replan_fail={m['replan_fail']} "
          f"completion={m['completion']:.0%} queue_timeout={m['queue_timeouts']} "
          f"collisions={sim.n_agv_collisions}")
    print(f"[RESULT] replan_reasons={dict(m['replan_reasons'])}")
    print(f"[RESULT] 共 {len(DUMPS)} 次 REPLAN-FAIL 现场, "
          f"{len(MOVE_ERRS)} 次 move_car Error, "
          f"{len({(e['car'],e['goal']) for e in MOVE_ERRS})} 个 (车,目标) 组合")

    # ── 结束 dump: 每车状态 ──
    print("\n===== [结束] 每车状态 =====", flush=True)
    reached = planner._reached
    for cid, car in planner.world.cars.items():
        got = len(reached.get(cid, set()))
        tag = "  <-- 少站!" if got < 2 else ""
        print(f"  car {cid}: 完成站={sorted(reached.get(cid, set()))} "
              f"current_goal={car.get('current_goal')} "
              f"route_q={planner._route_queue.get(cid)} 在_replan_failed="
              f"{cid in planner._replan_failed} "
              f"queue_goal={planner._queue_goal.get(cid)} "
              f"queue_spot={planner._queue_spot.get(cid)} "
              f"last_pos={car.get('last_pos')} last_time={car.get('last_time'):.1f}{tag}",
              flush=True)
    # 每个站台队列当前状态
    print("  站台队列 _station_queue:", flush=True)
    for S, sq in planner._station_queue.items():
        print(f"    {S}@{planner.world.lm_dict.get(S)}: {sq}", flush=True)
    # 各车轨迹末点 (停在哪)
    for cid, car in planner.world.cars.items():
        tr = car.get('trajectory', [])
        if tr:
            last = tr[-1]
            print(f"    车 {cid} 轨迹末点 t={last[0]:.1f} 格=({last[1]},{last[2]}) "
                  f"当前goal={car.get('current_goal')}", flush=True)


if __name__ == "__main__":
    main()
