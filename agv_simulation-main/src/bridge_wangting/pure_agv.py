"""
pure_agv.py — 纯 AGV 离线预规划 (bridge_wangting)
=================================================

把在线 planner_node 的机制套进离线预规划, 保留 QoS 引擎本体但全车同优先级 (无抢占):
  - 短停车窗 HOLD_SLOT (默认 5 tick): 站台只锁 5 tick, 车走即释放 → 后车能进空窗,
    不再出现"目标站台被 1000 tick 占死" (1000 tick 是之前离线 FAIL 的根因)
  - add 即 move: 每车 add 完立刻规划它的整条链, 首次 move_car 会清掉出生格预约,
    不挡别车必经通道
  - 目标被占 → 原地等到停车窗释放再试 (对齐"排队等待"), 最多 max_wait 次
  - 仍失败 → 跳站 (对齐 QUEUE_TIMEOUT 跳站), 不白搜几十万状态
  - 全部车 t=0 起 (同时发车), FCFS 预约顺序避让

规划本体不变: QoS 运动学时空 A* + ALT 差分启发式 (agv_world_qos.astar_with_time ← astar_alt)。
"""
from __future__ import annotations

import os
import sys
import time

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_DIR = os.path.join(SRC_DIR, "bridge")
MAPS_DIR = os.path.join(SRC_DIR, "..", "maps")
for p in (SRC_DIR, BRIDGE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from topic_bus import Bus                                          # noqa: E402
import mujoco_node as mjn                                           # noqa: E402
from mujoco_node import MujocoNode, AGV_RADIUS                      # noqa: E402
import map_scene                                                    # noqa: E402
from wangting_nodes import inflate_obstacles                        # noqa: E402

HOLD_SLOT = 5      # 站台短停车窗 (tick), 对齐 planner_node.HOLD_SLOT
MAX_WAIT = 5       # 目标被占时最多等 MAX_WAIT 次 (每次推进 hold+1 tick)


def preplan_chains(map_file, chains, inflate=None, hold=HOLD_SLOT, max_wait=MAX_WAIT,
                   stagger=0):
    """离线预规划 (QoS 引擎 + 在线机制)。返回 ({cid: [(t,x,y,dir)]}, 耗时)。
    stagger: 每车出发时刻 = idx*stagger (0=同时发车; >0 错峰, 预约更分散通常更快)。"""
    import agv_world_qos
    from astar import astar_waitfree
    agv_world_qos.astar_with_time = astar_waitfree     # 等待直到空闲 (时间折叠)

    md, frame, _ = map_scene.load_map_data(map_file)
    w = agv_world_qos.AGVWorld(map_file)
    occ = set(tuple(o) for o in md["obstacles"])
    radius = AGV_RADIUS if inflate is None else inflate
    w.obs_set = inflate_obstacles(occ, frame.res, radius) if radius > 0 else occ
    for lm in w.landmarks:
        w.obs_set.discard((lm["x"], lm["y"]))
    w.station_hold = hold             # 短停车窗
    w.lazy_lock = False               # move_car 写站台停车 (短窗)

    trajs = {}
    t0 = time.perf_counter()
    for idx, chain in enumerate(chains):
        cid = f"car{idx:02d}"
        w.world_time = idx * stagger  # 出发时刻
        w.add(cid, chain[0], "MEDIUM")
        for leg in chain[1:]:
            ok = _plan_leg(w, cid, leg, hold, max_wait)
            if not ok:
                print(f"[pure] car {cid} 段 {leg} 失败, 跳站")
                break
        trajs[cid] = [tuple(p) for p in w.cars[cid]["trajectory"]]
    elapsed = time.perf_counter() - t0
    return trajs, elapsed


def _plan_leg(w, cid, leg, hold, max_wait):
    """规划一段。只在【目标被占】(廉价失败) 时等停车窗释放重试;
    "could not find path"(几何/预约堵死, 等多久都一样) → 立即跳站, 不重复全图搜索。"""
    for attempt in range(max_wait + 1):
        w.world_time = w.cars[cid]["last_time"] + (hold + 1) * attempt
        res = w.move_car(cid, leg)
        if not str(res).startswith("Error"):
            return True
        if "could not find path" in str(res):
            return False
        # else: "Goal ... already reserved" → 等停车窗释放再试
    return False


def run(map_file, chains, headless=False, steps=0, speed=1.0, inflate=None,
        scene_out=None):
    """离线预规划 → 场景 → t=0 统一发布全部轨迹 → MujocoNode 渲染。"""
    trajs, plan_t = preplan_chains(map_file, chains, inflate=inflate)
    print(f"[pure] pre-plan {len(trajs)} cars 耗时 {plan_t:.2f}s "
          f"(t=0 同时发车, 短停车窗 HOLD_SLOT={HOLD_SLOT})")
    if not trajs:
        raise SystemExit("无任何可规划的车")

    md, frame, _ = map_scene.load_map_data(map_file)
    boxes = map_scene.obstacle_boxes(md, frame)
    garage = map_scene.pick_garage(md, frame)
    xml = map_scene.build_scene_xml(md, frame, boxes, garage)
    scene_path = scene_out or os.path.join(MAPS_DIR, "wangting_scene.xml")
    os.makedirs(os.path.dirname(scene_path), exist_ok=True)
    with open(scene_path, "w", encoding="utf-8") as f:
        f.write(xml)

    mjn.OBS_BOXES = boxes
    mjn.GARAGE_HOME = {f"agv_g{i}": (mx, my) for i, (mx, my) in enumerate(garage)}

    bus = Bus()
    sim = MujocoNode(bus, scene=scene_path, headless=headless,
                     wall_speed=speed, robot=False)
    for cid, traj in trajs.items():
        mtraj = [{"t": tt, "x": round(frame.cx2m(x), 3),
                  "y": round(frame.cy2m(y), 3), "dir": d}
                 for tt, x, y, d in traj]
        bus.publish("/planner/path", {"car_id": cid, "action": "add",
                                      "goal": None, "trajectory": mtraj},
                    stamp=0)
    print(f"[pure] published {len(trajs)} trajectories at t=0 (同时发车)")

    sim.run(steps=steps)
    print("[PURE-AGV] done.")