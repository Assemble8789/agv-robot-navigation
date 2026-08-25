"""
wangting_nodes.py — 阶段1 (纯 AGV): bridge 节点在米制帧下工作 (bridge_wangting)

不改原文件: 通过【继承 + 模块打补丁】复用原 bridge 节点。
  - WangtingPlanner(PlannerNode): 覆写发布/ARC 的 cell↔meter 换算
  - make_sim(): 给 mujoco_node.OBS_BOXES / GARAGE_HOME 打米制补丁, 生成新场景 XML

补丁只在调用方进程内生效; 原入口 (demo_bridge / validate_agv) 行为不变。
"""
from __future__ import annotations

import math
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_DIR = os.path.join(SRC_DIR, "bridge")
MAPS_DIR = os.path.join(SRC_DIR, "..", "maps")
for p in (SRC_DIR, BRIDGE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from planner_node import PlannerNode, ROBOT_OBS_R, BLOCK_HORIZON, \
    ROBOT_HOLD_R, STATION_QUEUE_R, PARK_BLOCK_DIST, MOVE_ASIDE_COOLDOWN, \
    ROBOT_BLOCK_DIST, CLEAR_DIST, RESUME_COOLDOWN, WAIT_TIMEOUT, \
    CHECK_INTERVAL                                                   # noqa: E402
import mujoco_node as mjn                                            # noqa: E402
from mujoco_node import MujocoNode, AGV_RADIUS, AGV_Z, AGV_MATERIALS   # noqa: E402
from mapframe import MapFrame                                        # noqa: E402
import map_scene                                                     # noqa: E402

# planner_node 在 import 时把 agv_world_qos.astar_with_time 换成 agv_world_qos_alt.astar_alt
# (USE_ALT=True), 但它缺 best_cost_to_state 时间剪枝, 0.1m 大图上会时间维爆掉撞 max_iter。
# 这里再换成修正版 (ASTAR_FIXED) —— 签名兼容, 只影响本进程。
import agv_world_qos                                                 # noqa: E402
from astar import astar_waitfree                                     # noqa: E402
agv_world_qos.astar_with_time = astar_waitfree
print("[wangting_nodes] A* patched to astar_waitfree (时间折叠等待直到空闲)")


def inflate_obstacles(occ, res, radius):
    """障碍按车半径膨胀: 球心离墙 ≥ radius → 球体(半径 radius)不穿墙,
    球无法进入宽度小于其直径 (2×radius) 的通道。返回膨胀后的格集合。"""
    R = max(1, int(round(radius / res)))
    inflated = set()
    for (x, y) in occ:
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                inflated.add((x + dx, y + dy))
    return inflated


class WangtingPlanner(PlannerNode):
    """PlannerNode 的米制帧版: A* 在 0.1m 格子里算, 对外发布/感知全是十进制米。
    构造时把障碍按 AGV_RADIUS 膨胀, 球无法穿过小于球体尺寸的通道。"""

    def __init__(self, bus, map_file, station_hold=None, inflate=None):
        super().__init__(bus, map_file, station_hold)
        self.frame = MapFrame(self.world.map_data)
        # 障碍膨胀 (仅规划层): inflate=None 用 AGV_RADIUS (0.2m); 0 关闭
        radius = AGV_RADIUS if inflate is None else inflate
        if radius > 0 and self.world.obs_set:
            inflated = inflate_obstacles(self.world.obs_set, self.frame.res, radius)
            for lm in self.world.landmarks:      # 地标格保持可达
                inflated.discard((lm["x"], lm["y"]))
            self.world.obs_set = inflated
            self.world.obstacles = list(inflated)
            print(f"[WangtingPlanner] obs inflated by {radius:.2f}m "
                  f"({self.frame.cell_radius(radius)} cells): "
                  f"{len(inflated)} cells")
        # 机器人相关逻辑全部在米制帧: 地标米制坐标表 (机器人路径点/站台判定用)
        self.lm_m = {name: self.frame.to_meters(x, y)
                     for name, (x, y) in self.world.lm_dict.items()
                     if not str(name).startswith('_')}

    # ── 发布: 格子 → 米制 ──
    def publish_path(self, car_id):
        car = self.world.cars.get(car_id)
        if not car:
            return
        fr = self.frame
        traj = [{"t": p[0], "x": fr.cx2m(p[1]), "y": fr.cy2m(p[2]), "dir": p[3]}
                for p in car['trajectory']]
        self.bus.publish("/planner/path", {
            "car_id": car_id,
            "action": "add" if len(traj) <= 1 else "update",
            "goal": car.get("current_goal"),
            "trajectory": traj,
        }, stamp=self.world.world_time)

    # ── ARC 让行: 车轨迹插值 → 米制 (与机器人米制同帧) ──
    def _car_pos(self, car_id, t):
        xy = super()._car_pos(car_id, t)
        if xy is None:
            return None
        return self.frame.to_meters(xy[0], xy[1])

    # ── 机器人占格: 米制 → 0.1m 格, 半径按 res 换算 ──
    def _robot_obs_cells(self):
        r = self._last_robot
        if not r:
            return set()
        fr = self.frame
        R = fr.cell_radius(ROBOT_OBS_R)
        cells = set()
        for px, py in ((r['x'], r['y']),
                       (r['x'] + r['vx'] * BLOCK_HORIZON,
                        r['y'] + r['vy'] * BLOCK_HORIZON)):
            cx, cy = fr.to_cell(px, py)
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    cells.add((cx + dx, cy + dy))
        return cells

    # ── 排队位: robot_pts 是米制, 换成米制距离比较 ──
    def _find_queue_spot(self, goal_name):
        gx, gy = self.world.lm_dict.get(goal_name, (None, None))
        if gx is None:
            return None
        wt = self.world.world_time
        robot_pts = []
        if self._last_robot_path is not None:
            idx = self._last_robot.get('wp_index', 0) if self._last_robot else 0
            wps = self._last_robot_path.get('waypoints', [])
            robot_pts = [(wps[i]['x'], wps[i]['y']) for i in range(idx, len(wps))]
        taken = set(self._queue_tried)
        taken.update(sp for sp in self._queue_spot.values() if sp)
        # 别的车【当前正停着】的格不选作排队位 (parked last_pos);
        # 瞬时的时空预约不排除 —— waitfree A* 会等待到空闲, 不用因为 wt+1 被占就放弃
        parked = {c['last_pos'] for c in self.world.cars.values() if c.get('last_pos')}
        fr = self.frame
        for r in range(1, 9):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if not (self.world.x_min <= nx < self.world.x_min + self.world.width
                            and self.world.y_min <= ny < self.world.y_min
                            + self.world.height):
                        continue
                    if (nx, ny) in self.world.obs_set or (nx, ny) in taken:
                        continue
                    if (nx, ny) in parked:
                        continue
                    if robot_pts:
                        mx, my = fr.to_meters(nx, ny)
                        if any(math.hypot(mx - px, my - py) < 0.8 for px, py in robot_pts):
                            continue
                    return (nx, ny)
        return None

    # ── 机器人相关方法: 全部转米制帧 (地标/车/机器人同一米制) ──

    def _robot_target_station(self):
        """机器人当前目标点对应的站台地标名 (距 < ROBOT_HOLD_R, 米制)。"""
        if not self._last_robot or not self._last_robot_path:
            return None
        idx = self._last_robot.get('wp_index', 0)
        wps = self._last_robot_path.get('waypoints', [])
        if idx >= len(wps):
            return None
        wx, wy = wps[idx]['x'], wps[idx]['y']
        for name, (x, y) in self.lm_m.items():
            if math.hypot(wx - x, wy - y) < ROBOT_HOLD_R:
                return name
        return None

    def _station_status(self):
        """{站台: 'docked'/'queuing'} —— 米制帧 (last_pos/在途位置/地标全转米制)。"""
        wt = self.world.world_time
        fr = self.frame
        status = {}
        for name, (x, y) in self.lm_m.items():
            for cid, car in self.world.cars.items():
                if cid in self._yield:
                    continue
                if self._queue_goal.get(cid) == name:
                    status[name] = 'queuing'
                    break
                cx, cy = None, None
                if wt >= car.get('last_time', -1):
                    lp = car.get('last_pos')
                    if lp:
                        cx, cy = fr.to_meters(lp[0], lp[1])
                else:
                    if car.get('current_goal') == name:
                        cur = self._car_pos(cid, wt)
                        if cur:
                            cx, cy = cur
                if cx is None:
                    continue
                if math.hypot(cx - x, cy - y) < STATION_QUEUE_R:
                    status[name] = 'docked' if wt >= car.get('last_time', -1) else 'queuing'
                    break
        return status

    def _handle_robot_blocking(self):
        """ARC 让行闭环 (米制): 预测撞机器人 → 停车; 让开 → 恢复; 超时 → 绕行;
        已到站车挡机器人目标点 → 挪开。逻辑同原版, 仅地标坐标换米制。"""
        wt = self.world.world_time
        if wt - self._last_block_check < CHECK_INTERVAL:
            return
        self._last_block_check = wt
        if self._last_robot is None:
            return
        target = self._robot_target_station()
        target_xy = self.lm_m.get(target) if target else None
        for car_id, car in list(self.world.cars.items()):
            if car.get('current_goal') is None or wt >= car['last_time']:
                continue
            if target_xy and car.get('current_goal') == target:
                cur = self._car_pos(car_id, wt)
                if cur and math.hypot(cur[0] - target_xy[0],
                                      cur[1] - target_xy[1]) < STATION_QUEUE_R:
                    continue
            d = self._closest_approach(car_id)
            if car_id not in self._yield:
                if d < ROBOT_BLOCK_DIST:
                    self._stop_car(car_id)
                    self._yield[car_id] = wt
            else:
                if d >= CLEAR_DIST:
                    if wt - self._last_resume.get(car_id, -9) >= RESUME_COOLDOWN:
                        self._last_resume[car_id] = wt
                        self._yield.pop(car_id, None)
                        self.replan_car(car_id, extra_obs=self._robot_obs_cells(),
                                        reason="机器人让开恢复")
                elif wt - self._yield[car_id] >= WAIT_TIMEOUT:
                    self._yield.pop(car_id, None)
                    self.replan_car(car_id, extra_obs=self._robot_obs_cells(),
                                    reason="让行超时绕行")

        if self._last_robot is not None and self._last_robot_path is not None:
            r = self._last_robot
            idx = r.get('wp_index', 0)
            wps = self._last_robot_path.get('waypoints', [])
            if idx < len(wps):
                wx, wy = wps[idx]['x'], wps[idx]['y']
                for car_id, car in list(self.world.cars.items()):
                    if wt < car['last_time'] or car_id in self._yield:
                        continue
                    if self._route_queue.get(car_id):
                        continue
                    if wt - self._moved_aside.get(car_id, -99) < MOVE_ASIDE_COOLDOWN:
                        continue
                    lx, ly = car['last_pos']
                    if math.hypot(lx - wx, ly - wy) < PARK_BLOCK_DIST:
                        self._move_agv_aside(car_id)

    def _move_agv_aside(self, car_id):
        """把停在机器人路径上的【已到站】AGV 挪到远处空位 (米制距离比较)。"""
        car = self.world.cars.get(car_id)
        if not car:
            return
        x, y = car['last_pos']
        wt = self.world.world_time
        fr = self.frame
        robot_pts = []
        if self._last_robot_path is not None:
            idx = self._last_robot.get('wp_index', 0) if self._last_robot else 0
            wps = self._last_robot_path.get('waypoints', [])
            robot_pts = [(wps[i]['x'], wps[i]['y']) for i in range(idx, len(wps))]
        best = None
        best_d = -1.0
        for r in range(2, 8):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (self.world.x_min <= nx < self.world.x_min + self.world.width
                            and self.world.y_min <= ny < self.world.y_min
                            + self.world.height):
                        continue
                    if (nx, ny) in self.world.obs_set:
                        continue
                    if (nx, ny, wt + 1) in self.world.reservation_info:
                        continue
                    mx, my = fr.to_meters(nx, ny)
                    dmin = min((math.hypot(mx - px, my - py) for px, py in robot_pts),
                               default=999.0)
                    if dmin > best_d:
                        best_d = dmin
                        best = (nx, ny)
            if best is not None and best_d >= 3.0:
                break
        if best is None:
            print(f"[MOVE-ASIDE-FAIL] car {car_id} 找不到空位")
            return
        nx, ny = best
        lm = "_MOVE_%s" % car_id
        self.world.lm_dict[lm] = (nx, ny)
        car['waiting_replan'] = True
        saved_obs = self.world.obs_set
        try:
            if self._last_robot is not None:
                self.world.obs_set = saved_obs | self._robot_obs_cells()
            res = self.world.move_car(car_id, lm)
        finally:
            self.world.obs_set = saved_obs
            car['waiting_replan'] = False
            self.world.lm_dict.pop(lm, None)
        if res != "busy" and not str(res).startswith("Error"):
            car['current_goal'] = None
            self._moved_aside[car_id] = wt
            self.metrics['move_aside'] += 1
            self.publish_path(car_id)
            print(f"[MOVE-ASIDE] car {car_id} 挪到 ({nx},{ny}) | {res}")
        else:
            self.metrics['move_aside_fail'] += 1
            print(f"[MOVE-ASIDE-FAIL] car {car_id}: {res}")


def make_sim(bus, map_file, headless=False, wall_speed=1.0, scene_out=None,
             robot=False, robot_plan=None):
    """构造 MujocoNode: 障碍盒/车库/场景全部从地图(米制)生成, 并打模块补丁。
    robot=True 时把 mujoco_node.build_scene 换成 elf3 + wangting 场景的合并版。"""
    md, frame, _ = map_scene.load_map_data(map_file)
    boxes = map_scene.obstacle_boxes(md, frame)
    garage = map_scene.pick_garage(md, frame)
    xml = map_scene.build_scene_xml(md, frame, boxes, garage)

    scene_path = scene_out or os.path.join(MAPS_DIR, "wangting_scene.xml")
    os.makedirs(os.path.dirname(scene_path), exist_ok=True)
    with open(scene_path, "w", encoding="utf-8") as f:
        f.write(xml)

    # 模块级补丁 (仅本进程): 原 15×10 行为不受影响
    mjn.OBS_BOXES = boxes
    mjn.GARAGE_HOME = {f"agv_g{i}": (mx, my) for i, (mx, my) in enumerate(garage)}
    print(f"[make_sim] OBS_BOXES={len(boxes)} garage={len(garage)} scene={scene_path} "
          f"robot={robot}")

    if robot:
        from robot_scene_wangting import build_scene as build_merged
        mjn.build_scene = lambda: build_merged(xml)
        print("[make_sim] robot scene merged (elf3 + wangting)")

    return WangtingMujoco(bus, scene=scene_path, headless=headless,
                          wall_speed=wall_speed, robot=robot, robot_plan=robot_plan)


def make_planner(bus, map_file, station_hold=None, inflate=None):
    return WangtingPlanner(bus, map_file, station_hold, inflate=inflate)


class WangtingMujoco(MujocoNode):
    """MujocoNode 的 wangting 版: 出生时把球立即摆到轨迹首点, 消除"车库→地标"闪现。"""

    def _spawn_car(self, car_id, traj):
        x, y = float(traj[0][1]), float(traj[0][2])
        if self._garage_free:
            sphere = self._garage_free.pop(0)
            src = "garage"
        else:
            sphere = "agv_" + car_id
            mat = AGV_MATERIALS[(5 + len(self._car_fragments)) % len(AGV_MATERIALS)]
            self._car_fragments[car_id] = (
                f'    <body name="{sphere}" pos="{x} {y} {AGV_Z}" '
                f'mocap="true">\n'
                f'      <geom name="{sphere}_geom" type="sphere" size="0.2" '
                f'material="{mat}" contype="1" conaffinity="1"/>\n'
                f'    </body>')
            self._pending_rebuild = True
            src = "dynamic"
        self._car_sphere[car_id] = sphere
        adr = self._sphere_adr.get(sphere)
        if adr is not None:
            self.data.mocap_pos[adr][0] = x
            self.data.mocap_pos[adr][1] = y
            self.data.mocap_pos[adr][2] = AGV_Z
        print(f"[SPAWN] car {car_id} -> sphere {sphere} (from {src}) at ({x}, {y})")