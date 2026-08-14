"""
MuJoCo 仿真节点: 只渲染 + 感知 + 时间主, 不 import AGVWorld
=============================================================

通过 topic 总线与 Planner 节点解耦:
  - 订阅 /planner/path   收到轨迹 → 自动建球 (车库球 / 动态球) 并插值渲染
  - 订阅 /planner/remove 删除车 → 回收球
  - 发布 /clock          时间主 (ROS /clock 模式, planner 同步 world_time)
  - 发布 /sim/car_state          当前每辆渲染车位置
  - 发布 /sim/collision          AGV-AGV 碰撞 (球心距 < 0.4, 边沿触发计数)
  - 发布 /sim/obstacle_distance  每车到最近障碍盒的解析距离

场景/常量/模型重建逻辑拷贝自 agv_world_qos_mujoco.py (去掉 AGVWorld 依赖):
  _rebuild_model / _launch_viewer / _car_xy(traj,...) / 车库+动态建球 / 碰撞段。
障碍盒是 contype=0, MuJoCo 不产生接触, 距离用解析式点到盒距离。

线程模型: 本节点状态 (model/data/_car_traj/_car_sphere/...) 只由 run 主循环
线程触碰; 订阅回调只 enqueue 到 incoming, 每帧开头 _drain_incoming 处理。
"""

import os
import sys
import time
import math
import queue
import threading

import numpy as np
import mujoco, mujoco.viewer

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BRIDGE_DIR)     # src/ (agv_scene.xml 所在)

SCENE = os.path.join(SRC_DIR, "agv_scene.xml")
ROBOT_PLAN = os.path.join(SRC_DIR, "..", "maps", "humanoid_plan.json")

if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)
from robot_scene import build_scene
from robot_walker import RobotWalker, RobotNavigator, ROBOT_RADIUS

AGV_Z = 0.04         # AGV 球心高度 (半径 0.2, 底在 z=-0.16=地面, 与工厂场景一致)
AGV_RADIUS = 0.2     # AGV 球半径 (两球球心距 < 0.4 = 触碰/碰撞)
AGV_MATERIALS = ["agv%d_mat" % i for i in range(8)]
DYN_OBST_R = 0.3     # 动态障碍默认半径 (红色球, 直径 0.6 > AGV 0.4, 明显需避让)
SAMPLE_DT = 0.1      # 速度采样步长 (有限差分, QoS 步 = 1 格/秒)
ROBOT_AVOID_DIST = 1.3   # 机器人避让 AGV 的感知距离 (策略行走有横向游走, 留足余量)
ROBOT_AVOID_AMP = 1.5   # 机器人侧向推开幅度 (m/s 级)

# 车库预置的 5 辆待命球 (与 agv_scene.xml 一致, 停在底部整数格 (0,0)..(4,0))
GARAGE_SPHERES = ["agv_g0", "agv_g1", "agv_g2", "agv_g3", "agv_g4"]
GARAGE_HOME = {"agv_g%d" % i: (float(i), 0.0) for i in range(5)}

# 场景里的 6 个障碍盒 (axis-aligned): (cx, cy, hx, hy)。
# 与 agv_scene.xml 的 obs_0..obs_5 保持同步:
#   obs_0@(9,8) obs_1@(9,5) obs_3@(4,5) obs_4@(4,8) 尺寸 1.5×0.35
#   obs_2@(6.5,1) obs_5@(6.5,2)                      尺寸 1.0×0.5
OBS_BOXES = [
    (9.0, 8.0, 1.5, 0.35),
    (9.0, 5.0, 1.5, 0.35),
    (6.5, 1.0, 1.0, 0.5),
    (4.0, 5.0, 1.5, 0.35),
    (4.0, 8.0, 1.5, 0.35),
    (6.5, 2.0, 1.0, 0.5),
]


class MujocoNode:
    def __init__(self, bus, scene=SCENE, headless=False, wall_speed=1.0,
                 robot=False, robot_plan=None):
        self.bus = bus
        self.headless = headless
        self.wall_speed = wall_speed        # 播放速度: 仿真秒 / 墙上秒
        self.running = True
        self.t_sim = 0.0                    # 仿真时钟 (时间主)
        self.robot = robot                  # 是否加载 elf3 人形机器人

        # MuJoCo 场景: 纯 AGV 版读 agv_scene.xml; 机器人版用文本合并场景。
        if robot:
            self._base_xml = build_scene()
        else:
            with open(scene, encoding="utf-8") as f:
                self._base_xml = f.read()
        self._garage_free = list(GARAGE_SPHERES)   # 仍在车库的球
        self._car_sphere = {}                      # car_id -> 球名
        self._car_traj = {}                        # car_id -> [(t,x,y,dir)]
        self._car_fragments = {}                   # 动态球 car_id -> <body> 片段
        self._sphere_adr = {}                      # 球名 -> mocap 地址
        self._pending_rebuild = False
        self._colliding = set()
        self.n_agv_collisions = 0
        # 验证指标 (validate_agv.py 读取)
        self.metrics = {'min_agv_agv': float('inf'), 'near_miss': 0}
        self._last_qos_time = 0
        self._dyn_obs = {}             # 动态障碍 id -> (x, y, radius)
        self._dyn_fragments = {}       # 动态障碍 id -> <body> XML 片段
        self._static_published = False

        self.incoming = queue.Queue()
        self._subs = [
            bus.subscribe("/planner/path",
                          lambda m: self.incoming.put(("path", m))),
            bus.subscribe("/planner/remove",
                          lambda m: self.incoming.put(("remove", m))),
            bus.subscribe("/planner/obstacle_cmd",
                          lambda m: self.incoming.put(("obstacle_cmd", m))),
            bus.subscribe("/planner/robot_hold",
                          lambda m: self.incoming.put(("robot_hold", m))),
        ]
        self._robot_hold = False           # 机器人是否被要求原地等待 (站台被占)

        # 机器人状态 (robot=False 时全部为 None)
        self._walker = None
        self._nav = None
        self._robot_prev = None            # 上一帧 (x,y), 算速度
        self._last_t_sim = 0.0             # 上一帧 t_sim (算帧长)
        self._robot_plan_idx_pub = -1      # 已发布的路径点序号
        self.n_robot_collisions = 0        # AGV↔机器人实体碰撞事件数
        self._robot_colliding = set()      # 当前正与机器人接触的 AGV 球 geom 名
        self.robot_fell_count = 0          # 机器人摔倒次数 (边沿触发)
        self._robot_fell_prev = False      # 上一帧是否摔倒 (判边沿)

        self.model = None
        self.data = None
        self._rebuild_model()
        if robot:
            self._setup_robot(robot_plan)

    # ── 机器人: 初始化 / 路径 ───────────────────────────────────────
    def _setup_robot(self, robot_plan):
        import json
        if robot_plan is None:
            with open(ROBOT_PLAN, encoding="utf-8") as f:
                robot_plan = json.load(f)
        elif isinstance(robot_plan, str):
            with open(robot_plan, encoding="utf-8") as f:
                robot_plan = json.load(f)
        wps = [(p["x"], p["y"]) for p in robot_plan["waypoints"]]
        self._nav = RobotNavigator(wps)
        self._walker = RobotWalker(self.model, self.data)
        # 起点: robot_plan 里可带 start (x,y); 否则用默认 (2,2.5) 附近。
        sx, sy = 2.0, 2.5
        start = robot_plan.get("start") if isinstance(robot_plan, dict) else None
        if isinstance(start, (list, tuple)) and len(start) >= 2:
            try:
                sx, sy = float(start[0]), float(start[1])
            except (TypeError, ValueError):
                pass
        self._walker.reset(sx, sy, z=1.1, yaw=0.0)
        self._robot_prev = (sx, sy)
        self._publish_robot_path()
        print(f"[ROBOT] elf3 行走策略加载完成: {len(wps)} 个路径点, "
              f"起于 ({sx}, {sy})")

    # ── 收到 planner 消息 (回调只 enqueue, 处理在 run 主循环) ───────
    def _drain_incoming(self):
        while not self.incoming.empty():
            kind, msg = self.incoming.get_nowait()
            if kind == "path":
                self._on_path(msg)
            elif kind == "remove":
                self._return_sphere(msg.get("car_id"))
            elif kind == "obstacle_cmd":
                self._on_obstacle_cmd(msg)
            elif kind == "robot_hold":
                self._robot_hold = bool(msg.get("hold", False))

    def _on_obstacle_cmd(self, msg):
        """注入/删除动态障碍 (来自 planner 的 /planner/obstacle_cmd)。
        动态障碍是红色半透明球, 惰性几何 (contype=0), 作为感知目标被
        /sim/obstacle_distance 上报, 也参与 /sim/dynamic_obstacles 广播。"""
        oid = str(msg.get("id"))
        if msg.get("action") == "add":
            x, y = float(msg["x"]), float(msg["y"])
            r = float(msg.get("radius", DYN_OBST_R))
            self._dyn_obs[oid] = (x, y, r)
            z = r - 0.16                    # 球底贴地 (地面 z=-0.16)
            self._dyn_fragments[oid] = (
                f'    <body name="dyn_{oid}" pos="{x} {y} {z:.2f}">\n'
                f'      <geom name="dyn_{oid}_geom" type="sphere" size="{r}" '
                f'material="dyn_mat" contype="0" conaffinity="0"/>\n'
                f'    </body>')
            self._pending_rebuild = True
            print(f"[OBST] + dyn obstacle {oid} at ({x},{y}) r={r}")
        elif msg.get("action") == "del":
            self._dyn_obs.pop(oid, None)
            self._dyn_fragments.pop(oid, None)
            self._pending_rebuild = True
            print(f"[OBST] - dyn obstacle {oid}")
        self._publish_dyn_obstacles()

    def _on_path(self, msg):
        car_id = msg["car_id"]
        traj = [(p["t"], p["x"], p["y"], p["dir"]) for p in msg["trajectory"]]
        if car_id not in self._car_sphere:
            self._spawn_car(car_id, traj)
        self._car_traj[car_id] = traj

    # ── 建球 / 回收球 ───────────────────────────────────────────────
    def _spawn_car(self, car_id, traj):
        """新车: 有车库球就用 (免重编译); 车库满 → 动态建球拼进 AGV_INSERT。
        车出生在轨迹首点 (即 add 地标)。"""
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
        print(f"[SPAWN] car {car_id} -> sphere {sphere} (from {src}) at ({x}, {y})")

    def _return_sphere(self, car_id):
        """回收车对应的球: 车库球回库复位; 动态球删片段并标记重编译。"""
        sphere = self._car_sphere.pop(car_id, None)
        self._car_traj.pop(car_id, None)
        if sphere is None:
            return
        if sphere in GARAGE_HOME:
            self._garage_free.append(sphere)
            adr = self._sphere_adr.get(sphere)
            if adr is not None:
                hx, hy = GARAGE_HOME[sphere]
                self.data.mocap_pos[adr][0] = hx
                self.data.mocap_pos[adr][1] = hy
                self.data.mocap_pos[adr][2] = AGV_Z
            print(f"[REMOVE] car {car_id} sphere {sphere} returned to garage")
        else:
            self._car_fragments.pop(car_id, None)
            self._pending_rebuild = True
            print(f"[REMOVE] car {car_id} sphere {sphere} removed")

    def _rebuild_model(self):
        # 重建前保存机器人物理状态 (动态加车/障碍会重建模型, 不能让机器人重置)
        saved = self._walker.save_state() if self._walker is not None else None

        fragments = (list(self._car_fragments.values())
                     + list(self._dyn_fragments.values()))
        xml = self._base_xml.replace(
            "<!-- AGV_INSERT -->", "\n".join(fragments))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._sphere_adr = {}
        for sphere in list(GARAGE_HOME) + list(self._car_sphere.values()):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, sphere)
            if bid >= 0:
                self._sphere_adr[sphere] = self.model.body_mocapid[bid]

        if self._walker is not None:
            self._walker = RobotWalker(self.model, self.data)
            if saved is not None:
                self._walker.restore_state(saved)

    def _launch_viewer(self):
        v = mujoco.viewer.launch_passive(self.model, self.data,
                                         show_left_ui=False)
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.lookat[:] = (7.5, 5.0, 0.6)
        v.cam.distance = 18.0
        v.cam.elevation = -35
        v.cam.azimuth = 180
        return v

    # ── 轨迹插值 ────────────────────────────────────────────────────
    def _car_xy(self, traj, t_sim):
        """轨迹 (t,x,y,dir) 在 t_sim 时刻的世界坐标 (连续插值, 首/末点钳位)。"""
        if not traj:
            return None
        prev = traj[0]
        nxt = None
        for pt in traj:
            if pt[0] <= t_sim:
                prev = pt
            else:
                nxt = pt
                break
        if nxt is None:
            return float(prev[1]), float(prev[2])
        dt = nxt[0] - prev[0]
        if dt <= 1e-9:
            return float(prev[1]), float(prev[2])
        f = min(1.0, max(0.0, (t_sim - prev[0]) / dt))
        return (prev[1] + f * (nxt[1] - prev[1]),
                prev[2] + f * (nxt[2] - prev[2]))

    # ── 感知反馈 ────────────────────────────────────────────────────
    def _nearest_obstacle(self, x, y):
        """(x,y) 到所有障碍 (静态盒 + 动态球) 的最近距离。

        返回 (dist, info):
          dist = 车心到障碍【表面】的距离 (未减 AGV_RADIUS; 净空=dist-AGV_RADIUS)
          info = dict(kind, cx, cy, hx, hy, px, py, dx, dy, radius)
            kind: 'box'(静态轴对齐盒) | 'dyn'(动态球)
            (px,py): 障碍上离车最近的点   (dx,dy): 车→最近点的向量 (避障躲开方向)
        """
        best = (float("inf"), None)
        for cx, cy, hx, hy in OBS_BOXES:
            px = min(max(x, cx - hx), cx + hx)
            py = min(max(y, cy - hy), cy + hy)
            dx = x - px
            dy = y - py
            d = math.hypot(dx, dy)
            if d < best[0]:
                best = (d, dict(kind="box", cx=cx, cy=cy, hx=hx, hy=hy,
                                px=px, py=py, dx=dx, dy=dy, radius=0.0))
        for oid, (ox, oy, r) in self._dyn_obs.items():
            dx = x - ox
            dy = y - oy
            d = max(math.hypot(dx, dy) - r, 0.0)
            if d < best[0]:
                best = (d, dict(kind="dyn", cx=ox, cy=oy, hx=r, hy=r,
                                px=ox, py=oy, dx=dx, dy=dy, radius=r))
        return best

    def _car_positions(self):
        pos = []
        for car_id, sphere in self._car_sphere.items():
            adr = self._sphere_adr.get(sphere)
            if adr is not None:
                pos.append((car_id,
                            float(self.data.mocap_pos[adr][0]),
                            float(self.data.mocap_pos[adr][1])))
        return pos

    def _car_sample(self, car_id, x, y):
        """车的感知采样: 速度 (有限差分), 航向, 计划偏差。
        位置来自 mocap (实际渲染位置); 速度 = 相邻两帧插值位置差分。
        tracking_err = 实际 vs 计划插值位置 (mocap 精确跟随计划时为 0;
        若日后加力偏移, 即反映出执行偏差)。"""
        traj = self._car_traj.get(car_id)
        if not traj:
            return 0.0, 0.0, 0.0, 0.0
        xy0 = self._car_xy(traj, self.t_sim - SAMPLE_DT) or (x, y)
        vx = (x - xy0[0]) / SAMPLE_DT
        vy = (y - xy0[1]) / SAMPLE_DT
        speed = math.hypot(vx, vy)
        heading = math.atan2(vy, vx)
        return vx, vy, heading, speed

    def _publish_feedback(self):
        pos = self._car_positions()
        stamp = self.t_sim

        # /sim/static_obstacles: 静态障碍几何自描述 (启动后发一次)
        if not self._static_published:
            self._static_published = True
            self.bus.publish("/sim/static_obstacles", {
                "boxes": [{"cx": c, "cy": cy, "hx": h, "hy": v}
                          for c, cy, h, v in OBS_BOXES],
                "agv_radius": AGV_RADIUS,
            }, stamp=stamp)

        # /sim/car_state: 每车 位置 + 速度/航向 + 计划偏差
        cars = []
        for cid, x, y in pos:
            vx, vy, heading, speed = self._car_sample(cid, x, y)
            traj = self._car_traj.get(cid)
            xy_planned = self._car_xy(traj, self.t_sim) if traj else (x, y)
            trk = math.hypot(x - xy_planned[0], y - xy_planned[1])
            cars.append({"car_id": cid, "x": x, "y": y,
                         "vx": vx, "vy": vy, "heading": heading, "speed": speed,
                         "tracking_err": trk})
        self.bus.publish("/sim/car_state", {"cars": cars}, stamp=stamp)

        # /sim/agv_distance: 全对连续距离 + 方向 (避障触发, 不等到接触)
        pairs = []
        min_d = float("inf")
        min_a = min_b = None
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx = pos[j][1] - pos[i][1]
                dy = pos[j][2] - pos[i][2]
                d = math.hypot(dx, dy)
                pairs.append({"a": pos[i][0], "b": pos[j][0],
                              "dist": d, "dx": dx, "dy": dy})
                if d < min_d:
                    min_d, min_a, min_b = d, pos[i][0], pos[j][0]
        if min_d < self.metrics['min_agv_agv']:
            self.metrics['min_agv_agv'] = min_d
        if min_d < 0.6:
            self.metrics['near_miss'] += 1      # 准碰撞 (0.4=接触, 0.6 留余量)
        self.bus.publish("/sim/agv_distance", {
            "pairs": pairs,
            "min_dist": min_d if pairs else float("inf"),
            "min_a": min_a, "min_b": min_b,
        }, stamp=stamp)

        # /sim/collision: 球心距 < 0.4 即触碰, 边沿触发计数
        pairs = []
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = math.hypot(pos[i][1] - pos[j][1], pos[i][2] - pos[j][2])
                if d < 2 * AGV_RADIUS:
                    pairs.append((pos[i][0], pos[j][0], d,
                                  (pos[i][1], pos[i][2]),
                                  (pos[j][1], pos[j][2])))
        self._colliding &= set((p[0], p[1]) for p in pairs)
        for (a, b, d, pa, pb) in pairs:
            if (a, b) not in self._colliding:
                self.n_agv_collisions += 1
                print(f"[COLLISION] {a} <-> {b} d={d:.3f} "
                      f"at {pa} / {pb} t={self.t_sim:.1f}")
        self._colliding |= set((p[0], p[1]) for p in pairs)
        self.bus.publish("/sim/collision", {
            "colliding_pairs": [{"a": a, "b": b, "dist": d} for (a, b, d, _, _) in pairs],
            "any": bool(pairs),
            "total_count": self.n_agv_collisions,
        }, stamp=stamp)

        # /sim/obstacle_distance: 每车到最近障碍 (静态盒或动态球)。
        #   dist = 车心到障碍表面 (未减 AGV_RADIUS)
        #   clearance = max(dist - AGV_RADIUS, 0)  (净空)
        #   kind: 'box' | 'dyn';  (px,py) 最近点;  (dx,dy) 车→障碍向量
        obs = []
        for cid, x, y in pos:
            d, info = self._nearest_obstacle(x, y)
            if info is not None:
                obs.append({
                    "car_id": cid, "dist": d,
                    "clearance": max(d - AGV_RADIUS, 0.0),
                    "kind": info["kind"],
                    "cx": info["cx"], "cy": info["cy"],
                    "hx": info["hx"], "hy": info["hy"],
                    "px": info["px"], "py": info["py"],
                    "dx": info["dx"], "dy": info["dy"],
                })
        self.bus.publish("/sim/obstacle_distance", {"cars": obs}, stamp=stamp)

    def _publish_dyn_obstacles(self):
        """广播当前动态障碍表 (变化时发, 供 planner/监听端对齐)。"""
        self.bus.publish("/sim/dynamic_obstacles", {
            "obstacles": [{"id": oid, "x": ox, "y": oy, "radius": r}
                          for oid, (ox, oy, r) in self._dyn_obs.items()],
        }, stamp=self.t_sim)

    # ── 机器人侧避让 ────────────────────────────────────────────────
    def _robot_detour(self, rx, ry, ryaw):
        """AGV 挡在机器人当前目标点附近 → 把目标点横向平移绕开 (工厂 waypoint shift)。
        已到达被平移的目标点 → 清偏移, 转向真实目标点。"""
        nav = self._nav
        if nav.arrived or nav.idx >= len(nav.wps):
            nav.offset[:] = 0.0
            return
        if nav._nav is not None and nav._nav.arrived:
            nav.offset[:] = 0.0                     # 到了平移目标 → 转真实目标
            return
        wx, wy = nav.wps[nav.idx]
        ax, ay = wx - rx, wy - ry
        d = math.hypot(ax, ay)
        if d < 1e-6:
            nav.offset[:] = 0.0
            return
        ux, uy = ax / d, ay / d
        px, py = -uy, ux                            # 垂直前进方向的单位向量
        SHIFT = 0.8
        best = None
        for cid, x, y in self._car_positions():
            if math.hypot(x - wx, y - wy) < 1.0:    # AGV 停在目标点附近
                lat = (x - wx) * px + (y - wy) * py
                if best is None or abs(lat) < abs(best):
                    best = lat
        if best is not None:
            side = 1.0 if best >= 0 else -1.0
            nav.offset[0] = -px * side * SHIFT
            nav.offset[1] = -py * side * SHIFT
        else:
            nav.offset[:] = 0.0

    def _robot_avoid_agvs(self, cmd, rx, ry, ryaw):
        """机器人侧避让 AGV (同 run_factory_full_agv_yield.py 的 lateral push):
        AGV 球太近时, 把机器人沿自身横向 (左轴) 推开, 加到 vy / omega 上。
        避免机器人自己走进停住让行的 AGV。"""
        cl, sl = math.cos(ryaw), math.sin(ryaw)
        for cid, x, y in self._car_positions():
            dx, dy = rx - x, ry - y
            dist = math.hypot(dx, dy)
            if 0.05 < dist < ROBOT_AVOID_DIST:
                force = (ROBOT_AVOID_DIST - dist) / ROBOT_AVOID_DIST
                # 车→机器人的横向分量 (沿机器人左轴, + = 推向左)
                push = -sl * dx + cl * dy
                amp = ROBOT_AVOID_AMP * force
                cmd[1] += push * amp          # vy 侧移
                cmd[2] += push * amp * 0.5    # omega 转向辅助
        return cmd

    # ── 机器人反馈 ──────────────────────────────────────────────────
    def _count_robot_collisions(self):
        """接触式 AGV↔机器人碰撞检测: 扫 data.contact (mj_step 刚算过),
        一边是机器人碰撞 geom (group 3), 另一边是 AGV 球 (名以 agv_ 开头)。
        边沿触发计数, 打印 [ROBOT-COLLISION]。"""
        ncon = self.data.ncon
        active = set()
        for i in range(ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1 = self.model.geom_group[g1] == 3
            r2 = self.model.geom_group[g2] == 3
            if r1 == r2:
                continue
            other = g2 if r1 else g1
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
            if nm and nm.startswith("agv_"):
                active.add(nm)
        self._robot_colliding &= active
        for nm in active:
            if nm not in self._robot_colliding:
                self.n_robot_collisions += 1
                print(f"[ROBOT-COLLISION] AGV {nm} hit robot  "
                      f"t={self.t_sim:.1f} (total={self.n_robot_collisions})")
        self._robot_colliding |= active

    def _publish_robot_feedback(self, frame_dt):
        """每帧发布机器人状态 / 机器人到各 AGV 的距离。"""
        rx, ry, ryaw = self._walker.get_pose()
        stamp = self.t_sim
        # 速度 (帧间有限差分)
        px, py = self._robot_prev
        dt = frame_dt if frame_dt > 1e-9 else 1e-6
        vx = (rx - px) / dt
        vy = (ry - py) / dt
        self._robot_prev = (rx, ry)

        self.bus.publish("/sim/robot_state", {
            "x": rx, "y": ry, "yaw": ryaw,
            "vx": vx, "vy": vy, "speed": math.hypot(vx, vy),
            "wp_index": self._nav.idx, "n_wp": len(self._nav.wps),
            "arrived": self._nav.arrived,
            "fell": bool(self._walker.fell()),     # numpy bool -> 原生 bool (JSON 可序列化)
            "radius": ROBOT_RADIUS,
        }, stamp=stamp)

        # 机器人 ↔ 每辆 AGV: dx,dy = 车→机器人 向量 (AGV 知道机器人在哪边)
        cars = []
        for cid, x, y in self._car_positions():
            dx, dy = rx - x, ry - y
            d = math.hypot(dx, dy)
            cars.append({"car_id": cid, "dist": d, "dx": dx, "dy": dy,
                         "clearance": max(d - AGV_RADIUS - ROBOT_RADIUS, 0.0)})
        self.bus.publish("/sim/robot_agv_distance", {"cars": cars}, stamp=stamp)

        # 路径只在前进时发布一次 (路径点序号变化)
        if self._nav.idx != self._robot_plan_idx_pub:
            self._robot_plan_idx_pub = self._nav.idx
            self._publish_robot_path()

    def _publish_robot_path(self):
        wps = self._nav.wps
        self.bus.publish("/sim/robot_path", {
            "waypoints": [{"x": p[0], "y": p[1]} for p in wps],
            "index": self._nav.idx,
            "arrived": self._nav.arrived,
        }, stamp=self.t_sim)

    # ── 主循环 (时间主, 渲染 + 反馈) ───────────────────────────────
    def run(self, steps=0):
        viewer = None
        if not self.headless:
            viewer = self._launch_viewer()
        _last_wall = time.perf_counter()

        while self.running:
            self._drain_incoming()          # 先处理 path/remove (只在此线程改状态)

            if self._pending_rebuild:
                self._pending_rebuild = False
                self._rebuild_model()
                if not self.headless and viewer is not None:
                    cam = (viewer.cam.lookat.copy(), float(viewer.cam.distance),
                           float(viewer.cam.elevation), float(viewer.cam.azimuth))
                    viewer.close()
                    viewer = self._launch_viewer()
                    viewer.cam.lookat[:] = cam[0]
                    viewer.cam.distance = cam[1]
                    viewer.cam.elevation = cam[2]
                    viewer.cam.azimuth = cam[3]
                if self.headless:
                    print("[REBUILD] model recompiled with "
                          f"{len(self._car_sphere)} AGV sphere(s) in use "
                          f"({len(self._garage_free)} in garage)")

            if not self.headless and not viewer.is_running():
                break

            # 时间推进: 交互模式跟墙上时钟走; headless 用固定步长快速跑
            if self.headless:
                frame_dt = 0.02
                self.t_sim += frame_dt
            else:
                _now = time.perf_counter()
                frame_dt = self.wall_speed * (_now - _last_wall)
                self.t_sim += frame_dt
                _last_wall = _now

            # 时间主: 广播 /clock, planner 据此同步 world_time
            self.bus.publish("/clock", {"time": self.t_sim}, stamp=self.t_sim)

            for car_id, sphere in list(self._car_sphere.items()):
                adr = self._sphere_adr.get(sphere)
                if adr is None:
                    continue
                xy = self._car_xy(self._car_traj.get(car_id), self.t_sim)
                if xy is None:
                    continue
                self.data.mocap_pos[adr][0] = xy[0]
                self.data.mocap_pos[adr][1] = xy[1]
                self.data.mocap_pos[adr][2] = AGV_Z

            # 机器人: 路径跟随 + AGV 避让 + 物理步进 (帧长/模型dt 个子步, 策略每 4 步一次)。
            # mj_step 会真实碰撞 AGV mocap 球 (AGV 是实体, 机器人会被顶/推开);
            # 所以机器人侧也要避让停住的 AGV (横向推开, 同工厂)。
            if self._walker is not None:
                rx, ry, ryaw = self._walker.get_pose()
                self._robot_detour(rx, ry, ryaw)            # 目标点平移绕开挡路 AGV
                cmd = self._nav.update(rx, ry, ryaw)
                cmd = self._robot_avoid_agvs(cmd, rx, ry, ryaw)   # 侧向推开兜底
                if self._robot_hold:
                    cmd = (0.0, 0.0, 0.0)   # 站台被 AGV 占用 → 机器人原地等待
                n_sub = max(1, int(round(frame_dt / self._walker.model.opt.timestep)))
                for _ in range(n_sub):
                    self._walker.step(*cmd)
                # 摔倒计数 (边沿触发): 供验证报告用
                fell_now = bool(self._walker.fell())
                if fell_now and not self._robot_fell_prev:
                    self.robot_fell_count += 1
                self._robot_fell_prev = fell_now
                self._count_robot_collisions()
                self._publish_robot_feedback(frame_dt)

            mujoco.mj_forward(self.model, self.data)   # mocap -> 渲染
            self._publish_feedback()

            if not self.headless:
                viewer.sync()
            else:
                if int(self.t_sim) != self._last_qos_time:
                    self._last_qos_time = int(self.t_sim)
                    if int(self.t_sim) % 5 == 0:
                        pos = {cid: tuple(round(v, 1) for v in
                                          (self._car_xy(self._car_traj.get(cid),
                                                        self.t_sim) or (0, 0)))
                               for cid in self._car_sphere}
                        print(f"  t={int(self.t_sim):>3}  cars={pos}")
            if self.headless and steps and self.t_sim >= steps:
                break

        self.running = False
        if viewer is not None:
            viewer.close()
        if self.headless:
            print("\n=== HEADLESS SUMMARY ===")
            print(f"garage free: {len(self._garage_free)} "
                  f"(dispatched {len(self._car_sphere)})")
            print(f"AGV-AGV collisions: {self.n_agv_collisions}")
            for cid, traj in self._car_traj.items():
                last_t, last_x, last_y, _ = traj[-1]
                print(f"  {cid}@{self._car_sphere.get(cid)}: "
                      f"last={last_x},{last_y} at_t={last_t} "
                      f"arrived={self.t_sim >= last_t}")
            if self._walker is not None:
                rx, ry, ryaw = self._walker.get_pose()
                print(f"  ROBOT: pos=({rx:.2f},{ry:.2f}) yaw={ryaw:.2f} "
                      f"wp={self._nav.idx}/{len(self._nav.wps)} "
                      f"arrived={self._nav.arrived} "
                      f"fell={self._walker.fell()} "
                      f"AGV-robot collisions: {self.n_robot_collisions}")
