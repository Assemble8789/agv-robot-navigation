"""
Planner 节点: 包 AGVWorld, 通过 topic 总线与 MuJoCo 仿真节点解耦
================================================================

职责 (只碰规划侧状态)
  - 订阅 /clock         → 同步 world.world_time (MuJoCo 是时间主, ROS /clock 模式)
  - 订阅 /sim/collision → 打印碰撞反馈
  - 订阅 /sim/obstacle_distance → 打印障碍距离反馈 (每秒摘要)
  - add / del / move / route 命令成功变更后, 把该车完整轨迹发布到 /planner/path;
    del 后发布 /planner/remove
  - CLI (stdin) 线程只往 world.command_queue 写, 不直接改 AGVWorld 状态

线程模型: worker 线程 (run) 独享 AGVWorld 全部状态; CLI 线程只写线程安全
队列; 订阅回调只 incoming.put —— 跨线程零锁。
"""

import os
import sys
import math
import queue
import threading

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BRIDGE_DIR)     # src/ (agv_world_qos.py 所在)
sys.path.insert(0, SRC_DIR)
from agv_world_qos import AGVWorld, QoS_LEVELS

# ── ALT 差分启发式规划 (Red Blob Games: landmark/differential heuristic) ──
# 旧行为 (保留注释, 设 USE_ALT=False 即可回退): AGVWorld.move_car 内部调用
#   agv_world_qos.astar_with_time —— Manhattan 启发式的运动学时空 A*。
# 新行为: 用 agv_world_qos_alt.astar_alt 全局替换 astar_with_time (签名/返回兼容,
#   drop-in, move_car 一行不用改)。 ALT 的 h(n)=max(Manhattan, |d(n,L)-d(goal,L)|),
#   预先对地标跑运动学 BFS 精确距离表 → 迷宫/走廊类大地图探索 -95%、耗时 -83%;
#   本仓库 15x10 开阔小地图收益≈0 (见 agv_world_qos_alt.py --benchmark / --maze)。
USE_ALT = True                        # ← 设 False 即回退旧 Manhattan 启发式
if USE_ALT:
    import agv_world_qos
    from agv_world_qos_alt import astar_alt
    # 旧的 astar_with_time 不删除 —— 只是模块全局引用换到 ALT 版 (原函数仍在模块里)
    agv_world_qos.astar_with_time = astar_alt
    print("[PlannerNode] ALT heuristic planning enabled (agv_world_qos_alt.astar_alt)")

# 危险预警阈值 (m): 低于该值打印 [DANGER]。 实际避障动作是后续闭环重规划的触发点。
AGV_SAFE = 0.60         # AGV-AGV 球心距 (接触是 0.4)
OBST_SAFE = 0.45        # 车心到障碍表面距离
ROBOT_AGV_SAFE = 0.90   # 机器人到 AGV 球心距 (机器人包络 0.3 + AGV 0.2 + 余量)

# ── AGV 让行机器人闭环参数 ────────────────────────────────────────
ROBOT_BLOCK_DIST = 1.2    # 预测最近距离 < 此值就停车让行。对穿相对速度~1.2m/s,
                         # 1.5 能 0 碰撞但让行过多 (yield 32-71), 1.2 折中提前量/让行次数。
ROBOT_OBS_R     = 1.0     # 机器人占的粗格半径 (阻塞 3x3 粗格, 绕行留 ~1m 余量)
BLOCK_HORIZON   = 2.0     # 预测视界 (秒)
WAIT_TIMEOUT    = 3.0     # 让行超时才绕行重规划 (机器人挡路不走)
CLEAR_DIST      = 1.1     # 机器人离开这个距离就恢复续走
CHECK_INTERVAL  = 0.5     # 闭环检查节流 (仿真秒)
RESUME_COOLDOWN = 1.0     # 恢复重规划冷却 (秒, 防反复重规划抖动)
PARK_BLOCK_DIST = 0.7     # 已到站 AGV 停在机器人目标点这个距离内 → 挪开
MOVE_ASIDE_COOLDOWN = 5.0 # 挪开冷却 (秒)

# ── 站台停留 / 进站队列 / 机器人站台等待 (对齐 run_factory_full_agv_yield) ──
DWELL_STOP      = 1.0     # AGV 每个经停点停留秒数 (到站后等这么久才派下一段)
STATION_QUEUE_R = 1.2     # 判定"站台被占 / 正在排队进站"的半径 (m)
ROBOT_HOLD_R    = 1.5     # 机器人目标点距站台 < 此值 且 站台被占 → 机器人等待
ROBOT_HOLD_WAIT = 5.0     # 机器人等待上限 (s, 安全网; 参考文件 WORK_WAIT=5.0)
QUEUE_TIMEOUT   = 15.0    # 进站排队超时 (s): 在等待点等这么久还没进才跳站。
                         # 配合站台交接串行化 (_handoff_buffer) + 轨迹连续修复,
                         # 长等待是安全的 (实测 timeout=15 全矩阵 0 碰撞, 完成率
                         # 5/10/15车 = 97%/73%/71%, 而 timeout=8 只有 93/72/64)。
DISPATCH_INTERVAL = 0.5   # 派发重试节流 (s): 排队车最多每 0.5s 试一次, 防 planner 落后
HOLD_SLOT        = 5      # 站台停车预约窗口 (s): 车到站只锁这么久, 离站释放 →
                         # 后车能规划进空窗 ("先到先进"); 车若停留超时会由
                         # _hold_station 每 tick 刷新续约, 保证物理占用安全。
LOCK_LEAD        = 2      # 临近锁提前量 (s): 车距目标站台 ≤ 这个到达时间才锁站台,
                         # 之前站台对别的车开放 (后来的车排队等待, 不派发即锁)。


class PlannerNode:
    def __init__(self, bus, map_file, station_hold=None):
        self.bus = bus
        self.world = AGVWorld(map_file)     # 规划器本体 (只在本节点 worker 线程触碰)
        # 站台停车预约窗口: None/默认 = 1000 (永久占住, 原行为);
        # 传 HOLD_SLOT (如 5) = 短窗口 (先到先进): 车到站只锁停留那几秒, 离站释放,
        # 后车能规划进空窗。车若停留超时由 _hold_station 每 tick 续约。
        self.world.station_hold = station_hold if station_hold is not None else 1000
        self.incoming = queue.Queue()       # 本地消息队列 (回调只 put 到这里)
        self.running = True
        self._route_queue = {}              # car_id -> [待走地标]
        self._last_arrive = {}              # car_id -> 已上报的 last_time
        self._reached = {}                  # car_id -> 已到达的真实站名集合 (完成率计数)
        self._last_obstacle = None          # 最近一次 /sim/obstacle_distance 消息
        self._last_agv_dist = None          # 最近一次 /sim/agv_distance 消息
        self._last_car_state = None         # 最近一次 /sim/car_state 消息
        self._last_dyn_obs = None           # 最近一次 /sim/dynamic_obstacles 消息
        self._last_static = None            # 最近一次 /sim/static_obstacles 消息
        self._last_robot = None             # 最近一次 /sim/robot_state 消息
        self._last_robot_agv = None         # 最近一次 /sim/robot_agv_distance 消息
        self._last_robot_path = None        # 最近一次 /sim/robot_path 消息
        self._last_danger_t = {}            # 危险预警节流: key -> int(stamp) 已打印
        self._summary_t = -1                # 上次打印反馈摘要的秒数
        # ── 让行机器人闭环状态 ──
        self._yield = {}                    # car_id -> wait_since (让行开始时刻)
        self._last_block_check = -1         # 上次闭环检查的整秒
        self._last_resume = {}              # car_id -> 上次恢复重规划时刻 (冷却)
        self._moved_aside = {}              # car_id -> 上次被挪开时刻 (冷却)
        # ── 站台等待状态 (机器人 hold) ──
        self._robot_hold_state = False      # 上次发布的 hold 值 (变化才发)
        self._robot_hold_since = None       # 本次 hold 开始的时刻 (超时释放用)
        self._queue_since = {}              # car_id -> 排队等待开始时刻 (超时跳过)
        # ── 进站队列状态: 车挪到目标站台【旁边】排队, 不占着上一站 ──
        self._queue_goal = {}               # car_id -> 正在排队的站台名
        self._queue_spot = {}               # car_id -> 排队停靠的格子 (x,y)
        self._queue_lm = {}                 # car_id -> 排队位临时地标名 (_Q_carX)
        self._queue_tried = set()           # 规划失败的排队格 (下次换一个)
        self._last_dispatch = {}            # car_id -> 上次派发尝试时刻 (节流)
        self._replan_failed = set()         # 重规划失败的卡住车 (原地停车, 周期重试)
        self._replan_stuck_since = {}       # car_id -> 卡住开始时刻 (超时跳站安全网)
        # ── 站台队列 (方案A): 每个站台一个 FIFO, 队首先停, 后来的车去等待点排队。
        # 只登记"谁有权停", 不锁站台格 → 路过车照常时空避让, 队首车临近才锁。
        self._station_queue = {}            # 站台名 -> [car_id, ...] (队首在前)

        # ── 验证指标采集 (validate_agv.py 读取) ──
        self.metrics = {
            'replan_total': 0,              # 中途重规划总数
            'replan_reasons': {},           # reason -> 次数 (含 "避开停车格"=AGV停下导致)
            'replan_fail': 0,               # 重规划失败 (堵死)
            'yield_events': 0,              # 停车让行次数
            'queue_events': 0,              # 进站排队次数 (挪到站旁)
            'queue_timeouts': 0,            # 排队超时跳站数
            'move_aside': 0,                # 挪开成功次数
            'move_aside_fail': 0,           # 挪开失败次数
            'goals_planned': 0,             # 派发的真实路段数
            'goals_reached': 0,             # 到站的真实路段数 (完成率 = reached/planned)
            'arrival_times': {},            # car_id -> 最后到站时刻
            'parked_time': 0.0,             # 经停点停车总时长 (等待开销)
        }
        self._stop_parked_since = {}        # car_id -> 到站时刻 (算停车时长)

        # 订阅 (回调只 enqueue, 便宜; 处理在 run 的 worker 线程)
        self._subs = [
            bus.subscribe("/clock",
                          lambda m: self.incoming.put(("clock", m))),
            bus.subscribe("/sim/collision",
                          lambda m: self.incoming.put(("collision", m))),
            bus.subscribe("/sim/obstacle_distance",
                          lambda m: self.incoming.put(("obstacle", m))),
            bus.subscribe("/sim/agv_distance",
                          lambda m: self.incoming.put(("agv_dist", m))),
            bus.subscribe("/sim/car_state",
                          lambda m: self.incoming.put(("car_state", m))),
            bus.subscribe("/sim/dynamic_obstacles",
                          lambda m: self.incoming.put(("dyn_obst", m))),
            bus.subscribe("/sim/static_obstacles",
                          lambda m: self.incoming.put(("static", m))),
            bus.subscribe("/sim/robot_state",
                          lambda m: self.incoming.put(("robot", m))),
            bus.subscribe("/sim/robot_path",
                          lambda m: self.incoming.put(("robot_path", m))),
            bus.subscribe("/sim/robot_agv_distance",
                          lambda m: self.incoming.put(("robot_agv", m))),
        ]

    # ── 对外接口 ────────────────────────────────────────────────────
    def publish_path(self, car_id):
        """把 car 的完整轨迹发布到 /planner/path (sim 据此渲染)。"""
        car = self.world.cars.get(car_id)
        if not car:
            return
        traj = [{"t": p[0], "x": p[1], "y": p[2], "dir": p[3]} for p in car['trajectory']]
        self.bus.publish("/planner/path", {
            "car_id": car_id,
            "action": "add" if len(traj) <= 1 else "update",
            "goal": car.get("current_goal"),
            "trajectory": traj,
        }, stamp=self.world.world_time)

    def route(self, car_id, *goals):
        """给 car 排一串地标, 到站自动派下一段 (等价原 QosMujocoApp.route)。"""
        if car_id not in self.world.cars:
            return "car not found"
        # 完成率分母只算真实站台: 车库/排队位等临时地标 (_ 开头) 不计
        # (回库是"清场"动作, 不该拉低经停完成率)。
        self.metrics['goals_planned'] += sum(1 for g in goals
                                             if not str(g).startswith('_'))
        self._route_queue.setdefault(car_id, []).extend(goals)
        car = self.world.cars[car_id]
        if self.world.world_time >= car['last_time']:
            return self._dispatch_next_route(car_id) or "queued"
        return f"queued {len(goals)} waypoints (dispatch on arrival)"

    def queue_commands(self, cmds):
        """批量注入命令 (demo 用)。 每项形如 (action, args_tuple), 例如
        ("add", ("car01", "LM000", "HIGH")) 或 ("move", ("car01", "LM001"))。"""
        for action, args in cmds:
            self.world.command_queue.put((action, args))

    def start(self):
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def start_cli(self):
        threading.Thread(target=self._cli, daemon=True).start()

    def stop(self):
        self.running = False
        self.incoming.put(("_stop", None))

    # ── worker 主循环 (规划侧唯一触碰 AGVWorld 的线程) ─────────────
    def run(self):
        last_clock_t = -1
        while self.running:
            kind, msg = self.incoming.get()
            if kind == "_stop":
                break
            if kind == "clock":
                # 时间主同步. 注意: sim 每帧发 /clock (500Hz+), 但 QoS 世界是
                # 整数秒粒度 —— 同一秒的重复 clock 直接跳过, 只处理一次.
                # 否则 planner 每帧跑整批处理 → 积压落后于 sim (重规划来不及,
                # 车会撞上刚停下的车)。
                new_t = int(msg["time"])
                if new_t == last_clock_t:
                    continue
                last_clock_t = new_t
                self.world.world_time = new_t
                self._drain_commands()
                self.world.process_pending_replans()        # QoS 抢占重规划
                self._dispatch_pending_routes()             # 到站自动派下一段 (含停留)
                self._handle_robot_blocking()               # 闭环: 让行机器人
                self._update_robot_hold()                   # 站台被占 → 机器人等待
                self._report_arrivals()
                self._maybe_summary()
            elif kind == "collision":
                if msg.get("any"):
                    pairs = ", ".join(f"{p['a']}<->{p['b']}"
                                      f" d={p['dist']:.3f}" for p in msg["colliding_pairs"])
                    print(f"[FB][COLLISION] total={msg['total_count']} "
                          f"t={msg['stamp']:.1f}  {pairs}")
            elif kind == "obstacle":
                self._last_obstacle = msg
                for c in msg.get("cars", []):
                    if c["dist"] < OBST_SAFE:
                        self._throttle_danger(
                            ("obs", c["car_id"]), msg["stamp"],
                            f"[DANGER] {c['car_id']} -> {c['kind']}@{c['cx']},{c['cy']} "
                            f"dist={c['dist']:.3f} clearance={c['clearance']:.3f} "
                            f"dir=({c['dx']:.2f},{c['dy']:.2f})")
            elif kind == "agv_dist":
                self._last_agv_dist = msg
                d = msg.get("min_dist", float("inf"))
                if d < AGV_SAFE and msg.get("min_a"):
                    self._throttle_danger(
                        ("agv", msg["min_a"], msg["min_b"]), msg["stamp"],
                        f"[DANGER] AGV {msg['min_a']} <-> {msg['min_b']} "
                        f"dist={d:.3f} t={msg['stamp']:.1f}")
            elif kind == "car_state":
                self._last_car_state = msg
            elif kind == "dyn_obst":
                self._last_dyn_obs = msg
            elif kind == "static":
                self._last_static = msg
            elif kind == "robot":
                self._last_robot = msg
                if msg.get("fell"):
                    self._throttle_danger(("robot_fell",), msg["stamp"],
                                          "[DANGER] 机器人摔倒了!")
            elif kind == "robot_path":
                self._last_robot_path = msg   # 机器人计划路径 (挪开挡路 AGV 用)
            elif kind == "robot_agv":
                self._last_robot_agv = msg
                for c in msg.get("cars", []):
                    if c["dist"] < ROBOT_AGV_SAFE:
                        self._throttle_danger(
                            ("ragv", c["car_id"]), msg["stamp"],
                            f"[DANGER] 机器人 -> {c['car_id']} "
                            f"dist={c['dist']:.3f} clearance={c['clearance']:.3f} "
                            f"dir=({c['dx']:.2f},{c['dy']:.2f})")

    def _drain_commands(self):
        """处理 CLI/demo 命令。 只对【成功变更】发布 path / remove。"""
        world = self.world
        while not world.command_queue.empty():
            action, args = world.command_queue.get_nowait()
            try:
                if action == "add":
                    res = world.add(*args)
                    if str(res).startswith("Error") or "already exists" in str(res):
                        print(f"[ERR] {res}")
                    else:
                        print(f"[PLAN] {res}")
                        self.publish_path(args[0])
                        # 新出生格将被这辆车占住 → 提前重规划未来路径穿过它的车
                        # (命令按序注入, 后面的车规划时可能没看到前面的出生预约)
                        _new_car = world.cars.get(args[0])
                        if _new_car:
                            sx, sy = _new_car['last_pos']
                            self._replan_cars_through(sx, sy, args[0])
                elif action == "move":
                    car_id = args[0]
                    res = world.move_car(*args)
                    if res == "busy":
                        print(f"[MOVE] car {car_id} busy (still on route)")
                    elif str(res).startswith("Error"):
                        print(f"[ERR] {res}")
                    else:
                        print(f"[PLAN] {res}")
                        self.publish_path(car_id)
                elif action == "del":
                    car_id = args[0]
                    res = world.del_car(car_id)
                    self.bus.publish("/planner/remove", {"car_id": car_id},
                                     stamp=world.world_time)
                    print(f"[DEL] {res}")
                elif action == "route":
                    car_id, goals = args[0], args[1]
                    if car_id not in world.cars:
                        print(f"[ERR] car {car_id} not added yet — 'add' first")
                    elif any(g not in world.lm_dict for g in goals):
                        print(f"[ERR] landmarks not found: "
                              f"{[g for g in goals if g not in world.lm_dict]}")
                    else:
                        print(f"[ROUTE] car {car_id} -> {list(goals)}")
                        self.route(car_id, *goals)
                elif action == "setqos":
                    car_id, qos = args[0], args[1].upper()
                    if car_id in world.cars and qos in QoS_LEVELS:
                        world.cars[car_id]['qos'] = QoS_LEVELS[qos]
                        world.cars[car_id]['qos_name'] = qos
                        print(f"[QOS] car {car_id} -> {qos}")
                    else:
                        print(f"[ERR] car {car_id} or QoS {qos}")
                elif action == "obst":
                    oid = str(args[0])
                    x, y = float(args[1]), float(args[2])
                    r = float(args[3]) if len(args) > 3 else 0.3
                    self.bus.publish("/planner/obstacle_cmd",
                                     {"action": "add", "id": oid,
                                      "x": x, "y": y, "radius": r},
                                     stamp=world.world_time)
                    print(f"[OBST] + dyn obstacle {oid} at ({x},{y}) r={r}")
                elif action == "delobst":
                    oid = str(args[0])
                    self.bus.publish("/planner/obstacle_cmd",
                                     {"action": "del", "id": oid},
                                     stamp=world.world_time)
                    print(f"[OBST] - dyn obstacle {oid}")
            except Exception as e:
                print(f"[ERR] {action}: {e}")

    def _dispatch_pending_routes(self):
        """到站后停留 DWELL_STOP 秒再派下一段 (每个经停点停车 1s)。
        派发尝试按每车 DISPATCH_INTERVAL 节流 —— 排队车每帧重试会让 planner
        落后于 sim (重规划来不及, 车撞上刚停下的车)。"""
        wt = self.world.world_time
        # 先刷新所有停靠车的当前格预约 (短窗口会过期, 但车还物理停着):
        # 必须在后续派发/规划前执行, 否则 _find_queue_spot 会把"还停着车"的格
        # 当成空位, 别的车规划进来相撞。
        # 仅在短窗口模式 (fifo) 需要: 永久预约 (station_hold>=100) 时格子已被
        # move_car 占满, 刷新是多余且会扰动调度的。
        if self.world.station_hold < 100:
            for cid, car in self.world.cars.items():
                if car.get('last_time', -1) <= wt and car.get('last_pos'):
                    self._hold_station(cid)
        for car_id, q in list(self._route_queue.items()):
            if not q:
                continue
            car = self.world.cars.get(car_id)
            if not car or wt < car['last_time'] + DWELL_STOP:
                continue
            # 只派发"当前目标已完成"的车: current_goal 为空, 或车已停在目标格。
            # 否则在途/让行/重规划失败卡住的车 (current_goal 没到) 会被当成到站,
            # 提前派发下一段 → 丢掉还没到的站。
            # 实测: 5车带机器人, car00 去 LM000 途中被让行停车 (6,4) 挡住 →
            # REPLAN-FAIL → _reserve_stop 把 last_time 重置成当前时刻 → 本循环
            # 误判"已到站"派发下一段, 而 LM000 早在派发时就已从队列 pop →
            # LM000 永久丢失 (completion 90%)。current_goal 判据能拦下这种情况。
            cg = car.get('current_goal')
            if cg is not None:
                gcoord = self.world.lm_dict.get(cg)
                if gcoord is None or car.get('last_pos') != gcoord:
                    continue
            if wt - self._last_dispatch.get(car_id, -9) < DISPATCH_INTERVAL:
                continue
            self._last_dispatch[car_id] = wt
            self._dispatch_next_route(car_id)
        # 站台队列推进: 队首临近锁站 (路过车避让); 队首在等待点且前面空了 → 进站
        for S, sq in list(self._station_queue.items()):
            if not sq:
                continue
            owner = sq[0]
            ocar = self.world.cars.get(owner)
            if not ocar:
                sq.pop(0)                    # 车已不存在 → 清出队首
                if not sq:
                    del self._station_queue[S]
                continue
            if ocar.get('current_goal') == S:
                self._near_station_lock(owner)   # 队首临近 S → 锁 S 格
            if (self._queue_goal.get(owner) == S
                    and ocar.get('last_time', -1) <= wt
                    and wt - self._last_dispatch.get(owner, -9) >= DISPATCH_INTERVAL):
                self._last_dispatch[owner] = wt
                self._dispatch_next_route(owner)   # 队首在等待点且前面空了 → 进站
        # 重规划失败卡住的车: 周期重试当前目标 (堵它的车离开后能恢复; 再失败会
        # 原地停车 + 补停车预约, 保持安全)。
        for car_id in list(self._replan_failed):
            car = self.world.cars.get(car_id)
            if not car:
                self._replan_failed.discard(car_id)
                self._replan_stuck_since.pop(car_id, None)
                continue
            goal = car.get('current_goal')
            if not goal:
                self._replan_failed.discard(car_id)
                self._replan_stuck_since.pop(car_id, None)
                continue
            if wt - self._last_dispatch.get(car_id, -9) < DISPATCH_INTERVAL:
                continue
            self._last_dispatch[car_id] = wt
            # 卡死安全网: 卡了 QUEUE_TIMEOUT 仍到不了真实站 → 像排队超时一样跳站,
            # 否则车会永久卡住 (dwell=0 时实测每 0.5s 重试一次, 卡到仿真结束)。
            since = self._replan_stuck_since.get(car_id)
            if since is None:
                self._replan_stuck_since[car_id] = wt
            elif wt - since >= QUEUE_TIMEOUT and not goal.startswith('_'):
                q = self._route_queue.get(car_id)
                if q and q and q[0] == goal:
                    q.pop(0)
                car['current_goal'] = None
                self._replan_failed.discard(car_id)
                self._replan_stuck_since.pop(car_id, None)
                self.metrics['queue_timeouts'] += 1
                print(f"[STUCK-TIMEOUT] car {car_id} 卡 {goal} 超时, 跳过, 剩余 "
                      f"{self._route_queue.get(car_id, [])}")
                continue
            self.replan_car(car_id, goal=goal, reason="恢复重试")
            if car_id not in self._replan_failed:
                self._replan_stuck_since.pop(car_id, None)   # 重试成功 → 清卡死计时

    def _find_queue_spot(self, goal_name):
        """在目标站台旁找一个空闲排队位 (环形搜索, 优先相邻格)。
        避开: 越界/障碍/预约/机器人剩余路径(≥0.8m)/已失败的格/别的车占用的排队位。"""
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
        for r in range(1, 4):
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
                    if (nx, ny, wt + 1) in self.world.reservation_info:
                        continue
                    if any(math.hypot(nx - px, ny - py) < 0.8 for px, py in robot_pts):
                        continue                          # 别挡机器人剩余路径
                    return (nx, ny)
        return None

    def _clear_queue(self, car_id):
        """清掉进站队列状态 + 移除排队位临时地标 (进站成功或超时跳过时调用)。"""
        lm = self._queue_lm.pop(car_id, None)
        if lm:
            self.world.lm_dict.pop(lm, None)
        self._queue_goal.pop(car_id, None)
        self._queue_spot.pop(car_id, None)
        self._queue_since.pop(car_id, None)

    def _dispatch_next_route(self, car_id):
        """派发车去路线里下一个目标。真实站台 → 走站台队列 (方案A: 队首先停,
        后来的车去等待点排队, 不困半路); 车库/临时地标 → 直接派发。"""
        q = self._route_queue.get(car_id)
        if not q:
            return None
        goal = q[0]
        wt = self.world.world_time
        car = self.world.cars.get(car_id)
        old_cell = car.get('last_pos') if car else None     # 离开前的停靠格
        # 真实站台 → 站台队列
        if goal in self.world.lm_dict and not str(goal).startswith('_'):
            return self._dispatch_to_station(car_id, goal, q, wt, old_cell)
        # 非站台 (车库/临时地标) → 直接派发
        res = self.world.move_car(car_id, goal)
        if res != "busy" and not str(res).startswith("Error"):
            # 车从【真实站台】派发离开 → 给站台格留"清空缓冲"预约 (防止排队车
            # 在离开车还在站台边插值时涌进, 对穿)。
            if old_cell is not None and self._is_real_station(old_cell):
                self._handoff_buffer(car_id, old_cell, wt)
            self._station_dequeue(car_id)   # 离开站台 → 出队
            q.pop(0)
            self._clear_queue(car_id)
            ps = self._stop_parked_since.pop(car_id, None)
            if ps is not None:
                self.metrics['parked_time'] += max(0.0, wt - ps)
            gx, gy = self.world.lm_dict.get(goal, (None, None))
            if gx is not None:
                self._replan_cars_through(gx, gy, car_id)
            print(f"[ROUTE] car {car_id} -> {goal} | {res}")
            self.publish_path(car_id)
            return res
        return res

    def _dispatch_to_station(self, car_id, S, q, wt, old_cell):
        """车去站台 S (站台队列): 队首/队列空 → 直接进站; 前面有人 → 去等待点排队。
        站台格不提前锁 (路过车照常时空避让), 队首车临近时由 _near_station_lock 锁。"""
        sq = self._station_queue.setdefault(S, [])
        idx = sq.index(car_id) if car_id in sq else len(sq)
        if any(sq[:idx]):
            # 前面有人排队 → 车已离开旧站台 (出队), 去 S 的等待点等 (不困半路)
            self._station_dequeue(car_id)
            self._enqueue_wait(car_id, S)
            return None
        # 本车是队首 → 尝试进 S
        res = self.world.move_car(car_id, S)
        if res == "busy" or str(res).startswith("Error"):
            self._station_dequeue(car_id)   # 进不了 → 离开旧站台, 去排队
            self._enqueue_wait(car_id, S)
            return res
        # 成功进站 → 从【旧】站台出队 (skip=S: 保留 S 队列), 放回 S 队首
        self._station_dequeue(car_id, skip=S)
        if car_id in sq:
            sq.remove(car_id)
        sq.insert(0, car_id)                # 队首 (不能 append 队尾, 否则被别的车顶掉)
        if old_cell is not None and self._is_real_station(old_cell):
            self._handoff_buffer(car_id, old_cell, wt)
        q.pop(0)
        self._clear_queue(car_id)
        ps = self._stop_parked_since.pop(car_id, None)
        if ps is not None:
            self.metrics['parked_time'] += max(0.0, wt - ps)
        gx, gy = self.world.lm_dict.get(S, (None, None))
        if gx is not None:
            self._replan_cars_through(gx, gy, car_id)
        print(f"[ROUTE] car {car_id} -> {S} | {res}")
        self.publish_path(car_id)
        return res

    def _station_dequeue(self, car_id, skip=None):
        """车从它所在的站台队列出队 (skip: 该站跳过, 刚登记为队首不清除)。"""
        for S, sq in list(self._station_queue.items()):
            if S == skip:
                continue
            if car_id in sq:
                sq.remove(car_id)
                if not sq:
                    del self._station_queue[S]
                break

    def _enqueue_wait(self, car_id, S):
        """车不是 S 队首 → 挪到等待点 (排队位) 排队, 登记进 S 队列。"""
        sq = self._station_queue.setdefault(S, [])
        if car_id not in sq:
            sq.append(car_id)
        if self._queue_goal.get(car_id) == S:
            return                              # 已在排队, 别重复挪
        car = self.world.cars.get(car_id)
        if car is None:
            return
        spot = self._find_queue_spot(S)
        if spot is None:
            print(f"[QUEUE-FAIL] car {car_id} 找不到 {S} 旁等待点")
            return
        lm = "_Q_%s" % car_id
        self.world.lm_dict[lm] = spot
        car['waiting_replan'] = True
        saved_obs = self.world.obs_set
        try:
            if self._last_robot is not None:
                self.world.obs_set = saved_obs | self._robot_obs_cells()
            r2 = self.world.move_car(car_id, lm)
        finally:
            self.world.obs_set = saved_obs
            car['waiting_replan'] = False
        if r2 != "busy" and not str(r2).startswith("Error"):
            self._queue_goal[car_id] = S
            self._queue_spot[car_id] = spot
            self._queue_lm[car_id] = lm
            self._queue_since[car_id] = self.world.world_time
            self.metrics['queue_events'] += 1
            self.publish_path(car_id)
            self._replan_cars_through(spot[0], spot[1], car_id)
            print(f"[QUEUE] car {car_id} 到 {S} 旁 ({spot[0]},{spot[1]}) 排队")
        else:
            self.world.lm_dict.pop(lm, None)
            self._queue_tried.add(spot)
            print(f"[QUEUE-FAIL] car {car_id} 去 {S} 旁 ({spot[0]},{spot[1]}) 失败: {r2}")

    def _report_arrivals(self):
        for car_id, car in self.world.cars.items():
            if car_id in self._yield:
                continue                          # 让行中, 不误报到站
            lt = car['last_time']
            if (lt <= self.world.world_time
                    and self._last_arrive.get(car_id, -1) < lt
                    and car.get('missions')):
                self._last_arrive[car_id] = lt
                # 任何到站停车 (真实站 / 排队位 / 挪开位) 都可能挡住别的车
                # 【已规划】的路径 → 重规划穿过它的车 (预约只挡未来新规划, 不追溯)
                lx, ly = car.get('last_pos', (None, None))
                if lx is not None:
                    self._replan_cars_through(lx, ly, car_id)
                # 车真正停在哪个真实站? 用 last_pos 命中的所有真实 mission 目标,
                # 不用 missions[-1] —— 重规划会 append 重复 mission, last mission
                # 可能已是后面的站, 导致车到站却没被计数。
                lp = car.get('last_pos')
                if lp is None:
                    continue
                new_stations = set()
                for (_sn, gn, _sc, gc) in car.get('missions', []):
                    if not str(gn).startswith('_') and gc == lp:
                        new_stations.add(gn)
                if not new_stations:
                    continue      # 没停在真实站 (重规划失败原地停车等), 不算到站
                reached = self._reached.setdefault(car_id, set())
                for gn in new_stations:
                    if gn in reached:
                        continue
                    reached.add(gn)
                    self.metrics['goals_reached'] += 1
                    self.metrics['arrival_times'][car_id] = lt
                    self._stop_parked_since[car_id] = lt      # 真实到站, 开始计时停车
                    print(f"[ARRIVED] car {car_id} -> {gn}  t={int(lt)}  "
                          f"qos={car.get('qos_name', 'MEDIUM')}")

    def metrics_summary(self):
        """返回验证指标 (供 validate_agv.py 读取), 含派生指标:
        makespan / 完成率 / 曼哈顿(完成路段直线距离) vs 实际走(含绕行/排队/挪开)。"""
        m = dict(self.metrics)
        m['replan_reasons'] = dict(self.metrics['replan_reasons'])
        # reason 以 "避开停车格" 开头都算 AGV 停下导致的重规划
        m['replan_by_stop'] = sum(v for k, v in self.metrics['replan_reasons'].items()
                                  if k.startswith("避开停车格"))
        manh, act = {}, {}
        for cid, car in self.world.cars.items():
            tot = 0.0
            for (_sn, gn, sc, gc) in car.get('missions', []):
                if isinstance(gn, str) and gn.startswith('_'):
                    continue                      # 排队位/挪开位不算路线
                tot += abs(gc[0] - sc[0]) + abs(gc[1] - sc[1])
            manh[cid] = tot
            # 实际路程只算到【最后一个真实站到达】为止: 之后是回库/收尾段
            # (到 _SP_i 车库), 不该算进"经停路上的绕行"。
            last_t = self.metrics['arrival_times'].get(cid, float('inf'))
            prev, d = None, 0.0
            for pt in car.get('trajectory', []):
                if pt[0] > last_t:
                    break
                x, y = pt[1], pt[2]
                if prev is not None and (x != prev[0] or y != prev[1]):
                    d += abs(x - prev[0]) + abs(y - prev[1])
                prev = (x, y)
            act[cid] = d
        m['manhattan_total'] = sum(manh.values())
        m['actual_total'] = sum(act.values())
        m['path_overhead'] = (m['actual_total'] / m['manhattan_total'] - 1.0) \
            if m['manhattan_total'] else 0.0
        arr = list(self.metrics['arrival_times'].values())
        m['makespan'] = max(arr) if arr else float(self.world.world_time)
        m['completion'] = (self.metrics['goals_reached'] / self.metrics['goals_planned']) \
            if self.metrics['goals_planned'] else 0.0
        return m

    def _maybe_summary(self):
        """每秒打印一次感知摘要: 每车速度 + 距障碍距离 + AGV-AGV 最近距 + 动态障碍数。"""
        t = self.world.world_time
        if t == self._summary_t:
            return
        self._summary_t = t
        bits = []
        if self._last_car_state:
            cars = self._last_car_state.get("cars", [])
            for c in cars:
                obs = next((x for x in (self._last_obstacle or {}).get("cars", [])
                            if x["car_id"] == c["car_id"]), None)
                d_obs = f"{obs['dist']:.2f}" if obs else "-"
                bits.append(f"{c['car_id']}:v={c['speed']:.2f} d_obs={d_obs}")
        if self._last_agv_dist and self._last_agv_dist.get("pairs"):
            bits.append("min_agv="
                        f"{self._last_agv_dist['min_dist']:.2f} "
                        f"({self._last_agv_dist.get('min_a')}"
                        f"<->{self._last_agv_dist.get('min_b')})")
        if self._last_dyn_obs is not None:
            bits.append(f"dyn_obst={len(self._last_dyn_obs.get('obstacles', []))}")
        if self._last_robot:
            r = self._last_robot
            ragv = ""
            if self._last_robot_agv and self._last_robot_agv.get("cars"):
                md = min((c["dist"] for c in self._last_robot_agv["cars"]),
                         default=float("inf"))
                ragv = f" min_r_agv={md:.2f}"
            bits.append(f"robot=({r['x']:.1f},{r['y']:.1f}) v={r['speed']:.2f} "
                        f"wp={r['wp_index']}/{r['n_wp']}{ragv}")
        if bits:
            print(f"[FB] t={t} " + " | ".join(bits))

    def _throttle_danger(self, key, stamp, text):
        """危险预警按 (key, 整秒) 节流, 同一威胁每秒最多打一条, 不刷屏。"""
        t0 = int(stamp)
        if self._last_danger_t.get(key) == t0:
            return
        self._last_danger_t[key] = t0
        print(text)

    # ── AGV 让行机器人闭环 ─────────────────────────────────────────
    def _car_pos(self, car_id, t):
        """线性插值车 QoS 轨迹在 t 时刻的位置 (同 sim 的 _car_xy, 首末点钳位)。"""
        traj = self.world.cars.get(car_id, {}).get('trajectory')
        if not traj:
            return None
        prev = traj[0]
        nxt = None
        for pt in traj:
            if pt[0] <= t:
                prev = pt
            else:
                nxt = pt
                break
        if nxt is None:
            return float(prev[1]), float(prev[2])
        dt = nxt[0] - prev[0]
        if dt <= 1e-9:
            return float(prev[1]), float(prev[2])
        f = min(1.0, max(0.0, (t - prev[0]) / dt))
        return (prev[1] + f * (nxt[1] - prev[1]),
                prev[2] + f * (nxt[2] - prev[2]))

    def _closest_approach(self, car_id):
        """预测车与机器人未来最近距离: min over tau of dist(car(t), robot+vel*t)。"""
        r = self._last_robot
        if not r:
            return float("inf")
        rx, ry = r['x'], r['y']
        rvx, rvy = r['vx'], r['vy']
        wt = float(self.world.world_time)
        best = float("inf")
        tau = 0.0
        while tau <= BLOCK_HORIZON + 1e-6:
            xy = self._car_pos(car_id, wt + tau)
            if xy is None:
                break
            px, py = rx + rvx * tau, ry + rvy * tau
            d = math.hypot(xy[0] - px, xy[1] - py)
            if d < best:
                best = d
            tau += 0.25
        return best

    def _robot_obs_cells(self):
        """机器人当前 + 预测位置周围 ROBOT_OBS_R 粗格, 作为临时静态障碍 (让 A* 绕行)。"""
        r = self._last_robot
        if not r:
            return set()
        cells = set()
        for px, py in ((r['x'], r['y']),
                       (r['x'] + r['vx'] * BLOCK_HORIZON,
                        r['y'] + r['vy'] * BLOCK_HORIZON)):
            cx, cy = int(round(px)), int(round(py))
            for dx in range(-int(ROBOT_OBS_R), int(ROBOT_OBS_R) + 1):
                for dy in range(-int(ROBOT_OBS_R), int(ROBOT_OBS_R) + 1):
                    cells.add((cx + dx, cy + dy))
        return cells

    def _stop_car(self, car_id):
        """停车让行: 截断轨迹到当前时刻 (sim 插值钳位 → 车停住), 清未来预约,
        把当前格预约 1000 tick (防别的车撞进来), 发布截断后的 path。"""
        car = self.world.cars.get(car_id)
        if not car:
            return
        self.metrics['yield_events'] += 1
        wt = self.world.world_time
        kept = [p for p in car['trajectory'] if p[0] <= wt]
        if not kept:
            return
        car['trajectory'] = kept
        cx, cy = kept[-1][1], kept[-1][2]
        for res in list(self.world.car_reservations.get(car_id, set())):
            if len(res) >= 3 and res[2] > wt:
                self.world.reservations.discard(res)
                self.world.reservation_info.pop(res, None)
                self.world.car_reservations[car_id].discard(res)
        qos = car['qos']
        for tp in range(wt + 1, wt + 1000):          # 停车位预约
            res = (cx, cy, tp)
            self.world.reservations.add(res)
            self.world.reservation_info[res] = (car_id, qos)
            self.world.car_reservations[car_id].add(res)
        self.publish_path(car_id)
        print(f"[YIELD] car {car_id} 停车让行 @({cx},{cy}) t={wt}")
        # 停住的格子可能挡在别的车【已规划】的路径上 → 重规划那些车
        self._replan_cars_through(cx, cy, car_id)

    def _replan_cars_through(self, cx, cy, blocker):
        """blocker 停在 (cx,cy), 别的车未来路径若穿过该格 (停车前规划的路径没避让)
        就重规划它们绕开新停泊位。"""
        wt = self.world.world_time
        for oid, ocar in list(self.world.cars.items()):
            if oid == blocker or wt >= ocar['last_time']:
                continue
            if not ocar.get('current_goal'):
                continue
            if wt - self._last_resume.get(oid, -9) < RESUME_COOLDOWN:
                continue
            # 直接检查车的未来轨迹是否经过停车格 (QoS 移动是格到格, 穿过某格
            # 必有一个轨迹点==该格; 时间采样会漏掉两个采样点之间穿过的车)
            hit = any(p[0] > wt and (p[1], p[2]) == (cx, cy)
                      for p in ocar.get('trajectory', []))
            if hit:
                self._last_resume[oid] = wt
                # 停车格作为【临时硬障碍】并入 obs_set —— QoS 时空预约检查在这种
                # 场景不可靠 (重规划 A* 仍会穿过预约格), 但 obs_set 是无条件避开的。
                extra = self._robot_obs_cells()
                extra.add((cx, cy))
                self.replan_car(oid, extra_obs=extra,
                                reason=f"避开停车格({cx},{cy})")

    def replan_car(self, car_id, goal=None, extra_obs=None, reason=""):
        """中途重规划: 截断轨迹到 <= world_time (move_car 只 append 不截断, 否则
        插值错乱), 临时把 extra_obs (机器人格) 并入 obs_set, move_car 到 goal,
        发布新 path。 中途重规划需 waiting_replan=True 绕过 move_car 的 busy 检查。
        reason 用于终端打印说明重规划原因。"""
        car = self.world.cars.get(car_id)
        if not car:
            return
        goal = goal or car.get('current_goal')
        if not goal:
            return
        wt = self.world.world_time
        kept = [p for p in car['trajectory'] if p[0] <= wt]
        # 重规划截断后补一个 wt 处的等待点: move_car 只 append path[1:](跳过 wt 起点),
        # 若旧轨迹末点 < wt, 轨迹会在 (旧末点, 新路径首动点) 之间出现时间空隙,
        # 仿真在空隙里线性插值 → 车"滑行"到不该到的位置 → 撞别的车。
        if kept and kept[-1][0] < wt:
            kept.append((wt, kept[-1][1], kept[-1][2], kept[-1][3]))
        car['trajectory'] = kept
        saved_obs = self.world.obs_set
        car['waiting_replan'] = True
        try:
            if extra_obs:
                self.world.obs_set = saved_obs | extra_obs
            res = self.world.move_car(car_id, goal)
        finally:
            self.world.obs_set = saved_obs
            car['waiting_replan'] = False
        tag = f"({reason}) " if reason else ""
        if res != "busy" and not str(res).startswith("Error"):
            self.metrics['replan_total'] += 1
            key = reason or "None"
            self.metrics['replan_reasons'][key] = self.metrics['replan_reasons'].get(key, 0) + 1
            self._replan_failed.discard(car_id)
            self.publish_path(car_id)
            print(f"[REPLAN] car {car_id} {tag}-> {goal} | {res}")
        else:
            self.metrics['replan_fail'] += 1
            # 失败: 车绝不能继续按旧轨迹走 — 预约已被 move_car 删掉, 再走就是不预约的
            # 格子 (别的车会规划进来 → 对头相撞)。发布截断轨迹让仿真停在原地, 并把
            # 停格占住 (parking), 让其他车绕开。
            self._replan_failed.add(car_id)
            self.publish_path(car_id)
            self._reserve_stop(car_id)
            print(f"[REPLAN-FAIL] car {car_id} {tag}: {res} (原地停车, 稍后重试)")

    def _reserve_stop(self, car_id):
        """重规划失败后把车停在当前位置: 截断轨迹已发布, 这里补停车预约 (parking)
        让别的车规划绕开, 并把 last_time/last_pos 对齐到"刚停下", 供派发重试判定。"""
        wt = self.world.world_time
        car = self.world.cars.get(car_id)
        if not car or not car.get('trajectory'):
            return
        _, x, y, _ = car['trajectory'][-1]
        car['last_pos'] = (x, y)
        car['last_time'] = wt
        qos = car.get('qos')
        added = False
        for t in range(int(wt) + 1, int(wt) + 1000):
            k = (x, y, t)
            if k in self.world.reservation_info:
                continue
            self.world.reservations.add(k)
            self.world.reservation_info[k] = (car_id, qos)
            self.world.car_reservations.setdefault(car_id, set()).add(k)
            added = True
        if added:
            self._replan_cars_through(x, y, car_id)

    def _hold_station(self, car_id):
        """已到站/停靠的车: 每 tick 刷新当前格的停车预约。等待点/失败停车的车可能
        停几十秒, 用【长期窗口】覆盖 (否则别的车能规划"短窗口之后"穿过, 而车还在那
        撞上); 正常站台停靠用短窗口 HOLD_SLOT。覆盖当前 wt 时刻 (别的车可能恰好
        下一 tick 到达该格)。"""
        car = self.world.cars.get(car_id)
        if not car or not car.get('last_pos'):
            return
        x, y = car['last_pos']
        wt = self.world.world_time
        qos = car.get('qos')
        # 等待点排队 / 重规划失败停住 → 可能长时间占用, 长期预约
        if self._queue_goal.get(car_id) or car_id in self._replan_failed:
            hold_end = int(wt) + 1000
        else:
            hold_end = int(wt) + 1 + HOLD_SLOT
        added = False
        for t in range(int(wt), hold_end):
            k = (x, y, t)
            if k in self.world.reservation_info:
                continue
            self.world.reservations.add(k)
            self.world.reservation_info[k] = (car_id, qos)
            self.world.car_reservations.setdefault(car_id, set()).add(k)
            added = True
        if added:
            self._replan_cars_through(x, y, car_id)

    def _is_real_station(self, cell):
        """cell 是不是真实站台格 (排除 _ 开头的排队位/挪开位/车库临时地标)。"""
        for name, coord in self.world.lm_dict.items():
            if not name.startswith('_') and coord == cell:
                return True
        return False

    def _handoff_buffer(self, car_id, cell, wt):
        """站台交接串行化: 车 car_id 刚从站台 cell 派发离开, 但它在 [wt, wt+1]
        还插值在站台格/相邻边上。给 cell 保留未来 2 tick 的预约当"清空缓冲",
        让排队车最早 wt+2 之后才规划进站, 避免"离站车 vs 进站车"在站台边对穿。"""
        car = self.world.cars.get(car_id)
        if not car:
            return
        qos = car.get('qos')
        for t in range(int(wt) + 1, int(wt) + 3):
            k = (cell[0], cell[1], t)
            if k in self.world.reservation_info:
                continue
            self.world.reservations.add(k)
            self.world.reservation_info[k] = (car_id, qos)
            self.world.car_reservations.setdefault(car_id, set()).add(k)

    def _near_station_lock(self, car_id):
        """lazy_lock 模式: 在途车按计划即将到达目标站台 (到达剩余 ≤ LOCK_LEAD) 时,
        才锁定站台停车格 —— 车还在远处时站台对别的车开放 (先到先进), 快到时锁住,
        后来的车规划时看到站台将忙 → 排队等待。"""
        car = self.world.cars.get(car_id)
        if not car:
            return
        goal = car.get('current_goal')
        if not goal or str(goal).startswith('_'):
            return                          # 临时地标 (排队位/挪开位/车库) 不锁
        gx, gy = self.world.lm_dict.get(goal, (None, None))
        if gx is None:
            return
        tr = car.get('trajectory', [])
        if not tr:
            return
        wt = self.world.world_time
        arrive = None
        for p in tr:
            if p[1] == gx and p[2] == gy and p[0] >= wt:
                arrive = p[0]
                break
        if arrive is None or arrive - wt > LOCK_LEAD:
            return                          # 还没临近, 不锁
        qos = car.get('qos')
        added = False
        for t in range(int(arrive) + 1, int(arrive) + 1 + HOLD_SLOT):
            k = (gx, gy, t)
            if k in self.world.reservation_info:
                continue
            self.world.reservations.add(k)
            self.world.reservation_info[k] = (car_id, qos)
            self.world.car_reservations.setdefault(car_id, set()).add(k)
            added = True
        if added:
            self._replan_cars_through(gx, gy, car_id)   # 锁定时重绕穿过站台的车

    def _move_agv_aside(self, car_id):
        """把停在机器人路径上的【已到站】AGV 挪到远处空位 (工厂的"挪开")。
        用临时地标 move_car 到离机器人剩余路径 ≥3 格的位置, 避免挪过去又被挡。"""
        car = self.world.cars.get(car_id)
        if not car:
            return
        x, y = car['last_pos']
        wt = self.world.world_time
        # 机器人剩余路径点 (当前目标点之后), 挪开要离它们尽量远
        robot_pts = []
        if self._last_robot_path is not None:
            idx = self._last_robot.get('wp_index', 0) if self._last_robot else 0
            wps = self._last_robot_path.get('waypoints', [])
            robot_pts = [(wps[i]['x'], wps[i]['y']) for i in range(idx, len(wps))]
        # 从近到远搜环形候选, 取离机器人路径最远的自由格; ≥3 格就停
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
                        continue                      # 越界格不能挪去 (A* 规划不到)
                    if (nx, ny) in self.world.obs_set:
                        continue
                    if (nx, ny, wt + 1) in self.world.reservation_info:
                        continue
                    dmin = min((math.hypot(nx - px, ny - py) for px, py in robot_pts),
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
            # 挪开路径也要避开机器人 (否则从 LM005 挪到远处会穿过正走近的机器人)
            if self._last_robot is not None:
                self.world.obs_set = saved_obs | self._robot_obs_cells()
            res = self.world.move_car(car_id, lm)
        finally:
            self.world.obs_set = saved_obs
            car['waiting_replan'] = False
            self.world.lm_dict.pop(lm, None)
        if res != "busy" and not str(res).startswith("Error"):
            car['current_goal'] = None          # 挪完仍视为停泊
            self._moved_aside[car_id] = wt
            self.metrics['move_aside'] += 1
            self.publish_path(car_id)
            print(f"[MOVE-ASIDE] car {car_id} 挪到 ({nx},{ny}) | {res}")
        else:
            self.metrics['move_aside_fail'] += 1
            print(f"[MOVE-ASIDE-FAIL] car {car_id}: {res}")

    def _robot_target_station(self):
        """机器人当前目标点对应的站台地标名 (距 < ROBOT_HOLD_R), 无则 None。"""
        if not self._last_robot or not self._last_robot_path:
            return None
        idx = self._last_robot.get('wp_index', 0)
        wps = self._last_robot_path.get('waypoints', [])
        if idx >= len(wps):
            return None
        wx, wy = wps[idx]['x'], wps[idx]['y']
        for name, (x, y) in self.world.lm_dict.items():
            if name.startswith('_'):
                continue                       # 临时地标 (排队位/挪开位), 不算站台
            if math.hypot(wx - x, wy - y) < ROBOT_HOLD_R:
                return name
        return None

    def _station_status(self):
        """返回 {站台地标名: 'docked'/'queuing'} —— 每个站台第一辆符合的车判定。
        'docked' : 有车已到站 (wt>=last_time) 且距地标 < STATION_QUEUE_R
        'queuing': 有车正在该站旁排队 (_queue_goal) 或 在途 current_goal==该地标
                   且距它 < STATION_QUEUE_R (正在进站)"""
        wt = self.world.world_time
        status = {}
        for name, (x, y) in self.world.lm_dict.items():
            if name.startswith('_'):
                continue                       # 临时地标不算站台
            for cid, car in self.world.cars.items():
                if cid in self._yield:
                    continue                   # 让行中的车不算站台占用
                if self._queue_goal.get(cid) == name:      # 正在站旁排队
                    status[name] = 'queuing'
                    break
                cx, cy = None, None
                if wt >= car.get('last_time', -1):
                    cx, cy = car.get('last_pos', (None, None))   # 已到站
                else:
                    if car.get('current_goal') == name:
                        cur = self._car_pos(cid, wt)             # 在途且正朝本站
                        if cur:
                            cx, cy = cur
                if cx is None:
                    continue
                if math.hypot(cx - x, cy - y) < STATION_QUEUE_R:
                    st = 'docked' if wt >= car.get('last_time', -1) else 'queuing'
                    status[name] = st
                    break
        return status

    def _update_robot_hold(self):
        """机器人当前目标站台被 AGV 占用/正在进站 → 发布 hold 让 sim 停住机器人。
        只在状态变化时发布; 超过 ROBOT_HOLD_WAIT 强制释放 (安全网)。"""
        target = self._robot_target_station()
        hold = False
        reason = ""
        if target:
            status = self._station_status().get(target)
            if status in ('docked', 'queuing'):
                hold = True
                reason = f"station {target} {status}"
        wt = self.world.world_time
        if hold:
            if self._robot_hold_since is None:
                self._robot_hold_since = wt
            elif wt - self._robot_hold_since >= ROBOT_HOLD_WAIT:
                hold = False                      # 等待超限, 释放 (靠 AGV 侧/机器人侧避让兜底)
                reason = "hold timeout"
        else:
            self._robot_hold_since = None
        if hold != self._robot_hold_state:
            self._robot_hold_state = hold
            self.bus.publish("/planner/robot_hold",
                             {"hold": bool(hold), "reason": reason},
                             stamp=wt)
            if hold:
                print(f"[ROBOT-HOLD] {reason}")

    def _handle_robot_blocking(self):
        """闭环状态机 (每 CHECK_INTERVAL 秒): 预测到撞机器人 → 停车让行;
        机器人让开 → 恢复; 让行超时 → 绕行重规划; 已到站车挡机器人目标点 → 挪开。"""
        wt = self.world.world_time
        if wt - self._last_block_check < CHECK_INTERVAL:
            return
        self._last_block_check = wt
        if self._last_robot is None:
            return
        # 机器人目标站台 (去往的站): 正在进站/已停泊的车优先, 不让行, 机器人改为等待
        target = self._robot_target_station()
        target_xy = self.world.lm_dict.get(target) if target else None
        for car_id, car in list(self.world.cars.items()):
            if car.get('current_goal') is None or wt >= car['last_time']:
                continue                          # 没任务 / 已到站
            # 进站优先: 该车正要进机器人目标站台 → 让它先停泊 (机器人等), 防死锁
            # 半径 2.5 会引入灾难性回归 (车去目标站途中横穿机器人路径但不让行 →
            # 10车P1 42 次碰撞), 必须保持 STATION_QUEUE_R。
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

        # 已到站的 AGV 停在机器人当前目标点附近 → 挪开 (否则机器人永远过不去)
        if self._last_robot is not None and self._last_robot_path is not None:
            r = self._last_robot
            idx = r.get('wp_index', 0)
            wps = self._last_robot_path.get('waypoints', [])
            if idx < len(wps):
                wx, wy = wps[idx]['x'], wps[idx]['y']
                for car_id, car in list(self.world.cars.items()):
                    if wt < car['last_time'] or car_id in self._yield:
                        continue                          # 在途/让行中, 由让行逻辑处理
                    if self._route_queue.get(car_id):
                        continue                          # 还有路线没走完, 不挪
                    if wt - self._moved_aside.get(car_id, -99) < MOVE_ASIDE_COOLDOWN:
                        continue
                    lx, ly = car['last_pos']
                    if math.hypot(lx - wx, ly - wy) < PARK_BLOCK_DIST:
                        self._move_agv_aside(car_id)

    # ── CLI (stdin) ─────────────────────────────────────────────────
    def _cli(self):
        print("AGV Bridge Planner CLI — 命令:")
        print("  add <car> <landmark> [qos]   新建车辆")
        print("  move <car> <landmark>        移动到目标地标")
        print("  route <car> <lm1> <lm2> ...  依次走多个点")
        print("  del <car>                    删除车辆")
        print("  setqos <car> <qos>           修改 QoS (HIGH/MEDIUM/LOW)")
        print("  obst <x> <y> [r]             放一个动态障碍 (红色球, 默认 r=0.3)")
        print("  delobst <id>                 删除动态障碍")
        print("  exit                         退出")
        while self.running:
            try:
                line = input(">> ").strip()
            except (EOFError, OSError):
                self.running = False
                break
            parts = line.split()
            if not parts:
                continue
            cmd = parts[0].lower()
            try:
                if cmd == "add" and len(parts) >= 3:
                    car_id, lm = parts[1], parts[2]
                    qos = parts[3].upper() if len(parts) > 3 else "MEDIUM"
                    self.world.command_queue.put(("add", (car_id, lm, qos)))
                elif cmd == "move" and len(parts) >= 3:
                    car_id, lm = parts[1], parts[2]
                    qos = None
                    for p in parts[3:]:
                        if p.upper() in QoS_LEVELS:
                            qos = p.upper()
                    self.world.command_queue.put(
                        ("move", (car_id, lm, False, qos)))
                elif cmd == "route" and len(parts) >= 3:
                    car_id = parts[1]
                    self.world.command_queue.put(
                        ("route", (car_id, tuple(parts[2:]))))
                elif cmd == "del" and len(parts) == 2:
                    self.world.command_queue.put(("del", (parts[1],)))
                elif cmd == "setqos" and len(parts) == 3:
                    self.world.command_queue.put(
                        ("setqos", (parts[1], parts[2].upper())))
                elif cmd == "obst" and len(parts) >= 4:
                    try:
                        x, y = float(parts[2]), float(parts[3])
                        r = float(parts[4]) if len(parts) > 4 else 0.3
                        n_obst = len((self._last_dyn_obs or {}).get("obstacles", []))
                        self.world.command_queue.put(
                            ("obst", ("o%d" % n_obst, x, y, r)))
                    except ValueError:
                        print("[ERR] obst <x> <y> [r] 需要数字")
                elif cmd == "delobst" and len(parts) == 2:
                    self.world.command_queue.put(("delobst", (parts[1],)))
                elif cmd == "exit":
                    self.running = False
                    break
                else:
                    print("usage: add <car> <lm> [qos] | move <car> <lm> | "
                          "route <car> <lm>... | del <car> | setqos <car> <qos> | exit")
            except Exception as e:
                print(f"[CLI-ERR] {e}")
