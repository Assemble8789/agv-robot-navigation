import json
import heapq
import random
import argparse
import datetime
import os
import sys
import threading
import queue
import time
import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ModuleNotFoundError:
    plt = None
    FuncAnimation = None

DIR_MARKERS = {
    'x+': r'$\rightarrow$',
    'x-': r'$\leftarrow$',
    'y+': r'$\uparrow$',
    'y-': r'$\downarrow$'
}

DIR_MAP = {'x+': 0, 'y+': 1, 'x-': 2, 'y-': 3}
REV_DIR_MAP = {0: 'x+', 1: 'y+', 2: 'x-', 3: 'y-'}

DX_DY = [(1, 0), (0, 1), (-1, 0), (0, -1)]

DEFAULT_COLOR_CYCLE = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
]


def ensure_matplotlib():
    if plt is None or FuncAnimation is None:
        raise RuntimeError(
            "matplotlib is required for visualization but is not installed. "
            "Install it with: pip install matplotlib"
        )

QoS_LEVELS = {
    'CRITICAL': 4,
    'HIGH': 3,
    'MEDIUM': 2,
    'LOW': 1,
    'IDLE': 0
}
QoS_NAMES = {v: k for k, v in QoS_LEVELS.items()}

def get_direction_label(dx, dy, current_dir):
    if dx > 0: return 'x+'
    if dx < 0: return 'x-'
    if dy > 0: return 'y+'
    if dy < 0: return 'y-'
    return current_dir


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_with_time(width, height, obs_set, start, goal, start_time, start_dir_str, reservation_info, x_min=0, y_min=0):
    """
    reservation_info: dict mapping (x, y, t) or (x1, y1, t, x2, y2) -> (car_id, qos_level)
    Returns: (path, conflicts) where conflicts is list of (x,y,t) or (x1,y1,t,x2,y2) that conflict with lower QoS
    Kinematic A*: State = (x, y, dir_idx)
    """
    start_dir = DIR_MAP.get(start_dir_str, 0)

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start_time, start[0], start[1], start_dir))

    came_from = {}
    visited = set()
    conflicts = []

    x_max, y_max = x_min + width, y_min + height

    max_iter = 100000
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
            return (list(reversed(path)), conflicts)

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

            # 被预约的时空点/边: 直接跳过 (不扩展) —— 保证 AGV 路径彼此不重叠,
            # 同 QoS 车辆也不会撞。 原实现只记录冲突却仍走进去, 导致同 QoS
            # 两车规划出重叠路径在运行时相撞。
            res_vertex = (nx, ny, nt)
            if res_vertex in reservation_info:
                if act_type == 'wait' and (nx, ny) == (cx, cy):
                    continue   # 原行为: 等待格被占则不能原地等
                continue       # 跳过被预约的时空点

            if act_type == 'move':
                res_edge = (nx, ny, nt, cx, cy)
                if res_edge in reservation_info:
                    continue   # 跳过被预约的移动边

            h = heuristic((nx, ny), goal)
            new_priority = nt + h

            heapq.heappush(open_set, (new_priority, nt, nx, ny, nd))

            new_node_key = (nx, ny, nd, nt)
            if new_node_key not in came_from:
                came_from[new_node_key] = (cx, cy, cdir, t)

    return (None, conflicts)


class AGVWorld:
    def __init__(self, map_file):
        with open(map_file, "r") as f:
            self.map_data = json.load(f)

        self.width = self.map_data["width"]
        self.height = self.map_data["height"]

        raw_obstacles = self.map_data.get("obstacles", [])
        self.obstacles = []
        has_negative = False
        for o in raw_obstacles:
            if isinstance(o, dict):
                ox, oy = o["x"], o["y"]
            else:
                ox, oy = o[0], o[1]
            self.obstacles.append((ox, oy))
            if ox < 0 or oy < 0:
                has_negative = True

        self.obs_set = set(self.obstacles)
        self.landmarks = self.map_data.get("landmarks", [])

        self.x_min, self.y_min = (-self.width // 2, -self.height // 2) if has_negative else (0, 0)

        self.lm_dict = {lm["name"]: (lm["x"], lm["y"]) for lm in self.landmarks}

        self.grid = np.zeros((self.height, self.width))
        for ox, oy in self.obstacles:
            iy, ix = int(oy - self.y_min), int(ox - self.x_min)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                self.grid[iy, ix] = 1

        self.cars = {}
        self.reservations = set()
        self.reservation_info = {}
        self.car_reservations = {}
        self.car_paths = {}
        self.car_waiting = {}
        self.command_queue = queue.Queue()
        # 目标格停车预约窗口 (tick): 默认 1000 = 永久占住 (车不走就一直锁)。
        # bridge 里 planner_node 会把它改成短窗口 (如 5) 实现"先到先进":
        # 车到站只锁停留那几秒, 离站释放, 后车能规划进空窗。
        self.station_hold = 1000
        # lazy_lock=True (bridge 设置): move_car 不再"派发即锁"目标站停车格,
        # 改为由 planner_node._near_station_lock 在车临近时才锁 (后来的车排队等待)。
        self.lazy_lock = False
        self.world_time = 0
        self.running = True
        self.pending_replan = queue.Queue()

        if plt is not None:
            self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        else:
            self.color_cycle = DEFAULT_COLOR_CYCLE
        self.color_idx = 0

    def get_next_color(self):
        color = self.color_cycle[self.color_idx % len(self.color_cycle)]
        self.color_idx += 1
        return color

    def get_nearest_obstacle_direction(self, x, y):
        if not self.obstacles:
            return None

        nearest = None
        min_dist_sq = float('inf')

        for ox, oy in self.obstacles:
            d = (ox - x)**2 + (oy - y)**2
            if d < min_dist_sq:
                min_dist_sq = d
                nearest = (ox, oy)

        if not nearest:
            return None

        nx, ny = nearest
        dx, dy = nx - x, ny - y

        if dx == 0 and dy == 0:
            return None

        if abs(dx) >= abs(dy):
            return 'x+' if dx > 0 else 'x-'
        else:
            return 'y+' if dy > 0 else 'y-'

    def get_state_at_time(self, car_id, t):
        car = self.cars.get(car_id)
        if not car or not car.get('trajectory'):
            return None

        state = car['trajectory'][0]
        for point in car['trajectory']:
            if point[0] > t:
                break
            state = point
        return state

    def find_goal_conflict(self, goal_coord, start_t, car_id):
        for other_car_id, other_car in self.cars.items():
            if other_car_id == car_id:
                continue
            # 只挡【物理上已停在目标格】的车。停靠顺序由 planner_node 的站台队列
            # (_station_queue) 显式管理: 谁先登记谁停, 后来的车去等待点排队。
            # 不再用"同站目标锁" (先派发者锁死站台, 远处的车虚拟占用)。
            if other_car.get('last_pos') == goal_coord and other_car.get('last_time', -1) <= start_t:
                return other_car_id, other_car.get('last_time', start_t)
        return None

    def generate_rotation_steps(self, start_t, x, y, start_dir_str, target_dir, car_id=None):
        steps = []
        if start_dir_str == target_dir:
            return steps

        curr_idx = DIR_MAP[start_dir_str]
        target_idx = DIR_MAP[target_dir]
        t = start_t

        while curr_idx != target_idx:
            t += 1
            dist_left = (target_idx - curr_idx) % 4
            dist_right = (curr_idx - target_idx) % 4

            if dist_left <= dist_right:
                 curr_idx = (curr_idx + 1) % 4
            else:
                 curr_idx = (curr_idx - 1) % 4

            new_dir_str = REV_DIR_MAP[curr_idx]
            steps.append((t, x, y, new_dir_str))

        return steps


    def add(self, car_id, landmark_name, qos='MEDIUM'):
        if car_id in self.cars:
            coord = self.lm_dict.get(landmark_name)
            existing_car = self.cars[car_id]
            if coord and existing_car['last_pos'] == coord:
                return f"Car {car_id} already exists at {landmark_name} with QoS {existing_car['qos_name']}."
            return f"Error: Car {car_id} already exists."

        coord = self.lm_dict.get(landmark_name)
        if not coord:
            return f"Error: Landmark {landmark_name} not found."

        qos_level = QoS_LEVELS.get(qos.upper(), QoS_LEVELS['MEDIUM'])

        color = self.get_next_color()
        self.cars[car_id] = {
            'color': color,
            'trajectory': [(self.world_time, coord[0], coord[1], 'x+')],
            'last_time': self.world_time,
            'last_pos': coord,
            'missions': [],
            'verbose': False,
            'qos': qos_level,
            'qos_name': qos.upper(),
            'current_goal': None,
            'waiting_replan': False
        }
        self.car_reservations[car_id] = set()
        self.car_paths[car_id] = []
        self.car_waiting[car_id] = False

        for t_park in range(self.world_time, self.world_time + 1000):
            res = (coord[0], coord[1], t_park)
            self.reservations.add(res)
            self.reservation_info[res] = (car_id, qos_level)
            self.car_reservations[car_id].add(res)

        return f"Car {car_id} added at {landmark_name} with QoS {qos.upper()}."

    def del_car(self, car_id):
        if car_id not in self.cars:
            return f"Error: Car {car_id} not found."

        if car_id in self.car_reservations:
            for res in self.car_reservations[car_id]:
                self.reservations.discard(res)
                if res in self.reservation_info:
                    del self.reservation_info[res]
            del self.car_reservations[car_id]

        if car_id in self.car_paths:
            del self.car_paths[car_id]

        if car_id in self.car_waiting:
            del self.car_waiting[car_id]

        del self.cars[car_id]
        return f"Car {car_id} deleted."

    def move_car(self, car_id, goal_name, verbose=False, qos_override=None):
        if car_id not in self.cars:
            return f"Error: Car {car_id} not found. Use 'addcar' first."

        car = self.cars[car_id]
        if self.world_time < car['last_time'] and not car.get('waiting_replan', False):
            return "busy"

        if qos_override:
            car['qos'] = QoS_LEVELS.get(qos_override.upper(), car['qos'])
            car['qos_name'] = qos_override.upper()

        goal_coord = self.lm_dict.get(goal_name)
        if not goal_coord:
            return f"Error: Landmark {goal_name} not found."

        start_t = self.world_time

        state_at_start = self.get_state_at_time(car_id, start_t)
        if state_at_start is not None:
            _, state_x, state_y, start_dir = state_at_start
            start_coord = (state_x, state_y)
        else:
            start_coord = car['last_pos']
            start_dir = car['trajectory'][-1][3] if car['trajectory'] else 'x+'

        if start_coord is None:
            return f"Error: Car {car_id} has no valid position for replanning."

        start_name = "Unknown"
        for name, coord in self.lm_dict.items():
            if coord == start_coord:
                start_name = name
                break

        removed_res = []          # (res, val) 被删的预约, 失败回滚用
        if car_id in self.car_reservations:
            to_remove = []
            for res in self.car_reservations[car_id]:
                if hasattr(res, '__len__') and len(res) >= 3:
                     if res[2] > start_t:
                         to_remove.append(res)

            # 记录被删的预约 (失败要回滚): 车仍在走【已发布】的旧轨迹, 预约若删了
            # 而不恢复, 别的车规划时看不到这些格 → 开进同一个格 → 对头相撞。
            for res in to_remove:
                self.reservations.discard(res)
                val = self.reservation_info.get(res)
                # 归属检查: 这个 (格,时刻) 键可能已被【别的车】接管 (本车旧预约
                # 被覆盖/回收后, 别的车 A* 在它为空时规划进来)。只删自己名下的,
                # 否则会把别车的预约删掉 → 那辆车继续走却没预约 → 对头相撞。
                if val is not None and val[0] == car_id:
                    self.reservation_info.pop(res, None)
                    removed_res.append((res, val))
                self.car_reservations[car_id].remove(res)

        def _rollback():
            for res, val in removed_res:
                self.reservations.add(res)
                self.reservation_info[res] = val
                self.car_reservations[car_id].add(res)

        goal_conflict = self.find_goal_conflict(goal_coord, start_t, car_id)
        if goal_conflict:
            other_car_id, conflict_time = goal_conflict
            _rollback()
            return (
                f"Error: Goal {goal_name} is already reserved by {other_car_id} "
                f"from t={conflict_time}"
            )

        path, conflicts = astar_with_time(
            self.width, self.height, self.obs_set, start_coord, goal_coord,
            start_t, start_dir, self.reservation_info, self.x_min, self.y_min
        )

        if not path:
            _rollback()
            return f"Error: Car {car_id} could not find path to {goal_name}"

        current_qos = car['qos']
        preempted_cars = []

        for conflict_info in conflicts:
            if len(conflict_info) == 3:
                res, existing_car, existing_qos = conflict_info
            else:
                continue

            if existing_car == car_id:
                continue

            if current_qos > existing_qos:
                preempted_cars.append((existing_car, res, existing_qos))

        preempted_cars_unique = {}
        for existing_car, res, existing_qos in preempted_cars:
            if existing_car not in preempted_cars_unique:
                preempted_cars_unique[existing_car] = existing_qos

        for existing_car, existing_qos in preempted_cars_unique.items():
            if existing_car not in self.cars:
                continue

            preempted_car = self.cars[existing_car]
            old_qos_name = preempted_car['qos_name']
            new_qos = max(0, existing_qos - 1)
            preempted_car['qos'] = new_qos
            preempted_car['qos_name'] = QoS_NAMES.get(new_qos, 'IDLE')

            if existing_car in self.car_reservations:
                for res in list(self.car_reservations[existing_car]):
                    if len(res) >= 3 and res[2] > self.world_time:
                        self.reservations.discard(res)
                        if res in self.reservation_info:
                            del self.reservation_info[res]
                        self.car_reservations[existing_car].discard(res)

            preempted_car['waiting_replan'] = True
            state_at_preempt = self.get_state_at_time(existing_car, self.world_time)
            if state_at_preempt is not None:
                _, px, py, _ = state_at_preempt
                preempted_car['last_pos'] = (px, py)
            preempted_car['last_time'] = self.world_time

            self.pending_replan.put((existing_car, old_qos_name, preempted_car['qos_name']))

            if verbose:
                print(f"[QoS] Car {existing_car} preempted by {car_id}: QoS {old_qos_name} -> {preempted_car['qos_name']}")

        final_t, final_x, final_y, final_dir = path[-1]
        target_face_dir = self.get_nearest_obstacle_direction(final_x, final_y)

        if target_face_dir and target_face_dir != final_dir:
            if verbose:
                print(f"[Plan] Reached goal, rotating from {final_dir} to {target_face_dir}")
            rotation_steps = self.generate_rotation_steps(final_t, final_x, final_y, final_dir, target_face_dir)
            path.extend(rotation_steps)

        # 保证轨迹连续: move_car 通常只 append path[1:] (path[0]=起点, 已在旧轨迹里)。
        # 但若旧轨迹末点时刻 < 新路径起点 (dispatch 直接调 move_car 不截断时),
        # 不补 path[0] 会在 (旧末点, path[1]) 之间出现时间空隙, 仿真线性插值
        # → 车"滑行"到不该到的位置 → 撞别的车 (实测 car10 (4,2)->(5,2) 3s 滑行撞车)。
        if car['trajectory'] and path and car['trajectory'][-1][0] < path[0][0]:
            car['trajectory'].append(path[0])

        for i in range(1, len(path)):
            t, x, y, d = path[i]
            res_v = (x, y, t)
            self.reservations.add(res_v)
            self.reservation_info[res_v] = (car_id, current_qos)
            self.car_reservations[car_id].add(res_v)

            prev_t, prev_x, prev_y, prev_d = path[i-1]
            if (x != prev_x or y != prev_y):
                res_e = (prev_x, prev_y, t, x, y)
                self.reservations.add(res_e)
                self.reservation_info[res_e] = (car_id, current_qos)
                self.car_reservations[car_id].add(res_e)

            car['trajectory'].append((t, x, y, d))

        # lazy_lock 模式: 真实站台 (非 _ 开头) 不随派发写停车预约 (车临近站台时由
        # planner_node._near_station_lock 补), 让远处的车不虚拟占用站台;
        # 但【临时地标】(等待点 _Q_carX / 车库 _SP_i) 仍写 —— 车停那要保护,
        # 否则别的车会规划穿过它 (实测 car13 停在等待点 (5,0) 被 car05 穿过撞车)。
        if (not self.lazy_lock) or str(goal_name).startswith('_'):
            for t_park in range(path[-1][0] + 1, path[-1][0] + self.station_hold):
                res_p = (goal_coord[0], goal_coord[1], t_park)
                self.reservations.add(res_p)
                self.reservation_info[res_p] = (car_id, current_qos)
                self.car_reservations[car_id].add(res_p)

        car['last_time'] = path[-1][0]
        car['last_pos'] = goal_coord
        car['missions'].append((start_name, goal_name, start_coord, goal_coord))
        car['verbose'] = verbose
        car['current_goal'] = goal_name
        car['waiting_replan'] = False

        self.car_paths[car_id] = path

        if preempted_cars_unique:
            return f"Car {car_id} moving from {start_name} to {goal_name} (preempted {len(preempted_cars_unique)} cars)"
        return f"Car {car_id} moving from {start_name} to {goal_name}."

    def process_pending_replans(self):
        while not self.pending_replan.empty():
            car_id, old_qos, new_qos = self.pending_replan.get()
            if car_id in self.cars and self.cars[car_id].get('waiting_replan', False):
                if self.cars[car_id]['current_goal']:
                    goal_name = self.cars[car_id]['current_goal']
                    print(f"[REPLAN] car {car_id} (QoS抢占 {old_qos}->{new_qos}) -> {goal_name}")
                    if self.cars[car_id]['verbose']:
                        print(f"[Replan] Car {car_id} replanning with QoS {new_qos} to {goal_name}")
                    self.move_car(car_id, goal_name, verbose=self.cars[car_id]['verbose'])

    def run_cli(self):
        print("AGV World QoS CLI started.")
        print("Commands:")
        print("  add <carID> <landmark> [qos]     - Add car with optional QoS (CRITICAL/HIGH/MEDIUM/LOW)")
        print("  del <carID>                       - Delete car")
        print("  move <carID> <goalLM> [-v] [qos]  - Move car, optionally set QoS")
        print("  setqos <carID> <qos>              - Change car QoS level")
        print("  interval <ms>                     - Set animation interval")
        print("  help")
        print("  exit")

        while self.running:
            try:
                cmd_input = input(">> ").strip()
                if not cmd_input:
                    continue

                parts = cmd_input.split()
                cmd = parts[0].lower()

                if cmd == "add" and len(parts) >= 3:
                    car_id = parts[1]
                    landmark = parts[2]
                    qos = parts[3] if len(parts) > 3 else 'MEDIUM'
                    self.command_queue.put(("add", (car_id, landmark, qos)))
                elif cmd == "del" and len(parts) == 2:
                    car_id = parts[1]
                    self.command_queue.put(("del", (car_id,)))
                elif cmd == "move" and len(parts) >= 3:
                    car_id = parts[1]
                    goal = parts[2]
                    verbose = "-v" in parts
                    qos_override = None
                    for p in parts:
                        if p.upper() in QoS_LEVELS and p.upper() not in ['ADD', 'DEL', 'MOVE', 'SETQOS', 'INTERVAL', 'HELP', 'EXIT']:
                            if p.upper() != 'MEDIUM' or parts.index(p) != 0:
                                qos_override = p.upper()
                                break
                    self.command_queue.put(("move", (car_id, goal, verbose, qos_override)))
                elif cmd == "setqos" and len(parts) == 3:
                    car_id = parts[1]
                    qos = parts[2].upper()
                    if qos in QoS_LEVELS:
                        self.command_queue.put(("setqos", (car_id, qos)))
                    else:
                        print(f"Invalid QoS level. Use: CRITICAL, HIGH, MEDIUM, LOW")
                elif cmd == "interval" and len(parts) == 2:
                    try:
                        ms = int(parts[1])
                        self.command_queue.put(("set_interval", (ms,)))
                    except ValueError:
                        print("Invalid interval value.")
                elif cmd == "exit":
                    self.running = False
                    if plt is not None:
                        plt.close('all')
                    break
                elif cmd == "help":
                    print("Examples:")
                    print("  add car01 LM001 HIGH")
                    print("  add car02 LM002 CRITICAL")
                    print("  del car01")
                    print("  move car01 LM005")
                    print("  move car01 LM005 -v")
                    print("  move car01 LM005 HIGH")
                    print("  move car01 LM005 -v CRITICAL")
                    print("  setqos car01 HIGH")
                else:
                    print("Unknown command or wrong arguments.")
            except EOFError:
                break
            except Exception as e:
                print(f"CLI Error: {e}")

    def visualize(self):
        ensure_matplotlib()
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        x_max = self.x_min + self.width
        y_max = self.y_min + self.height

        ax.set_xlim(self.x_min - 1, x_max)
        ax.set_ylim(self.y_min - 1, y_max)

        if self.width <= 50:
            ax.set_xticks(range(self.x_min, x_max))
            ax.set_yticks(range(self.y_min, y_max))
            ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        else:
            step = max(1, self.width // 10)
            ax.set_xticks(range(self.x_min, x_max, step))
            ax.set_yticks(range(self.y_min, y_max, step))
            ax.grid(False)

        ax.set_aspect('equal')

        ax.imshow(self.grid, extent=[self.x_min - 0.5, x_max - 0.5, self.y_min - 0.5, y_max - 0.5],
                  origin='lower', cmap='Greys', alpha=0.3, interpolation='nearest', zorder=1)

        for lm in self.landmarks:
            ax.plot(lm["x"], lm["y"], 'o', color='#0066cc', markersize=3, alpha=0.6, zorder=3)
            ax.text(lm["x"], lm["y"]+0.25, lm["name"], fontsize=7, ha='center', color='#0066cc', alpha=0.8, zorder=3)

        car_artists = {}
        mission_markers = {}

        def update(frame):
            try:
                while not self.command_queue.empty():
                    action, args = self.command_queue.get_nowait()
                    if action == "add":
                        res = self.add(*args)
                        print(f"\033[92m[System]\033[0m {res}")
                    elif action == "del":
                        car_id = args[0]
                        res = self.del_car(car_id)
                        if car_id in car_artists:
                            arts = car_artists[car_id]
                            arts['scatter'].remove()
                            arts['text'].remove()
                            arts['trail'].remove()
                            del car_artists[car_id]
                        if car_id in mission_markers:
                            for mm in mission_markers[car_id]:
                                mm.remove()
                            del mission_markers[car_id]
                        print(f"\033[92m[System]\033[0m {res}")
                    elif action == "move":
                        res = self.move_car(*args)
                        print(f"\033[92m[System]\033[0m {res}")
                    elif action == "setqos":
                        car_id, new_qos = args
                        if car_id in self.cars:
                            old_qos = self.cars[car_id]['qos_name']
                            self.cars[car_id]['qos'] = QoS_LEVELS.get(new_qos, self.cars[car_id]['qos'])
                            self.cars[car_id]['qos_name'] = new_qos
                            print(f"\033[92m[System]\033[0m Car {car_id} QoS changed: {old_qos} -> {new_qos}")
                        else:
                            print(f"\033[91m[Error]\033[0m Car {car_id} not found")
                    elif action == "set_interval":
                        new_ms = args[0]
                        ani.event_source.interval = new_ms
                        self.interval = new_ms
                        print(f"\033[92m[System]\033[0m Interval set to {new_ms}ms")
            except queue.Empty:
                pass

            self.process_pending_replans()

            self.world_time += 1
            ax.set_title(f"AGV World QoS | Time: {self.world_time}s | Map: {self.width}x{self.height}",
                         color='black', fontsize=12, pad=20)

            for car_id, car_data in self.cars.items():
                arrived_this_step = False
                if self.world_time == car_data['last_time']:
                    arrived_this_step = True
                    goal_str = ""
                    if car_data.get('missions'):
                        goal_str = f" at {car_data['missions'][-1][1]}"
                    qos_str = car_data.get('qos_name', 'MEDIUM')
                    print(f"\033[92m[System]\033[0m Car {car_id} Arrived{goal_str}. QoS: {qos_str}")
                    car_data['verbose'] = False

                    if car_data['qos'] < QoS_LEVELS['HIGH']:
                        car_data['qos'] = QoS_LEVELS['MEDIUM']
                        car_data['qos_name'] = 'MEDIUM'
                        print(f"\033[92m[System]\033[0m Car {car_id} QoS restored to MEDIUM")

                pos = None
                direction = 'x+'
                found = False

                traj = car_data['trajectory']
                if not traj: continue

                path_pts = []
                for p in traj:
                    if p[0] <= self.world_time:
                         path_pts.append((p[1], p[2]))

                if self.world_time < traj[0][0]:
                    pos = (traj[0][1], traj[0][2])
                    direction = traj[0][3] if len(traj[0]) > 3 else 'x+'
                    found = False
                elif self.world_time >= traj[-1][0]:
                    pos = (traj[-1][1], traj[-1][2])
                    direction = traj[-1][3] if len(traj[-1]) > 3 else 'x+'
                    found = True
                else:
                    for item in traj:
                        t = item[0]
                        if t == self.world_time:
                            pos = (item[1], item[2])
                            direction = item[3] if len(item) > 3 else 'x+'
                            found = True
                            if car_data.get('verbose'):
                                qos_str = car_data.get('qos_name', 'MEDIUM')
                                print(f"\033[92m[System]\033[0m Car {car_id} at ({pos[0]}, {pos[1]}), Dir: {direction}, QoS: {qos_str}")
                            break

                if pos:
                    marker_sym = DIR_MARKERS.get(direction, r'$\rightarrow$')

                    if car_id not in car_artists:
                        scat = ax.scatter([pos[0]], [pos[1]], s=400, color=car_data['color'], marker=marker_sym,
                                         linewidths=1.5, zorder=20)
                        txt = ax.text(pos[0], pos[1], car_id, fontsize=8, fontweight='bold',
                                      ha='center', va='center', color='black', zorder=21)
                        trail, = ax.plot([], [], color=car_data['color'], alpha=0.3, linewidth=1, zorder=10)
                        car_artists[car_id] = {'scatter': scat, 'text': txt, 'trail': trail, 'marker': marker_sym}
                    else:
                        arts = car_artists[car_id]

                        if arts.get('marker') != marker_sym:
                            arts['scatter'].remove()
                            new_scat = ax.scatter([pos[0]], [pos[1]], s=400, color=car_data['color'], marker=marker_sym,
                                         linewidths=1.5, zorder=20)
                            arts['scatter'] = new_scat
                            arts['marker'] = marker_sym
                        else:
                            arts['scatter'].set_offsets([[pos[0], pos[1]]])

                        arts['text'].set_position((pos[0], pos[1]))
                        if path_pts:
                            px, py = zip(*path_pts)
                            arts['trail'].set_data(px, py)

                        if self.world_time < traj[0][0]:
                            arts['scatter'].set_alpha(0.2)
                            arts['text'].set_alpha(0.2)
                        else:
                            arts['scatter'].set_alpha(1.0)
                            arts['text'].set_alpha(1.0)

                if car_id not in mission_markers:
                    mission_markers[car_id] = []
                for m in mission_markers[car_id]:
                    m.remove()
                mission_markers[car_id] = []

                if car_data['missions']:
                    s_nm, g_nm, s_c, g_c = car_data['missions'][-1]
                    m1 = ax.scatter(s_c[0], s_c[1], s=100, facecolors='none',
                                   edgecolors=car_data['color'], marker='o', alpha=0.6, linestyle='--')
                    m2 = ax.scatter(g_c[0], g_c[1], s=120, color=car_data['color'],
                                   marker='X', alpha=0.8, edgecolors='black', linewidths=0.5)
                    mission_markers[car_id].extend([m1, m2])

            return []

        ani = FuncAnimation(fig, update, interval=self.interval, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
        self.running = False

def main():
    parser = argparse.ArgumentParser(description="AGV World Simulation with Dynamic QoS")
    parser.add_argument("map_file", help="The map JSON file")
    parser.add_argument("--interval", type=int, default=200, help="Animation interval in ms (default: 200)")
    args = parser.parse_args()

    if not os.path.exists(args.map_file):
        print(f"Map file {args.map_file} not found.")
        return

    world = AGVWorld(args.map_file)
    world.interval = args.interval

    cli_thread = threading.Thread(target=world.run_cli, daemon=True)
    cli_thread.start()

    world.visualize()

if __name__ == "__main__":
    main()
