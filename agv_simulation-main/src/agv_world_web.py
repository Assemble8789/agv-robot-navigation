import json
import heapq
import random
import argparse
import os
import sys
import threading
import queue
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from agv_map_common import load_normalized
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# --- Constants ---
DIR_MARKERS = {
    'x+': r'$\rightarrow$',
    'x-': r'$\leftarrow$',
    'y+': r'$\uparrow$',
    'y-': r'$\downarrow$'
}

DIR_MAP = {'x+': 0, 'y+': 1, 'x-': 2, 'y-': 3}
REV_DIR_MAP = {0: 'x+', 1: 'y+', 2: 'x-', 3: 'y-'}
# 0:x+, 1:y+, 2:x-, 3:y-
DX_DY = [(1, 0), (0, 1), (-1, 0), (0, -1)]

def get_direction_label(dx, dy, current_dir):
    if dx > 0: return 'x+'
    if dx < 0: return 'x-'
    if dy > 0: return 'y+'
    if dy < 0: return 'y-'
    return current_dir

# --- Inherited from agv_planner.py / agv_world.py ---

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_with_time(width, height, obs_set, start, goal, start_time, start_dir_str, reservations, x_min=0, y_min=0):
    """
    reservations: set of (x, y, t) or (x1, y1, t, x2, y2) for edge collisions
    Kinematic A*: State = (x, y, dir_idx)
    """
    start_dir = DIR_MAP.get(start_dir_str, 0)
    
    open_set = []
    # (priority, time, x, y, dir_idx)
    heapq.heappush(open_set, (heuristic(start, goal), start_time, start[0], start[1], start_dir))
    
    came_from = {} # (x, y, dir, t) -> (px, py, pdir, pt)
    visited = set()

    x_max, y_max = x_min + width, y_min + height

    # Search limit
    max_iter = 1000000
    iterations = 0

    while open_set and iterations < max_iter:
        iterations += 1
        priority, t, cx, cy, cdir = heapq.heappop(open_set)
        
        if (cx, cy) == goal:
            # Reconstruct
            path = []
            curr_state = (cx, cy, cdir, t)
            while curr_state in came_from:
                # Store (t, x, y, dir_label)
                x, y, d, tm = curr_state
                path.append((tm, x, y, REV_DIR_MAP[d]))
                curr_state = came_from[curr_state]
            
            # push start
            x, y, d, tm = curr_state
            path.append((tm, x, y, REV_DIR_MAP[d]))
            return list(reversed(path))

        state_key = (cx, cy, cdir, t)
        if state_key in visited:
            continue
        visited.add(state_key)

        # Possible Actions
        next_actions = []
        
        # 1. Wait
        next_actions.append(('wait', cx, cy, cdir))
        
        # 2. Turn Left
        next_actions.append(('turn', cx, cy, (cdir + 1) % 4))
        
        # 3. Turn Right
        next_actions.append(('turn', cx, cy, (cdir - 1) % 4))
        
        # 4. Move Forward
        dx, dy = DX_DY[cdir]
        nx, ny = cx + dx, cy + dy
        if x_min <= nx < x_max and y_min <= ny < y_max and (nx, ny) not in obs_set:
             next_actions.append(('move', nx, ny, cdir))

        for act_type, nx, ny, nd in next_actions:
            nt = t + 1
            
            # Check Reservations
            # Vertex
            if (nx, ny, nt) in reservations: 
                continue
            
            # Edge (only if moving)
            if act_type == 'move':
                if (nx, ny, nt, cx, cy) in reservations:
                    continue
            
            # Heuristic
            h = heuristic((nx, ny), goal)
            new_priority = nt + h
            
            heapq.heappush(open_set, (new_priority, nt, nx, ny, nd))
            
            new_node_key = (nx, ny, nd, nt)
            if new_node_key not in came_from:
                came_from[new_node_key] = (cx, cy, cdir, t)

    return None

class AGVWorldWeb:
    def __init__(self, map_file):
        self.map_data = load_normalized(map_file)
        
        self.width = self.map_data["width"]
        self.height = self.map_data["height"]
        
        self.obstacles = [tuple(o) for o in self.map_data["obstacles"]]
        self.obs_set = set(self.obstacles)
        self.landmarks = self.map_data["landmarks"]
        
        # Centered map support
        has_negative = any(ox < 0 or oy < 0 for ox, oy in self.obstacles)
        self.x_min, self.y_min = (-self.width // 2, -self.height // 2) if has_negative else (0, 0)
        
        self.lm_dict = {lm["name"]: (lm["x"], lm["y"]) for lm in self.landmarks}
        
        # 转换为 numpy 矩阵进行极速渲染
        self.grid = np.zeros((self.height, self.width))
        for ox, oy in self.obstacles:
            iy, ix = int(oy - self.y_min), int(ox - self.x_min)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                self.grid[iy, ix] = 1

        
        self.cars = {} 
        self.reservations = set()
        self.car_reservations = {} # carID -> set of reservation tuples
        self.command_queue = queue.Queue()
        self.world_time = 0
        self.running = True
        
        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.color_idx = 0

    def get_next_color(self):
        color = self.color_cycle[self.color_idx % len(self.color_cycle)]
        self.color_idx += 1
        return color

    def get_nearest_obstacle_direction(self, x, y):
        """Find the cardinal direction pointing to the nearest obstacle from (x, y)"""
        if not self.obstacles:
            return None
            
        nearest = None
        min_dist_sq = float('inf')
        
        for ox, oy in self.obstacles:
            # Squared Euclidean distance
            d = (ox - x)**2 + (oy - y)**2
            if d < min_dist_sq:
                min_dist_sq = d
                nearest = (ox, oy)
        
        if not nearest:
            return None
            
        nx, ny = nearest
        dx, dy = nx - x, ny - y
        
        if dx == 0 and dy == 0:
            return None # On top of an obstacle?

        # Prefer the dominant axis for direction
        if abs(dx) >= abs(dy):
            return 'x+' if dx > 0 else 'x-'
        else:
            return 'y+' if dy > 0 else 'y-'

    def generate_rotation_steps(self, start_t, x, y, start_dir_str, target_dir, car_id=None):
        """Generate trajectory steps to rotate from start_dir to target_dir in place"""
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
                 curr_idx = (curr_idx + 1) % 4 # Turn Left
            else:
                 curr_idx = (curr_idx - 1) % 4 # Turn Right
                 
            new_dir_str = REV_DIR_MAP[curr_idx]
            steps.append((t, x, y, new_dir_str))
        
        return steps

    def add(self, car_id, landmark_name):
        if car_id in self.cars:
            return f"Error: Car {car_id} already exists."
        
        coord = self.lm_dict.get(landmark_name)
        if not coord:
            return f"Error: Landmark {landmark_name} not found."

        color = self.get_next_color()
        self.cars[car_id] = {
            'color': color,
            'trajectory': [(self.world_time, coord[0], coord[1], 'x+')],
            'last_time': self.world_time,
            'last_pos': coord,
            'missions': [],
            'verbose': False
        }
        self.car_reservations[car_id] = set()

        # Reserve the spot
        for t_park in range(self.world_time, self.world_time + 1000):
            res = (coord[0], coord[1], t_park)
            self.reservations.add(res)
            self.car_reservations[car_id].add(res)
            
        return f"Car {car_id} added at {landmark_name}."

    def del_car(self, car_id):
        if car_id not in self.cars:
            return f"Error: Car {car_id} not found."
        
        if car_id in self.car_reservations:
            for res in self.car_reservations[car_id]:
                self.reservations.discard(res)
            del self.car_reservations[car_id]
        
        del self.cars[car_id]
        return f"Car {car_id} deleted."

    def move_car(self, car_id, goal_name, verbose=False):
        if car_id not in self.cars:
            return f"Error: Car {car_id} not found. Use 'addcar' first."
        
        car = self.cars[car_id]
        # Check if stopped
        if self.world_time < car['last_time']:
            return "busy"

        goal_coord = self.lm_dict.get(goal_name)
        if not goal_coord:
            return f"Error: Landmark {goal_name} not found."
        
        start_coord = car['last_pos']
        # Find start name for mission logging
        start_name = "Unknown"
        for name, coord in self.lm_dict.items():
            if coord == start_coord:
                start_name = name
                break

        # Plan path from current position
        start_t = self.world_time
        
        # Clear future reservations for this car
        if car_id in self.car_reservations:
            to_remove = []
            for res in self.car_reservations[car_id]:
                if hasattr(res, '__len__') and len(res) >= 3:
                     if res[2] > start_t:
                         to_remove.append(res)
            
            for res in to_remove:
                self.reservations.discard(res)
                self.car_reservations[car_id].remove(res)

        # Get current direction from last trajectory point
        start_dir = car['trajectory'][-1][3] if car['trajectory'] else 'x+'
        
        path = astar_with_time(self.width, self.height, self.obs_set, start_coord, goal_coord, start_t, start_dir, self.reservations, self.x_min, self.y_min)
        
        if not path:
            return f"Error: Car {car_id} could not find path to {goal_name} with rotation."

        # --- Automatic Rotation to Nearest Obstacle at Goal ---
        final_t, final_x, final_y, final_dir = path[-1]
        target_face_dir = self.get_nearest_obstacle_direction(final_x, final_y)
        
        if target_face_dir and target_face_dir != final_dir:
            if verbose:
                print(f"[Plan] Reached goal, rotating from {final_dir} to {target_face_dir} towards obstacle.")
            rotation_steps = self.generate_rotation_steps(final_t, final_x, final_y, final_dir, target_face_dir)
            path.extend(rotation_steps)

        # Update reservations and trajectory
        for i in range(1, len(path)):
            t, x, y, d = path[i]
            res_v = (x, y, t)
            self.reservations.add(res_v)
            self.car_reservations[car_id].add(res_v)
            
            # Check for move vs wait/turn for edge reservation
            prev_t, prev_x, prev_y, prev_d = path[i-1]
            if (x != prev_x or y != prev_y):
                res_e = (prev_x, prev_y, t, x, y)
                self.reservations.add(res_e)
                self.car_reservations[car_id].add(res_e)
            
            car['trajectory'].append((t, x, y, d))
        
        # Reserve goal spot
        for t_park in range(path[-1][0] + 1, path[-1][0] + 1000):
            res_p = (goal_coord[0], goal_coord[1], t_park)
            self.reservations.add(res_p)
            self.car_reservations[car_id].add(res_p)

        car['last_time'] = path[-1][0]
        car['last_pos'] = goal_coord
        car['missions'].append((start_name, goal_name, start_coord, goal_coord))
        car['verbose'] = verbose
        
        return f"Car {car_id} moving from {start_name} to {goal_name}."

    def visualize(self):
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        x_max = self.x_min + self.width
        y_max = self.y_min + self.height
        
        ax.set_xlim(self.x_min - 1, x_max)
        ax.set_ylim(self.y_min - 1, y_max)
        
        # 动态调整网格密度，减少渲染负担
        if self.width <= 50:
            ax.set_xticks(range(self.x_min, x_max))
            ax.set_yticks(range(self.y_min, y_max))
            ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        else:
            # 大地图不画详细网格线，只留边框 and 主要刻度
            step = max(1, self.width // 10)
            ax.set_xticks(range(self.x_min, x_max, step))
            ax.set_yticks(range(self.y_min, y_max, step))
            ax.grid(False) 
            
        ax.set_aspect('equal')


        # 使用 imshow 替代成千上万个 Rectangle，性能大幅提升
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
                    action, args, resp_event, resp_data = self.command_queue.get_nowait()
                    res = ""
                    if action == "add":
                        res = self.add(*args)
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
                    elif action == "move":
                        res = self.move_car(*args)
                    elif action == "set_interval":
                        new_ms = args[0]
                        ani.event_source.interval = new_ms
                        self.interval = new_ms
                        res = f"Interval set to {new_ms}ms"
                    print(f"\033[92m[System API]\033[0m {res}")

                    resp_data['message'] = res
                    resp_event.set()
            except queue.Empty:
                pass

            self.world_time += 1
            ax.set_title(f"AGV World (Web API) | Time: {self.world_time}s | Map: {self.width}x{self.height}", 
                         color='black', fontsize=12, pad=20)

            for car_id, car_data in self.cars.items():
                if self.world_time == car_data['last_time']:
                    print(f"\033[92m[System API]\033[0m Car {car_id} Arrived.")
                    car_data['verbose'] = False
                pos = None
                direction = 'x+' # Default
                
                traj = car_data['trajectory']
                if not traj: continue
                path_pts = [(p[1], p[2]) for p in traj if p[0] <= self.world_time]
                
                if self.world_time < traj[0][0]:
                    pos = (traj[0][1], traj[0][2])
                    direction = traj[0][3] if len(traj[0]) > 3 else 'x+'
                    active = False
                elif self.world_time >= traj[-1][0]:
                    pos = (traj[-1][1], traj[-1][2])
                    direction = traj[-1][3] if len(traj[-1]) > 3 else 'x+'
                    active = True
                else:
                    active = True
                    for item in traj:
                        t = item[0]
                        if t == self.world_time:
                            pos = (item[1], item[2])
                            direction = item[3] if len(item) > 3 else 'x+'
                            if car_data.get('verbose'):
                                print(f"\033[92m[System API]\033[0m Car {car_id} at ({pos[0]}, {pos[1]}), Dir: {direction}")
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
                        
                        # Check if marker needs update
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
                        alpha = 1.0 if active else 0.2
                        arts['scatter'].set_alpha(alpha)
                        arts['text'].set_alpha(alpha)

                if car_id not in mission_markers:
                    mission_markers[car_id] = []
                for m in mission_markers[car_id]:
                    m.remove()
                mission_markers[car_id] = []

                if car_data['missions']:
                    _, _, s_c, g_c = car_data['missions'][-1]
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

# --- FastAPI App ---

app = FastAPI(title="AGV World API")
world = None

class AddCarRequest(BaseModel):
    carID: str
    landmark: str

class MoveCarRequest(BaseModel):
    carID: str
    goalLM: str
    verbose: bool = False

class DelCarRequest(BaseModel):
    carID: str
    
class UpdateIntervalRequest(BaseModel):
    interval: int


@app.post("/add")
async def api_add_car(req: AddCarRequest):
    event = threading.Event()
    resp = {}
    world.command_queue.put(("add", (req.carID, req.landmark), event, resp))
    event.wait(timeout=2.0)
    if "Error" in resp.get("message", ""):
        raise HTTPException(status_code=400, detail=resp["message"])
    return {"status": "success", "message": resp.get("message")}

@app.post("/del")
async def api_del_car(req: DelCarRequest):
    event = threading.Event()
    resp = {}
    world.command_queue.put(("del", (req.carID,), event, resp))
    event.wait(timeout=2.0)
    if "Error" in resp.get("message", ""):
        raise HTTPException(status_code=400, detail=resp["message"])
    return {"status": "success", "message": resp.get("message")}

@app.post("/move")
async def api_move_car(req: MoveCarRequest):
    event = threading.Event()
    resp = {}
    world.command_queue.put(("move", (req.carID, req.goalLM, req.verbose), event, resp))
    event.wait(timeout=2.0)
    msg = resp.get("message", "")
    if "Error" in msg:
        raise HTTPException(status_code=400, detail=msg)
    if msg == "busy":
        return {"status": "busy", "message": "Car is currently moving"}
    return {"status": "success", "message": msg}

@app.post("/interval")
async def api_set_interval(req: UpdateIntervalRequest):
    event = threading.Event()
    resp = {}
    world.command_queue.put(("set_interval", (req.interval,), event, resp))
    event.wait(timeout=2.0)
    return {"status": "success", "message": resp.get("message")}


@app.get("/status")
async def api_get_status():
    cars_status = {}
    for cid, cdata in world.cars.items():
        # Find current direction
        direction = 'x+'
        traj = cdata['trajectory']
        if traj:
            if world.world_time < traj[0][0]:
                direction = traj[0][3] if len(traj[0]) > 3 else 'x+'
            elif world.world_time >= traj[-1][0]:
                direction = traj[-1][3] if len(traj[-1]) > 3 else 'x+'
            else:
                for item in traj:
                    if item[0] == world.world_time:
                        direction = item[3] if len(item) > 3 else 'x+'
                        break
        
        cars_status[cid] = {
            "last_pos": cdata["last_pos"],
            "last_time": cdata["last_time"],
            "direction": direction,
            "moving": world.world_time < cdata["last_time"]
        }
    return {
        "world_time": world.world_time,
        "cars": cars_status
    }

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8001)

def main():
    global world
    parser = argparse.ArgumentParser(description="AGV World Web API")
    parser.add_argument("map_file", help="The map JSON file")
    parser.add_argument("--interval", type=int, default=200, help="Animation interval in ms (default: 200)")
    args = parser.parse_args()

    if not os.path.exists(args.map_file):
        print(f"Map file {args.map_file} not found.")
        return

    world = AGVWorldWeb(args.map_file)
    world.interval = args.interval
    
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    print("API server running on http://0.0.0.0:8001")
    print("Endpoints:")
    print("  POST /add    {\"carID\": \"...\", \"landmark\": \"...\"}")
    print("  POST /del    {\"carID\": \"...\"}")
    print("  POST /move   {\"carID\": \"...\", \"goalLM\": \"...\"}")
    print("  POST /interval {\"interval\": 100}")
    print("  GET  /status")

    
    world.visualize()

if __name__ == "__main__":
    main()
