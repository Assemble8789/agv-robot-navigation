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

def get_direction_label(dx, dy, current_dir):
    if dx > 0: return 'x+'
    if dx < 0: return 'x-'
    if dy > 0: return 'y+'
    if dy < 0: return 'y-'
    return current_dir


# --- Inherited from agv_planner.py ---

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
    # Check visited states at specific times to avoid cycles/redundancy?
    # Actually, minimal time to reach (x, y, dir) is useful, but checks with reservations mean time is state.
    # We use closed_set logic implicitly via `came_from` keys? No, (x,y,d,t) is unique node.
    # But we need to prune.
    visited = set()

    x_max, y_max = x_min + width, y_min + height

    # Search limit
    max_iter = 100000
    iterations = 0

    best_cost_to_state = {} # (x, y, dir) -> min_time

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

        # Optimization: if we reached this spatial state earlier with same dir, usually better.
        # But dynamic obstacles mean arriving later might be necessary. 
        # So we can't strict prune based on (x,y,dir) unless we know future is free?
        # Safe to just limit max_iter.

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
            # We assume turn cost is handled by the steps taken.
            h = heuristic((nx, ny), goal)
            new_priority = nt + h
            
            heapq.heappush(open_set, (new_priority, nt, nx, ny, nd))
            
            new_node_key = (nx, ny, nd, nt)
            if new_node_key not in came_from:
                came_from[new_node_key] = (cx, cy, cdir, t)

    return None

# --- AGV World Class ---

class AGVWorld:
    def __init__(self, map_file):
        with open(map_file, "r") as f:
            self.map_data = json.load(f)
        
        self.width = self.map_data["width"]
        self.height = self.map_data["height"]
        
        # Handle both list and dict formats for obstacles
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
        
        # Centered map support
        self.x_min, self.y_min = (-self.width // 2, -self.height // 2) if has_negative else (0, 0)
        
        self.lm_dict = {lm["name"]: (lm["x"], lm["y"]) for lm in self.landmarks}
        
        # 转换为 numpy 矩阵进行极速渲染
        self.grid = np.zeros((self.height, self.width))
        for ox, oy in self.obstacles:
            iy, ix = int(oy - self.y_min), int(ox - self.x_min)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                self.grid[iy, ix] = 1

        
        self.cars = {} # carID -> {color, trajectory, last_time, last_pos}
        self.reservations = set()
        self.car_reservations = {} # carID -> set of reservation tuples
        self.command_queue = queue.Queue()
        self.world_time = 0
        self.running = True
        
        # Colors for cars
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
        
        # Determine shortest turn direction
        # left diff = (target - current) % 4
        # right diff = (current - target) % 4
        
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

    def find_goal_conflict(self, goal_coord, start_t, car_id):
        for other_car_id, other_car in self.cars.items():
            if other_car_id == car_id:
                continue
            if other_car.get('last_pos') == goal_coord and other_car.get('last_time', -1) <= start_t:
                return other_car_id, other_car.get('last_time', start_t)
            missions = other_car.get('missions', [])
            if missions and missions[-1][3] == goal_coord and other_car.get('last_time', -1) >= start_t:
                return other_car_id, other_car.get('last_time', start_t)
        return None


    def add(self, car_id, landmark_name, qos=None):
        if car_id in self.cars:
            coord = self.lm_dict.get(landmark_name)
            if coord and self.cars[car_id]['last_pos'] == coord:
                return f"Car {car_id} already exists at {landmark_name}."
            return f"Error: Car {car_id} already exists."
        
        coord = self.lm_dict.get(landmark_name)
        if not coord:
            return f"Error: Landmark {landmark_name} not found."

        color = self.get_next_color()
        self.cars[car_id] = {
            'color': color,
            'color': color,
            'trajectory': [(self.world_time, coord[0], coord[1], 'x+')],
            'last_time': self.world_time,
            'last_pos': coord,
            'missions': [], # List of (start_name, goal_name, start_coord, goal_coord)
            'verbose': False
        }
        self.car_reservations[car_id] = set()

        # Reserve the spot to prevent others from crashing into a parked car.
        # For simplicity, let's reserve for next 1000 steps
        for t_park in range(self.world_time, self.world_time + 1000):
            res = (coord[0], coord[1], t_park)
            self.reservations.add(res)
            self.car_reservations[car_id].add(res)
        
        return f"Car {car_id} added at {landmark_name}."

    def del_car(self, car_id):
        if car_id not in self.cars:
            return f"Error: Car {car_id} not found."
        
        # Remove reservations
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
        
        # Clear future reservations for this car to avoid self-collision during planning
        # (e.g. blocking its own turns because of previous parking reservation)
        if car_id in self.car_reservations:
            # Identify future reservations
            to_remove = []
            for res in self.car_reservations[car_id]:
                # res is (x, y, t) or (x, y, t, x2, y2)
                # Time is always at index 2 for vertex, index 2 for edge?
                # Vertex: (x, y, t) -> t is index 2
                # Edge: (x1, y1, t, x2, y2) -> t is index 2
                if hasattr(res, '__len__') and len(res) >= 3:
                     if res[2] > start_t:
                         to_remove.append(res)
            
            # Remove them
            for res in to_remove:
                self.reservations.discard(res)
                self.car_reservations[car_id].remove(res)

        goal_conflict = self.find_goal_conflict(goal_coord, start_t, car_id)
        if goal_conflict:
            other_car_id, conflict_time = goal_conflict
            return (
                f"Error: Goal {goal_name} is already reserved by {other_car_id} "
                f"from t={conflict_time}."
            )

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
        # path is list of (t, x, y, dir)
        # Skip the first point (current state) as it's already in history/reservation?
        # No, new path starts at start_t (current). 
        # But checking 'astar' it returns start node too.
        # We append from 2nd point onwards because current point already in traj?
        # Actually `start_t` is `self.world_time`. The car is already at `start_coord` at `start_t`.
        # So we should append points from `start_t + 1`.

        for i in range(1, len(path)):
            t, x, y, d = path[i]
            res_v = (x, y, t)
            self.reservations.add(res_v)
            self.car_reservations[car_id].add(res_v)
            
            # Check for move vs wait/turn for edge reservation
            prev_t, prev_x, prev_y, prev_d = path[i-1]
            if (x != prev_x or y != prev_y):
                # It was a move
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

    def run_cli(self):
        print("AGV World CLI started.")
        print("Commands:")
        print("  add <carID> <landmark>")
        print("  del <carID>")
        print("  move <carID> <goalLM>")
        print("  interval <ms>")
        print("  help")

        print("  exit")
        
        while self.running:
            try:
                cmd_input = input(">> ").strip()
                if not cmd_input:
                    continue
                
                parts = cmd_input.split()
                cmd = parts[0].lower()
                
                if cmd == "add" and len(parts) == 3:
                    car_id = parts[1]
                    landmark = parts[2]
                    self.command_queue.put(("add", (car_id, landmark)))
                elif cmd == "del" and len(parts) == 2:
                    car_id = parts[1]
                    self.command_queue.put(("del", (car_id,)))
                elif cmd == "move" and len(parts) >= 3:
                    car_id = parts[1]
                    goal = parts[2]
                    verbose = "-v" in parts
                    self.command_queue.put(("move", (car_id, goal, verbose)))
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
                    print("Example:")
                    print("  add car01 LM001")
                    print("  del car01")
                    print("  move car01 LM005")
                    print("  move car01 LM005 -v")
                else:
                    print("Unknown command or wrong arguments.")
            except EOFError:
                break
            except Exception as e:
                print(f"CLI Error: {e}")

    def visualize(self):
        ensure_matplotlib()
        # Changed to default for white background
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
            # 大地图不画详细网格线，只留边框和主要刻度
            step = max(1, self.width // 10)
            ax.set_xticks(range(self.x_min, x_max, step))
            ax.set_yticks(range(self.y_min, y_max, step))
            ax.grid(False) 
            
        ax.set_aspect('equal')


        # 使用 imshow 替代成千上万个 Rectangle，性能大幅提升
        ax.imshow(self.grid, extent=[self.x_min - 0.5, x_max - 0.5, self.y_min - 0.5, y_max - 0.5], 
                  origin='lower', cmap='Greys', alpha=0.3, interpolation='nearest', zorder=1)


        # Draw landmarks - adjusted for white background
        for lm in self.landmarks:
            ax.plot(lm["x"], lm["y"], 'o', color='#0066cc', markersize=3, alpha=0.6, zorder=3)
            ax.text(lm["x"], lm["y"]+0.25, lm["name"], fontsize=7, ha='center', color='#0066cc', alpha=0.8, zorder=3)

        car_artists = {} # car_id -> {scatter, text, trail}
        mission_markers = {} # car_id -> [list of marker artists]

        def update(frame):
            # Process commands
            try:
                while not self.command_queue.empty():
                    action, args = self.command_queue.get_nowait()
                    if action == "add":
                        res = self.add(*args)
                        print(f"\033[92m[System]\033[0m {res}")
                    elif action == "del":
                        car_id = args[0]
                        res = self.del_car(car_id)
                        # Clean up artists if deleted
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
                    elif action == "set_interval":
                        new_ms = args[0]
                        ani.event_source.interval = new_ms
                        self.interval = new_ms
                        print(f"\033[92m[System]\033[0m Interval set to {new_ms}ms")
            except queue.Empty:

                pass

            self.world_time += 1
            ax.set_title(f"AGV World | Time: {self.world_time}s | Map: {self.width}x{self.height}", 
                         color='black', fontsize=12, pad=20)


            for car_id, car_data in self.cars.items():
                # Check if just arrived
                # Check if just arrived
                if self.world_time == car_data['last_time']:
                    goal_str = ""
                    if car_data.get('missions'):
                        goal_str = f" at {car_data['missions'][-1][1]}"
                    print(f"\033[92m[System]\033[0m Car {car_id} Arrived{goal_str}.")
                    car_data['verbose'] = False
                # Find position at world_time
                pos = None
                direction = 'x+' # Default
                found = False
                
                # Check trajectory
                traj = car_data['trajectory']
                if not traj: continue

                # Get path points up to world_time for trail
                path_pts = []
                for p in traj:
                    if p[0] <= self.world_time:
                         path_pts.append((p[1], p[2]))
                
                if self.world_time < traj[0][0]:
                    # Future car
                    pos = (traj[0][1], traj[0][2])
                    direction = traj[0][3] if len(traj[0]) > 3 else 'x+'
                    found = False # Don't show yet or show dimmed
                elif self.world_time >= traj[-1][0]:
                    # Finished mission, staying at end
                    pos = (traj[-1][1], traj[-1][2])
                    direction = traj[-1][3] if len(traj[-1]) > 3 else 'x+'
                    found = True
                else:
                    # Moving
                    for item in traj:
                        t = item[0]
                        if t == self.world_time:
                            pos = (item[1], item[2])
                            direction = item[3] if len(item) > 3 else 'x+'
                            found = True
                            if car_data.get('verbose'):
                                print(f"\033[92m[System]\033[0m Car {car_id} at ({pos[0]}, {pos[1]}), Dir: {direction}")
                            break
                
                if pos:
                    marker_sym = DIR_MARKERS.get(direction, r'$\rightarrow$')
                    
                    if car_id not in car_artists:
                        # Character-based marker for AGV
                        scat = ax.scatter([pos[0]], [pos[1]], s=400, color=car_data['color'], marker=marker_sym,
                                         linewidths=1.5, zorder=20)
                        txt = ax.text(pos[0], pos[1], car_id, fontsize=8, fontweight='bold', 
                                      ha='center', va='center', color='black', zorder=21)
                        # Trail line
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
                            # Restore alpha if needed (though not strictly tracked here, simplified)
                        else:
                            arts['scatter'].set_offsets([[pos[0], pos[1]]])
                        
                        arts['text'].set_position((pos[0], pos[1]))
                        if path_pts:
                            px, py = zip(*path_pts)
                            arts['trail'].set_data(px, py)
                        
                        # Set alpha based on whether it's active
                        if self.world_time < traj[0][0]:
                            arts['scatter'].set_alpha(0.2)
                            arts['text'].set_alpha(0.2)
                        else:
                            arts['scatter'].set_alpha(1.0)
                            arts['text'].set_alpha(1.0)

                # Show the latest mission's start and end
                if car_id not in mission_markers:
                    mission_markers[car_id] = []
                for m in mission_markers[car_id]:
                    m.remove()
                mission_markers[car_id] = []

                if car_data['missions']:
                    s_nm, g_nm, s_c, g_c = car_data['missions'][-1]
                    # Start: empty ring
                    m1 = ax.scatter(s_c[0], s_c[1], s=100, facecolors='none', 
                                   edgecolors=car_data['color'], marker='o', alpha=0.6, linestyle='--')
                    # Goal: X marker
                    m2 = ax.scatter(g_c[0], g_c[1], s=120, color=car_data['color'], 
                                   marker='X', alpha=0.8, edgecolors='black', linewidths=0.5)
                    mission_markers[car_id].extend([m1, m2])

            return []

        ani = FuncAnimation(fig, update, interval=self.interval, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
        self.running = False

def main():
    parser = argparse.ArgumentParser(description="AGV World Simulation")
    parser.add_argument("map_file", help="The map JSON file")
    parser.add_argument("--interval", type=int, default=200, help="Animation interval in ms (default: 200)")
    args = parser.parse_args()

    if not os.path.exists(args.map_file):
        print(f"Map file {args.map_file} not found.")
        return

    world = AGVWorld(args.map_file)
    world.interval = args.interval
    
    # Start CLI thread
    cli_thread = threading.Thread(target=world.run_cli, daemon=True)
    cli_thread.start()
    
    # Run Visualization in main thread
    world.visualize()

if __name__ == "__main__":
    main()
