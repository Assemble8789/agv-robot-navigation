"""
AGV Planner v2 — fixed bidirectional edge collision detection + multi-stop chains.

Changes from agv_planner.py:
  1. Edge reservation now blocks BOTH directions (A→B also blocks B→A).
  2. A* check tests both edge directions.
  3. Start-time staggering improved: each car starts only after its start cell
     is clear of previous cars.
  4. Multi-destination chains: each car visits 3 destinations, stopping
     1 tick (1s) at each before moving on.

Usage:
  # Single stop pairs
  python src/agv_planner_v2.py maps/map_*.json --tasks "LM000,LM002;LM003,LM001;LM000,LM004;LM001,LM002"
  # Chains: each ;-separated entry is one car: start,stop1,stop2,stop3
  python src/agv_planner_v2.py maps/map_*.json --tasks "LM000,LM002,LM004,LM001;LM003,LM001,LM000,LM002;LM001,LM004,LM003,LM000;LM002,LM000,LM003,LM004"
"""

import json
import heapq
import random
import argparse
import datetime
import os


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_with_time(width, height, obs_set, start, goal, start_time,
                    reservations, x_min=0, y_min=0, dwell=0):
    """
    Time-aware A* with bidirectional edge collision detection.

    reservations: set of
      (x, y, t)         — vertex occupied at time t
      (x1, y1, t, x2, y2) — edge from A→B (also blocks B→A)

    dwell: extra ticks to stay at goal after arriving.  The dwell times
    participate in reservation checking, so a later car cannot enter the
    goal while an earlier car is still dwelling there.
    """
    open_set = [(0, 0, start_time, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    x_max, y_max = x_min + width, y_min + height

    while open_set:
        _, current_g, current_t, current = heapq.heappop(open_set)

        if current == goal:
            # Dwell feasibility: all dwell ticks must be free
            dwell_ok = all(
                (goal[0], goal[1], current_t + d) not in reservations
                for d in range(1, dwell + 1))
            if dwell_ok:
                path = []
                path_times = []
                curr = current
                curr_t = current_t
                while curr in came_from:
                    path.append(curr)
                    path_times.append(curr_t)
                    curr, _, curr_t = came_from[curr]
                path.append(start)
                path_times.append(start_time)
                path.reverse()
                path_times.reverse()
                # Append dwell ticks
                for d in range(1, dwell + 1):
                    path.append(goal)
                    path_times.append(path_times[-1] + 1)
                return [(t, pos) for t, pos in zip(path_times, path)]
            # Dwell blocked — keep searching (wait then retry)

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (x_min <= neighbor[0] < x_max and y_min <= neighbor[1] < y_max):
                continue
            if neighbor in obs_set:
                continue

            nt = current_t + 1

            # Vertex: destination occupied at arrival time?
            if (neighbor[0], neighbor[1], nt) in reservations:
                continue

            # Edge: bidirectionally blocked?
            edge_fwd = (current[0], current[1], nt, neighbor[0], neighbor[1])
            edge_rev = (neighbor[0], neighbor[1], nt, current[0], current[1])
            if edge_fwd in reservations or edge_rev in reservations:
                continue

            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = (current, g_score[current], current_t)
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], tentative_g, nt, neighbor))

    return None


def plan_paths(map_file, num_cars=0, tasks=None, stagger=2):
    """Multi-car path planning with bidirectional edge reservations."""
    with open(map_file, 'r') as f:
        map_data = json.load(f)

    width = map_data.get('width', 100)
    height = map_data.get('height', 100)
    obstacles = map_data.get('obstacles', [])
    landmarks = map_data.get('landmarks', [])

    obs_set = set()
    for obs in obstacles:
        if isinstance(obs, list) and len(obs) == 2:
            obs_set.add((obs[0], obs[1]))
        elif isinstance(obs, dict) and 'x' in obs and 'y' in obs:
            obs_set.add((obs['x'], obs['y']))

    lm_dict = {}
    for lm in landmarks:
        if 'name' in lm and 'x' in lm and 'y' in lm:
            lm_dict[lm['name']] = (lm['x'], lm['y'])

    x_min, y_min = 0, 0

    # ── Build tasks: each task is a chain of landmarks ──
    planned_chains = []  # list of [name0, name1, name2, ...] — first is start
    if tasks:
        for chain_names in tasks:
            coords = [lm_dict.get(n) for n in chain_names]
            if all(coords):
                planned_chains.append(list(chain_names))
            else:
                print(f"Warning: some landmark in {chain_names} not found.")
    elif num_cars:
        if landmarks:
            lm_names = [lm["name"] for lm in landmarks]
            for i in range(num_cars):
                chain = [random.choice(lm_names)]
                while len(chain) < 4:  # start + 3 destinations
                    g = random.choice(lm_names)
                    if g != chain[-1]:
                        chain.append(g)
                planned_chains.append(chain)
        else:
            def get_free():
                while True:
                    c = (random.randint(x_min, width - 1),
                         random.randint(y_min, height - 1))
                    if c not in obs_set:
                        return c
            for i in range(num_cars):
                chain = [get_free(), get_free(), get_free(), get_free()]
                planned_chains.append(chain)

    STOP_TICKS = 1  # dwell time at each destination (1s)

    # ── Plan sequentially, car by car, leg by leg ──
    reservations = set()
    car_paths = []

    for car_id, chain in enumerate(planned_chains):
        start_coord = lm_dict[chain[0]]
        start_time = car_id * stagger
        while (start_coord[0], start_coord[1], start_time) in reservations:
            start_time += 1

        full_traj = []  # (t, x, y) — complete multi-leg trajectory
        current = start_coord
        t = start_time
        ok = True

        for stop_name in chain[1:]:
            goal = lm_dict[stop_name]
            path = astar_with_time(width, height, obs_set, current, goal,
                                   t, reservations, x_min, y_min,
                                   dwell=STOP_TICKS)
            if not path:
                print(f"Car {car_id} {chain} failed at leg → {stop_name} (t={t}).")
                ok = False
                break

            # Register this leg's vertex + bidirectional edge reservations
            # path items are (t, (x, y)); normalize to (t, x, y)
            leg_pts = [(pt_, x, y) for pt_, (x, y) in path]
            for pt_, x, y in leg_pts:
                reservations.add((x, y, pt_))
            for i in range(1, len(leg_pts)):
                pt_, x, y = leg_pts[i]
                ppt_, px, py = leg_pts[i - 1]
                reservations.add((px, py, pt_, x, y))
                reservations.add((x, y, pt_, px, py))

            # Append leg to full trajectory (skip duplicate start point)
            if full_traj:
                full_traj.extend(leg_pts[1:])
            else:
                full_traj.extend(leg_pts)

            # Move on from this stop (path already includes dwell ticks)
            current = goal
            t = full_traj[-1][0] + 1  # resume after dwell

        if not ok:
            continue

        trajectory = [{"t": tt, "x": xx, "y": yy} for tt, xx, yy in full_traj]
        car_paths.append({
            "car_id": car_id,
            "start_landmark": chain[0],
            "goal_landmark": chain[-1],
            "stops": chain[1:],
            "trajectory": trajectory,
        })

    # ── Output ──
    result = {"map_file": map_file, "cars": car_paths}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plan_v2_{timestamp}.json"
    maps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps")
    os.makedirs(maps_dir, exist_ok=True)
    output_path = os.path.join(maps_dir, filename)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Planning complete: maps/{filename}  ({len(car_paths)}/{len(planned_chains)} cars planned)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGV Path Planner v2 (bidirectional edges)")
    parser.add_argument("map_file", help="Map JSON file")
    parser.add_argument("--cars", type=int, help="Number of cars (random)")
    parser.add_argument("--tasks", type=str,
                        help="Specific tasks, e.g. 'LM001,LM005;LM002,LM006'")
    parser.add_argument("--stagger", type=int, default=2,
                        help="Ticks between car start times (default: 2)")

    args = parser.parse_args()

    task_list = None
    if args.tasks:
        # Each ';'-entry is ONE CAR's chain: start,stop1,stop2,stop3
        task_list = []
        for chain_str in args.tasks.split(';'):
            names = [n.strip() for n in chain_str.split(',') if n.strip()]
            if names:
                task_list.append(names)

    num_cars = args.cars if args.cars else (0 if task_list else 2)

    plan_paths(args.map_file, num_cars=num_cars, tasks=task_list, stagger=args.stagger)
