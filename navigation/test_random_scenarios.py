"""
Robustness test: run the ARC yield version on RANDOM scenarios.

Each scenario:
  - random AGV chains (4 cars, random landmark chains) -> new plan_v2_*.json
  - random robot route: start LM005 -> random via -> random goal
    (robot teleports to LM005 in the sim)
  - the robot path is re-planned against the new AGV plan
Then the full sim is run headless and key metrics are parsed.

Usage:
  D:/download/anaconda3/envs/tutorial_for_mujoco/python.exe navigation/test_random_scenarios.py [N]
"""

import json
import os
import sys
import glob
import re
import random
import subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agv_simulation-main", "src"))

MAPS_DIR = os.path.join(ROOT, "agv_simulation-main", "maps")
map_file = sorted(glob.glob(os.path.join(MAPS_DIR, "map_*.json")))[0]

from agv_planner_v2 import plan_paths as agv_plan          # noqa: E402
from navigation import navigate_factory as nf              # noqa: E402

# Landmarks the robot may use as via/goal (LM005 is the fixed start)
GOALS = ["LM000", "LM001", "LM002", "LM003", "LM004", "LM006"]
# AGVs must NOT start at LM005 — the robot teleports there, so a car spawned
# on that cell overlaps the robot from t=0 and every contact frame counts as
# a false "collision" (scenario seed 0 had car1 start at (1,2) → 98 hits).
AGV_LMS = ["LM000", "LM001", "LM002", "LM003", "LM004", "LM006"]
START = "LM005"
START_XY = (1.125, 2.125)   # matches the sim teleport to (1,2)


def gen_agv_plan(seed):
    random.seed(seed)
    for _try in range(5):
        tasks = [random.sample(AGV_LMS, 4) for _ in range(4)]
        out = agv_plan(map_file, num_cars=0, tasks=tasks, stagger=2)
        if sum(1 for line in open(out) if '"car_id"' in line) >= 4:
            return out
        # one car failed to plan — retry with a fresh random draw
    return out


def gen_robot_route(seed):
    random.seed(seed + 1000)
    return tuple(random.sample(GOALS, 2))   # (via, goal)


def plan_robot_path(via, goal, agv_cars):
    factory = nf.load_factory_map(map_file)
    via_xy, goal_xy = factory["landmarks"][via], factory["landmarks"][goal]
    legs = []
    for s, g in [(START_XY, via_xy), (via_xy, goal_xy)]:
        leg = nf.plan_spatial_path((float(s[0]), float(s[1])),
                                   (float(g[0]), float(g[1])),
                                   factory["obstacles"], agv_cars,
                                   factory["width"], factory["height"],
                                   obstacle_margin=3, agv_block=True)
        if not leg:
            return None
        legs.append(leg)
    path = legs[0] + legs[1][1:]
    # correct_path pushes waypoints out of collision zones and re-smooths;
    # its Catmull-Rom pass is now also collision-checked (it used to dip
    # waypoints INSIDE a shelf box, wedging the robot).
    path_t = [(i, x, y) for i, (x, y) in enumerate(path)]
    path_c = nf.correct_path(path_t)
    return simplify_path([(x, y) for _, x, y in path_c], tol=0.1)


def simplify_path(path, tol=0.3):
    """Douglas-Peucker — collapse a dense fine-grid path to its corners."""
    if len(path) < 3:
        return path
    pts = np.array(path, dtype=float)

    def _dist(a, b, c):
        ab = b - a
        return abs(ab[0] * (c[1] - a[1]) - ab[1] * (c[0] - a[0])) / np.hypot(*ab)

    def _rec(lo, hi):
        dmax, idx = 0.0, -1
        for i in range(lo + 1, hi):
            d = _dist(pts[lo], pts[hi], pts[i])
            if d > dmax:
                dmax, idx = d, i
        if dmax > tol:
            return _rec(lo, idx)[:-1] + _rec(idx, hi)
        return [tuple(map(float, pts[lo])), tuple(map(float, pts[hi]))]

    return _rec(0, len(pts) - 1)


def write_robot_plan(path, via, goal):
    hplan = {"waypoints": [{"x": float(x), "y": float(y)} for x, y in path],
             "start": START, "via": via, "goal": goal}
    with open(os.path.join(MAPS_DIR, "humanoid_plan.json"), "w") as f:
        json.dump(hplan, f)


def run_headless():
    cmd = [sys.executable, os.path.join(ROOT, "navigation", "run_factory_full.py"),
           "--headless"]
    out = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout

    def _g(k):
        m = re.search(rf"{re.escape(k)}\s*:\s*([\d.]+)", out)
        return float(m.group(1)) if m else float("nan")

    return {
        "triggers": _g("ARC triggers"),
        "collisions": _g("collisions"),
        "shelf": _g("shelf_coll"),
        "falls": _g("falls"),
        "min_agv": _g("min AGV dist"),
        "cost": _g("COST"),
        "arrived": "arrived      : True" in out,
    }


def main(N=5):
    print(f"{'#':>2}  {'route':>22}  {'wp':>3}  {'trig':>4}  {'agv_col':>7}  "
          f"{'shelf':>5}  {'falls':>5}  {'min_d':>6}  {'time':>5}  {'cost':>9}  ok")
    ok = 0
    for i in range(N):
        gen_agv_plan(i)
        via, goal = gen_robot_route(i)
        plan_file = sorted(glob.glob(os.path.join(MAPS_DIR, "plan_v2_*.json")))[-1]
        agv_cars = nf.load_agv_plan(plan_file)
        path = plan_robot_path(via, goal, agv_cars)
        if path is None:
            print(f"{i:>2}  LM005->{via}->{goal}  ROBOT PATH PLAN FAILED")
            continue
        write_robot_plan(path, via, goal)
        r = run_headless()
        time_s = (r["cost"] - 300 * r["shelf"]) / 3.0 if r["cost"] == r["cost"] else float("nan")
        flag = "OK" if (r["arrived"] and r["collisions"] == 0 and r["falls"] == 0) else "FAIL"
        if flag == "OK":
            ok += 1
        print(f"{i:>2}  LM005->{via}->{goal}  {len(path):>3}  {r['triggers']:>4.0f}  "
              f"{r['collisions']:>7.0f}  {r['shelf']:>5.0f}  {r['falls']:>5.0f}  "
              f"{r['min_agv']:>6.2f}  {time_s:>5.0f}  {r['cost']:>9.0f}  {flag}")
    print(f"\n{ok}/{N} scenarios passed (arrived, 0 AGV collisions, 0 falls)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n)
