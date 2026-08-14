"""
Auto-tune ARC parameters by minimizing a scalar cost over headless runs.

Each candidate is a full headless run of run_factory_full.py with --arc
overrides; the sim prints "COST: <x>" (see arc_cost() there).  Uses
coordinate descent: for each parameter try a few candidate values, keep the
best, iterate until no improvement or max rounds.

Run with the mujoco python, e.g.:
  D:/download/anaconda3/envs/tutorial_for_mujoco/python.exe navigation/tune_arc.py

Deterministic: the policy + plan are fixed, so a given param set yields the
same cost every time (no repeated trials needed).
"""

import subprocess
import sys
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SCRIPT = os.path.join(ROOT, "navigation", "run_factory_full.py")

DEFAULTS = dict(vy_dodge=0.5, vx_yield=0.3, lat_thresh=0.6, lookahead=2.0, hold=2.0)

# (param, [candidate values]) — tried in coordinate descent
SEARCH = {
    "vy_dodge":   [0.3, 0.4, 0.5, 0.6, 0.7],
    "vx_yield":   [0.2, 0.25, 0.3, 0.35, 0.4],
    "hold":       [1.0, 1.5, 2.0, 2.5, 3.0],
    "lat_thresh": [0.45, 0.5, 0.6, 0.7],
    "lookahead":  [1.5, 2.0, 2.5],
}


def run(params) -> tuple[float, str]:
    """Run one headless sim with the given ARC params, return (cost, stdout)."""
    arc = ",".join(f"{k}={v}" for k, v in params.items())
    cmd = [PY, SCRIPT, "--headless", "--arc", arc]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         errors="replace").stdout
    m = re.search(r"COST\s*:\s*([\d.]+)", out)
    cost = float(m.group(1)) if m else float("inf")
    return cost, out


def main(max_rounds: int = 3):
    best = dict(DEFAULTS)
    cost, out = run(best)
    print(f"[baseline]  cost={cost:8.2f}  params={best}")
    improved = True
    rnd = 0
    while improved and rnd < max_rounds:
        improved = False
        rnd += 1
        print(f"\n--- round {rnd} ---")
        for param in SEARCH:
            for v in SEARCH[param]:
                if abs(best[param] - v) < 1e-9:
                    continue
                cand = dict(best)
                cand[param] = v
                c, _ = run(cand)
                flag = "  [new best]" if c < cost else ""
                print(f"  {param}={v:>5}: cost={c:8.2f}{flag}", flush=True)
                if c < cost:
                    cost, best = c, cand
                    improved = True

    # Final confirmation run
    cost, out = run(best)
    print("\n" + "=" * 56)
    print(f"BEST cost={cost:.2f}  params={best}")
    print("=" * 56)
    # Print the metrics behind the best cost
    for line in out.splitlines():
        if any(k in line for k in ("triggers", "collisions", "falls",
                                   "min AGV", "arrived", "COST")):
            print("  " + line.strip())
    print("\nTo use these: python navigation/run_factory_full.py --arc "
          + ",".join(f"{k}={v}" for k, v in best.items()))


if __name__ == "__main__":
    rounds = 3
    if "--rounds" in sys.argv:
        rounds = int(sys.argv[sys.argv.index("--rounds") + 1])
    main(max_rounds=rounds)
