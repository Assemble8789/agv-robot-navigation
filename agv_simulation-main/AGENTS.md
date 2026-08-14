# AGV 多机路径规划系统 — AGENTS.md

## Repo Facts
- **Flat Python scripts** — no package, no pyproject.toml, no venv activation needed. Dependencies installed once into `.venv/` via `pip install -r requirements.txt`.
- **Python 3.12** (venv present). All `.py` files are standalone entry points.
- **No linter/formatter/config** — no pyproject.toml, setup.cfg, .pre-commit-config.yaml.
- **`.gitignore` excludes `*.json` and `*.png`** — all map files, plan files, and image artifacts are gitignored. Data lives outside Git.

## Architecture (one sentence each)

| File | Role |
|---|---|
| `agv_map_edit.py` | GUI map editor — left-click places obstacles, Ctrl+left-click places landmarks (LM001…LM999). Generates `map_*.json` + `map_*.png`. |
| `agv_planner.py` | **Batch planner** — offline A\* with Space-Time Reservation (vertex+edge). Input: map JSON + car count or task list. Output: `plan_*.json`. |
| `agv_world.py` | **Core AGV engine** — online, interactive simulation. CLI (stdin thread) + matplotlib animation in main thread. |
| `agv_world_web.py` | **Web API wrapper** — wraps `AGVWorld` with FastAPI (port 8001). Same engine, HTTP endpoints instead of CLI. |
| `agv_world_qos.py` | **QoS-extended engine** — forks from `agv_world.py`, adds priority preemption (CRITICAL > HIGH > MEDIUM > LOW). Lower-priority cars get preempted and must replan. |
| `agv_visualizer.py` | **Standalone replay** — reads `plan_*.json`, plays back trajectories with Play/Pause/Step buttons. No world logic, pure visualization. |

## Path Planning Algorithms

Three **independent** A* implementations coexist — they are NOT imported by each other:

| File | A* variant | Key trait |
|---|---|---|
| `agv_planner.py` | Std A\* + time reservations | 8-directional movement, no heading. Pre-plans N cars sequentially (stagger start times by 2). |
| `agv_world.py` / `agv_world_web.py` | Kinematic A\* — state is `(x, y, dir_idx)` | 4-directional + turn/wait/move actions. Handles multi-car real-time conflicts. |
| `agv_world_qos.py` | Kinematic A\* + reservation_info dict | Reservation entries map `(key -> car_id, qos_level)`. Higher-QoS cars preempt lower-QoS reservations and trigger pending replans. |

**⚠️ When you modify one copy of A\*, the other two do NOT auto-update.** Changes are manual.

## Running Things

```bash
# Install deps (one-time)
pip install -r requirements.txt

# Map editor (blocks — GUI)
python agv_map_edit.py                          # new 15x10 map
python agv_map_edit.py map_20260113_153602.json # load existing

# Batch planner (generates plan_*.json)
python agv_planner.py map_20260113_153602.json --cars 5
python agv_planner.py map_20260113_153602.json --tasks "LM001,LM008;LM003,LM007"

# QoS-enabled planner test (no GUI — writes JSON to test_results/)
python test_qos_scenario.py --mode both

# Standalone replay
python agv_visualizer.py plan_20260128_152331.json

# Interactive CLI simulation (GUI + CLI in threads)
python agv_world.py map_20260113_153602.json

# Web API simulation (FastAPI on :8001 + GUI)
python agv_world_web.py map_20260113_153602.json

# Tests
python test_multi_agv.py           # needs a map file present
python test_web_api.py             # needs agv_world_web.py already running on :8001
```

### QoS Test Script Modes
```bash
python test_qos_scenario.py --mode original    # first-come-first-served
python test_qos_scenario.py --mode qos         # priority preemption
python test_qos_scenario.py --mode both        # side-by-side comparison (default)
```

## Important Constraints
- **All `.json` and `.png` files are gitignored** — generated data artifacts. Don't commit them.
- **No test framework** — tests are hand-rolled scripts. No pytest, no assertions.
- **`agv_world.py` blocks** — `plt.show()` blocks the main thread. CLI runs in daemon thread. You cannot send commands after the GUI closes.
- **`agv_world_web.py`** — `uvicorn.run()` + `plt.show()` both block. In practice the GUI blocks first, but both spin in threads.
- **Map coordinate systems** — maps with negative obstacle coords (`has_negative`) use a centered origin `(–width/2, –height/2)`. Standard maps use `(0, 0)`. The code auto-detects this from obstacle data.
- **Landmark naming** — auto-incremented `LM001`, `LM002`, … based on existing LM prefix numbers.

## Testing
- Tests require **interactive processes** — they spawn `subprocess.Popen` or use `requests` against a live server.
- `test_multi_agv.py` — launches `agv_world.py`, pipes `stdin` commands (add/move 4 cars), waits 15s.
- `test_web_api.py` — requires `agv_world_web.py` already running in another terminal before executing.
- `test_qos_scenario.py` — imports world modules via `importlib`. Runs headless simulation (no GUI). Writes JSON snapshots to `test_results/`.

## Known Issues / Gotchas
- **Three separate A\* implementations** — no shared module. Refactoring one requires manual porting to the others.
- **Empty `except Exception`** in `agv_world_web.py` `visualize()` update loop swallows all animation errors silently.
- **Three `except queue.Empty`** blocks across world modules are bare (no logging).
- **100-step parking reservation** (hardcoded `range(self.world_time, self.world_time + 1000)`) — cars block landmarks for 1000 ticks after parking, preventing other cars from claiming occupied spots.
- **`.gitignore` pattern `*.json`** covers ALL JSON files including any future project config. If a real `pyproject.json` or similar is needed, add an explicit exception.
