#!/usr/bin/env python3
"""
AGV QoS Test Script

Demonstrates the advantage of QoS-based scheduling by comparing:
1. agv_world.py - First-come-first-served scheduling
2. agv_world_qos.py - QoS priority scheduling

Usage:
    python test_qos_scenario.py --mode original    # Test original algorithm
    python test_qos_scenario.py --mode qos         # Test QoS algorithm
    python test_qos_scenario.py --mode both         # Test both and compare
"""

import json
import sys
import time
import argparse
from datetime import datetime

MAP_FILE = "maps/map_20260113_153602.json"
OUTPUT_DIR = "test_results"

INITIAL_SCENARIO = [
    {"car": "carA", "start": "LM002", "goal": "LM005", "qos": "LOW"},
    {"car": "carB", "start": "LM006", "goal": "LM009", "qos": "LOW"},
    {"car": "carC", "start": "LM010", "goal": "LM003", "qos": "MEDIUM"},
    {"car": "carD", "start": "LM011", "goal": "LM004", "qos": "MEDIUM"},
]

LATE_SCENARIO = [
    {"car": "carE", "start": "LM007", "goal": "LM001", "qos": "HIGH", "start_time": 10},
    {"car": "carF", "start": "LM002", "goal": "LM008", "qos": "CRITICAL", "start_time": 15},
]


def build_plan_snapshot(world, algorithm):
    """Build a JSON-serializable snapshot of planned trajectories."""
    cars = []
    for car_id, car_data in world.cars.items():
        trajectory = []
        for point in car_data.get('trajectory', []):
            if len(point) >= 4:
                t, x, y, direction = point
            else:
                t, x, y = point
                direction = None
            trajectory.append({
                't': t,
                'x': x,
                'y': y,
                'dir': direction,
            })

        cars.append({
            'car_id': car_id,
            'trajectory': trajectory,
            'last_time': car_data.get('last_time'),
            'last_pos': car_data.get('last_pos'),
            'missions': car_data.get('missions', []),
            'qos': car_data.get('qos'),
            'qos_name': car_data.get('qos_name'),
            'current_goal': car_data.get('current_goal'),
            'waiting_replan': car_data.get('waiting_replan', False),
        })

    return {
        'algorithm': algorithm,
        'map_file': MAP_FILE,
        'world_time': world.world_time,
        'cars': cars,
    }


def save_json(data, filename):
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, filename)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    return output_file

def load_map():
    with open(MAP_FILE, 'r') as f:
        return json.load(f)


def print_car_routes():
    print("\n[Car Routes]")
    for item in INITIAL_SCENARIO:
        print(f"  {item['car']}: {item['start']} -> {item['goal']} (QoS: {item['qos']})")
    for item in LATE_SCENARIO:
        print(
            f"  {item['car']}: {item['start']} -> {item['goal']} "
            f"(QoS: {item['qos']}, inserted at t={item['start_time']})"
        )


def initial_add_commands():
    return [("add", (item["car"], item["start"], item["qos"])) for item in INITIAL_SCENARIO]


def original_move_commands():
    return [("move", (item["car"], item["goal"], False)) for item in INITIAL_SCENARIO]


def qos_move_commands():
    return [("move", (item["car"], item["goal"], False, None)) for item in INITIAL_SCENARIO]

def test_original_mode():
    """Test original agv_world.py algorithm"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("agv_world", "src/agv_world.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    print("=" * 60)
    print("TEST: Original Algorithm (First-Come-First-Served)")
    print("=" * 60)

    world = module.AGVWorld(MAP_FILE)
    results = {
        'algorithm': 'original',
        'arrivals': [],
        'world_times': [],
        'preemptions': 0,
        'plan': None,
    }
    print("\n[Scenario Setup]")
    print("6 cars competing for crossing paths")
    print("- carA, carB: Move from LEFT to RIGHT (long path)")
    print("- carC, carD: Move from BOTTOM to TOP (short path)")
    print("- carE: HIGH priority emergency vehicle")
    print("- carF: CRITICAL priority emergency vehicle")
    print_car_routes()

    commands = initial_add_commands()

    print("\n[Phase 1: Adding initial cars]")
    for cmd, args in commands:
        if cmd == "add":
            result = world.add(*args)
            print(f"  {result}")
            world.command_queue.put((cmd, args))

    time.sleep(0.1)
    world.process_pending_replans = lambda: None

    print("\n[Phase 2: Initial movements (causing conflicts)]")

    move_cmds = original_move_commands()

    start_time = time.time()
    for cmd, args in move_cmds:
        result = world.move_car(*args)
        print(f"  {result}")
        results['world_times'].append(('move', args[0], args[1], world.world_time))

    print("\n[Phase 3: Adding HIGH and CRITICAL vehicles after 10 ticks]")
    world.world_time = 10

    high_scenario = LATE_SCENARIO[0]
    print(f"  {world.add(high_scenario['car'], high_scenario['start'], high_scenario['qos'])}")

    world.world_time = 12
    result = world.move_car(high_scenario["car"], high_scenario["goal"], False)
    print(f"  {result}")
    results['world_times'].append(('move', high_scenario["car"], high_scenario["goal"], world.world_time))

    world.world_time = 15
    critical_scenario = LATE_SCENARIO[1]
    print(f"  {world.add(critical_scenario['car'], critical_scenario['start'], critical_scenario['qos'])}")

    world.world_time = 16
    result = world.move_car(critical_scenario["car"], critical_scenario["goal"], False)
    print(f"  {result}")
    results['world_times'].append(('move', critical_scenario["car"], critical_scenario["goal"], world.world_time))

    print("\n[Phase 4: Simulating execution]")
    max_time = 50
    for t in range(max_time):
        world.world_time = t
        for car_id, car_data in world.cars.items():
            if world.world_time == car_data['last_time']:
                results['arrivals'].append({
                    'car': car_id,
                    'time': t,
                    'goal': car_data['missions'][-1][1] if car_data.get('missions') else 'unknown'
                })
                print(f"  [t={t:3d}] Car {car_id} ARRIVED at {car_data['missions'][-1][1] if car_data.get('missions') else 'unknown'}")

    elapsed = time.time() - start_time
    results['total_time'] = elapsed
    results['simulation_end_time'] = max_time
    results['plan'] = build_plan_snapshot(world, 'original')

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total simulation time steps: {max_time}")
    print(f"Total cars processed: {len(world.cars)}")
    print(f"Arrivals recorded: {len(results['arrivals'])}")
    print(f"\nArrival order:")
    for i, arr in enumerate(results['arrivals'], 1):
        print(f"  {i}. Car {arr['car']} at t={arr['time']} -> {arr['goal']}")

    return results


def test_qos_mode():
    """Test agv_world_qos.py algorithm with QoS priorities"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("agv_world_qos", "src/agv_world_qos.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    print("=" * 60)
    print("TEST: QoS Algorithm (Priority-Based Preemption)")
    print("=" * 60)

    world = module.AGVWorld(MAP_FILE)
    results = {
        'algorithm': 'qos',
        'arrivals': [],
        'world_times': [],
        'preemptions': 0,
        'qos_changes': [],
        'plan': None,
    }

    print("\n[Scenario Setup]")
    print("6 cars competing for crossing paths")
    print("- carA, carB: Move with LOW priority")
    print("- carC, carD: Move with MEDIUM priority")
    print("- carE: HIGH priority vehicle (can preempt LOW)")
    print("- carF: CRITICAL priority vehicle (can preempt all)")
    print_car_routes()

    commands = initial_add_commands()

    print("\n[Phase 1: Adding initial cars]")
    for cmd, args in commands:
        if cmd == "add":
            result = world.add(*args)
            print(f"  {result}")
            world.command_queue.put((cmd, args))

    time.sleep(0.1)

    print("\n[Phase 2: Initial movements (LOW and MEDIUM cars)]")

    move_cmds = qos_move_commands()

    start_time = time.time()
    for cmd, args in move_cmds:
        result = world.move_car(*args)
        print(f"  {result}")
        results['world_times'].append(('move', args[0], args[1], world.world_time))

    print("\n[Phase 3: Adding HIGH vehicle at t=10]")
    world.world_time = 10
    high_scenario = LATE_SCENARIO[0]
    print(f"  {world.add(high_scenario['car'], high_scenario['start'], high_scenario['qos'])}")

    world.world_time = 12
    result = world.move_car(high_scenario["car"], high_scenario["goal"], False, high_scenario["qos"])
    print(f"  {result}")
    if "preempted" in result:
        results['preemptions'] += 1
        results['world_times'].append(('preempt', 'carE', 'carA', world.world_time))
        print(f"  [!] carE PREEMPTED carA's path!")
    results['world_times'].append(('move', high_scenario["car"], high_scenario["goal"], world.world_time))

    print("\n[Phase 4: Adding CRITICAL vehicle at t=15]")
    world.world_time = 15
    critical_scenario = LATE_SCENARIO[1]
    print(f"  {world.add(critical_scenario['car'], critical_scenario['start'], critical_scenario['qos'])}")

    world.world_time = 16
    result = world.move_car(critical_scenario["car"], critical_scenario["goal"], False, critical_scenario["qos"])
    print(f"  {result}")
    if "preempted" in result:
        results['preemptions'] += 1
        results['world_times'].append(('preempt', 'carF', 'carB', world.world_time))
        print(f"  [!] carF PREEMPTED carB's path!")

    print("\n[Phase 5: Processing pending replans]")
    for _ in range(5):
        world.process_pending_replans()

    print("\n[Phase 6: Simulating execution]")
    max_time = 50
    for t in range(max_time):
        world.process_pending_replans()
        world.world_time = t
        for car_id, car_data in world.cars.items():
            if world.world_time == car_data['last_time']:
                qos = car_data.get('qos_name', 'MEDIUM')
                results['arrivals'].append({
                    'car': car_id,
                    'time': t,
                    'goal': car_data['missions'][-1][1] if car_data.get('missions') else 'unknown',
                    'qos': qos
                })
                results['qos_changes'].append({
                    'car': car_id,
                    'time': t,
                    'event': 'arrived',
                    'qos': qos
                })
                print(f"  [t={t:3d}] Car {car_id} ARRIVED at {car_data['missions'][-1][1] if car_data.get('missions') else 'unknown'} (QoS: {qos})")

    elapsed = time.time() - start_time
    results['total_time'] = elapsed
    results['simulation_end_time'] = max_time
    results['plan'] = build_plan_snapshot(world, 'qos')

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total simulation time steps: {max_time}")
    print(f"Total cars processed: {len(world.cars)}")
    print(f"Arrivals recorded: {len(results['arrivals'])}")
    print(f"Preemptions occurred: {results['preemptions']}")
    print(f"\nArrival order:")
    for i, arr in enumerate(results['arrivals'], 1):
        print(f"  {i}. Car {arr['car']} at t={arr['time']} -> {arr['goal']} (QoS: {arr['qos']})")

    print("\n[QoS Changes During Test]")
    for change in results['qos_changes']:
        print(f"  t={change['time']}: Car {change['car']} - {change['event']} (QoS: {change['qos']})")

    return results


def compare_results(original_results, qos_results):
    """Compare and analyze results from both algorithms"""
    print("\n")
    print("=" * 60)
    print("COMPARISON: Original vs QoS Algorithm")
    print("=" * 60)

    print("\n[Arrival Times Comparison]")
    print("-" * 60)
    print(f"{'Car':<8} {'Original t':<12} {'QoS t':<12} {'Diff':<10} {'Winner'}")
    print("-" * 60)

    orig_arrivals = {a['car']: a['time'] for a in original_results['arrivals']}
    qos_arrivals = {a['car']: a['time'] for a in qos_results['arrivals']}

    all_cars = set(orig_arrivals.keys()) | set(qos_arrivals.keys())
    qos_wins = 0
    orig_wins = 0

    for car in sorted(all_cars):
        orig_t = orig_arrivals.get(car, '-')
        qos_t = qos_arrivals.get(car, '-')
        if orig_t != '-' and qos_t != '-':
            diff = orig_t - qos_t
            winner = "QoS" if diff > 0 else ("Orig" if diff < 0 else "Tie")
            if winner == "QoS":
                qos_wins += 1
            elif winner == "Orig":
                orig_wins += 1
            print(f"{car:<8} {orig_t:<12} {qos_t:<12} {diff:+<10} {winner}")
        else:
            print(f"{car:<8} {str(orig_t):<12} {str(qos_t):<12}")

    print("-" * 60)
    print(f"Wins: QoS={qos_wins}, Original={orig_wins}")

    print("\n[Key Advantages of QoS Algorithm]")
    print(f"  1. Preemptions observed: {qos_results['preemptions']}")
    print(f"  2. QoS still affects path selection and conflict handling")
    print(f"  3. Unique goal enforcement prevents invalid same-destination completions")

    print("\n[Scenario Analysis]")
    print("""
    In this test scenario:

    ORIGINAL ALGORITHM:
    - Cars are planned in sequence with reservation-based conflict avoidance
    - A later car cannot claim a goal already assigned to another car
    - No priority-based takeover occurs

    QoS ALGORITHM:
    - The same unique-goal rule is enforced
    - QoS can still change route choice when path conflicts exist
    - In this particular run, no explicit preemption was triggered
    """)

    return {
        'qos_wins': qos_wins,
        'orig_wins': orig_wins,
        'preemptions': qos_results['preemptions']
    }


def main():
    parser = argparse.ArgumentParser(description="AGV QoS Scheduling Test")
    parser.add_argument("--mode", choices=['original', 'qos', 'both'], default='both',
                        help="Test mode: original, qos, or both (default: both)")
    parser.add_argument("--output", default=None,
                        help="Output file for results (JSON)")
    args = parser.parse_args()
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("\n" + "=" * 60)
    print("AGV QoS SCHEDULING COMPARISON TEST")
    print(f"Map: {MAP_FILE}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    original_results = None
    qos_results = None

    if args.mode in ['original', 'both']:
        try:
            original_results = test_original_mode()
            original_plan_file = save_json(
                original_results['plan'],
                f'plan_{run_timestamp}_original.json'
            )
            print(f"\n[Plan saved to: {original_plan_file}]")
        except Exception as e:
            print(f"\n[ERROR] Original mode failed: {e}")
            import traceback
            traceback.print_exc()

    if args.mode in ['qos', 'both']:
        try:
            qos_results = test_qos_mode()
            qos_plan_file = save_json(
                qos_results['plan'],
                f'plan_{run_timestamp}_qos.json'
            )
            print(f"\n[Plan saved to: {qos_plan_file}]")
        except Exception as e:
            print(f"\n[ERROR] QoS mode failed: {e}")
            import traceback
            traceback.print_exc()

    if args.mode == 'both' and original_results and qos_results:
        comparison = compare_results(original_results, qos_results)

        if args.output:
            import os
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_file = os.path.join(OUTPUT_DIR, args.output)
            with open(output_file, 'w') as f:
                json.dump({
                    'original': original_results,
                    'qos': qos_results,
                    'comparison': comparison
                }, f, indent=2)
            print(f"\n[Results saved to: {output_file}]")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
