"""
Robot navigation: walk from current pose to a target pose.

Improved control algorithm:
  - Continuous nonlinear steering (tanh) instead of hard threshold.
  - Speed profile: constant speed at distance > decel_distance, quadratic deceleration near target.
  - Orientation alignment at final stage.
  - Low‑pass filtering on velocity commands to reduce jerks.

Usage:
  python navigate_improved.py                        # default target
  python navigate_improved.py --tx 2.0 --ty 1.0      # custom target
"""

import sys
import numpy as np
import mujoco
import mujoco.viewer
import os
# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_env import EnvConfig, SimpleEnv


def get_robot_pose(env: SimpleEnv):
    """Return robot base (x, y, yaw) from qpos."""
    x, y = env.data.qpos[0], env.data.qpos[1]
    qw, qx, qy, qz = env.data.qpos[3:7]
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny, cosy)
    return float(x), float(y), float(yaw)


def normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _add_geom_to_scene(scene, geom_type: int, size, pos, rgba):
    """Add one geom to an MjvScene, no-op if full."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    sz = np.asarray(size, dtype=np.float64).reshape(-1, 1)
    ps = np.asarray(pos, dtype=np.float64).reshape(-1, 1)
    mt = np.eye(3, dtype=np.float64).ravel().reshape(-1, 1)
    cl = np.asarray(rgba, dtype=np.float32).reshape(-1, 1)
    mujoco.mjv_initGeom(g, geom_type, sz, ps, mt, cl)
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def add_target_marker(scene, x: float, y: float, z: float = 0.0,
                      radius: float = 0.06, height: float = 0.6):
    """Draw a red pin marker at target (x,y)."""
    # Vertical cylinder
    _add_geom_to_scene(
        scene, mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[radius, height / 2, 0],
        pos=[x, y, z + height / 2],
        rgba=[1.0, 0.1, 0.1, 0.9],
    )
    # Sphere on top
    _add_geom_to_scene(
        scene, mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius * 1.8, 0, 0],
        pos=[x, y, z + height],
        rgba=[1.0, 0.2, 0.2, 0.95],
    )
    # Small ring at base
    _add_geom_to_scene(
        scene, mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[radius * 2.5, 0.02, 0],
        pos=[x, y, z + 0.02],
        rgba=[1.0, 0.3, 0.3, 0.6],
    )


class Navigator:
    """
    Improved navigation controller.

    Parameters:
      target_x, target_y: goal position in world frame.
      fwd_speed: maximum forward speed (m/s).
      turn_speed: maximum angular speed (rad/s).
      arrival_threshold: distance below which robot stops (m).
      decel_distance: distance at which deceleration begins (m).
      align_distance: distance at which orientation alignment is emphasized (m).
      alpha: low‑pass filter coefficient (0~1, larger = more responsive).
    """

    def __init__(self, target_x: float, target_y: float,
                 fwd_speed: float = 0.7,
                 turn_speed: float = 1.0,
                 arrival_threshold: float = 0.10,
                 decel_distance: float = 0.5,
                 align_distance: float = 0.35,
                 alpha: float = 0.7,
                 min_fwd_speed: float = 0.25):
        self.target = np.array([target_x, target_y], dtype=np.float32)
        self.fwd_speed = fwd_speed
        self.turn_speed = turn_speed
        self.arrival_threshold = arrival_threshold
        self.decel_distance = decel_distance
        self.align_distance = align_distance
        self.alpha = alpha
        self.min_fwd_speed = min_fwd_speed  # floor to avoid policy dead zone

        self.last_cmd = np.zeros(3, dtype=np.float32)
        self.distance = float('inf')
        self.heading_error = 0.0
        self.arrived = False

    def update(self, x: float, y: float, yaw: float) -> np.ndarray:
        """
        Compute velocity command (vx, vy, omega) with improved control laws.

        Key insight: the RL policy has a minimum perceptible forward speed
        (~0.25 m/s). Any vx below this dead zone is ignored → robot stops
        short. We enforce a floor on vx for ALL non-arrived states, applied
        AFTER the low-pass filter, so the actual command never drops into
        the dead zone — even on the first step after a cold start.
        """
        dx = self.target[0] - x
        dy = self.target[1] - y
        self.distance = np.sqrt(dx * dx + dy * dy)

        if self.distance < self.arrival_threshold:
            self.arrived = True
            self.last_cmd = np.zeros(3)
            return self.last_cmd.copy()

        target_heading = np.arctan2(dy, dx)
        self.heading_error = normalize_angle(target_heading - yaw)

        # ----- Large heading error: stop and turn in place -----
        # When heading error > 60°, walking forward just creates a
        # circling orbit that never converges (especially for 90° targets).
        # Hysteresis: once in turn-in-place, need to get well-aligned (<45°)
        # before walking, to prevent bearing-change oscillations for
        # close behind targets.
        if not hasattr(self, '_in_turn_in_place'):
            self._in_turn_in_place = False
        enter_thresh = np.radians(60)   # enter turn-in-place
        exit_thresh  = np.radians(45)   # exit  turn-in-place (stricter)
        threshold = exit_thresh if self._in_turn_in_place else enter_thresh

        if abs(self.heading_error) > threshold:
            self._in_turn_in_place = True
            vx_raw = 0.0
            omega_raw = self.turn_speed * np.sign(self.heading_error)
        else:
            self._in_turn_in_place = False
            # ----- Steering: continuous nonlinear (tanh) -----
            omega_raw = self.turn_speed * np.tanh(2.0 * self.heading_error)

            # ----- Forward speed: distance‑based profile × heading penalty -----
            # heading_penalty: cos(heading_error) — reduces speed when not
            # facing the target, preventing "run past" that causes oscillation.
            heading_penalty = np.cos(self.heading_error)  # 1.0 at 0°, 0.5 at 60°
            heading_penalty = max(0.15, heading_penalty)   # floor: don't stall

            if self.distance > self.decel_distance:
                vx_raw = self.fwd_speed * heading_penalty
            else:
                # Quadratic deceleration from decel_distance down to 0
                factor = (self.distance / self.decel_distance) ** 2
                vx_raw = self.fwd_speed * max(0.0, min(1.0, factor))
                vx_raw *= heading_penalty

        # ----- Final alignment: boost steering for precise heading -----
        if self.distance < self.align_distance and not self._in_turn_in_place:
            # Boost steering gain only — do NOT scale down vx here
            # (scaling vx drops it into the policy dead zone → robot stops short)
            omega_raw = self.turn_speed * np.tanh(3.0 * self.heading_error)

        # Assemble raw command (vy=0, no lateral motion)
        cmd_raw = np.array([vx_raw, 0.0, omega_raw], dtype=np.float32)

        # ----- Low‑pass filter to smooth commands -----
        cmd = self.alpha * cmd_raw + (1.0 - self.alpha) * self.last_cmd

        # ----- Floor: avoid policy dead zone -----
        # Applied AFTER the filter so the very first step (when last_cmd=0)
        # also stays above the dead zone.  Covers the whole approach —
        # from far away down to the arrival threshold.
        if cmd[0] > 0.01 and self.distance > self.arrival_threshold:
            cmd[0] = max(cmd[0], self.min_fwd_speed)

        self.last_cmd = cmd.copy()
        return cmd


def render_navigation(target_x: float, target_y: float):
    """Launch viewer and navigate robot to target using improved controller."""
    config = EnvConfig()
    env = SimpleEnv(config)
    env.reset()

    nav = Navigator(target_x, target_y)

    print("╔══════════════════════════════════════════════╗")
    print("║  机器人导航 (改进版)                         ║")
    print(f"║  目标: ({target_x:.2f}, {target_y:.2f})                      ║")
    print("║  ESC : 退出                                  ║")
    print("╚══════════════════════════════════════════════╝")

    with mujoco.viewer.launch_passive(
        env.model, env.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 4.0
        viewer.cam.lookat[:] = (0.3, 0.0, 0.9)
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 160

        add_target_marker(viewer.user_scn, target_x, target_y)

        print_tick = 0
        arrived_printed = False
        while viewer.is_running():
            x, y, yaw = get_robot_pose(env)
            cmd = nav.update(x, y, yaw)

            env.step()

            if env.counter % env.control_decimation == 0:
                env.cmd = cmd
                env.calc_obs()
                env.policy_inference()

            viewer.sync()
            env.rate.sleep()

            print_tick += 1
            if print_tick % 200 == 0:
                deg = np.degrees(nav.heading_error)
                robot_speed = float(np.linalg.norm(env.data.qvel[0:3]))
                print(f"\rpos=({x:.3f},{y:.3f})  yaw={np.degrees(yaw):.1f}°  "
                      f"dist={nav.distance:.3f}m  heading_err={deg:+.1f}°  "
                      f"speed={robot_speed:.3f} m/s  cmd=({cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f})  "
                      f"arrived={nav.arrived}  ", end="")

            if nav.arrived:
                if not arrived_printed:
                    print(f"\n✓ Arrived at target! Final pos=({x:.3f},{y:.3f})  standing by...")
                    arrived_printed = True


def run_headless(target_x: float, target_y: float,
                 fwd_speed: float = 0.7,
                 turn_speed: float = 1.0,
                 arrival_threshold: float = 0.10,
                 decel_distance: float = 0.5,
                 align_distance: float = 0.35,
                 alpha: float = 0.7,
                 min_fwd_speed: float = 0.25,
                 max_time: float = 30.0) -> dict:
    """Run one navigation episode headless, return metrics."""
    config = EnvConfig()
    env = SimpleEnv(config)
    env.reset()

    nav = Navigator(target_x, target_y,
                    fwd_speed=fwd_speed,
                    turn_speed=turn_speed,
                    arrival_threshold=arrival_threshold,
                    decel_distance=decel_distance,
                    align_distance=align_distance,
                    alpha=alpha,
                    min_fwd_speed=min_fwd_speed)

    start_pos = get_robot_pose(env)
    traj = [(0.0, *start_pos)]
    t = 0.0
    dt = config.simulation_dt
    stuck_counter = 0

    while t < max_time:
        x, y, yaw = get_robot_pose(env)
        cmd = nav.update(x, y, yaw)
        traj.append((t, x, y, yaw))

        env.step()
        if env.counter % env.control_decimation == 0:
            env.cmd = cmd
            env.calc_obs()
            env.policy_inference()

        t += dt

        # Detect stuck: no progress in last N seconds
        if len(traj) >= 3:
            prev = traj[-50] if len(traj) >= 50 else traj[0]
            moved = np.sqrt((x - prev[1])**2 + (y - prev[2])**2)
            if moved < 0.01 and nav.distance > arrival_threshold:
                stuck_counter += 1
            else:
                stuck_counter = 0

        if stuck_counter > 200:  # ~1s stuck
            break
        if nav.arrived:
            break

    final_x, final_y, final_yaw = get_robot_pose(env)
    final_dist = float(np.sqrt((target_x - final_x)**2 + (target_y - final_y)**2))
    path_length = float(np.sum(np.sqrt(
        np.diff(np.array([p[1] for p in traj]))**2 +
        np.diff(np.array([p[2] for p in traj]))**2
    ))) if len(traj) > 1 else 0.0

    return {
        'arrived': nav.arrived,
        'stuck': stuck_counter > 200,
        'final_dist': final_dist,
        'elapsed': t,
        'path_length': path_length,
        'start_x': start_pos[0], 'start_y': start_pos[1],
        'final_x': final_x, 'final_y': final_y,
    }


def tune_navigation(target_x: float = 2.0, target_y: float = 1.0):
    """Grid-search navigation parameters to minimize final distance error."""
    print(f"\n{'='*64}")
    print(f"  Tuning navigation params — target=({target_x},{target_y})")
    print(f"{'='*64}\n")

    # Search space
    fwd_speeds    = [0.3, 0.5, 0.7]
    min_fwd_speeds = [0.15, 0.20, 0.25]
    decel_dists   = [0.4, 0.6, 0.9]
    arrival_thresh = [0.10, 0.15, 0.20]
    alphas        = [0.5, 0.7]

    best = None
    best_score = -999.0

    trial = 0
    total = (len(fwd_speeds) * len(min_fwd_speeds) * len(decel_dists)
             * len(arrival_thresh) * len(alphas))
    print(f"  Total combinations: {total}\n")
    print(f"  {'#':>3}  {'fwd':>5}  {'min':>5}  {'decel':>5}  {'arr':>5}  {'α':>5}  {'final_d':>8}  {'arrived':>7}  {'elapsed':>7}")
    print(f"  {'─'*3}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*7}")

    for fs in fwd_speeds:
        for mfs in min_fwd_speeds:
            for dd in decel_dists:
                for at in arrival_thresh:
                    for al in alphas:
                        trial += 1
                        m = run_headless(target_x, target_y,
                                        fwd_speed=fs,
                                        min_fwd_speed=mfs,
                                        decel_distance=dd,
                                        arrival_threshold=at,
                                        alpha=al,
                                        max_time=20.0)

                        # Score: lower final_dist is better;
                        # heavy penalty for not arriving
                        score = -m['final_dist']
                        if not m['arrived'] and not m['stuck']:
                            score -= 1.0  # ran out of time
                        if m['stuck']:
                            score -= 2.0  # got stuck

                        print(f"  {trial:>3}  {fs:>5.2f}  {mfs:>5.2f}  {dd:>5.2f}  {at:>5.2f}  {al:>5.2f}  "
                              f"{m['final_dist']:>8.3f}  {str(m['arrived']):>7}  {m['elapsed']:>6.1f}s"
                              f"{'  ★' if score > best_score else ''}")

                        if score > best_score:
                            best_score = score
                            best = {**m, 'fwd_speed': fs, 'min_fwd_speed': mfs,
                                    'decel_distance': dd, 'arrival_threshold': at, 'alpha': al}

    print(f"\n{'='*64}")
    print(f"  BEST PARAMS")
    print(f"  fwd_speed={best['fwd_speed']}  min_fwd_speed={best['min_fwd_speed']}")
    print(f"  decel_distance={best['decel_distance']}  arrival_threshold={best['arrival_threshold']}")
    print(f"  alpha={best['alpha']}")
    print(f"  final_dist={best['final_dist']:.4f}m  arrived={best['arrived']}  elapsed={best['elapsed']:.1f}s")
    print(f"{'='*64}\n")
    return best


# ═══════════════════════════════════════════════════════════════════════════
# Test Suite
# ═══════════════════════════════════════════════════════════════════════════

TEST_TARGETS = [
    # (label, x, y) — robot starts at ~(0,0) heading ~0
    ("near straight",        0.5,  0.0),
    ("near right",           0.4, -0.3),
    ("near left",            0.4,  0.3),
    ("medium straight",      1.0,  0.0),
    ("medium right-45°",     0.7, -0.7),
    ("medium left-45°",      0.7,  0.7),
    ("left 90°",             0.0,  1.0),
    ("right 90°",            0.0, -1.0),
    ("far straight",         2.0,  0.0),
    ("far diagonal",         1.5,  1.0),
    ("far right",            1.5, -1.0),
    ("far left",             1.0,  1.5),
    ("very far",             3.0,  0.5),
    ("far behind-right",    -0.3, -1.5),
    ("far diagonal-left",    2.0,  2.0),
]


def generate_50_targets():
    """Generate 50 test targets covering near→far distances, all directions.

    Includes (0, 0.5) — a known problematic 90° lateral target.
    Robot starts at (0,0) heading +X (0°). Targets are in world frame.
    """
    targets = []

    # --- Helper: add target at (distance, angle_deg) ---
    def add(dist, angle_deg):
        rad = np.radians(angle_deg)
        tx = dist * np.cos(rad)
        ty = dist * np.sin(rad)
        label = f"d={dist:.2f} θ={angle_deg}°"
        targets.append((label, round(tx, 4), round(ty, 4)))

    # Ring 1: 0.30m — 4 cardinal directions (very near)
    for a in [0, 90, 180, -90]:
        add(0.30, a)

    # Ring 2: 0.50m — 8 directions, includes (0, 0.5) at 90°
    for a in [0, 45, 90, 135, 180, -135, -90, -45]:
        add(0.50, a)

    # Ring 3: 0.75m — 8 directions
    for a in [0, 30, 60, 90, 120, 180, -120, -60]:
        add(0.75, a)

    # Ring 4: 1.00m — 8 directions
    for a in [0, 45, 90, 135, 180, -135, -90, -45]:
        add(1.00, a)

    # Ring 5: 1.50m — 8 directions
    for a in [0, 30, 60, 90, 120, 180, -120, -60]:
        add(1.50, a)

    # Ring 6: 2.00m — 8 directions
    for a in [0, 45, 90, 135, 180, -135, -90, -45]:
        add(2.00, a)

    # Ring 7: 2.50m — 4 directions
    for a in [0, 90, 180, -90]:
        add(2.50, a)

    # Ring 8: 3.00m — 2 directions (front, diagonal)
    for a in [0, 45]:
        add(3.00, a)

    # Sort by distance, then angle
    targets.sort(key=lambda t: (float(t[0].split('=')[1].split()[0]),
                                 float(t[0].split('=')[-1].replace('°',''))))
    return targets


def run_test_50(max_time: float = 30.0):
    """Generate 50 targets and test them all headless."""
    targets = generate_50_targets()
    print(f"\n{'='*78}")
    print(f"  50-TARGET NAVIGATION TEST SUITE  (max {max_time}s each)")
    print(f"  Defaults: fwd=0.7 min_fwd=0.25 decel=0.5 arr=0.10 α=0.7")
    print(f"{'='*78}\n")
    print(f"  {'#':>2}  {'target':>22}  {'(x,y)':>16}  {'arr':>5}  {'final_d':>8}  {'start→end':>22}  {'time':>6}")
    print(f"  {'─'*2}  {'─'*22}  {'─'*16}  {'─'*5}  {'─'*8}  {'─'*22}  {'─'*6}")

    results = []
    for i, (label, tx, ty) in enumerate(targets, 1):
        m = run_headless(tx, ty, max_time=max_time)
        results.append((label, tx, ty, m))
        print(f"  {i:>2}  {label:>22}  ({tx:>6.2f},{ty:>6.2f})  "
              f"{'YES' if m['arrived'] else 'NO':>5}  "
              f"{m['final_dist']:>8.4f}  "
              f"({m['start_x']:>5.1f},{m['start_y']:>5.1f})→({m['final_x']:>6.2f},{m['final_y']:>6.2f})  "
              f"{m['elapsed']:>5.1f}s")

    # Summary
    arrived = [r for r in results if r[3]['arrived']]
    not_arrived = [r for r in results if not r[3]['arrived']]
    stuck = [r for r in results if r[3]['stuck']]
    dists = [r[3]['final_dist'] for r in results]
    times = [r[3]['elapsed'] for r in results]

    print(f"\n{'─'*78}")
    print(f"  SUMMARY")
    print(f"  ───────")
    print(f"  Arrived: {len(arrived)}/{len(results)} ({100*len(arrived)/len(results):.0f}%)")
    print(f"  Stuck:   {len(stuck)}/{len(results)}")
    if not_arrived:
        print(f"  Failed targets:")
        for label, tx, ty, m in not_arrived:
            print(f"    {label:>22}  ({tx:.2f},{ty:.2f})  final_d={m['final_dist']:.4f}  "
                  f"stuck={m['stuck']}")
    print(f"  Final distance: mean={np.mean(dists):.4f}m  median={np.median(dists):.4f}m  "
          f"min={np.min(dists):.4f}m  max={np.max(dists):.4f}m")
    print(f"  Elapsed time:   mean={np.mean(times):.1f}s  max={np.max(times):.1f}s")
    print(f"{'='*78}\n")

    return results


def run_test_suite(max_time: float = 25.0):
    """Run all test targets headless, print summary table."""
    print(f"\n{'='*72}")
    print(f"  NAVIGATION TEST SUITE  ({len(TEST_TARGETS)} targets, max {max_time}s each)")
    print(f"  Defaults: fwd=0.7 min_fwd=0.25 decel=0.5 arr=0.10 α=0.7")
    print(f"{'='*72}\n")
    print(f"  {'#':>2}  {'target':>20}  {'(x,y)':>16}  {'arrived':>7}  {'final_d':>8}  {'start→end':>18}  {'time':>6}")
    print(f"  {'─'*2}  {'─'*20}  {'─'*16}  {'─'*7}  {'─'*8}  {'─'*18}  {'─'*6}")

    results = []
    for i, (label, tx, ty) in enumerate(TEST_TARGETS, 1):
        m = run_headless(tx, ty, max_time=max_time)
        results.append((label, tx, ty, m))
        print(f"  {i:>2}  {label:>20}  ({tx:>5.1f},{ty:>5.1f})   "
              f"{'YES' if m['arrived'] else 'NO':>7}  "
              f"{m['final_dist']:>8.4f}  "
              f"({m['start_x']:>5.1f},{m['start_y']:>5.1f})→({m['final_x']:>5.1f},{m['final_y']:>5.1f})  "
              f"{m['elapsed']:>5.1f}s")

    # Summary
    arrived_count = sum(1 for _, _, _, m in results if m['arrived'])
    stuck_count = sum(1 for _, _, _, m in results if m['stuck'])
    dists = [m['final_dist'] for _, _, _, m in results]
    times = [m['elapsed'] for _, _, _, m in results]

    print(f"\n{'─'*72}")
    print(f"  SUMMARY")
    print(f"  ───────")
    print(f"  Arrived: {arrived_count}/{len(results)} ({100*arrived_count/len(results):.0f}%)")
    print(f"  Stuck:   {stuck_count}/{len(results)}")
    print(f"  Final distance: mean={np.mean(dists):.4f}m  median={np.median(dists):.4f}m  "
          f"min={np.min(dists):.4f}m  max={np.max(dists):.4f}m")
    print(f"  Elapsed time:   mean={np.mean(times):.1f}s  max={np.max(times):.1f}s")
    print(f"{'='*72}\n")

    return results


if __name__ == "__main__":
    target_x = 2
    target_y = 3

    if "--tx" in sys.argv:
        idx = sys.argv.index("--tx")
        target_x = float(sys.argv[idx + 1])
    if "--ty" in sys.argv:
        idx = sys.argv.index("--ty")
        target_y = float(sys.argv[idx + 1])

    if "--tune" in sys.argv:
        tune_navigation(target_x, target_y)
    elif "--test50" in sys.argv:
        run_test_50()
    elif "--test" in sys.argv:
        run_test_suite()
    elif "--headless" in sys.argv:
        m = run_headless(target_x, target_y, max_time=30.0)
        print(f"\nHeadless result: arrived={m['arrived']}  final_dist={m['final_dist']:.4f}m  "
              f"elapsed={m['elapsed']:.1f}s")
    else:
        render_navigation(target_x, target_y)