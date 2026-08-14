"""
Topic 监视器: 打印总线上每条消息 (类似 `ros2 topic echo`)
=========================================================

订阅全部 topic, 把到达的原始消息打出来, 直观看到 Planner ↔ MuJoCo 之间
实时传了哪些信息。

节流策略: 高频 topic (/clock /sim/car_state /sim/obstacle_distance
/sim/collision) 每个整秒只打一条, 避免刷屏; 事件型 topic
(/planner/path /planner/remove) 每条都打。 碰撞事件 (any=True) 必打。

用法:
  monitor = TopicMonitor(bus)        # 订阅全部默认 topic
"""

DEFAULT_TOPICS = [
    "/clock", "/planner/path", "/planner/remove",
    "/sim/car_state", "/sim/collision", "/sim/obstacle_distance",
    "/sim/agv_distance", "/sim/dynamic_obstacles", "/sim/static_obstacles",
    "/sim/robot_state", "/sim/robot_path", "/sim/robot_agv_distance",
]

# 高频 topic: 每个整秒 (int(stamp)) 只打一条
THROTTLED = {"/clock", "/sim/car_state", "/sim/collision",
             "/sim/obstacle_distance", "/sim/agv_distance",
             "/sim/robot_state", "/sim/robot_agv_distance"}


class TopicMonitor:
    def __init__(self, bus, topics=None):
        self._last = {}          # topic -> 上次打印的 int(stamp)
        self._subs = []
        for t in (topics or DEFAULT_TOPICS):
            self._subs.append(
                bus.subscribe(t, (lambda tt: (lambda m: self._on(tt, m)))(t)))

    def _on(self, topic, msg):
        t0 = int(msg.get("stamp", 0.0))
        if topic in THROTTLED:
            if topic == "/sim/collision" and msg.get("any"):
                pass                      # 碰撞事件必打
            elif self._last.get(topic) == t0:
                return                    # 本秒已打过, 跳过
            self._last[topic] = t0
        self._print(topic, msg)

    def _print(self, topic, msg):
        seq = msg.get("seq")
        if topic == "/clock":
            print(f"[ECHO] {topic:<24} t={msg['time']:7.2f}  seq={seq}")
        elif topic == "/planner/path":
            traj = msg.get("trajectory", [])
            first = traj[0] if traj else None
            last = traj[-1] if traj else None
            src = f"({first['x']},{first['y']})" if first else "-"
            dst = f"({last['x']},{last['y']})" if last else "-"
            print(f"[ECHO] {topic:<24} car={msg['car_id']} "
                  f"action={msg['action']} goal={msg['goal']} "
                  f"npts={len(traj)} {src}->{dst}  seq={seq}")
        elif topic == "/planner/remove":
            print(f"[ECHO] {topic:<24} car={msg['car_id']}  seq={seq}")
        elif topic == "/sim/car_state":
            cars = ", ".join(
                f"{c['car_id']}:({c['x']:.1f},{c['y']:.1f})v={c['speed']:.2f}"
                for c in msg.get("cars", []))
            print(f"[ECHO] {topic:<24} cars={{{cars}}}  seq={seq}")
        elif topic == "/sim/collision":
            pairs = [(p['a'], p['b'], round(p['dist'], 3))
                     for p in msg.get("colliding_pairs", [])]
            print(f"[ECHO] {topic:<24} any={msg.get('any')} "
                  f"total={msg['total_count']} pairs={pairs}  seq={seq}")
        elif topic == "/sim/obstacle_distance":
            cars = ", ".join(
                f"{c['car_id']} d={c['dist']:.2f} "
                f"clr={c['clearance']:.2f}({c['kind']})"
                for c in msg.get("cars", []))
            print(f"[ECHO] {topic:<24} {cars}  seq={seq}")
        elif topic == "/sim/agv_distance":
            m = msg
            if m.get("pairs"):
                print(f"[ECHO] {topic:<24} min={m['min_dist']:.3f} "
                      f"({m['min_a']}<->{m['min_b']}) "
                      f"pairs={len(m['pairs'])}  seq={seq}")
            else:
                print(f"[ECHO] {topic:<24} (no cars)  seq={seq}")
        elif topic == "/sim/dynamic_obstacles":
            obs = [(o['id'], o['x'], o['y'], o['radius'])
                   for o in msg.get("obstacles", [])]
            print(f"[ECHO] {topic:<24} {obs}  seq={seq}")
        elif topic == "/sim/static_obstacles":
            print(f"[ECHO] {topic:<24} boxes={msg.get('boxes')} "
                  f"agv_r={msg.get('agv_radius')}  seq={seq}")
        elif topic == "/sim/robot_state":
            print(f"[ECHO] {topic:<24} pos=({msg['x']:.2f},{msg['y']:.2f}) "
                  f"yaw={msg['yaw']:.2f} v={msg['speed']:.2f} "
                  f"wp={msg['wp_index']}/{msg['n_wp']} "
                  f"arrived={msg['arrived']} fell={msg['fell']}  seq={seq}")
        elif topic == "/sim/robot_path":
            print(f"[ECHO] {topic:<24} n_wp={len(msg.get('waypoints', []))} "
                  f"index={msg['index']}  seq={seq}")
        elif topic == "/sim/robot_agv_distance":
            cars = ", ".join(
                f"{c['car_id']}:d={c['dist']:.2f}"
                for c in msg.get("cars", []))
            print(f"[ECHO] {topic:<24} {cars}  seq={seq}")
