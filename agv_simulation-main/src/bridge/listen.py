"""
另开终端的 topic 监听工具 (类似 `ros2 topic echo`)
====================================================

连上 bridge 进程里的 TCP 中继, 只打印你要的话题消息, 不在 bridge 终端刷屏。

用法 (两个终端):
  终端1: python src/bridge/demo_bridge.py --port 5557
  终端2: python src/bridge/listen.py --port 5557 --topic /sim/obstacle_distance
          python src/bridge/listen.py --port 5557 --topic /planner/path --topic /sim/collision
          python src/bridge/listen.py --port 5557               # 默认除 /clock 外全听
          python src/bridge/listen.py --port 5557 --topic /sim/obstacle_distance --raw  # 完整 JSON
"""

import sys
import json
import argparse
import socket

DEFAULT_TOPICS = [
    "/planner/path", "/planner/remove",
    "/sim/car_state", "/sim/collision", "/sim/obstacle_distance",
    "/sim/agv_distance", "/sim/dynamic_obstacles", "/sim/static_obstacles",
    "/sim/robot_state", "/sim/robot_path", "/sim/robot_agv_distance",
]

# 已知 topic 前缀 (用于从 Git Bash 转成的路径里恢复出原始 topic)
KNOWN_ROOTS = ("/clock", "/planner/", "/sim/")


def norm_topic(t):
    """恢复 --topic 参数。 Git Bash 会把 `/sim/x` 转成 `<git-root>/sim/x`,
    这里从已知前缀截断还原; 正常以 / 开头的直接返回。"""
    if t.startswith("/"):
        return t
    for root in KNOWN_ROOTS:
        i = t.find(root)
        if i >= 0:
            return t[i:]
    return t


def format_msg(topic, msg):
    """紧凑人类可读格式 (与 TopicMonitor 一致)。"""
    if topic == "/clock":
        return f"time={msg.get('time'):.2f}  seq={msg.get('seq')}"
    if topic == "/planner/path":
        traj = msg.get("trajectory", [])
        first = traj[0] if traj else None
        last = traj[-1] if traj else None
        src = f"({first['x']},{first['y']})" if first else "-"
        dst = f"({last['x']},{last['y']})" if last else "-"
        return (f"car={msg['car_id']} action={msg['action']} "
                f"goal={msg['goal']} npts={len(traj)} {src}->{dst} "
                f"seq={msg.get('seq')}")
    if topic == "/planner/remove":
        return f"car={msg['car_id']}  seq={msg.get('seq')}"
    if topic == "/sim/car_state":
        cars = ", ".join(
            f"{c['car_id']}:({c['x']:.1f},{c['y']:.1f})v={c['speed']:.2f}"
            for c in msg.get("cars", []))
        return f"cars={{{cars}}}  seq={msg.get('seq')}"
    if topic == "/sim/collision":
        pairs = [(p['a'], p['b'], round(p['dist'], 3))
                 for p in msg.get("colliding_pairs", [])]
        return (f"any={msg.get('any')} total={msg['total_count']} "
                f"pairs={pairs}  seq={msg.get('seq')}")
    if topic == "/sim/obstacle_distance":
        cars = ", ".join(
            f"{c['car_id']} d={c['dist']:.2f} "
            f"clr={c['clearance']:.2f}({c['kind']})"
            for c in msg.get("cars", []))
        return f"{cars}  seq={msg.get('seq')}"
    if topic == "/sim/agv_distance":
        m = msg
        if m.get("pairs"):
            return (f"min={m['min_dist']:.3f} ({m['min_a']}<->{m['min_b']}) "
                    f"pairs={len(m['pairs'])}  seq={m.get('seq')}")
        return f"(no cars)  seq={m.get('seq')}"
    if topic == "/sim/dynamic_obstacles":
        obs = [(o['id'], o['x'], o['y'], o['radius'])
               for o in msg.get("obstacles", [])]
        return f"obstacles={obs}  seq={msg.get('seq')}"
    if topic == "/sim/static_obstacles":
        return (f"boxes={msg.get('boxes')} agv_r={msg.get('agv_radius')} "
                f"seq={msg.get('seq')}")
    if topic == "/sim/robot_state":
        return (f"pos=({msg['x']:.2f},{msg['y']:.2f}) yaw={msg['yaw']:.2f} "
                f"v={msg['speed']:.2f} wp={msg['wp_index']}/{msg['n_wp']} "
                f"arrived={msg['arrived']} fell={msg['fell']}  "
                f"seq={msg.get('seq')}")
    if topic == "/sim/robot_path":
        return (f"n_wp={len(msg.get('waypoints', []))} "
                f"index={msg['index']}  seq={msg.get('seq')}")
    if topic == "/sim/robot_agv_distance":
        cars = ", ".join(f"{c['car_id']}:d={c['dist']:.2f}"
                         for c in msg.get("cars", []))
        return f"{cars}  seq={msg.get('seq')}"
    return json.dumps(msg, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="监听 bridge 的 topic (类似 ros2 topic echo)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--topic", action="append", default=[],
                        help="要监听的话题 (可多次); 缺省=除 /clock 外全部")
    parser.add_argument("--raw", action="store_true", help="打印完整 JSON")
    args = parser.parse_args()

    if args.topic:
        topics = [norm_topic(t) for t in args.topic]
    else:
        topics = DEFAULT_TOPICS
    bad = [t for t in topics if not t.startswith("/")]
    if bad:
        print(f"[WARN] 这些话题可能被 Git Bash 路径转换搞坏了: {bad}")
        print(f"[WARN] 可用 MSYS_NO_PATHCONV=1 前缀运行, 如:")
        print(f"       MSYS_NO_PATHCONV=1 python src/bridge/listen.py "
              f"--port {args.port} --topic /sim/obstacle_distance")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((args.host, args.port))
    except (ConnectionRefusedError, OSError):
        print(f"[ERR] 连不上 {args.host}:{args.port} —— "
              f"先在另一个终端运行: python src/bridge/demo_bridge.py --port {args.port}")
        sys.exit(1)

    s.sendall((json.dumps({"op": "sub", "topics": topics}) + "\n").encode("utf-8"))
    print(f"[LISTEN] {args.host}:{args.port} subscribed: {topics}")
    print("[LISTEN] Ctrl-C 退出\n")

    buf = b""
    try:
        while True:
            data = s.recv(65536)
            if not data:
                print("[LISTEN] 连接已关闭 (bridge 退出了?)")
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    frame = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if frame.get("op") == "ok":
                    continue
                topic = frame.get("topic", "?")
                msg = frame.get("msg", {})
                if args.raw:
                    print(f"[{topic}] {json.dumps(msg, ensure_ascii=False)}")
                else:
                    print(f"[{topic}] {format_msg(topic, msg)}", flush=True)
    except KeyboardInterrupt:
        print("\n[LISTEN] stopped")
    finally:
        try:
            s.close()
        except OSError:
            pass


if __name__ == "__main__":
    main()
