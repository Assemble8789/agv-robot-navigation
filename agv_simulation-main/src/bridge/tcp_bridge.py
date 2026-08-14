"""
TCP 中继: 把进程内 Bus 的消息转发给远程监听端 (另开终端 `listen.py`)
====================================================================

在 bridge 进程内启动一个 localhost TCP 服务器, 订阅 Bus 上所有 topic,
把每条消息以 newline-delimited JSON 转发给已连接并订阅了对应 topic 的
客户端。 节点代码零改动 —— 节点仍用进程内 Bus, 中继只是加一个旁路出口。

协议 (每行一个 JSON):
  client -> relay:  {"op": "sub", "topics": ["/sim/obstacle_distance"]}
  relay  -> client: {"op": "ok", "topics": [...]}
  relay  -> client: {"topic": "/sim/...", "msg": {...stamp, seq, ...}}

用法:
  relay = TcpRelay(bus, port=5557)     # 在 bridge 进程内启动
"""

import os
import sys
import json
import socket
import threading

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)


def _json_default(o):
    """JSON 兜底编码: numpy 标量 (bool/int/float) 转原生类型, 避免
    'not JSON serializable' 把整条消息丢掉 (如 /sim/robot_state 的 fell)。"""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

DEFAULT_FORWARD = [
    "/clock", "/planner/path", "/planner/remove", "/planner/robot_hold",
    "/sim/car_state", "/sim/collision", "/sim/obstacle_distance",
    "/sim/agv_distance", "/sim/dynamic_obstacles", "/sim/static_obstacles",
    "/sim/robot_state", "/sim/robot_path", "/sim/robot_agv_distance",
]


class TcpRelay:
    def __init__(self, bus, host="127.0.0.1", port=5557, topics=None):
        self.bus = bus
        self.host = host
        self.port = port
        self.topics = topics or DEFAULT_FORWARD
        self.running = True
        self._clients = {}                 # socket -> set(subscribed topics)
        self._lock = threading.Lock()

        # 订阅 Bus 全部 topic: 回调只转发, 不阻塞发布者太久 (发送有超时兜底)
        self._subs = [
            bus.subscribe(t, (lambda tt: (lambda m: self._forward(tt, m)))(t))
            for t in self.topics
        ]

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(8)
        self._server.settimeout(0.5)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        print(f"[RELAY] TCP relay on {host}:{port} | topics={self.topics}")
        print(f"[RELAY] 另开终端: python {os.path.basename(__file__).replace('tcp_bridge','listen')} "
              f"--port {port} --topic <topic>")

    def stop(self):
        self.running = False
        try:
            self._server.close()
        except OSError:
            pass

    # ── 服务端 ──────────────────────────────────────────────────────
    def _accept_loop(self):
        while self.running:
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(0.5)   # 读写都有超时兜底, 慢客户端最多拖 0.5s
            threading.Thread(target=self._client_loop, args=(conn,),
                             daemon=True).start()

    def _client_loop(self, conn):
        sub_topics = set()
        with self._lock:
            self._clients[conn] = sub_topics
        buf = b""
        try:
            while self.running:
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue                 # 静默客户端: 保持连接
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        req = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    if req.get("op") == "sub":
                        ts = req.get("topics", [])
                        with self._lock:
                            sub_topics.update(ts)
                        self._send(conn, {"op": "ok", "topics": list(sub_topics)})
        except (OSError, ConnectionError):
            pass
        finally:
            with self._lock:
                self._clients.pop(conn, None)
            try:
                conn.close()
            except OSError:
                pass

    def _send(self, conn, obj):
        try:
            conn.sendall((json.dumps(obj, default=_json_default) + "\n").encode("utf-8"))
        except (OSError, ConnectionError):
            with self._lock:
                self._clients.pop(conn, None)
            try:
                conn.close()
            except OSError:
                pass

    def _forward(self, topic, msg):
        frame = json.dumps({"topic": topic, "msg": msg},
                           ensure_ascii=False, default=_json_default) + "\n"
        with self._lock:
            targets = [c for c, ts in list(self._clients.items())
                       if topic in ts]
        for c in targets:
            self._send(c, {"topic": topic, "msg": msg})
