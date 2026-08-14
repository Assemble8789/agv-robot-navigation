"""
进程内 ROS-topic 风格的发布/订阅总线
=====================================

Planner 节点和 MuJoCo 仿真节点之间解耦通信的传输层。 语义对齐 ROS topic:
  - publish(topic, msg)  发布消息
  - subscribe(topic, cb) 订阅, 返回句柄 (可 unsubscribe)

消息是 JSON 可序列化的 dict, 总线自动附加两个字段:
  - stamp: 发布者提供的仿真时间 (无则 time.time())
  - seq:   该 topic 上的单调递增序号

设计纪律: 订阅回调在【发布者线程】上同步执行, 但总线保证在锁外调用
(回调列表锁内快照)。 因此回调必须"便宜"——标准做法是只 enqueue 到订阅方
自己的 queue.Queue, 真正处理由订阅方 worker 线程完成。 这样各节点的可变
状态零跨线程竞争, 不需要在节点代码里加锁。

Transport 抽象层保证以后换传输 (跨进程 TCP / JSON 文件) 时, 节点代码
零改动 —— 节点只依赖 publish/subscribe/unsubscribe 三个方法。
"""

import abc
import time
import threading


class Transport(abc.ABC):
    """传输层接口。 任何节点只依赖这个抽象。"""

    @abc.abstractmethod
    def publish(self, topic, msg, stamp=None):
        """向 topic 发布一条消息 (dict)。 stamp 缺省用墙上时钟。"""

    @abc.abstractmethod
    def subscribe(self, topic, callback):
        """订阅 topic。 callback(msg) 在发布者线程被调用 (必须便宜)。
        返回订阅句柄, 可用于 unsubscribe。"""

    @abc.abstractmethod
    def unsubscribe(self, handle):
        """取消订阅。 重复取消 / 无效句柄为 no-op。"""


class Topic:
    """单个 topic: 订阅者表 + 序号计数器。"""

    def __init__(self, name):
        self.name = name
        self._lock = threading.RLock()
        self._subs = {}      # handle -> callback
        self._next_id = 1
        self.seq = 0         # 已发布消息数 (单调递增)

    def publish(self, msg, stamp=None):
        with self._lock:
            self.seq += 1
            full = dict(msg)
            full.setdefault("stamp", time.time() if stamp is None else stamp)
            full["seq"] = self.seq
            subs = list(self._subs.values())       # 锁内快照
        for cb in subs:                            # 锁外调用 (不阻塞其它发布/订阅)
            try:
                cb(full)
            except Exception as e:
                print(f"[bus] {self.name}: subscriber error: {e}")

    def subscribe(self, callback):
        with self._lock:
            handle = self._next_id
            self._next_id += 1
            self._subs[handle] = callback
            return handle

    def unsubscribe(self, handle):
        with self._lock:
            self._subs.pop(handle, None)


class Bus(Transport):
    """进程内总线: topic 名字 -> Topic。 线程安全, 惰性建 topic。"""

    def __init__(self):
        self._topics = {}
        self._lock = threading.Lock()

    def _get_topic(self, name):
        with self._lock:
            topic = self._topics.get(name)
            if topic is None:
                topic = Topic(name)
                self._topics[name] = topic
            return topic

    def publish(self, topic, msg, stamp=None):
        self._get_topic(topic).publish(msg, stamp=stamp)

    def subscribe(self, topic, callback):
        return self._get_topic(topic).subscribe(callback)

    def unsubscribe(self, handle):
        for topic in list(self._topics.values()):
            topic.unsubscribe(handle)
