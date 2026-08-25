# AGV 仿真系统核心 (src/)

## 这个文件夹是干啥的

`src/` 是整个 **AGV 多机路径规划系统**的代码本体，包含两大块：

1. **AGV 调度核心算法** — QoS 时空 A\* 规划 + 预约表 + 站台队列，纯逻辑（不碰渲染）。
2. **`bridge/` 协同桥接** — 把 AGV 调度接到 MuJoCo 物理仿真 + 人形机器人，形成 AGV↔机器人闭环（ARC）。

```
┌─ src/ ────────────────────────────────────────────────────────────┐
│  AGV 调度核心 (纯逻辑, 不渲染)                                       │
│  ├─ agv_world_qos.py    QoS 时空 A* 引擎 (规划本体, move_car)       │
│  ├─ agv_world.py        实时仿真引擎 (交互/CLI)                     │
│  ├─ agv_planner.py      离线批量规划器                              │
│  ├─ agv_map_edit.py     地图编辑器                                 │
│  ├─ agv_world_web.py    Web API 封装 (端口 8001)                   │
│  ├─ agv_visualizer.py   动画回放                                   │
│  │                                                                 │
│  ├─ bridge/  AGV ↔ MuJoCo + 机器人 桥接层 (推荐入口)               │
│  │  ├─ planner_node.py   ★规划节点 (大脑: 调度/让行/排队决策)        │
│  │  ├─ mujoco_node.py    仿真节点 (物理引擎 + 感知反馈)             │
│  │  └─ ...               总线/机器人/验证/可视化                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 整体架构: 两节点 + 总线

`bridge/` 把系统拆成**两个解耦节点**，通过进程内 **ROS-topic 风格 pub/sub 总线**通信。
语义对齐 ROS topic，方便以后换跨进程 TCP/JSON 传输。

```
         /clock (时间主: MuJoCo 是时间主, planner 同步 world_time) ►
  ┌──────────────┐   /planner/path (发布轨迹)   ┌──────────────┐
  │ PlannerNode  │ ────────────────────────────► │  MujocoNode  │
  │  (规划/大脑)  │  /planner/remove             │  (物理/感知)  │
  │  AGVWorld    │ ◄──────────────────────────── │ MuJoCo+机器人 │
  └──────────────┘   /sim/* (每帧反馈)            └──────────────┘
```

### 分工总原则

| | **PlannerNode** (大脑) | **MujocoNode** (身体) |
|---|---|---|
| 职责 | 所有**决策/规划/调度** | 所有**物理/渲染/感知** |
| 管 | AGV 路径规划、让行、排队、站台、机器人 hold 决策 | AGV mocap 渲染、机器人物理行走、碰撞检测 |
| 不看 | 物理步进、渲染 | AGV 的 A\* 规划、调度、让行决策 |
| 输出 | `/planner/path`(轨迹)、`/planner/robot_hold` | `/clock`、`/sim/*` 反馈 |

> **一句话：PlannerNode 想，MujocoNode 动。** MujocoNode 只是 planner 决策的执行器 + 传感器，
> 它**不参与任何 AGV 规划/让行/排队决策**——那些全在 planner 里。

---

## PlannerNode 干什么（核心）

`planner_node.py` 包 `AGVWorld`（QoS 时空 A\*），是系统**唯一的决策者**。每 tick 处理：

```
订阅 /clock 同步 world_time
  ├─ 处理命令: add/del/move/route → move_car (QoS 时空 A*) → 发布 /planner/path
  ├─ 到站派发: 车到站停留 DWELL_STOP(1s) 后自动派下一段
  ├─ 进站队列: 站台被占 → 挪到等待点排队 (FIFO, 队首先停)
  ├─ ARC 让行: _handle_robot_blocking 每 0.5s 检查每辆在途车
  │    ├─ 预测最近距离 < ROBOT_BLOCK_DIST(1.2m) → 停车让行 _stop_car
  │    ├─ 机器人走远 → 恢复 replan_car; 让行超时 → 绕行
  │    ├─ 已到站车挡机器人目标点 → 挪开 _move_agv_aside
  │    └─ 机器人目标站台被 AGV 占用 → 发布 robot_hold (机器人等)
  ├─ REPLAN-FAIL 处理: 重规划失败的车原地停车 + 周期重试 + 超时跳站
  └─ 站台队列: 队首临近锁站, 后来的车排队等待
```

**ARC (AGV-Robot Closure) 让行闭环** — 机器人路径横穿站台区, AGV 进出站必经:

```
预测: _closest_approach = 车未来 0~2s 轨迹 vs 机器人位置+速度外推 的最小距离 d
状态机 (每 0.5s 对每辆在途车):
  d < ROBOT_BLOCK_DIST(1.2m)  → 停车让行 _stop_car (截断轨迹停住, 当前格预约 1000tick)
  已在让行:
    机器人走开 (d >= CLEAR_DIST 1.1m) → 恢复续走 replan_car
    让行超 WAIT_TIMEOUT(3s)            → 绕行重规划 (机器人格并入 obs_set)
  已到站车停在机器人目标点附近           → 挪开 _move_agv_aside
站台交互:
  机器人目标站台被 AGV 停泊/进站 → 发布 /planner/robot_hold → 机器人原地等
  (进站 AGV 优先, 防"互相等"死锁)
```

- **ROBOT_BLOCK_DIST=1.2m**: 对穿相对速度 ~1.22m/s, 0.9m 提前量 0.74s 走完 <
  检查间隔+延迟 → 车对穿前停住 (对穿碰撞清零)。
- **机器人终点非站台** (`humanoid_plan.json` 终点 = 空位): 避免车被随机路线派去
  机器人终点站死锁。

**关键点**:
- **只做决策，不做物理**。`move_car` 只算轨迹（`(t,x,y,dir)` 点列），车怎么动由 sim 按轨迹插值渲染。
- 所有 AGV-AGV / AGV-机器人 的**避让决策**都在这：
  - AGV-AGV：A\* 预约表（规划时跳过别人已占的时空点/边）
  - AGV-机器人：ARC 让行（见上）
- 它读机器人的**状态**（`/sim/robot_state`、`/sim/robot_path`）来做让行决策，但**不控制机器人物理**。

## MujocoNode 干什么（物理引擎）

`mujoco_node.py` 是纯**物理 + 渲染 + 感知**，把 planner 的轨迹变成真实世界：

```
每帧 (500Hz):
  ├─ mj_step 物理步进:
  │    ├─ AGV: 按 /planner/path 的轨迹把 mocap 球插值到位 (车就是 mj 里的 mocap 球)
  │    └─ 机器人: RobotWalker 用 ONNX 行走策略 → mj_step 真实物理行走 (会被 AGV 顶开)
  ├─ 发布 /clock (时间主)
  ├─ 发布 /sim/* 反馈: car_state / collision / agv_distance / obstacle_distance / robot_state
  └─ 机器人侧兜底避让: 目标点平移 (_robot_detour) + 侧向推开 (_robot_avoid_agvs)
```

**关键点**:
- **只负责物理引擎**：推进 MuJoCo、渲染 AGV mocap、机器人物理行走、碰撞/距离感知。
- **不做 AGV 规划/调度/让行决策**。它只是"身体 + 眼睛"，把结果报告给 planner（`/sim/*`）。
- 机器人**走路**在这里（物理），但机器人**什么时候该等**（hold）由 planner 决策（`/planner/robot_hold`）。

---

## 消息流 (Message Flow And Message Format)

两个端点的消息流，格式为 `topic {payload}`。所有消息经总线自动附 `stamp`(仿真时间) + `seq`(序号)。

```mermaid
sequenceDiagram
  autonumber
  participant S as MujocoNode
  participant P as PlannerNode

  loop 每帧 500Hz
    S->>P: /clock
    S->>P: /sim/car_state
    S->>P: /sim/robot_state
    S->>P: /sim/robot_path
    S->>P: /sim/collision
  end

  Note over P: 1 命令 add/route 后 move_car 规划
  P->>S: /planner/path 新轨迹

  Note over P: 2 车到站后派下一段
  P->>S: /planner/path 下一段轨迹

  Note over P: 3 ARC 预测到撞机器人 停车让行
  P->>S: /planner/path 截断轨迹 停车

  Note over P: 4 ARC 机器人走远恢复 或 让行超时绕行
  P->>S: /planner/path 重规划轨迹 (replan)

  Note over P: 5 机器人目标站台被 AGV 占用
  P->>S: /planner/robot_hold

  Note over P: 6 别的车停下挡路 重规划绕开
  P->>S: /planner/path 绕行轨迹 (replan)

  Note over S: 按轨迹插值 mocap 机器人物理行走
```

### 关键消息格式

**`/planner/path` (Planner → Sim)** — 发布 AGV 轨迹, 同一 topic 不同时机不同语义:

| 时机 (什么时候调用) | action | 传什么信息 | 是否 replan |
|---|---|---|---|
| add/move/route 命令 | `add` | 完整轨迹 `trajectory:[{t,x,y,dir}]` | 否 (首次规划) |
| 车到站 DWELL_STOP 后 | `update` | 下一段轨迹 | 否 (派发) |
| ARC 预测撞 (每 0.5s) | `update` | 截断到当前时刻的轨迹 = **停车让行** | 否 (停车) |
| ARC 机器人走远/让行超时 | `update` | 重规划轨迹 = 恢复续走 / 绕行 | **是** |
| 别的车停下挡路 | `update` | 绕开停车格的新轨迹 | **是** |
| 排队/进站失败重试 | `update` | 新轨迹 | **是** |

**`/planner/robot_hold` (Planner → Sim)** — 机器人目标站台被 AGV 占用时, 让机器人原地等:

| 时机 | 传什么信息 |
|---|---|
| 机器人当前目标点距站台 <1.5m 且该站被 AGV 停泊/进站 | `{hold:true, reason:"station LM005 docked"}` |
| 等待超 ROBOT_HOLD_WAIT(5s) 安全网 | `{hold:false, reason:"hold timeout"}` |

**`/sim/robot_state` (Sim → Planner)** — ARC 让行的感知输入 (每帧):

```
header:  /sim/robot_state
content: {x, y, yaw, vx, vy, speed, wp_index, n_wp, arrived, fell, radius}
```

planner 用 `x,y,vx,vy` 直线外推预测机器人未来位置, 结合车 QoS 轨迹算最近距离, 决定是否让行。

**`/sim/robot_path` (Sim → Planner)** — 机器人剩余路径 (每帧):

```
header:  /sim/robot_path
content: {waypoints:[{x,y}], index, arrived}
```

planner 用它判断机器人目标站台 (进站优先/站台 hold)。

### replan 的 4 个时机

1. **机器人让开恢复** — ARC 让行中机器人走远(>1.1m)→ `replan_car` 续走
2. **让行超时绕行** — 让行 3s 机器人还挡 → `replan_car`(机器人格并入 obs_set)
3. **避开停车格** — 别的车停下挡路 → `_replan_cars_through` 重规划穿过它的车
4. **排队/进站失败重试** — `_dispatch_next_route` / `_replan_failed` 周期重试

---

## Topic / 消息总表

| Topic | 发布者 | 内容 |
|---|---|---|
| `/clock` | MujocoNode | `{time}` — 仿真时间, planner 同步 |
| `/planner/path` | PlannerNode | `{car_id, action, goal, trajectory}` — 变更时发完整轨迹 |
| `/planner/remove` | PlannerNode | `{car_id}` — del 后回收球 |
| `/planner/robot_hold` | PlannerNode | `{hold, reason}` — 机器人站台被占 → 原地等 |
| `/sim/car_state` | MujocoNode | `{cars:[{car_id,x,y,vx,vy,...}]}` |
| `/sim/agv_distance` | MujocoNode | `{pairs:[{a,b,dist}], min_dist}` |
| `/sim/collision` | MujocoNode | `{colliding_pairs, any, total}` — 球心距<0.4 |
| `/sim/robot_state` | MujocoNode | `{x,y,yaw,vx,vy,wp_index,n_wp,arrived,fell}` |
| `/sim/robot_path` | MujocoNode | `{waypoints, index, arrived}` — 机器人计划路径 |
| `/sim/robot_agv_distance` | MujocoNode | `{cars:[{car_id,dist,dx,dy}]}` — 机器人↔AGV |

---

## 文件清单

### AGV 调度核心 (src/)

| 文件 | 作用 |
|---|---|
| `agv_world_qos.py` | QoS 时空 A\* 引擎：预约表、优先级抢占、`move_car` 规划本体 |
| `agv_world.py` | 实时仿真引擎（交互/CLI 版） |
| `agv_planner.py` / `agv_planner_v2.py` | 离线批量规划器 (v2 支持 `--heuristic manhattan/chebyshev/alt`) |
| `agv_map_common.py` | **归一化加载层**: `load_normalized` 修复损坏 JSON → 浮点转 0.1m 整数格 → 自动补地标 |
| `agv_map_convert.py` | 外部地图转换器 (`--res/--landmarks/--seed/--to-meters`) |
| `agv_map_edit.py` | 地图编辑器（15×10, 障碍/停靠点） |
| `agv_world_web.py` | Web API 封装（/add /move /status, 端口 8001） |
| `agv_visualizer.py` | 动画回放 |

### bridge/ 核心节点

| 文件 | 干什么 |
|---|---|
| `planner_node.py` | **规划节点 (大脑)**: 包 AGVWorld, 处理命令/到站派发/进站队列/ARC 让行/REPLAN-FAIL, 发布 `/planner/path` |
| `mujoco_node.py` | **仿真节点 (物理引擎)**: MuJoCo 步进 + AGV mocap 渲染 + 机器人物理行走, 发 `/clock` + `/sim/*` 反馈 |

### bridge_wangting/ (小数版 0.1m 地图桥接)

| 文件 | 干什么 |
|---|---|
| `astar.py` | **等待直到空闲 A\*** `astar_waitfree` (时间折叠: 状态键 `(x,y,dir)`, 被占→跳最早空闲) + 修正 ALT `astar_alt_fixed` (best_cost_to_state 剪枝) |
| `mapframe.py` | 米制帧: `cell ↔ meter` 精确双射 (`res=0.1, offset=(-14.9,-0.4)`) |
| `map_scene.py` | 从地图生成障碍盒/车库/MuJoCo 场景 XML (0.1m 占用格按行聚类) |
| `wangting_nodes.py` | `WangtingPlanner`/`WangtingMujoco`: 帧换算 + 障碍膨胀(球不穿墙) + 出生不闪现 + A\* 再 patch |
| `run_wangting.py` | 入口: `--demo/--demo4/--demo5/--cars/--robot/--robot-wp/--pure/--headless` |
| `pure_agv.py` | 离线预规划: 短停车窗 + 等待重试 + 跳站 + t=0 同时发车 |
| `robot_scene_wangting.py` | elf3 人形 + wangting 工厂场景文本合并 |
| `robot_path.py` | 机器人路径: 细格 A\* 绕障碍 + 避开窄走廊 (<1.0m 不走) |
| `record_traj.py` | 录制 AGV+机器人每整秒位置 → `bridge/_playback.py` 2D 回放 |

### 机器人

| 文件 | 干什么 |
|---|---|
| `robot_scene.py` | 合并 elf3 + AGV 场景 XML (`robot_scene.build_scene`), 绕过 `<include>` 解析问题 |
| `robot_walker.py` | `RobotWalker`(elf3 行走策略, ONNX 推理 + mj_step) + `RobotNavigator`(P2P 导航, 包 navigation.navigate.Navigator) |

### 入口

| 文件 | 干什么 |
|---|---|
| `demo_bridge.py` | 入口: 建 Bus + 两节点, 支持 `--demo/--demo4/--demo5/--robot/--headless/--speed/--steps/--obstacle/--port/--echo` |

### 总线 / 网络

| 文件 | 干什么 |
|---|---|
| `topic_bus.py` | 传输层: `Transport` 抽象 + 进程内 `Bus`/`Topic` (线程安全 pub/sub, 自动附 stamp+seq) |
| `topic_monitor.py` | `--echo`: 本终端打印总线上每条消息 (类似 `ros2 topic echo`) |
| `tcp_bridge.py` | TCP 中继: 把 Bus 消息转发给另开终端的监听端 |
| `listen.py` | 另开终端的 topic 监听工具 (连 TCP 中继, 只打印你要的话题) |

### 验证 / 播放

| 文件 | 干什么 |
|---|---|
| `validate_agv.py` | 验证矩阵: 车数×路径×重复 headless 实验, 输出 markdown 报告 (完成率/碰撞/让行/排队) |
| `_record_trajectory.py` | 播放前置: 跑一遍 headless, 采样线程每整秒记录所有 AGV + 机器人位置到 JSON |
| `_playback.py` | 播放: 加载记录 JSON, 交互回放 (点击/←→ 过 1 秒, 空格播放, 参考 agv_world_qos 画法) |

### 调试 / 可视化 (下划线前缀 = 调试工具)

| 文件 | 干什么 |
|---|---|
| `_debug_route_map.py` | 2D 路线图: 任意车数×路径的路线 + 机器人路径 (PNG + 终端 ASCII) |
| `_debug_visual.py` | 实时可视化: 任意车数×路径带机器人开 MuJoCo 窗口 |
| `_debug_robot_collision.py` | 每次机器人碰撞 dump 现场 (位置/速度/接触点/剩余路径), 诊断碰撞用 |
| `_debug_replan_fail.py` | REPLAN-FAIL / 派发现场 dump (ASCII 网格 + 队列状态), 诊断卡死用 |

---

## 快速开始

```powershell
# cd 到 agv_simulation-main, PY = D:/download/anaconda3/envs/tutorial_for_mujoco/python.exe

# 纯 AGV
PY src/bridge/demo_bridge.py --demo5

# 带机器人协同
PY src/bridge/demo_bridge.py --robot --demo5

# 验证矩阵 (5/10/15 车 × 3 路径, 输出 markdown)
PY src/bridge/validate_agv.py --cars 5,10,15 --paths 3 --reps 1 --robot 1

# 播放回放 (记录 → 点一下过一秒)
PY src/bridge/_record_trajectory.py --cars 5 --path 0 --steps 150 --out docs/traj_5_p0.json
PY src/bridge/_playback.py docs/traj_5_p0.json

# 2D 路线图 / 实时窗口
PY src/bridge/_debug_route_map.py --cars 5 --path 0 --out docs/route_map_5_p0.png
PY src/bridge/_debug_visual.py --cars 5 --path 1
```

## 验证结果 (`ROBOT_BLOCK_DIST=1.2` + 机器人终点非站台)

- 5 车 93% / 10 车 95% / 15 车 86%，**机器人碰撞 0~2 次** (基线 0~13, 对穿清零)
- 5车P1 死锁消除 90→100% (机器人终点非站台), 15车P1 63→100%
- 详见 [docs/agv_validation_robot.md](../../docs/agv_validation_robot.md)
