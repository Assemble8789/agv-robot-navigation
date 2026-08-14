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

`bridge/` 把系统拆成**两个解耦节点**，通过进程内 ROS-topic 风格 pub/sub 总线通信：

```
         /clock (时间主: 每秒 tick) ───────────────►
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

## 文件清单

| 文件 | 作用 |
|---|---|
| `agv_world_qos.py` | QoS 时空 A\* 引擎：预约表、优先级抢占、`move_car` 规划本体 |
| `agv_world.py` | 实时仿真引擎（交互/CLI 版） |
| `agv_planner.py` / `agv_planner_v2.py` | 离线批量规划器 |
| `agv_map_edit.py` | 地图编辑器（15×10, 障碍/停靠点） |
| `agv_world_web.py` | Web API 封装（/add /move /status, 端口 8001） |
| `agv_world_qos_mujoco.py` / `agv_world_qos_robot.py` | 早期整合版（demo 用） |
| `agv_visualizer.py` | 动画回放 |
| `bridge/` | **推荐入口**：AGV ↔ MuJoCo + 机器人 桥接（见上） |

## 快速开始

```powershell
# 纯 AGV 交互 (MuJoCo 窗口 + CLI)
PY src/bridge/demo_bridge.py --demo5

# 带机器人协同 (AGV + elf3 机器人)
PY src/bridge/demo_bridge.py --robot --demo5

# 验证矩阵 (5/10/15 车 × 3 路径, markdown 报告)
PY src/bridge/validate_agv.py --cars 5,10,15 --paths 3 --reps 1 --robot 1

# 播放回放 (点一下过一秒)
PY src/bridge/_record_trajectory.py --cars 5 --path 0 --steps 150
PY src/bridge/_playback.py docs/traj_5_p0.json
```

> `PY` = `D:/download/anaconda3/envs/tutorial_for_mujoco/python.exe`
