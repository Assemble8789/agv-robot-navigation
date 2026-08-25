# 人形机器人 × AGV 协同仿真实验室

研究**人形机器人(BXI_elf3)运动控制**与**AGV 多机调度**，以及两者在工厂场景中的**协同避让闭环**的仿真仓库。
MuJoCo 物理仿真 + ONNX 行走策略 + QoS 时空 A\* 调度。

```
人形机器人控制 (model/ + navigation/)     AGV 多机调度 (agv_simulation-main/src/)
        │                                        │
        └────────────── 协同避让 (ARC) ───────────┘
              agv_simulation-main/src/bridge/   ← 推荐入口
              navigation/run_factory_full_agv_yield.py
```

## 核心内容

### 1. 人形机器人运动控制

BXI_elf3 双足机器人，ONNX 行走策略(RL 训练)驱动 29 关节，MuJoCo 物理步进：

| 文件 | 作用 |
|---|---|
| `simple_env.py` | 行走策略 lite 仿真（单机器人原地行走/避障） |
| `WPPwrapper.py` | mock 步态 + 任务 IK 可视化 |
| `model/bxi_elf3/` | 机器人模型/场景（`bxi_elf3_scene.xml` 等）+ 行走策略 onnx |
| `model/unitree_g1/` | Unitree G1 机器人模型 |

### 2. 机器人导航 (navigation/)

基于行走策略的 P2P 导航（纯反馈控制，实时算速度指令驱动 RL 策略）：

```bash
python navigation/navigate.py --tx 2.0 --ty 1.0    # 走到目标点
python navigation/navigate.py --test50             # 50 目标全向测试
```

工厂协同（AGV + 机器人同场）：`navigation/run_factory_full_agv_yield.py`、`navigate_factory.py`。

### 3. AGV 多机调度 (agv_simulation-main/src/)

QoS 时空 A\* 规划 + 预约表（顶点/边时空预约）+ 优先级抢占 + 站台队列：

- `agv_world_qos.py` — QoS 仿真引擎（A\* 本体）
- `agv_world.py` / `agv_planner.py` / `agv_map_edit.py` — 实时仿真 / 离线规划 / 地图编辑

### 4. AGV + 人形机器人协同 (agv_simulation-main/src/bridge/) ★推荐入口

把 QoS 规划器和 MuJoCo 仿真 + 机器人解耦成两个节点（ROS-topic 风格总线）：

```
PlannerNode (AGVWorld) ──/planner/path──► MujocoNode (MuJoCo + 机器人)
      ▲                                   │
      └─────────── /clock ◄───────────────┘  (MuJoCo 时间主)
           ◄── /sim/* 反馈 (位置/碰撞/障碍/机器人)
```

**ARC（AGV-Robot Closure）避让闭环**：机器人路径横穿站台区，AGV 进出站必经——
预测最近距离 < 1.2m → 停车让行（对穿提前量）；机器人走远恢复 / 超时绕行 / 站台等待；
机器人终点设非站台避免车被派去终点站死锁。

```bash
# 带机器人可视化 (AGV + 机器人一起跑)
PY src/bridge/demo_bridge.py --robot --demo5

# 验证矩阵 (5/10/15 车 × 3 路径, markdown 报告)
PY src/bridge/validate_agv.py --cars 5,10,15 --paths 3 --reps 1 --robot 1

# 播放回放 (点一下过一秒)
PY src/bridge/_record_trajectory.py --cars 5 --path 0 --steps 150
PY src/bridge/_playback.py docs/traj_5_p0.json
```

> `PY` = `D:/download/anaconda3/envs/tutorial_for_mujoco/python.exe`（`tutorial_for_mujoco` 环境）。
> 详细说明见 `agv_simulation-main/src/README.md`。

### 5. 小数版地图 (0.1m) + 修正 A* 桥接 (agv_simulation-main/src/bridge_wangting/)

把商用 AGV 地图导出文件（0.1m 浮点点云，如 `wangting...workflow2.json`）无损接入
整数格 A*，并在 AGV↔MuJoCo bridge 里跑通（含机器人、ARC 让行）：

- **归一化层** `agv_map_common.load_normalized`：修复损坏 JSON → 浮点转 0.1m 整数格 → 自动补地标
- **等待直到空闲 A\*** `bridge_wangting/astar.astar_waitfree`：状态键 `(x,y,dir)` 不含时间，
  移动到被占邻居时原地算最早空闲时刻直接跳过去 → 时间维折叠，0.1m 大图不再爆状态
- 障碍按 AGV 半径膨胀（球不穿墙）、米制帧换算、机器人避窄走廊、2D 录制回放

```bash
PY src/bridge_wangting/run_wangting.py --robot --demo5 --speed 3      # 交互可视化
PY src/bridge_wangting/run_wangting.py --pure --demo5 --headless --steps 150  # 离线预规划
PY src/bridge_wangting/record_traj.py --demo5 --robot --steps 150     # 录制 → 2D 回放
PY src/bridge/_playback.py docs/traj_wangting_v2.json
```

> 详细说明见 `agv_simulation-main/src/bridge_wangting/README.md` 和 `agv_simulation-main/docs/decimal_2d_version.md`。

## 目录结构

```
├── simple_env.py               # 行走策略 lite 仿真
├── WPPwrapper.py / model_inference.py
├── model/                      # 机器人模型 (bxi_elf3 / unitree_g1) + 行走策略
├── navigation/                 # 机器人导航 + 工厂 AGV 协同
├── agv_simulation-main/
│   ├── src/                    # AGV QoS 调度核心
│   ├── src/bridge/             # AGV ↔ MuJoCo 桥接 + ARC 协同 (推荐)
│   ├── src/bridge_wangting/    # 小数版 0.1m 地图 + 等待直到空闲 A* 桥接 (wangting 工厂)
│   ├── maps/                   # 地图 / 机器人路径
│   └── docs/                   # 验证报告
└── docs/                       # 仓库级文档 / 报告
```

## 依赖

`mujoco`、`onnxruntime`、`numpy`、`mink`（IK）、`loop_rate_limiters`。
用 `tutorial_for_mujoco` conda 环境跑（系统 Python 缺依赖）。

## 子模块 README（摘要 + 完整版链接）

各子模块有自己的 README。下面摘录关键内容，点击链接看完整版。

### 📦 navigation/ — [完整 README](navigation/README.md)

**BXI_ELF3 双足机器人导航**（P2P 走到目标，纯反馈控制，无路径规划）：
- 实时算速度指令驱动 RL 行走策略，`RobotNavigator` 包 `navigation.navigate.Navigator`
- 控制模式：到达 / 原地转身（大角度偏差）/ 行走+转向（tanh 非线性）
- 死区地板 + 最终对齐（arrival_threshold=0.10m，50 目标测试）
- 命令：`python navigation/navigate.py --tx 2.0 --ty 1.0`

### 📦 agv_simulation-main/ — [完整 README](agv_simulation-main/README.md)

**AGV 多机路径规划 + 人形机器人协同**：
- QoS 时空 A\* 调度：预约表（顶点/边）+ 优先级抢占 + 站台队列
- 核心：`src/agv_world_qos.py`（引擎）、`src/agv_map_edit.py`（地图）、`src/agv_world_web.py`（API）
- 推荐入口是 `src/bridge/`（见下）

### 📦 agv_simulation-main/src/ — [完整 README](agv_simulation-main/src/README.md)

**AGV 仿真系统核心**（QoS 调度 + bridge 协同）：
- AGV 调度核心：QoS 时空 A\*（`agv_world_qos.py`）、地图编辑、Web API
- bridge 协同：`planner_node.py`（规划大脑：调度/让行/排队决策）↔ `mujoco_node.py`（物理引擎）
- ARC 避让：预测距离 <1.2m → 停车让行（对穿提前量）→ 恢复/绕行/站台 hold
- 验证：5/10/15 车完成率 93/95/86%，机器人碰撞 0~2（详见 [验证报告](agv_simulation-main/docs/agv_validation_robot.md)）

> 每个子 README 都是模块的权威文档；根 README 只做总览和导航。
