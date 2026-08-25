# AGV 多机路径规划 + 人形机器人协同系统

AGV（自动导引运输车）仿真实验平台，支持地图编辑、多车路径规划、实时仿真、Web API 控制，
以及与**人形机器人（elf3）**协同的避让闭环。

## 目录结构

```
├── src/                      # 核心脚本
│   ├── agv_map_edit.py       地图编辑器
│   ├── agv_planner.py        离线批量规划器
│   ├── agv_world.py          实时仿真引擎
│   ├── agv_world_web.py      Web API 封装
│   ├── agv_world_qos.py      QoS 优先仿真引擎 (AGV 时空 A* 本体)
│   ├── agv_map_common.py     地图归一化加载层 (外部浮点地图 → 0.1m 整数格)
│   ├── agv_map_convert.py    外部地图转换器 (--res/--landmarks/--to-meters)
│   ├── agv_visualizer.py     动画回放
│   ├── bridge/               AGV ↔ MuJoCo 桥接 + 人形机器人协同 (推荐入口)
│   │   ├── planner_node.py   规划节点: 包 AGVWorld, 调度/让行/排队
│   │   ├── mujoco_node.py    仿真节点: MuJoCo 渲染 + 机器人, /clock 时间主
│   │   ├── demo_bridge.py    入口: 建 Bus + 两节点
│   │   ├── validate_agv.py   验证矩阵 (车数×路径, markdown 报告)
│   │   ├── _record_trajectory.py / _playback.py   播放回放
│   │   └── ...               (详见 src/README.md)
│   └── bridge_wangting/      小数版 0.1m 地图桥接 (等待直到空闲 A* + 膨胀 + 机器人)
│       ├── astar.py          修正 ALT 时空 A* / 等待直到空闲 A* (时间折叠)
│       ├── run_wangting.py   入口: --demo5/--cars/--robot/--pure/--headless
│       ├── pure_agv.py       离线预规划 (t=0 同时发车)
│       ├── record_traj.py    录制位置 → 2D 回放
│       └── ...               (详见 src/bridge_wangting/README.md)
├── test/             # 测试脚本
├── maps/             # 地图 / 规划 / 图片数据
├── docs/             # 设计文档 + 验证报告 (agv_validation_robot.md, decimal_2d_version.md)
├── requirements.txt
└── README.md
```

> **推荐入口是 `src/bridge/`**：AGV（QoS 时空 A\*）和人形机器人走同一套 ROS-topic 风格的
> pub/sub 总线，带完整的让行/排队/站台等待闭环。纯 AGV 的核心算法在 `src/` 下（QoS 等）。

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 所有命令在激活 venv 后使用 python 运行
python src/agv_map_edit.py
```

> ⚠️ **必须使用 venv 的 Python**，系统 Python 缺少依赖且 numpy / matplotlib 版本冲突。

## 核心功能

### 1. 地图编辑器

```bash
python src/agv_map_edit.py                           # 新建 15×10
python src/agv_map_edit.py --width 20 --height 15     # 指定尺寸
python src/agv_map_edit.py maps/map_20260113_153602.json  # 编辑已有地图
```

**操作：** 左键放置障碍物 | 右键删除障碍物 | Ctrl+左键放置停靠点 | Ctrl+右键删除 | Save 保存到 `maps/map_*.json` + `map_*.png`

### 2. 离线规划器

```bash
python src/agv_planner.py maps/map_20260113_153602.json --cars 5
python src/agv_planner.py maps/map_20260113_153602.json --tasks "LM001,LM008;LM003,LM007"
```

输出 `maps/plan_*.json`。

### 3. 动画回放

```bash
python src/agv_visualizer.py maps/plan_20260128_152331.json
```

### 4. 实时仿真

```bash
python src/agv_world.py maps/map_20260113_153602.json
```

**CLI：** `add <carID> <landmark>` | `del <carID>` | `move <carID> <goalLM> [-v]` | `interval <ms>` | `exit`

### 5. Web API（端口 8001）

```bash
python src/agv_world_web.py maps/map_20260113_153602.json
```

| 端点 | 方法 | Body |
|---|---|---|
| `/add` | POST | `{"carID": "car01", "landmark": "LM001"}` |
| `/del` | POST | `{"carID": "car01"}` |
| `/move` | POST | `{"carID": "car01", "goalLM": "LM008", "verbose": true}` |
| `/interval` | POST | `{"interval": 100}` |
| `/status` | GET | — |

### 6. QoS 仿真

优先级：CRITICAL > HIGH > MEDIUM > LOW。高优先级可抢占低优先级车辆的预约并触发重规划。

```bash
# CLI 用法（示例）
add car01 LM002 LOW
add car02 LM001 CRITICAL
move car01 LM005 -v HIGH
```

### 7. AGV + 人形机器人协同 (src/bridge/)

把 **QoS 规划器 (`AGVWorld`)** 和 **MuJoCo 仿真 + elf3 人形机器人** 解耦成两个节点，
通过进程内 ROS-topic 风格 pub/sub 总线通信：

```
PlannerNode (AGVWorld)  ──/planner/path──►  MujocoNode (MuJoCo 渲染)
      ▲                                     │
      └──────────────/clock ◄───────────────┘  (MuJoCo 是时间主)
            ◄────── /sim/* 反馈 (位置/碰撞/障碍/机器人)
```

**两个节点各干什么**:
- **`planner_node.py`** — 规划节点: 包 `AGVWorld`(QoS 时空 A\*), 订阅 `/clock` 同步时间;
  处理 add/move/route、到站派发、**进站队列**、**AGV 让行机器人闭环 (ARC)**、站台等待。
- **`mujoco_node.py`** — 仿真节点: MuJoCo 渲染 + elf3 机器人, 发布 `/clock`(时间主)
  和每帧反馈 (位置/速度/碰撞/障碍距离/机器人位姿路径)。

```powershell
# 前置: cd 到 agv_simulation-main, PY = D:/download/anaconda3/envs/tutorial_for_mujoco/python.exe

# 带机器人窗口 (AGV + 机器人一起跑)
PY src/bridge/demo_bridge.py --robot --demo5

# 验证矩阵 (5/10/15 车 × 3 路径, headless, 输出 markdown 报告)
PY src/bridge/validate_agv.py --cars 5,10,15 --paths 3 --reps 1 --robot 1 --out docs/agv_validation_robot.md

# 播放回放 (记录 → 点一下过一秒)
PY src/bridge/_record_trajectory.py --cars 5 --path 0 --steps 150 --out docs/traj_5_p0.json
PY src/bridge/_playback.py docs/traj_5_p0.json
```

**ARC (AGV-Robot Closure) 避让闭环** — 机器人路径横穿站台区, AGV 进出站必经:
- 预测车与机器人未来 2s 最近距离 < `ROBOT_BLOCK_DIST=1.2m` → 停车让行
  (1.2m 对穿提前量: 相对速度 ~1.22m/s, 0.9m 只够 0.74s < 检查间隔+延迟)
- 机器人走远 → 恢复; 让行 3s 超时 → 绕行; 已到站车挡机器人目标点 → 挪开
- 机器人目标站台被 AGV 占用 → 机器人 hold 等待 (进站 AGV 优先, 防互相等死锁)
- **机器人终点非站台** (humanoid_plan.json 终点 = 空位): 避免车被随机路线派去
  机器人终点站 → 消除"车进不了站"死锁 (5车P1 90→100%, 15车P1 63→100%)

**验证结果** (`ROBOT_BLOCK_DIST=1.2` + 终点非站台, 详见 `docs/agv_validation_robot.md`):

- 5 车 93% / 10 车 95% / 15 车 86%, **机器人碰撞 0~2 次** (基线 0~13, 对穿清零)

### 8. 小数版地图 + 等待直到空闲 A* (src/bridge_wangting/)

商用 AGV 地图导出文件（0.1m 浮点点云）→ 整数格 A* → AGV↔MuJoCo bridge（含机器人）：

- **归一化** `agv_map_common.load_normalized`：修损坏 JSON → 浮点转 0.1m 整数格（round 防
  二进制误差）→ 裁包围盒平移到 (0,0) → 自动补地标，附 `_resolution_m`/`_origin_offset_m`
- **等待直到空闲 A\*** `bridge_wangting/astar.astar_waitfree`：状态键 `(x,y,dir)` 不含 t，
  移动到被占邻居 → 原地算最早空闲时刻跳过去（等待隐式）→ 时间维折叠，0.1m 大图不爆状态
  （修复原 `_astar_core` 缺 `best_cost_to_state` 剪枝导致的大图 max_iter 失败）
- 障碍按 AGV 半径膨胀（球不穿墙）、米制帧 cell↔meter 精确换算、机器人避窄走廊、2D 录制回放

```powershell
# PY = D:/download/anaconda3/envs/tutorial_for_mujoco/python.exe

# 交互可视化 (AGV + 机器人 + ARC 让行)
PY src/bridge_wangting/run_wangting.py --robot --demo5 --speed 3

# headless 离线预规划 (t=0 同时发车, 5 车 ~2-8s)
PY src/bridge_wangting/run_wangting.py --pure --demo5 --headless --steps 150

# 录制 → 2D matplotlib 回放 (不卡)
PY src/bridge_wangting/record_traj.py --demo5 --robot --steps 150 --out docs/traj_wangting_v2.json
PY src/bridge/_playback.py docs/traj_wangting_v2.json
```

> 详细说明见 `src/bridge_wangting/README.md` 与 `docs/decimal_2d_version.md`。

### 9. 测试

```bash
python test/test_multi_agv.py                         # 多车并发（stdin 控制，需 GUI 环境）
python test/test_qos_scenario.py --mode both           # QoS vs 原始调度对比
python test/test_web_api.py                           # API 测试（需先启动 agv_world_web.py）
```

## 算法说明

项目包含三个独立实现的 A* 算法，**互不导入**：

| 位置 | 变体 | 特性 |
|---|---|---|
| `agv_planner.py` | 标准 A\* + 时空预约 | 8 方向，无朝向，逐车预规划（起始间隔 2 tick） |
| `agv_world.py` / `_web.py` | 运动学 A\* | 4 方向 + 转向/等待/前进，实时多车冲突处理 |
| `agv_world_qos.py` | 运动学 A\* + reservation_info | 预约关联优先级，高优先级可抢占 |

## 已知问题

**核心算法 (src/)**
- **三个独立 A\* 实现** — 无共享模块，修改需手动同步。
- **`agv_world_web.py` 空 `except Exception`** — 静默吞掉所有动画错误。
- **三处 `except queue.Empty` 无日志** — 调试时需手动添加。
- **停车预约 1000 tick** — 让行/排队停车时预约 1000 tick 保护停格（有意为之，避免别的车穿入）。
- **`.gitignore` 覆盖全部 `*.json`** — 机器人路径 `humanoid_plan.json` 等配置改动不入库（本地生效）。
- **地图坐标系** — 障碍物含负坐标时自动切换为中心原点，否则 `(0, 0)` 原点。

**AGV+机器人协同 (src/bridge/)**
- **模式C 碰撞 0~2 次** — 机器人要进的站台 (LM002 via) 被已到站 AGV 占用, 机器人 hold 5s 超时后撞上 (结构冲突, 非对穿)。
- **5 车 P0/P2 偶发 90%** — 时序波动: planner/sim 两线程非严格同步, 让行检查时机偶发让某车差几秒没在 150s 内完成 (纯 AGV 全 100%)。
- **15 车完成率 86%** — AGV-AGV 拥堵 (站台过热/连锁重规划失败), 与机器人无关。方向: 路线生成避撞 (每站 ≤3 车)。
- **`humanoid_plan.json` 终点非站台** — 若改回站台终点, 随机路线把车派去机器人终点站会死锁。
