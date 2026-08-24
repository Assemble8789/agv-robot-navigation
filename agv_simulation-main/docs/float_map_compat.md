# 外部浮点地图接入 AGV 系统兼容方案

> 目标：让商用 AGV 地图导出文件（浮点坐标、如 `maps/wangting.cmrn59bz10027gg3srcnhp03d.workflow2.json`）
> 能直接被本仓库的 AGV 规划/仿真工具加载运行，且不修改源文件、不丢失 0.1m 精度。

## 1. 为什么原文件不兼容

以 `wangting...workflow2.json` 为例，它是一个 0.1m 精度的浮点点云地图：

```json
{"width":100.1,"height":100.1,"obstacles":[{"x":-14.9,"y":18.8}, ...],"landmarks":[]}
```

AGV 引擎（`agv_planner.py` / `agv_planner_v2.py` / `agv_world.py` / `agv_world_qos.py` / `agv_world_web.py` / `agv_map_edit.py` / `agv_visualizer.py`）内部是**整数格子 A\***，存在三个不兼容点：

| 问题 | 表现 |
|---|---|
| **JSON 损坏** | 文件有两处字符丢失（`"y".7`、`{"x:18.4}`），`json.load` 直接崩溃 |
| **浮点坐标** | `np.zeros((height,width))`、`range()`、`obs_set` 精确匹配都要求整数；0.1m 浮点喂进去必报错 |
| **缺 landmarks** | `landmarks` 为空，AGV 派车没有装卸货点 |

## 2. 解决方案：共享归一化加载层

新增 **`src/agv_map_common.py`**，提供唯一入口 `load_normalized(map_file, res=0.1, n_landmarks=4)`，
所有 AGV 入口统一改用它替代各自手写的 `json.load`。归一化三步：

```
load_normalized(map_file)
  │
  ├─ 1. 自动修复损坏 JSON（仅内存，绝不写回源文件）
  │       内置 CORRUPTIONS 损坏段补齐表，失败则报清晰错误
  │
  ├─ 2. 浮点坐标 → 整数格子
  │       cell = round(x / res)          # 用 round 不用 floor，
  │                                       # 因为 -14.9/0.1 在二进制里是 -148.999...
  │       去重 → 裁包围盒 → 平移到 (0,0)
  │       附带 _resolution_m / _origin_offset_m 元数据（用于转回真实米制坐标）
  │
  └─ 3. 自动补 landmarks（n_landmarks=4 取四角最近自由格；>4 随机取自由格）
```

- **浮点地图**：量化成 0.1m 整数格，信息零丢失（源数据本身就是 0.1m 离散网格）。
- **原生整数地图**（如 `map_20260730_162111.json`）：原样透传，行为零变化（含负坐标 centered 图）。

### 返回值格式（引擎可直接消化）

```python
{
  "width": 367, "height": 253,          # 整数
  "obstacles": [[0,0], [1,0], ...],      # 整数格，已平移到 (0,0)
  "landmarks": [{"name":"LM001","x":..,"y":..}, ...],
  "_resolution_m": 0.1,                   # 米制元数据
  "_origin_offset_m": [-14.9, -0.4],
  "_normalized": True,
}
```

## 3. 改动范围

所有地图加载入口都从各自 `json.load` 切换为 `load_normalized`：

| 文件 | 改动 |
|---|---|
| `src/agv_map_common.py` | **新增**，归一化加载层（修复/量化/补地标/米制元数据） |
| `src/agv_planner.py` / `agv_planner_v2.py` | 规划器加载改归一化 |
| `src/agv_world.py` / `agv_world_web.py` / `agv_world_qos.py` / `agv_world_qos_alt.py` | 仿真引擎加载改归一化 |
| `src/agv_map_edit.py` / `agv_visualizer.py` | 编辑器/回放器加载改归一化 |
| `src/agv_map_convert.py` | 复用共享层，`--res` 默认 0.1，新增 `--to-meters` |

另外四份 world A* 的 `max_iter` 从 `100000` 调到 `1000000`（0.1m 网格路径变长）。

## 4. 用法

`PY = D:/download/anaconda3/envs/tutorial_for_mujoco/python.exe`（`tutorial_for_mujoco` 环境），
在 `agv_simulation-main/` 下执行。

### 4.1 源文件直跑（推荐，源文件不改）

```bash
# 规划（自动修复损坏 + 归一化 + 自动补地标）
PY src/agv_planner.py maps/wangting.cmrn59bz10027gg3srcnhp03d.workflow2.json --cars 5

# 交互仿真
PY src/agv_world.py maps/wangting.cmrn59bz10027gg3srcnhp03d.workflow2.json

# 地图编辑器打开
PY src/agv_map_edit.py maps/wangting.cmrn59bz10027gg3srcnhp03d.workflow2.json
```

### 4.2 生成干净的原生地图

```bash
# 10 个随机地标（--seed 固定随机），导出原生 map_*.json
PY src/agv_map_convert.py maps/wangting.cmrn59bz10027gg3srcnhp03d.workflow2.json \
    --res 0.1 --out map_wangting_10lm.json --landmarks 10 --seed 42

# 换分辨率（--res 0.5 = 0.5m 一格，格子更少更快但细节更粗）
```

### 4.3 plan 导出真实米制浮点坐标

```bash
# 格子帧 plan → 真实厂房米制坐标（x*0.1 + offset）
PY src/agv_map_convert.py --to-meters maps/plan_v2_xxx.json
```

## 5. 启发式（ALT 默认）

`agv_planner_v2.py` 支持三种启发式，**默认 `alt`（最快且最优）**：

| 启发式 | 求解时间* | 展开节点 | 最优性 |
|---|---|---|---|
| manhattan（原版） | 0.66s | 39,601 | 理论可能次优（对 8 方向高估） |
| chebyshev | 2.16s | 125,617 | ✅ |
| **alt（地标差分，默认）** | **0.60s** | **23,238** | ✅ |

\* 同一 5 车 20 腿任务、367×253 格。ALT 用 6 个地标的运动学 BFS 代价表，
`h = max(Chebyshev, |d(n,Lᵢ)-d(goal,Lᵢ)|)`，既紧又 admissible。

```bash
PY src/agv_planner_v2.py maps/map_wangting_10lm.json --cars 5                 # 默认 ALT
PY src/agv_planner_v2.py maps/map_wangting_10lm.json --cars 5 --heuristic alt # 显式指定
PY src/agv_planner_v2.py maps/map_wangting_10lm.json --cars 5 --heuristic manhattan  # 对比
PY src/agv_planner_v2.py maps/map_wangting_10lm.json --alt-landmarks 10        # 调地标数
```

## 6. 验证结果

- 源文件直跑：自动修复 2 处损坏 → 367×253 格 / 17486 障碍 / 自动补地标 → 规划成功
- 连通性：99.9% 空闲格在同一连通区，所有地标两两可达
- 最优性：逐腿与无预约 BFS 真实最短距离对比，A* 路径长度全部相等（最优）
- 老地图回归：原生整数地图障碍集合逐格一致（透传，零行为变化）
- 米制导出：起点 (-14.9,-0.4) → 终点 (21.7,24.8)，每步 +0.1m，与源数据吻合

## 7. 注意事项 / 限制

- **源文件永不修改**：损坏修复只在内存做；浮点坐标保留原样。
- **0.1m 网格代价**：格子多（约 9 万），单条长路径 A* 约几秒；想更快用 `--res 0.5` 或 `--heuristic alt`。
- **AGV 坐标与真实米制**：引擎内部是"相对格"坐标（已平移到 0,0），真实厂房坐标在 `_origin_offset_m` 里；
  要真实米制 plan 用 `--to-meters` 导出。
- **新损坏段**：若未来有新文件损坏且不匹配 `CORRUPTIONS`，需按障碍排列规律补充坏→好映射（见 `agv_map_common.py`）。
- **bridge/ 不在范围**：AGV×机器人协作层用米制机器人坐标，另有自己的 `load_map`，如需支持再单独接入。
- 三处 A*（planner / world / qos）仍各自独立，改动需手动同步（见仓库 AGENTS.md）。