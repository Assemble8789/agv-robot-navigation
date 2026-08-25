# 小数版二维地图 — 相对原始版改了什么

> **Git 定位**：HEAD = `5fee0f6 小数版二维地图`，对比基线 = 原始仓库 `13b8f40`。
> `src/bridge_wangting/`（AGV↔MuJoCo bridge 阶段 1/2）**不在此版**，是后续未提交的独立工作。
>
> **一句话**：让 0.1m 浮点小数地图（如 `wangting...workflow2.json`）能直接喂给原本只吃
> 整数格子地图的 AGV 引擎 —— 新增统一归一化层，8 个入口全部接入，并顺手加了 8 方向
> 最优启发式（Chebyshev）与 ALT 差分启发式。

---

## 1. 改动文件清单（11 个，+550/−139）

### 新增（3 个）

| 文件 | 作用 |
|---|---|
| `src/agv_map_common.py` | **核心**：统一地图归一化加载层 `load_normalized()` |
| `src/agv_map_convert.py` | 外部地图 → 原生 `map_*.json` 转换器（`--res`/`--landmarks`/`--seed`/`--to-meters`） |
| `docs/float_map_compat.md` | 归一化方案说明文档 |

### 修改（8 个，全部是"加载入口接入归一化层"）

| 文件 | 改动点 |
|---|---|
| `src/agv_world.py` / `src/agv_world_web.py` | `AGVWorld.__init__`：`json.load` → `load_normalized`；`max_iter` 100000→1000000 |
| `src/agv_world_qos.py` | 同上（QoS 引擎加载归一化 + max_iter 提升） |
| `src/agv_world_qos_alt.py` | 3 处地图加载改归一化；max_iter 提升 |
| `src/agv_planner.py` | `plan_paths` 加载改归一化，障碍解析简化（已归一化为整数） |
| `src/agv_planner_v2.py` | 加载改归一化 + **新增启发式**（见 §3） |
| `src/agv_map_edit.py` | `load_map` 改归一化（兼容浮点地图打开） |
| `src/agv_visualizer.py` | 地图解析改归一化 |

> 原版整数地图（15×10 等）走 `load_normalized` 时**原样透传，行为零变化**。

---

## 2. 新算法：统一归一化加载层 `load_normalized`

原来 8 个入口各自 `json.load` + 各自解析，对浮点/损坏/空地标地图无处理。新增唯一入口：

```
load_normalized(map_file)
  ├─ 1. 修复损坏 JSON（仅内存，绝不写回源文件）
  │        内置 CORRUPTIONS 损坏段补齐表，失败则报清晰错误
  ├─ 2. 浮点坐标 → 0.1m 整数格
  │        cell = round(x / 0.1)        # round 而非 floor，规避 -14.9/0.1 的二进制误差
  │        去重 → 裁占用包围盒 → 平移到 (0,0)
  │        附带 _resolution_m / _origin_offset_m 米制元数据（供转回真实米制）
  └─ 3. 自动补地标
         空 landmarks → 四角最近自由格生成 LM001…；可随机 N 个
```

返回值就是引擎直接消化的原生格式：整数 `width/height`、`[[int,int],…]` 障碍、带地标。
`cell_to_meters(x,y,res,offset)` 提供格→米换算（`--to-meters` 导出 plan 用）。

**精度**：源数据本身是 0.1m 离散点，整数格与浮点完全等价，零信息丢失。

---

## 3. 新算法：agv_planner_v2 的启发式升级

原版 8 方向移动、每步代价 1，但用 Manhattan 启发式 —— **对 8 方向等代价会高估**（真实最优是
Chebyshev），不满足 admissible，理论上可能返回次优路径。本版：

| 启发式 | 公式 | 说明 |
|---|---|---|
| `manhattan`（原版改名） | `|dx|+|dy|` | 保留作对比，可能次优 |
| `chebyshev`（新增） | `max(|dx|,|dy|)` | 8 方向等代价的真实最优（admissible+consistent） |
| `alt`（**默认**） | `max(Chebyshev, 地标差分)` | 用 `agv_world_qos_alt.build_differential` 的运动学 BFS 地标表，障碍感知更紧 |

- `astar_with_time` 增加 `hfun` / `stats` 参数（可注入启发式、统计展开节点数）
- CLI：`--heuristic {manhattan,chebyshev,alt}`、`--alt-landmarks N`
- 输出追加耗时与展开节点统计

**实测**（0.1m 网格 5 车×2 段）：

| 启发式 | 求解时间 | 展开节点 | 最优性 |
|---|---|---|---|
| manhattan | 0.66s | 39,601 | 理论可能次优 |
| chebyshev | 2.16s | 125,617 | ✅ |
| **alt（默认）** | **0.60s** | **23,238** | ✅ |

---

## 4. 其他改动

- **`max_iter` 100000 → 1000000**（4 个 world 引擎）：0.1m 网格路径变长，防止 A* 提前放弃
- **`agv_visualizer.py` / `agv_map_edit.py`** 兼容浮点地图：现在能直接打开 `wangting...json` 查看/编辑

---

## 5. 验证结果

| 项 | 结果 |
|---|---|
| 源文件直跑规划（自动修损坏→归一化→补地标→A*） | ✅ |
| 0.1m 网格连通性 | ✅ 99.9% 空闲格连通，地标两两可达 |
| 路径最优性 | ✅ 逐腿与 BFS 真实最短对比全部相等 |
| 老地图回归（15×10） | ✅ 障碍集合逐格一致，透传零变化 |
| `--to-meters` 米制导出 | ✅ LM001(-14.9,-0.4)→LM004(21.7,24.8)，每步 +0.1m |

---

## 6. 运行命令

```powershell
# PY = D:/download/anaconda3/envs/tutorial_for_mujoco/python.exe

# 源文件直跑（自动修复+归一化+补地标）
PY src/agv_planner.py maps/wangting.cmrn59bz10027gg3srcnhp03d.workflow2.json --cars 5

# 生成 10 随机地标原生地图
PY src/agv_map_convert.py maps/wangting.cmrn59bz10027gg3srcnhp03d.workflow2.json --res 0.1 --out map_wangting_10lm.json --landmarks 10 --seed 42

# ALT 启发式规划（默认）
PY src/agv_planner_v2.py maps/map_wangting_10lm.json --cars 5
PY src/agv_planner_v2.py maps/map_wangting_10lm.json --cars 5 --heuristic chebyshev

# 导出米制浮点 plan
PY src/agv_map_convert.py --to-meters maps/plan_v2_xxx.json
```