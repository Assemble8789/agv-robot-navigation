# 工厂 AGV 调度 + 人形机器人导航 —— 技术总览

## 总体目标

工厂环境（静态货架/设备已知，多辆 AGV 按预规划时空轨迹搬运）中，**BXI_ELF3 人形机器人**从 LM005(1,2) 出发 → LM002(9,7) → LM006(5,1) 完成巡检任务，**不改全局路径**，用**预测式让行（减速爬行 + 温和侧移）**动态避开 AGV；同时 4 辆 AGV 互相避让、不穿模。

## 已实现

- ✅ **AGV 调度规划**（`agv_planner_v2.py`）：时空 A* + 双向边预约 + 链式任务 + 站点停留，输出 plan_v2 轨迹
- ✅ **AGV 运行时**：Catmull-Rom 样条平滑、阻塞绕行、移动-移动互斥，mocap 控制位置（`run_factory_full.py` AGV 层）
- ✅ **机器人路径规划**：空间 A*（障碍膨胀 + AGV 停留点阻塞）→ `humanoid_plan.json`（`plan_spatial_path` + `correct_path`）
- ✅ **机器人行走控制**：RL 策略 + PD，转向/速度剖面/死区下限（`navigate.py` Navigator）
- ✅ **Level 2 预测式让行（ARC）**：时空对齐检测 → vx 减速爬行 + 温和 vy 侧移，**实测 0 AGV 碰撞、0 摔倒、全程到达**（A/B：ARC 关 77 次碰撞）
- ✅ **主仿真闭环**：真实碰撞检测（`data.contact`）+ 摔倒检测 + `--headless`/`--no-arc` 诊断开关

---

## 1. 数据流（当前在用）

```
agv_planner_v2.py ──► maps/plan_v2_*.json（AGV 时空轨迹 (t,x,y)）
navigate_factory.py ──► maps/humanoid_plan.json（机器人空间 waypoint）

run_factory_full.py（主仿真）
  ├─ AGV 层：clock 推进 → Catmull-Rom 样条 → 排斥/互斥 → mocap 写入
  ├─ 机器人层：SpatialNav（复用 navigate.py）→ RL 策略(model_normal.onnx) + PD
  ├─ Level 2 ARC：时空检测 → vx 减速让行 + 温和 vy
  └─ 碰撞检测（data.contact，含 AGV + 货架）+ 摔倒检测
```

---

## 2. AGV 规划：`agv_planner_v2.py`

**时空 A\*（`astar_with_time`）**，状态 `(x,y,t)`，8 方向，曼哈顿启发式：

- **双向边预约**：预约表同时存顶点 `(x,y,t)` 和边 `(x1,y1,t,x2,y2)`；扩展时正反向边都查 → **A→B 占用也阻塞 B→A**，解决 AGV 对穿
- **dwell-aware 目标**：到达站点后停留 1 tick（1s），规划时所有停留 tick 都须空闲，否则等待重试
- **链式任务**：每车一条链 `start,stop1,stop2,stop3`，逐段规划，前段预约全注册进全局表
- **逐车顺序规划 + 起始错峰**（stagger=2），后车自动避开前车

## 3. AGV 运行时：`run_agv_factory.py` / 主仿真 AGV 层

- **Catmull-Rom 样条**：网格 waypoint → 连续曲线（200 点），逐点做静态障碍排斥（`SAFE_DIST=0.8m`）防擦货架
- **逐车时钟**：clock 按 `dt·SPEED·AGV_SPEED` 推进；下一格被静止车占用则**沿垂直方向 lateral 偏移绕行**
- **移动-移动互斥**：两车距离 <0.7m 互相推开
- **mocap 控制**：位置写 `data.mocap_pos`，物理不演化；但 `contype=1` 可碰撞 → AGV 是实心障碍

---

## 4. 机器人路径规划：`navigate_factory.py`

只用了两个函数：

- **`plan_spatial_path`**：纯空间 A\*（0.25m 细网格，8 方向，octile 启发式）
  - 障碍按 `obstacle_margin` 格膨胀（当前 margin=3 ≈ 0.75m）
  - **AGV 停留点/终点当静态障碍**（运动中的 AGV 忽略——它们比机器人快 4 倍，机器人到时早走了）
- **`correct_path`**：闭环碰撞修正——waypoint 被解析距离推离货架/地标停车区，再 Catmull-Rom 重平滑

## 5. 机器人底层控制：`navigate.py` Navigator

- 连续非线性转向 `omega = turn_speed·tanh(2·heading_error)`
- 速度剖面：远距恒速 / 近距二次减速 / `heading_penalty` 防冲过头
- 原地转向滞回（>60° 先转再走），低通滤波（α=0.7）
- **死区下限** `min_fwd_speed=0.25`：策略对 <0.25m/s 前向指令不响应，保底避免停住

---

## 6. 主仿真：`run_factory_full.py`

### 6.1 每帧（10× 加速）

```
AGV 位置（clock→样条→排斥→mocap）
机器人位姿 → nav.update() → (vx,vy,omega)
ARC 预测让行（时空检测→vx 爬行 + 温和 vy）
货架盒体排斥 + AGV 近距排斥（兜底）
10 次物理步（PD + 策略推理）
真实碰撞检测 + 摔倒检测（torso z<0.6）
```

### 6.2 Level 2：预测式让行（vx 减速 + 温和 vy）

**思想**：检测到 AGV 将横穿路径时，**减速爬行（vx→0.2）让 AGV 先通过横穿点**，同时加**温和 vy 侧移（0.5）**处理需要横向让开的场合。保持机器人基本在货架安全的路径上，不滑向两侧。

- **时空对齐检测**：采样 AGV 未来 2s 位置（`clocks + t_ahead·AGV_SPEED`），把机器人**沿真实 waypoint 路径**投影（直线投影会漏——机器人转弯），取未来最小距离 `<0.7m` 且**逼近**（距离在缩小）才触发。滤掉平行经过、远离的过度反应（触发 7→3 次）
- **执行机制**：`vx = min(vx, 0.2)`（爬行让行）+ `vy = sign·0.5`（温和侧移），持续 `hold=2.5s`
- **货架防护**：基于**真实货架盒体几何**的排斥（早期格点排斥有 bug——会把人推进货架）。纯 vy 侧滑在 2.3m 窄走廊是几何死局（躲 AGV 撞货架 / 躲货架撞 AGV），**vx 让行**才是正解
- **自动调参**：`navigation/tune_arc.py` 用代价函数（含货架碰撞惩罚）坐标下降，`--arc "k=v"` 覆盖参数。当前默认参数（调优）：`vy_dodge=0.5, vx_yield=0.2, lat_thresh=0.7, hold=2.5`

### 6.3 已修复的关键 Bug

| # | 问题 | 修复 |
|---|---|---|
| 1 | 触发用欧氏距离<0.9m，AGV 前方 4m 横穿不触发 | 改时空对齐检测（沿路径投影未来最小距离） |
| 2 | AGV 时钟双倍推进（跑 2 倍速），预测全错位 | 只保留一条 clock 推进 |
| 3 | `clocks+t_ahead` 把 sim 秒当 plan 秒（超前 1.43×） | 改 `+t_ahead·AGV_SPEED` |
| 4 | 平行经过 / 远离的 AGV 误触发（过度反应，触发 7 次） | 加**逼近过滤**（未来距离在缩小才躲）→ 3 次 |
| 5 | 纯 vy 侧滑在 2.3m 窄走廊几何死局（躲 AGV 撞货架 472 次 / 躲货架撞 AGV） | 改 **vx 减速爬行让行** + 温和 vy → 货架擦碰 7 次 |
| 6 | 货架排斥用格点坐标，把机器人往货架里推 | 改用**真实盒体几何**（`OBS_BOXES`） |
| 7 | vy 命令 0.2~0.4 在策略死区（几乎无侧移） | 实测策略 vy 增益 ~1.0、死区 ~0.4，命令提到 >0.4 |

### 6.4 验证（headless A/B）

| 指标 | ARC ON | ARC OFF |
|---|---|---|
| AGV 碰撞 | **0** | 77 |
| 货架擦碰 | 7（亚毫米） | — |
| 最小间距 | 0.523m | 0.177m |
| 到达 LM006 | ✅ | ✅ |
| 耗时 | ~40s | 33s |

---

## 7. 运行命令

```bash
# 生成 AGV 计划（链式任务，每车 起点,站1,站2,站3）
python agv_simulation-main/src/agv_planner_v2.py agv_simulation-main/maps/map_*.json \
  --tasks "LM000,LM002,LM004,LM001;LM003,LM001,LM000,LM002;LM001,LM004,LM003,LM000;LM002,LM000,LM003,LM004"

# 仅 AGV 可视化
python navigation/run_agv_factory.py

# 主仿真（AGV + 机器人 + ARC）
python navigation/run_factory_full.py

# headless 诊断 / 关 ARC 对比 / 覆盖参数
python navigation/run_factory_full.py --headless [--no-arc] [--arc "vy_dodge=0.5,vx_yield=0.2"]

# 自动调参（坐标下降，每轮约 10 分钟）
python navigation/tune_arc.py --rounds 3

# 对照版：纯停不让（实测会撞，用于对比）
python navigation/run_factory_full_stop.py --headless
```

## 8. 已知代价

- 让行需减速 → 全程约 40s（vs 无 ARC 33s，慢 ~7s）；3 次真威胁让行，另有 7 次亚毫米级货架擦边（物理上等于没碰）
- 循环结束后 AGV 全停在 LM004，机器人返回段无相遇场景，未覆盖
