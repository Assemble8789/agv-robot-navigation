# bridge_wangting — 小数版 0.1m 地图桥接 (AGV + 机器人)

把商用 AGV 地图导出文件（0.1m 浮点点云，如 `maps/wangting.cmrn59bz10027gg3srcnhp03d.workflow2.json`）
无损接入整数格 A*，并在 AGV↔MuJoCo bridge 里跑通（含 elf3 人形机器人、ARC 让行）。

零侵入：不改任何原 `bridge/` 文件，通过继承 + 运行时打模块补丁复用原节点。

## 文件

| 文件 | 作用 |
|---|---|
| `astar.py` | **等待直到空闲 A\*** `astar_waitfree`（时间折叠）+ 修正 ALT `astar_alt_fixed` |
| `mapframe.py` | 米制帧：`cell ↔ meter` 精确双射（`res=0.1, offset=(-14.9,-0.4)`） |
| `map_scene.py` | 从地图生成障碍盒（0.1m 占用格按行聚类）、车库、MuJoCo 场景 XML |
| `wangting_nodes.py` | `WangtingPlanner`/`WangtingMujoco`：帧换算 + 障碍膨胀 + 出生不闪现 + A\* 再 patch |
| `run_wangting.py` | 入口（demo / cars / robot / pure / headless） |
| `pure_agv.py` | 离线预规划（t=0 同时发车） |
| `robot_scene_wangting.py` | elf3 人形 + wangting 工厂场景文本合并 |
| `robot_path.py` | 机器人路径（细格 A* 绕障碍 + 避开窄走廊 <1.0m） |
| `record_traj.py` | 录制每整秒位置 → `bridge/_playback.py` 2D 回放 |

## 核心算法：等待直到空闲 A*（时间折叠）

原 `agv_world_qos_alt._astar_core` 状态含时间 `(x,y,dir,t)`，0.1m 大图上同一格同一朝向被
无数不同 t 反复 push → 撞 `max_iter=1e6` 返回 None（"could not find path"）。15 车小图
状态空间小不触发，大图必然触发。

`astar_waitfree` 修复：
- **状态键 = `(x,y,dir)` 不含 t**；`g[(x,y,dir)]` = 最早到达时刻
- 移动到被占邻居 → 原地算**最早空闲时刻**直接跳过去（等待隐式，不生成等待态）
- 状态数从无限时间维塌缩到 `(x,y,dir)` ~272k

效果（0.1m 网格，差分表缓存后）：

| 项 | 结果 |
|---|---|
| 5 车离线预规划 | 2.2~8s（此前 53s） |
| 随机 25 对自由格 | 100% 成功，avg 0.27s/对 |
| 15 对累积预约（模拟多车） | 100% 成功，4.2s |
| 端到端（AGV+机器人） | 0 碰撞、机器人没摔 |

## 运行（PY = D:/download/anaconda3/envs/tutorial_for_mujoco/python.exe）

```powershell
# 交互可视化 (AGV + 机器人 + ARC 让行)
PY src/bridge_wangting/run_wangting.py --robot --demo5 --speed 3

# headless (纯 AGV)
PY src/bridge_wangting/run_wangting.py --demo5 --headless --steps 120

# 离线预规划 (t=0 同时发车)
PY src/bridge_wangting/run_wangting.py --pure --demo5 --headless --steps 150

# 源文件直跑 (自动修损坏+归一化)
PY src/bridge_wangting/run_wangting.py --map maps/wangting.cmrn59bz10027gg3srcnhp03d.workflow2.json --cars 3 --headless --steps 60

# 录制 → 2D 回放
PY src/bridge_wangting/record_traj.py --demo5 --robot --steps 150 --out docs/traj_wangting_v2.json
PY src/bridge/_playback.py docs/traj_wangting_v2.json
```

## 关键参数

- `--inflate`：障碍膨胀半径（默认 AGV_RADIUS=0.2m，0 关闭）→ A* 只走球能放下的格，球不穿墙
- `--robot-wp "LM006,LM008,LM002,LM010"`：机器人路径地标，窄走廊段自动跳过
- `--cars N --seed S`：N 车随机链

详见 `agv_simulation-main/docs/decimal_2d_version.md`（二维版改动）与仓库根/子 README。