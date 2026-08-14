# AGV World QoS 动态优先级路径规划

## 概述

`agv_world_qos.py` 是基于 `agv_world.py` 的增强版本，引入了动态 QoS（Quality of Service）优先级调度机制。原有的时间A*+预约算法在多车并发时采用"先到先得"的策略，新版本在此基础上增加了基于优先级的路径抢占机制，实现了更灵活的资源调度。

---

## 一、原有算法回顾

### 1.1 时间A*算法

原始算法使用 Kinematic A* 进行路径规划，状态空间为 `(x, y, dir, t)`，考虑车辆的方向和到达时间。

```python
def astar_with_time(width, height, obs_set, start, goal, start_time, start_dir_str, reservations, x_min=0, y_min=0):
    # 状态: (x, y, dir_idx, time)
    # 动作: 等待、左转、右转、前进
    # 启发式: 曼哈顿距离
```

### 1.2 预约系统

- **顶点预约**: `(x, y, t)` - 某时刻某格子被占用
- **边预约**: `(x1, y1, t, x2, y2)` - 某时刻某条边被占用

### 1.3 局限性

- 所有车辆公平竞争，无优先级区分
- 高优先级任务无法优先使用资源
- 紧急任务可能因路径被占用而延迟

---

## 二、QoS增强设计

### 2.1 QoS等级定义

```python
QoS_LEVELS = {
    'CRITICAL': 4,  # 最高优先级，用于紧急任务
    'HIGH': 3,      # 高优先级
    'MEDIUM': 2,    # 默认优先级
    'LOW': 1,       # 低优先级
    'IDLE': 0       # 空闲/已降级
}
```

### 2.2 数据结构变更

| 数据结构 | 原版 | QoS版 |
|---------|------|-------|
| `reservations` | `set` | `set` (不变) |
| `reservation_info` | 无 | `dict` - 预约→(car_id, qos_level) |
| `car_reservations` | 同上 | 同上 |
| `car_paths` | 无 | `dict` - 记录当前规划路径 |
| `car_waiting` | 无 | `dict` - 标记等待重规划 |
| `pending_replan` | 无 | `queue` - 待重规划队列 |

### 2.3 预约信息增强

```python
# 原有预约
self.reservations = set()
# {(x, y, t), (x1, y1, t, x2, y2), ...}

# 新增预约信息索引
self.reservation_info = {}
# {
#   (x, y, t): (car_id, qos_level),
#   (x1, y1, t, x2, y2): (car_id, qos_level),
#   ...
# }
```

---

## 三、核心算法改进

### 3.1 A*冲突检测

**原版**: 遇到预约直接跳过该节点
```python
if (nx, ny, nt) in reservations:
    continue
```

**QoS版**: 记录冲突但继续搜索，用于后续抢占判断
```python
res_vertex = (nx, ny, nt)
if res_vertex in reservation_info:
    existing_car, existing_qos = reservation_info[res_vertex]
    # 记录冲突但继续搜索
    if (res_vertex, existing_car, existing_qos) not in conflicts:
        conflicts.append((res_vertex, existing_car, existing_qos))
```

### 3.2 抢占机制流程

```
┌─────────────────────────────────────────────────────────────┐
│                     move_car(car_id, goal)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. 规划路径 (astar_with_time)                              │
│     - 返回 (path, conflicts)                               │
│     - conflicts: 与其他车辆预约冲突的列表                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 分析冲突 (QoS比较)                                      │
│                                                             │
│  for conflict in conflicts:                                 │
│      if current_qos > existing_qos:                        │
│          preempted_cars.add(existing_car)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 执行抢占 (高QoS抢占低QoS)                               │
│                                                             │
│  for car in preempted_cars:                                 │
│      - 清除其未来预约                                       │
│      - QoS 降一档 (e.g., HIGH → MEDIUM)                     │
│      - 设置 waiting_replan = True                          │
│      - 加入 pending_replan 队列                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 注册新路径预约                                          │
│     - 当前车辆按新QoS注册预约                               │
│     - 更新 reservation_info                                 │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 抢占时的QoS降级规则

```python
# 被抢占车辆降一档
new_qos = max(0, existing_qos - 1)  # 最低降到 IDLE(0)
preempted_car['qos'] = new_qos
preempted_car['qos_name'] = QoS_NAMES[new_qos]
```

---

## 四、重规划机制

### 4.1 待重规划队列

```python
self.pending_replan = queue.Queue()
# 队列元素: (car_id, old_qos, new_qos)
```

### 4.2 自动重规划流程

```python
def process_pending_replans(self):
    while not self.pending_replan.empty():
        car_id, old_qos, new_qos = self.pending_replan.get()
        if car_id in self.cars and car['waiting_replan']:
            # 使用降级后的QoS重新规划
            self.move_car(car_id, car['current_goal'])
```

### 4.3 重规划触发时机

每帧可视化更新时自动调用：
```python
def update(frame):
    # ... 命令处理 ...
    self.process_pending_replans()  # 新增
    self.world_time += 1
```

---

## 五、任务完成QoS恢复

### 5.1 规则

任务完成后：
- 如果当前 QoS < HIGH，则恢复为 MEDIUM
- 如果已是 HIGH 或 CRITICAL，保持不变

```python
if self.world_time == car_data['last_time']:
    # 任务到达处理...

    if car_data['qos'] < QoS_LEVELS['HIGH']:
        car_data['qos'] = QoS_LEVELS['MEDIUM']
        car_data['qos_name'] = 'MEDIUM'
```

---

## 六、CLI命令扩展

### 6.1 新增命令

| 命令 | 格式 | 说明 |
|------|------|------|
| `add` | `add <carID> <landmark> [qos]` | 添加车辆时可指定QoS |
| `move` | `move <carID> <goal> [-v] [qos]` | 移动时可临时提升QoS |
| `setqos` | `setqos <carID> <qos>` | 动态修改车辆QoS |

### 6.2 QoS级别

```
CRITICAL  - 最高优先级，可抢占所有其他级别
HIGH      - 可抢占 MEDIUM、LOW
MEDIUM    - 可抢占 LOW（默认）
LOW       - 最低优先级
```

### 6.3 使用示例

```bash
# 添加不同优先级的车辆
add car01 LM001 HIGH
add car02 LM002 CRITICAL
add car03 LM003 LOW
add car04 LM004 MEDIUM

# 高优先级抢占
move car02 LM005          # CRITICAL级别移动，可抢占路径

# 动态调整优先级
setqos car01 CRITICAL

# 移动时临时提升优先级
move car01 LM005 HIGH

# 查看帮助
help
```

---

## 七、算法对比

| 特性 | agv_world.py | agv_world_qos.py |
|------|--------------|------------------|
| 路径规划 | 时间A* | 时间A* (冲突感知) |
| 调度策略 | 先到先得 | QoS优先 + 先到先得 |
| 冲突处理 | 避让/等待 | 抢占 + 重规划 |
| 优先级 | 无 | 5级 (CRITICAL→IDLE) |
| 降级机制 | 无 | 被抢占者QoS降一档 |
| 重规划 | 无 | 自动触发 |
| 恢复机制 | 无 | 任务完成恢复MEDIUM |

---

## 八、典型场景

### 场景1: 紧急任务插入

```
t=0: car01 (MEDIUM) 从 LM001 → LM005
t=5: car02 (CRITICAL) 从 LM002 → LM005
     → car01 被抢占，QoS: MEDIUM → LOW
     → car01 路径被取消，加入重规划队列
t=6: car01 以 LOW 级别重新规划，绕道或等待
```

### 场景2: 同级公平竞争

```
car01 (MEDIUM) 和 car02 (MEDIUM) 同时申请同一路径
→ 先规划者获得路径
→ 后规划者寻找替代路径
```

### 场景3: 任务完成恢复

```
car01 (LOW) 完成任务到达目标
→ QoS 恢复为 MEDIUM
→ 下次任务可正常参与竞争
```

---

## 九、文件结构

```
agv_simulation/
├── agv_world.py      # 原始版本（无QoS）
├── agv_world_qos.py  # QoS增强版本
├── agv_map_edit.py   # 地图编辑器
├── agv_planner.py    # 独立规划器
└── *.json           # 地图文件
```

---

## 十、运行方式

```bash
# 原始版本
python agv_world.py map.json

# QoS版本
python agv_world_qos.py map.json

# 可选参数
python agv_world_qos.py map.json --interval 100  # 加速动画
```
