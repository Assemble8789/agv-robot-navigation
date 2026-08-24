import json
import heapq
import random
import argparse
import datetime
import os
from agv_map_common import load_normalized

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_with_time(width, height, obs_set, start, goal, start_time, reservations, x_min=0, y_min=0):
    """
    带时间约束的A*算法实现
    参数:
    - width, height: 地图尺寸
    - obs_set: 障碍物集合
    - start, goal: 起点和终点坐标
    - start_time: 起始时间
    - reservations: 时间-位置预约集合，用于防碰撞
    - x_min, y_min: 地图坐标原点
    """
    # 优先队列，存储 (f_score, g_score, 时间, 位置)
    open_set = [(0, 0, start_time, start)]
    came_from = {}  # position -> (prev_position, prev_t)
    
    # g_score: 从起点到当前节点的实际代价
    g_score = {start: 0}
    
    # f_score: g_score + 启发函数估计值
    f_score = {start: heuristic(start, goal)}
    
    # 用于防碰撞的预约集合
    # reservations: (x, y, t) 表示在时间t位置(x,y)已被占用
    # reservations: (px, py, t, x, y) 表示在时间t从(px,py)移动到(x,y)的路径已被占用
    
    x_max, y_max = x_min + width, y_min + height
    
    while open_set:
        # 取出f_score最小的节点
        _, current_g, current_t, current = heapq.heappop(open_set)
        
        # 如果到达目标点
        if current == goal:
            path = []
            path_times = []
            curr = current
            curr_t = current_t
            while curr in came_from:
                path.append(curr)
                path_times.append(curr_t)
                curr, _, curr_t = came_from[curr]
            path.append(start)
            path_times.append(start_time)
            path.reverse()
            path_times.reverse()
            return [(t, pos) for t, pos in zip(path_times, path)]
        
        # 8方向移动
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # 检查边界
            if not (x_min <= neighbor[0] < x_max and y_min <= neighbor[1] < y_max):
                continue
                
            # 检查障碍物
            if neighbor in obs_set:
                continue
                
            # 检查时间-位置预约（防碰撞）
            if (neighbor[0], neighbor[1], current_t + 1) in reservations:
                continue
                
            # 检查路径预约（防碰撞）
            # 如果从当前节点移动到邻居节点的路径已被占用
            if (current[0], current[1], current_t + 1, neighbor[0], neighbor[1]) in reservations:
                continue
                
            # 计算新的g_score
            tentative_g_score = g_score[current] + 1
            
            # 如果找到更短的路径
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = (current, g_score[current], current_t)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], current_t + 1, neighbor))
    
    # 没有找到路径
    return None

def plan_paths(map_file, num_cars=0, tasks=None):
    """
    路径规划主函数
    """
    # 读取地图文件（自动修复/归一化/补地标）
    map_data = load_normalized(map_file)
    
    # 解析地图数据
    width = map_data['width']
    height = map_data['height']
    obstacles = map_data['obstacles']
    landmarks = map_data['landmarks']
    
    # 构建障碍物集合
    obs_set = set()
    for obs in obstacles:
        obs_set.add((obs[0], obs[1]))
    
    # 构建地标字典
    lm_dict = {}
    for lm in landmarks:
        if 'name' in lm and 'x' in lm and 'y' in lm:
            lm_dict[lm['name']] = (lm['x'], lm['y'])
    
    # 计算地图坐标范围
    x_min, y_min = 0, 0
    x_max, y_max = width, height
    
    # 任务规划
    planned_tasks = []
    
    if tasks:
        # 指定任务
        for start_node, goal_node in tasks:
            s_coord = lm_dict.get(start_node)
            g_coord = lm_dict.get(goal_node)
            if s_coord and g_coord:
                planned_tasks.append((s_coord, g_coord, start_node, goal_node))
            else:
                print(f"Warning: Landmark {start_node} or {goal_node} not found.")
    elif num_cars:
        # 随机任务
        if landmarks:
            for i in range(num_cars):
                s_lm = random.choice(landmarks)
                g_lm = random.choice(landmarks)
                while g_lm == s_lm and len(landmarks) > 1:
                    g_lm = random.choice(landmarks)
                planned_tasks.append(((s_lm["x"], s_lm["y"]), (g_lm["x"], g_lm["y"]), s_lm["name"], g_lm["name"]))
        else:
            # 兼容老版本：随机自由格
            def get_free():
                while True:
                    c = (random.randint(x_min, x_max - 1), random.randint(y_min, y_max - 1))
                    if c not in obs_set: return c
            for i in range(num_cars):
                start = get_free()
                goal = get_free()
                while goal == start:
                    goal = get_free()
                planned_tasks.append((start, goal, "Random", "Random"))

    # 防碰撞预约集合
    reservations = set()
    car_paths = []

    # 为每辆车规划路径
    for car_id, (start, goal, s_name, g_name) in enumerate(planned_tasks):
        # 为每辆车设置不同的起始时间，避免冲突
        start_time = car_id * 2  # 时间间隔为2
        
        # 调用A*算法进行路径规划
        path = astar_with_time(width, height, obs_set, start, goal, start_time, reservations, x_min, y_min)
        
        if not path:
            print(f"Car {car_id} ({s_name} -> {g_name}) planning failed.")
            continue

        # 记录路径轨迹
        trajectory = []
        for t, (x, y) in path:
            trajectory.append({"t": t, "x": x, "y": y})
            # 将当前位置加入预约集合
            reservations.add((x, y, t))
            
        # 将路径移动轨迹加入预约集合
        for i in range(1, len(path)):
            t, (x, y) = path[i]
            pt, (px, py) = path[i-1]
            # 将从(px,py)到(x,y)的路径加入预约集合
            reservations.add((px, py, t, x, y))

        car_paths.append({
            "car_id": car_id,
            "start_landmark": s_name,
            "goal_landmark": g_name,
            "trajectory": trajectory
        })

    # 生成结果
    result = {
        "map_file": map_file,
        "cars": car_paths
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plan_{timestamp}.json"
    maps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps")
    os.makedirs(maps_dir, exist_ok=True)
    output_path = os.path.join(maps_dir, filename)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Planning complete: maps/{filename} (based on {map_file})")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGV Path Planner")
    parser.add_argument("map_file", help="The map JSON file")
    parser.add_argument("--cars", type=int, help="Number of cars (random start/goal from landmarks)")
    parser.add_argument("--tasks", type=str, help="Specific tasks, e.g. 'LM001,LM005;LM002,LM006'")
    
    args = parser.parse_args()
    
    task_list = None
    if args.tasks:
        # 解析任务字符串
        task_list = []
        for pair in args.tasks.split(';'):
            if ',' in pair:
                s, g = pair.split(',')
                task_list.append((s.strip(), g.strip()))
    
    # 默认值
    num_cars = args.cars if args.cars else (0 if task_list else 2)
    
    plan_paths(args.map_file, num_cars=num_cars, tasks=task_list)
