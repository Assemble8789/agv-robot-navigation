import json
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

def visualize(filename, interval=100):
    with open(filename, "r") as f:
        data = json.load(f)
    
    if "map_file" not in data:
        print("Error: Input must be a plan file (plan_*.json).")
        sys.exit(1)
        
    plan_data = data
    map_file = plan_data["map_file"]
    if not os.path.exists(map_file):
        map_file = os.path.join(os.path.dirname(filename), os.path.basename(map_file))
    
    if not os.path.exists(map_file):
        print(f"Error: Map file '{map_file}' not found.")
        sys.exit(1)

    with open(map_file, "r") as f:
        map_data = json.load(f)
    
    cars = plan_data.get("cars", [])
    width = map_data["width"]
    height = map_data["height"]
    
    raw_obstacles = map_data.get("obstacles", [])
    has_negative = False
    
    # 转换为 numpy 矩阵进行极速渲染
    # 处理坐标原点
    for o in raw_obstacles:
        ox = o["x"] if isinstance(o, dict) else o[0]
        oy = o["y"] if isinstance(o, dict) else o[1]
        if ox < 0 or oy < 0:
            has_negative = True
            break
            
    x_min, y_min = (-width // 2, -height // 2) if has_negative else (0, 0)
    x_max, y_max = x_min + width, y_min + height

    # 创建背景掩码矩阵
    grid = np.zeros((height, width))
    for o in raw_obstacles:
        ox = o["x"] if isinstance(o, dict) else o[0]
        oy = o["y"] if isinstance(o, dict) else o[1]
        # 映射到矩阵索引 (y从上往下，或者配合 origin='lower')
        iy, ix = int(oy - y_min), int(ox - x_min)
        if 0 <= ix < width and 0 <= iy < height:
            grid[iy, ix] = 1

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)
    
    # 使用 imshow 替代成千上万个 Rectangle，性能大幅提升
    # cmap 可自定义颜色，这里用灰阶表示障碍物
    ax.imshow(grid, extent=[x_min - 0.5, x_max - 0.5, y_min - 0.5, y_max - 0.5], 
              origin='lower', cmap='Greys', alpha=0.3, interpolation='nearest', zorder=1)

    ax.set_xlim(x_min - 0.5, x_max - 0.5)
    ax.set_ylim(y_min - 0.5, y_max - 0.5)
    
    # 动态调整网格密度，减少渲染负担
    if width <= 50:
        ax.set_xticks(range(x_min, x_max))
        ax.set_yticks(range(y_min, y_max))
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    else:
        # 大地图不画详细网格线，只留边框和主要刻度
        step = max(1, width // 10)
        ax.set_xticks(range(x_min, x_max, step))
        ax.set_yticks(range(y_min, y_max, step))
        ax.grid(False) 

    time_map = {}
    max_time = 0
    scatters = {}
    
    for car in cars:
        car_id = car["car_id"]
        scat = ax.scatter([], [], s=150, label=f"Car {car_id}", zorder=10)
        scatters[car_id] = scat
        color = scat.get_facecolor()[0]
        traj = car["trajectory"]
        if not traj: continue
        
        start_pt, end_pt = traj[0], traj[-1]
        # 绘制起点和终点（作为背景，不参与每一帧重绘）
        ax.scatter(start_pt["x"], start_pt["y"], s=80, facecolors='none', edgecolors=color, marker='o', alpha=0.4, zorder=2)
        ax.scatter(end_pt["x"], end_pt["y"], s=100, color=color, marker='X', alpha=0.4, zorder=2)
        
        for p in traj:
            t = p["t"]
            max_time = max(max_time, t)
            time_map.setdefault(t, []).append((car_id, p["x"], p["y"]))

    ax.legend(loc="upper right", fontsize='small', ncol=2 if len(cars) > 5 else 1)

    class AnimationControl:
        def __init__(self):
            self.current_frame = 0
            self.is_playing = True
            self.max_frames = max_time
            
        def update_plot(self):
            # 高效更新 Scatter 坐标
            for car_id, scatter in scatters.items():
                found = False
                for cid, x, y in time_map.get(self.current_frame, []):
                    if cid == car_id:
                        scatter.set_offsets([[x, y]])
                        scatter.set_visible(True)
                        found = True
                        break
                if not found:
                    # 如果该时间点没有该车数据，隐藏或者保持不动
                    pass
            ax.set_title(f"Plan: {os.path.basename(filename)} | Time: {self.current_frame}s", fontsize=10)

        def next(self):
            if self.current_frame < self.max_frames:
                self.current_frame += 1
            else:
                self.current_frame = 0
            self.update_plot()

    ctrl = AnimationControl()

    def animation_step(i):
        if ctrl.is_playing:
            ctrl.next()
        return list(scatters.values())

    # 创建控制按钮
    ax_play = plt.axes([0.35, 0.05, 0.12, 0.075])
    ax_step = plt.axes([0.52, 0.05, 0.12, 0.075])
    
    btn_play = Button(ax_play, 'Play/Pause')
    btn_step = Button(ax_step, 'Step >')

    def toggle(event):
        ctrl.is_playing = not ctrl.is_playing
    
    def step_once(event):
        ctrl.is_playing = False
        ctrl.next()
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle)
    btn_step.on_clicked(step_once)

    # interval 降低到 20ms 现在应该有明显效果了
    ani = FuncAnimation(fig, animation_step, interval=interval, blit=True, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast AGV Visualizer")
    parser.add_argument("filename", nargs="?", help="Plan JSON file")
    parser.add_argument("--interval", type=int, default=100, help="ms per step")
    args = parser.parse_args()
    
    target = args.filename
    if not target:
        plans = sorted([f for f in os.listdir('.') if f.startswith('plan_') and f.endswith('.json')])
        if plans: target = plans[-1]
        else: print("No plan file found."); sys.exit(1)
            
    visualize(target, interval=args.interval)
