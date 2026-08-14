import json
import os
import argparse
import datetime
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

class MapEditor:
    def __init__(self, width=15, height=10, filename=None):
        self.x_min, self.y_min = 0, 0
        self.is_centered = False
        if filename and os.path.exists(filename):
            self.load_map(filename)
        else:
            self.width = width
            self.height = height
            self.obstacles = set()
            self.landmarks = [] # [{"name": "LM001", "x": 1, "y": 2}, ...]
            self.filename = None

        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.2)
        
        self.rects = {} # (x, y) -> Rectangle object
        self.landmark_plots = {} # (x, y) -> (Scatter plot, Text plot)
        self.setup_plot()
        self.draw_existing_obstacles()

        # 事件绑定
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 按钮创建
        ax_random = plt.axes([0.3, 0.05, 0.1, 0.075])
        ax_save = plt.axes([0.6, 0.05, 0.1, 0.075])
        
        self.btn_random = Button(ax_random, 'Random')
        self.btn_save = Button(ax_save, 'Save')
        
        self.btn_random.on_clicked(self.generate_random)
        self.btn_save.on_clicked(self.save_map)

    def setup_plot(self):
        x_max = self.x_min + self.width
        y_max = self.y_min + self.height
        
        self.ax.set_xlim(self.x_min - 0.5, x_max - 0.5)
        self.ax.set_ylim(self.y_min - 0.5, y_max - 0.5)
        
        # Grid density control
        if self.width <= 50:
            self.ax.set_xticks(range(self.x_min, x_max))
        else:
            step = max(1, self.width // 10)
            self.ax.set_xticks(range(self.x_min, x_max, step))
            
        if self.height <= 50:
            self.ax.set_yticks(range(self.y_min, y_max))
        else:
            step = max(1, self.height // 10)
            self.ax.set_yticks(range(self.y_min, y_max, step))
            
        self.ax.grid(True, linestyle='--', alpha=0.6)
        title = "Map Editor (Left: Obs, Right: Del Obs)\n(Ctrl+Left: Add Landmark, Ctrl+Right: Del Landmark)"
        if self.filename:
            title += f"\nEditing: {os.path.basename(self.filename)}"
        self.ax.set_title(title)

    def draw_existing_obstacles(self):
        for (x, y) in self.obstacles:
            self.add_rect_at(x, y)
        for lm in self.landmarks:
            self.add_landmark_plot(lm["x"], lm["y"], lm["name"])

    def add_rect_at(self, x, y):
        if (x, y) not in self.rects:
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1, color="gray", alpha=0.8)
            self.ax.add_patch(rect)
            self.rects[(x, y)] = rect

    def remove_rect_at(self, x, y):
        if (x, y) in self.rects:
            self.rects[(x, y)].remove()
            del self.rects[(x, y)]

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        # 获取整数坐标
        x = int(round(event.xdata))
        y = int(round(event.ydata))

        # 检查边界
        x_max = self.x_min + self.width
        y_max = self.y_min + self.height
        if not (self.x_min <= x < x_max and self.y_min <= y < y_max):
            return

        is_ctrl = event.key == 'control'

        if event.button == 1: # 左键
            if is_ctrl: # Add Landmark
                # Check if there's an obstacle here
                if (x, y) in self.obstacles:
                    print(f"Cannot add landmark at ({x}, {y}): Obstacle present.")
                    return
                if not any(lm['x'] == x and lm['y'] == y for lm in self.landmarks):
                    # 获取现有的最大编号，确保新编号唯一
                    existing_ids = [int(lm['name'][2:]) for lm in self.landmarks if lm['name'].startswith('LM') and lm['name'][2:].isdigit()]
                    next_id = max(existing_ids) + 1 if existing_ids else 0
                    lm_name = f"LM{next_id:03d}"
                    
                    self.landmarks.append({"name": lm_name, "x": x, "y": y})
                    self.add_landmark_plot(x, y, lm_name)
            else: # Add Obstacle
                # Check if there's a landmark here
                if any(lm['x'] == x and lm['y'] == y for lm in self.landmarks):
                    print(f"Cannot add obstacle at ({x}, {y}): Landmark present.")
                    return
                if (x, y) not in self.obstacles:
                    self.obstacles.add((x, y))
                    self.add_rect_at(x, y)
        elif event.button == 3: # 右键
            if is_ctrl: # Remove Landmark
                self.landmarks = [lm for lm in self.landmarks if not (lm['x'] == x and lm['y'] == y)]
                self.remove_landmark_plot(x, y)
            else: # Remove Obstacle
                if (x, y) in self.obstacles:
                    self.obstacles.remove((x, y))
                    self.remove_rect_at(x, y)
        
        self.fig.canvas.draw_idle()

    def add_landmark_plot(self, x, y, name):
        scat = self.ax.scatter(x, y, s=100, color="blue", marker="D", zorder=10)
        txt = self.ax.text(x, y + 0.3, name, color="blue", fontsize=8, ha='center', fontweight='bold', zorder=11)
        self.landmark_plots[(x, y)] = (scat, txt)

    def remove_landmark_plot(self, x, y):
        if (x, y) in self.landmark_plots:
            scat, txt = self.landmark_plots[(x, y)]
            scat.remove()
            txt.remove()
            del self.landmark_plots[(x, y)]

    def generate_random(self, event=None):
        import random
        # 1. 清空当前所有内容
        for (x, y) in list(self.obstacles):
            self.remove_rect_at(x, y)
        self.obstacles.clear()
        
        for (x, y) in list(self.landmark_plots.keys()):
            self.remove_landmark_plot(x, y)
        self.landmarks.clear()

        # 2. 随机生成数量 (例如 10% ~ 20% 的填充率)
        total_cells = self.width * self.height
        num_to_gen = random.randint(int(total_cells * 0.1), int(total_cells * 0.2))
        
        all_cells = [(x, y) for x in range(self.x_min, self.x_min + self.width) 
                           for y in range(self.y_min, self.y_min + self.height)]
        random_obs = random.sample(all_cells, num_to_gen)
        
        # 3. 添加新障碍
        for (x, y) in random_obs:
            self.obstacles.add((x, y))
            self.add_rect_at(x, y)
            
        self.ax.set_title(f"Random Generated: {num_to_gen} obstacles")
        self.fig.canvas.draw_idle()

    def load_map(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.width = data["width"]
            self.height = data["height"]
            
            # Handle both list and dict formats for obstacles
            raw_obstacles = data.get("obstacles", [])
            self.obstacles = set()
            has_negative = False
            for o in raw_obstacles:
                if isinstance(o, dict):
                    ox, oy = o["x"], o["y"]
                else:
                    ox, oy = o[0], o[1]
                self.obstacles.add((ox, oy))
                if ox < 0 or oy < 0:
                    has_negative = True
            
            self.landmarks = data.get("landmarks", [])
            for lm in self.landmarks:
                if lm["x"] < 0 or lm["y"] < 0:
                    has_negative = True

            # If origin is at middle, coordinates range from -width//2 to width//2-1
            if has_negative:
                self.is_centered = True
                self.x_min = -self.width // 2
                self.y_min = -self.height // 2
            else:
                self.is_centered = False
                self.x_min = 0
                self.y_min = 0
                
            self.filename = filename
            print(f"Loaded map: {filename} (Centered: {self.is_centered})")

    @property
    def _out_dir(self):
        """Maps directory relative to project root (parent of src/)."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        maps_dir = os.path.join(project_root, "maps")
        os.makedirs(maps_dir, exist_ok=True)
        return maps_dir

    def save_map(self, event=None):
        map_data = {
            "width": self.width,
            "height": self.height,
            "obstacles": sorted([list(o) for o in self.obstacles]),
            "landmarks": self.landmarks
        }
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"map_{timestamp}.json"
        img_name = f"map_{timestamp}.png"
        out_path = os.path.join(self._out_dir, out_name)
        img_path = os.path.join(self._out_dir, img_name)
        
        with open(out_path, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        # 保存图片，去除周围的按钮
        # 暂时隐藏按钮所在的 axes
        axes_to_hide = [self.btn_save.ax, self.btn_random.ax]
        for b_ax in axes_to_hide:
            b_ax.set_visible(False)
        
        self.fig.savefig(img_path, bbox_inches='tight', dpi=150)
        
        # 恢复显示按钮
        for b_ax in axes_to_hide:
            b_ax.set_visible(True)

        rel_json = os.path.join("maps", out_name)
        rel_png  = os.path.join("maps", img_name)
        
        print(f"Map saved successfully as: {rel_json}")
        print(f"Image saved successfully as: {rel_png}")
        self.ax.set_title(f"Saved! -> {rel_json}\nImage -> {rel_png}")
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGV Map Editor")
    parser.add_argument("filename", nargs="?", help="Existing map JSON to edit")
    parser.add_argument("--width", type=int, default=15, help="Width of new map")
    parser.add_argument("--height", type=int, default=10, help="Height of new map")
    
    args = parser.parse_args()
    
    editor = MapEditor(width=args.width, height=args.height, filename=args.filename)
    plt.show()
