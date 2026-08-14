"""回放 AGV+机器人 位置记录 (从 _record_trajectory.py 的 JSON)。
交互:
  ←/→  后退/前进 1 秒     (鼠标点击 = 前进 1 秒)
  空格  播放 / 暂停         Home/End 首帧/末帧
用法:
  PY src/bridge/_record_trajectory.py --cars 5 --path 0
  PY src/bridge/_playback.py docs/traj_5_p0.json
"""
import os
import sys
import json
import argparse

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BRIDGE_DIR)
sys.path.insert(0, BRIDGE_DIR)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

CAR_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
              "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
              "#e6beff", "#9a6324", "#800000", "#808000", "#000075"]


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="traj_*.json (from _record_trajectory.py)")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    traj = data["traj"]
    times = sorted(int(k) for k in traj)
    t_max = max(times) if times else 0
    W, H = data["width"], data["height"]
    obs = [tuple(o) for o in data["obstacles"]]
    lms = data["landmarks"]
    robot_path = data.get("robot_path", [])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, W))
    ax.set_yticks(range(0, H))
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

    grid_arr = np.zeros((H, W))
    for (x, y) in obs:
        grid_arr[y, x] = 1
    ax.imshow(grid_arr, extent=[-0.5, W - 0.5, -0.5, H - 0.5], origin="lower",
              cmap="Greys", alpha=0.35, interpolation="nearest", zorder=1)
    for name, (x, y) in lms.items():
        ax.plot(x, y, "o", color="#0066cc", markersize=8, alpha=0.85, zorder=4)
        ax.text(x, y + 0.3, name, fontsize=8, ha="center", color="#0066cc", zorder=5)

    # 机器人计划路径 (背景)
    if robot_path:
        rxp = [p[0] for p in robot_path]
        ryp = [p[1] for p in robot_path]
        ax.plot(rxp, ryp, color="red", linewidth=1.2, alpha=0.25, zorder=2, label="robot plan")

    # 车 ID 集合
    car_ids = sorted({cid for t in times for cid in traj[str(t)]["cars"]})
    car_col = {cid: CAR_COLORS[i % len(CAR_COLORS)] for i, cid in enumerate(car_ids)}

    car_scat = {cid: ax.scatter([], [], s=260, color=c, zorder=6,
                                edgecolors="black", linewidths=1.2) for cid, c in car_col.items()}
    car_txt = {cid: ax.text(0, 0, cid, fontsize=7, ha="center", va="center",
                            color="white", fontweight="bold", zorder=7) for cid in car_ids}
    car_trail = {cid: ax.plot([], [], color=c, alpha=0.4, linewidth=1.5, zorder=3)[0]
                 for cid, c in car_col.items()}
    robot_scat = ax.scatter([], [], s=320, color="red", marker="D", zorder=6,
                            edgecolors="black", linewidths=1.2, label="robot")
    robot_trail = ax.plot([], [], color="red", alpha=0.5, linewidth=2, zorder=3)[0]
    title = ax.set_title("", fontsize=12, pad=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.6)

    state = {"t": 0, "idx": 0, "playing": False}
    anim = {"timer": None}

    def draw():
        idx = state["idx"]
        t = times[idx] if times else 0
        state["t"] = t
        rec = traj.get(str(t), {"cars": {}, "robot": []})
        # 轨迹 = 0..t 的所有记录点 (车从出生到当前)
        for cid in car_ids:
            pts = []
            for tt in times:
                if tt > t:
                    break
                if cid in traj.get(str(tt), {}).get("cars", {}):
                    pts.append(traj[str(tt)]["cars"][cid])
            if pts:
                car_trail[cid].set_data([p[0] for p in pts], [p[1] for p in pts])
                px, py = pts[-1]
                car_scat[cid].set_offsets([[px, py]])
                car_txt[cid].set_position((px, py))
                car_txt[cid].set_text(cid)
            else:
                car_scat[cid].set_offsets([[np.nan, np.nan]])
                car_txt[cid].set_text("")
        # 机器人轨迹
        rpts = []
        for tt in times:
            if tt > t:
                break
            if traj.get(str(tt), {}).get("robot"):
                rpts.append(traj[str(tt)]["robot"])
        if rpts:
            robot_trail.set_data([p[0] for p in rpts], [p[1] for p in rpts])
            robot_scat.set_offsets([rpts[-1]])
        else:
            robot_scat.set_offsets([[np.nan, np.nan]])
        title.set_text(f"t = {t}s / {t_max}s   (idx {idx}/{len(times)-1})"
                       + ("   [SPACE=pause]" if state["playing"] else "   [SPACE=play, <-/-> step, click=+1s]"))
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["idx"] = min(state["idx"] + 1, len(times) - 1)
            draw()
        elif event.key == "left":
            state["idx"] = max(state["idx"] - 1, 0)
            draw()
        elif event.key == " ":
            state["playing"] = not state["playing"]
            if state["playing"]:
                anim["timer"] = fig.canvas.new_timer(interval=1000)
                anim["timer"].add_callback(step_play)
                anim["timer"].start()
            else:
                if anim["timer"]:
                    anim["timer"].stop()
            draw()
        elif event.key == "home":
            state["idx"] = 0
            draw()
        elif event.key == "end":
            state["idx"] = len(times) - 1
            draw()

    def on_click(event):
        if event.inaxes == ax:
            state["idx"] = min(state["idx"] + 1, len(times) - 1)
            draw()

    def step_play():
        if state["playing"]:
            if state["idx"] < len(times) - 1:
                state["idx"] += 1
                draw()
            else:
                state["playing"] = False
                if anim["timer"]:
                    anim["timer"].stop()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    draw()
    print("回放: ←/→ 步进1s, 点击图=+1s, 空格=播放/暂停, Home/End=首/末")
    plt.show()


if __name__ == "__main__":
    main()
