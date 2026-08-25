"""
map_scene.py — 从 (0.1m) 地图生成米制物理场景 (bridge_wangting)

纯 AGV 阶段需要:
  - OBS_BOXES: 障碍盒 (米制, [(cx,cy,hx,hy)]), 供 /sim/obstacle_distance 感知
  - GARAGE_HOME: 车库球出生位 (米制)
  - MuJoCo 场景 XML: 地面 / 障碍盒 / 地标 marker / 车库球 (镜像 agv_scene.xml 结构)

障碍盒 = 0.1m 占用格【按行 run-length + 竖直合并】成 AABB 矩形 → 无损覆盖所有
占用格 (每格是中心 ± res/2 的见方), 盒边落在十进制米上 (如墙边在 4.1m 就是 4.1m)。
A* 仍用 0.1m 格子避障, 物理盒只影响感知/渲染。

不修改原文件: agv_scene.xml 不碰, 场景是另写的新 XML。
"""
from __future__ import annotations

import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agv_map_common import load_normalized          # noqa: E402
from mapframe import MapFrame                        # noqa: E402

# 物理常量 (与 mujoco_node 对齐)
AGV_Z = 0.04
BOX_H = 2.0        # 障碍盒高度 (半高 1.0), 距离计算是 2D, 高度不影响感知
GARAGE_N = 5


def load_map_data(map_file):
    """加载归一化地图 + 米制帧 + 占用格集合。"""
    md = load_normalized(map_file)
    frame = MapFrame(md)
    occ = set((o[0], o[1]) for o in md["obstacles"])
    return md, frame, occ


# ───────────────────── 障碍盒聚类 (无损覆盖占用格) ─────────────────────

def obstacle_boxes(md, frame):
    """把占用格合并成米制 AABB 盒。返回 [(cx, cy, hx, hy), ...]。"""
    occ = set((o[0], o[1]) for o in md["obstacles"])
    if not occ:
        return []

    # 1) 每行做 run-length: cy -> [(x0, x1), ...] (x1 含)
    rows = {}
    by_y = {}
    for cx, cy in occ:
        by_y.setdefault(cy, []).append(cx)
    for cy, xs in by_y.items():
        xs.sort()
        runs = []
        x0 = x1 = xs[0]
        for x in xs[1:]:
            if x == x1 + 1:
                x1 = x
            else:
                runs.append((x0, x1))
                x0 = x1 = x
        runs.append((x0, x1))
        rows[cy] = runs

    # 2) 竖直合并: 相邻行相同 (x0,x1) 的 run 合成一块矩形
    groups = []                      # [x0, x1, y0, y1]
    for cy in sorted(rows):
        for (x0, x1) in rows[cy]:
            merged = None
            for g in groups:
                if g[2] == cy - 1 and g[0] == x0 and g[1] == x1:
                    merged = g
                    break
            if merged is not None:
                merged[3] = cy
            else:
                groups.append([x0, x1, cy, cy])

    # 3) 格子 footprint 中心 ± res/2 → 米制盒
    boxes = []
    for x0, x1, y0, y1 in groups:
        xm0 = frame.cx2m(x0) - frame.res / 2.0
        xm1 = frame.cx2m(x1) + frame.res / 2.0
        ym0 = frame.cy2m(y0) - frame.res / 2.0
        ym1 = frame.cy2m(y1) + frame.res / 2.0
        boxes.append(((xm0 + xm1) / 2.0, (ym0 + ym1) / 2.0,
                      (xm1 - xm0) / 2.0, (ym1 - ym0) / 2.0))
    return boxes


# ───────────────────── 车库 (互相最远空闲格, 米制) ─────────────────────

def pick_garage(md, frame, n=GARAGE_N):
    """选 n 个互相最远的空闲格 (贪心), 转米制, 按 (y,x) 排序成整齐一排。"""
    occ = set((o[0], o[1]) for o in md["obstacles"])
    w, h = md["width"], md["height"]
    free = [(x, y) for y in range(h) for x in range(w) if (x, y) not in occ]
    picks = []
    if free:
        # 第一个取左下角最近空闲格, 之后贪心取离已有最远的格
        corner = min(free, key=lambda c: c[0] + c[1])
        picks.append(corner)
        while len(picks) < n and len(picks) < len(free):
            best, best_d = None, -1
            for f in free:
                if f in picks:
                    continue
                d = min(abs(f[0] - p[0]) + abs(f[1] - p[1]) for p in picks)
                if d > best_d:
                    best_d, best = d, f
            if best is None or best_d <= 0:
                break
            picks.append(best)
    picks.sort(key=lambda c: (c[1], c[0]))
    return [frame.to_meters(cx, cy) for cx, cy in picks]


# ───────────────────── MuJoCo 场景 XML 生成 ─────────────────────

def _factory_bounds(md, frame):
    occ = set((o[0], o[1]) for o in md["obstacles"])
    if not occ:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [o[0] for o in occ]
    ys = [o[1] for o in occ]
    x0 = frame.cx2m(min(xs)) - frame.res
    x1 = frame.cx2m(max(xs)) + frame.res
    y0 = frame.cy2m(min(ys)) - frame.res
    y1 = frame.cy2m(max(ys)) + frame.res
    return (x0, x1, y0, y1)


def build_scene_xml(md, frame, boxes, garage, out_name="wangting_scene.xml"):
    """生成 MuJoCo 场景 XML 字符串 (含 <!-- AGV_INSERT -->)。"""
    x0, x1, y0, y1 = _factory_bounds(md, frame)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    span_x = max(x1 - x0, 1.0)
    span_y = max(y1 - y0, 1.0)
    MARGIN = 6.0

    def b(i, name, x, y, z, gx, gy, gz, mat, extra=""):
        return (f'    <body name="{name}" pos="{x:.3f} {y:.3f} {z:.3f}"{extra}>\n'
                f'      <geom name="{name}_geom" type="box" size="{gx:.3f} {gy:.3f} {gz:.3f}" '
                f'material="{mat}" contype="1" conaffinity="1"/>\n'
                f'    </body>')

    obs_xml = []
    for i, (bx, by, hx, hy) in enumerate(boxes):
        obs_xml.append(b(i, f"obs_{i}", bx, by, BOX_H / 2.0, hx, hy, BOX_H / 2.0,
                          "shelf_mat"))

    lm_xml = []
    for lm in md["landmarks"]:
        lx, ly = frame.to_meters(lm["x"], lm["y"])
        nm = f"lm_{lm['name']}"
        lm_xml.append(
            f'    <body name="{nm}" pos="{lx:.3f} {ly:.3f} -0.12">\n'
            f'      <geom name="{nm}_geom" type="cylinder" size="0.15 0.02" '
            f'material="lm_mat" contype="0" conaffinity="0"/>\n'
            f'    </body>')

    garage_xml = []
    gx0 = min((p[0] for p in garage), default=0.0)
    gy0 = min((p[1] for p in garage), default=0.0)
    if garage:
        pad_x = max(abs(p[0] - gx0) for p in garage) + 0.6
        pad_y = max(abs(p[1] - gy0) for p in garage) + 0.6
        garage_xml.append(
            f'    <geom name="garage_pad" type="box" pos="{gx0 + (max(p[0] for p in garage) - gx0) / 2:.3f} '
            f'{gy0 + (max(p[1] for p in garage) - gy0) / 2:.3f} 0.02" '
            f'size="{pad_x:.3f} {pad_y:.3f} 0.02" material="garage_mat" '
            f'contype="0" conaffinity="0"/>')
    for i, (mx, my) in enumerate(garage):
        garage_xml.append(
            f'    <body name="agv_g{i}" pos="{mx:.3f} {my:.3f} {AGV_Z}" mocap="true">\n'
            f'      <geom name="agv_g{i}_geom" type="sphere" size="0.2" '
            f'material="agv{i}_mat" contype="1" conaffinity="1"/>\n'
            f'    </body>')

    xml = f'''<mujoco model="wangting_scene">
  <statistic center="{cx:.2f} {cy:.2f} 0.6" extent="{max(span_x, span_y) / 2 + MARGIN:.2f}" meansize="0.1"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
    <global azimuth="180" elevation="-35"/>
  </visual>

  <asset>
    <texture name="floor_tex" type="2d" builtin="flat" rgb1=".58 .62 .60" rgb2=".50 .54 .52" width="128" height="128" mark="random" markrgb=".52 .56 .54"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="20 15" texuniform="true" reflectance="0.03"/>
    <material name="shelf_mat" rgba="0.5 0.45 0.4 1"/>
    <material name="equip_mat" rgba="0.4 0.5 0.55 1"/>
    <material name="lm_mat" rgba="0.2 0.8 0.2 0.6"/>
    <material name="garage_mat" rgba="0.4 0.4 0.5 0.25"/>
    <material name="dyn_mat" rgba="0.95 0.15 0.2 0.6"/>
    <material name="agv0_mat" rgba="0.9 0.3 0.2 1"/>
    <material name="agv1_mat" rgba="0.2 0.5 0.9 1"/>
    <material name="agv2_mat" rgba="0.2 0.7 0.3 1"/>
    <material name="agv3_mat" rgba="0.9 0.7 0.1 1"/>
    <material name="agv4_mat" rgba="0.6 0.2 0.8 1"/>
    <material name="agv5_mat" rgba="0.2 0.8 0.8 1"/>
    <material name="agv6_mat" rgba="0.9 0.5 0.6 1"/>
    <material name="agv7_mat" rgba="0.5 0.5 0.5 1"/>
  </asset>

  <worldbody>
    <light pos="{cx:.2f} {cy:.2f} 8" directional="true"/>
    <geom name="ground" type="plane" pos="{cx:.2f} {cy:.2f} -0.16"
          size="{span_x / 2 + MARGIN:.2f} {span_y / 2 + MARGIN:.2f} 0.1"
          material="floor_mat" friction="1.0 0.1 0.1"/>

{chr(10).join(obs_xml)}

{chr(10).join(lm_xml)}

{chr(10).join(garage_xml)}

    <!-- 超过车库容量的新车球体: 由 mujoco_node 拼进此处 -->
    <!-- AGV_INSERT -->
  </worldbody>

  <option gravity="0 0 -9.81"/>
</mujoco>'''
    return xml


def write_scene(map_file, out_path):
    """生成场景 XML 写入文件。返回 (boxes, garage, scene_path)。"""
    md, frame, _ = load_map_data(map_file)
    boxes = obstacle_boxes(md, frame)
    garage = pick_garage(md, frame)
    xml = build_scene_xml(md, frame, boxes, garage)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"[scene] {out_path}  boxes={len(boxes)} garage={garage}")
    return boxes, garage, out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从地图生成米制场景 XML")
    parser.add_argument("map_file")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or os.path.join(SRC_DIR, "..", "maps", "wangting_scene.xml")
    boxes, garage, path = write_scene(args.map_file, out)
    print(f"[ok] {len(boxes)} boxes, garage {garage}")