"""
robot_scene_wangting.py — elf3 人形机器人 + wangting 工厂场景的文本合并
=======================================================================

MujocoNode(robot=True) 内部调用 `from robot_scene import build_scene`, 把 elf3.xml
和 agv_scene.xml 合并。wangting 版把底座换成【地图生成】的 wangting 场景 XML:
复用 bridge/robot_scene.py 的标签扫描/合并工具 (只读 import, 不改原文件),
用法:
    from robot_scene_wangting import build_scene
    xml = build_scene(map_scene_xml_string)   # elf3 + 工厂场景合并
"""
from __future__ import annotations

import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_DIR = os.path.join(SRC_DIR, "bridge")
for p in (SRC_DIR, BRIDGE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import robot_scene as _rs          # noqa: E402  (复用 _extract/_inner/ELF3/ASSETS)


def build_scene(map_scene_xml):
    """把 elf3 与地图生成场景 XML 合并成单个 <mujoco> (含 AGV_INSERT 标记)。
    逻辑 = bridge/robot_scene.build_scene, 仅底座场景换成参数。"""
    elf = open(_rs.ELF3, encoding="utf-8").read()
    scene = map_scene_xml

    elf_compiler = _rs._extract(elf, 'compiler').replace(
        'meshdir="assets"', 'meshdir="%s"' % _rs.ASSETS)
    elf_default = _rs._extract(elf, 'default')
    elf_sensor = _rs._extract(elf, 'sensor')
    elf_actuator = _rs._extract(elf, 'actuator')
    elf_keyframe = _rs._extract(elf, 'keyframe')

    scene_stat = _rs._extract(scene, 'statistic')
    scene_visual = _rs._extract(scene, 'visual')
    scene_option = _rs._extract(scene, 'option')

    asset_inner = (_rs._inner(_rs._extract(elf, 'asset'))
                   + '\n' + _rs._inner(_rs._extract(scene, 'asset')))
    world_inner = (_rs._inner(_rs._extract(elf, 'worldbody'))
                   + '\n' + _rs._inner(_rs._extract(scene, 'worldbody')))

    merged = f'''<mujoco model="wangting_scene_robot">
  {elf_compiler}
  {elf_default}
  {scene_stat}
  {scene_visual}
  <asset>
{asset_inner}
  </asset>
  <worldbody>
{world_inner}
  </worldbody>
  {elf_sensor}
  {elf_actuator}
  {scene_option}
  {elf_keyframe}
</mujoco>'''
    assert '<!-- AGV_INSERT -->' in merged, 'AGV_INSERT marker lost in merge!'
    return merged


if __name__ == "__main__":
    import argparse
    import map_scene
    parser = argparse.ArgumentParser(description="elf3 + wangting 场景合并")
    parser.add_argument("map_file")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    md, frame, _ = map_scene.load_map_data(args.map_file)
    boxes = map_scene.obstacle_boxes(md, frame)
    garage = map_scene.pick_garage(md, frame)
    base = map_scene.build_scene_xml(md, frame, boxes, garage)
    merged = build_scene(base)

    out = args.out or os.path.join(SRC_DIR, "..", "maps", "wangting_robot_scene.xml")
    with open(out, "w", encoding="utf-8") as f:
        f.write(merged)
    import mujoco
    m = mujoco.MjModel.from_xml_string(merged)
    print(f"[ok] {out} nbody={m.nbody} ngeom={m.ngeom}")