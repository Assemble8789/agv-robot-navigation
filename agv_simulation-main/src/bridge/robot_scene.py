"""
elf3 人形机器人 + AGV QoS 场景的文本合并
=========================================

MuJoCo 的 `MjModel.from_xml_string` 不处理 `<include>`, 而跨目录 `<include>`
时被包含文件的 `meshdir` 解析会坏掉 (变成 `assets/model/bxi_elf3/...`)。
所以这里在运行时做【文本合并】: 用平衡标签扫描器提取 `elf3.xml` 和
`agv_scene.xml` 的顶层区块, 重新组装成单个 `<mujoco>`, 并把 elf3 的
`meshdir="assets"` 改成绝对路径。

不用标准 XML 解析器 (agv_scene.xml 顶部注释里有嵌套 `<!-- ... -->`,
是无效 XML 但 MuJoCo 容忍); 扫描器只认标签、忽略注释内容。

参照 model/bxi_elf3/bxi_elf3_scene_factory.xml 的做法 (那里用 include,
这里改成文本合并以便 from_xml_string 能用)。

用法:
    xml = build_scene()            # 返回合并后的场景 XML 字符串
"""

import os
import re

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BRIDGE_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(SRC_DIR))

ELF3 = os.path.join(REPO_ROOT, "model", "bxi_elf3", "elf3.xml")
ASSETS = os.path.join(REPO_ROOT, "model", "bxi_elf3", "assets").replace(os.sep, "/")
BASE_SCENE = os.path.join(SRC_DIR, "agv_scene.xml")


def _extract(xml, tag):
    """返回第一个 `<tag ...>...</tag>` 完整块 (正确处理同名嵌套, 忽略注释)。
    自闭合标签只取自身。 找不到返回 ''。"""
    m = re.search(r'<' + tag + r'(?=[\s>])', xml)
    if not m:
        return ''
    i = m.start()
    gt = xml.index('>', i)
    if xml[gt - 1] == '/':
        return xml[i:gt + 1]                 # 自闭合
    depth = 1
    pos = gt + 1
    close = re.compile(r'</' + tag + r'\s*>|<' + tag + r'(?=[\s>])')
    while depth > 0:
        m2 = close.search(xml, pos)
        if not m2:
            return ''                        # 不平衡 (多半被注释干扰)
        depth += -1 if m2.group(0).startswith('</') else 1
        pos = m2.end()
    return xml[i:pos]


def _inner(block):
    """去掉块的开头 `<tag...>` 和结尾 `</tag>`。"""
    gt = block.index('>')
    # 从末尾找 `</` 对应闭合
    close = block.rfind('</')
    return block[gt + 1:close]


def build_scene():
    """返回合并 elf3 机器人 + AGV QoS 场景的 XML 字符串 (含 <!-- AGV_INSERT -->)。"""
    elf = open(ELF3, encoding="utf-8").read()
    scene = open(BASE_SCENE, encoding="utf-8").read()

    # elf3: compiler(改绝对 meshdir) / default / asset / worldbody / sensor /
    #       actuator / keyframe。 注意 elf3 的 <option> 是注释掉的 (默认配置),
    #       绝不能带进合并结果 —— 所以这里不取 elf 的 option。
    elf_compiler = _extract(elf, 'compiler').replace(
        'meshdir="assets"', 'meshdir="%s"' % ASSETS)
    elf_default = _extract(elf, 'default')
    elf_sensor = _extract(elf, 'sensor')
    elf_actuator = _extract(elf, 'actuator')
    elf_keyframe = _extract(elf, 'keyframe')

    # scene: statistic / visual / asset / worldbody / option
    scene_stat = _extract(scene, 'statistic')
    scene_visual = _extract(scene, 'visual')
    scene_option = _extract(scene, 'option')

    asset_inner = (_inner(_extract(elf, 'asset'))
                   + '\n' + _inner(_extract(scene, 'asset')))
    world_inner = (_inner(_extract(elf, 'worldbody'))
                   + '\n' + _inner(_extract(scene, 'worldbody')))

    merged = f'''<mujoco model="agv_qos_scene_robot">
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
    import mujoco
    xml = build_scene()
    m = mujoco.MjModel.from_xml_string(xml)
    print("ROBOT SCENE OK: nbody=%d nq=%d ngeom=%d nact=%d nkey=%d"
          % (m.nbody, m.nq, m.ngeom, m.nu, m.nkey))
    print(" timestep =", m.opt.timestep)
    for nm in ["torso_link", "agv_g0", "ground", "obs_0"]:
        print(" body", nm, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm))
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")
    print(" world_joint qposadr =", m.jnt_qposadr[jid] if jid >= 0 else None)
