"""
agv_map_common.py — 统一地图加载/归一化层。

AGV 系统（planner / world / visualizer / map_edit）各自的 json.load + 各自解析，
本模块提供唯一入口 load_normalized()，让所有入口一致地处理：

1. JSON 损坏自动修复（已知损坏模式的字符丢失，仅内存修复，不写回源文件）。
2. 浮点坐标地图自动归一化：按 0.1m 分辨率量化成整数格子、去重、裁包围盒、平移到 (0,0)，
   附带 _resolution_m / _origin_offset_m 元数据以便转回真实米制坐标。
   原生整数地图（含负坐标 centered 图）原样透传，行为零变化。
3. landmarks 为空时自动在四角最近的自由格生成 LM001..LM004。

返回值是引擎直接可用的原生格式 dict：
    {width, height, obstacles: [[int,int],...], landmarks: [{name,x,y},...],
     _resolution_m, _origin_offset_m, _normalized}
"""
import json
import math
import os
import random

# 已知的源文件损坏段（字符丢失）。key 为损坏文本，value 为按障碍排列规律补齐的正确文本。
CORRUPTIONS = [
    (
        '{"x":-13.7,"y".7,"y":18.5}',
        '{"x":-13.7,"y":18.4},{"x":-13.7,"y":18.5}',
    ),
    (
        '{"x":-13.3,"y":20.8},{"x:18.4},{"x":-13":-13.3,"y":20.9}',
        '{"x":-13.3,"y":20.8},{"x":-13.3,"y":18.4},{"x":-13.3,"y":20.9}',
    ),
]

DEFAULT_RES_M = 0.1  # 外部浮点地图的默认量化分辨率 (m/cell)


def _read_map_text(map_file):
    with open(map_file, "r", encoding="utf-8") as f:
        return f.read()


def _parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        fixed = text
        repaired = 0
        for bad, good in CORRUPTIONS:
            if bad in fixed:
                fixed = fixed.replace(bad, good, 1)
                repaired += 1
        if repaired == 0:
            raise ValueError(
                f"Map JSON is corrupted and no known repair pattern matched: {e}"
            ) from e
        print(f"[agv_map_common] repaired {repaired} corrupted segment(s) (char {e.pos})")
        return json.loads(fixed)


def _is_float_coord(v):
    return not isinstance(v, int) or (isinstance(v, float) and not v.is_integer())


def _find_nearest_free(start, occupied, w, h):
    """从 (x, y) 开始螺旋搜索最近的空闲格。"""
    x, y = start
    for r in range(max(w, h)):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in occupied:
                    return (nx, ny)
    return None


def _auto_landmarks(occupied, w, h, n=4):
    """生成 n 个地标：
    - n<=4 时取四角最近自由格（保证边缘可达、互不拥挤）。
    - n>4 时随机取 n 个互不相同的自由格。
    """
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    if n <= 4:
        landmarks = []
        for i, corner in enumerate(corners[:n]):
            pos = _find_nearest_free(corner, occupied, w, h)
            if pos is None:
                continue
            landmarks.append({"name": f"LM{i + 1:03d}", "x": pos[0], "y": pos[1]})
        return landmarks

    free = [(x, y) for x in range(w) for y in range(h) if (x, y) not in occupied]
    if len(free) < n:
        raise ValueError(f"Only {len(free)} free cells, cannot place {n} landmarks.")
    picks = random.sample(free, n)
    return [{"name": f"LM{i + 1:03d}", "x": x, "y": y}
            for i, (x, y) in enumerate(picks)]


def load_normalized(map_file, res=None, n_landmarks=4):
    """加载地图并归一化为原生整数格子格式。

    - 已知损坏自动修复（仅内存）。
    - 浮点坐标地图按 res (默认 0.1m) 量化、裁包围盒、平移到 (0,0)，附米制元数据。
    - landmarks 为空时自动生成 n_landmarks 个（默认 4 个四角）。
    """
    text = _read_map_text(map_file)
    data = _parse(text)

    width = data.get("width")
    height = data.get("height")
    raw_obstacles = data.get("obstacles", [])
    landmarks = data.get("landmarks", [])

    obstacles = []
    for o in raw_obstacles:
        ox, oy = (o["x"], o["y"]) if isinstance(o, dict) else (o[0], o[1])
        obstacles.append((ox, oy))

    # 检测是否有浮点坐标 → 外部连续坐标地图，需要量化
    coords = obstacles + [(lm.get("x"), lm.get("y")) for lm in landmarks]
    has_float = any(
        ox is not None and (not isinstance(ox, int) and _is_float_coord(ox))
        for ox, _ in coords
    ) or any(
        oy is not None and (not isinstance(oy, int) and _is_float_coord(oy))
        for _, oy in coords
    )

    res = data.get("_resolution_m", DEFAULT_RES_M) if res is None else res
    offset = [0.0, 0.0]
    normalized = False

    if has_float:
        # 浮点 → 0.1m 整数格：用 round 避免 floor(x/0.1) 的二进制误差
        quant = lambda v: int(round(v / res))  # noqa: E731
        occ = {(quant(ox), quant(oy)) for ox, oy in obstacles}
        min_cx = min(c for c, _ in occ)
        max_cx = max(c for c, _ in occ)
        min_cy = min(c for _, c in occ)
        max_cy = max(c for _, c in occ)
        width = max_cx - min_cx + 1
        height = max_cy - min_cy + 1
        obstacles = [(cx - min_cx, cy - min_cy) for cx, cy in occ]
        offset = [min_cx * res, min_cy * res]
        normalized = True

        # 浮点地图的 landmarks（若有）同样量化
        quant_lms = []
        for lm in landmarks:
            quant_lms.append({"name": lm["name"],
                              "x": quant(lm["x"]) - min_cx,
                              "y": quant(lm["y"]) - min_cy})
        landmarks = quant_lms

        print(f"[agv_map_common] normalized float map: grid={width}x{height}, "
              f"cells={len(obstacles)}, offset=({offset[0]:.1f},{offset[1]:.1f})m, "
              f"res={res}m/cell")
    else:
        # 原生整数地图：原样透传（含负坐标 centered 图），保留可能携带的米制元数据
        obstacles = [(int(ox), int(oy)) for ox, oy in obstacles]
        offset = list(data.get("_origin_offset_m", [0.0, 0.0]))
        res = data.get("_resolution_m", 1.0)
        print(f"[agv_map_common] loaded native map: {width}x{height}, "
              f"cells={len(obstacles)}, landmarks={len(landmarks)}")

    if not landmarks:
        landmarks = _auto_landmarks(set(obstacles), width, height, n=n_landmarks)
        print(f"[agv_map_common] auto-generated landmarks: "
              f"{[lm['name'] for lm in landmarks]}")
        if not landmarks:
            raise ValueError("Map has no free cell for landmarks — fully blocked.")

    return {
        "width": int(width),
        "height": int(height),
        "obstacles": [list(o) for o in obstacles],
        "landmarks": landmarks,
        "_resolution_m": res,
        "_origin_offset_m": offset,
        "_normalized": normalized,
    }


def cell_to_meters(x, y, res, offset):
    """整数格 → 真实米制坐标。"""
    return x * res + offset[0], y * res + offset[1]