"""
mapframe.py — 米制帧换算 (bridge_wangting)

bridge 物理世界 (MuJoCo / 机器人 / 障碍盒 / ARC 距离) 全部用十进制米；
AGV A* 用 0.1m 整数格。本模块在两者之间做精确双射换算：

    to_meters(cell) = cell * res + offset
    to_cell(meter)  = round((meter - offset) / res)

因为 res 和 offset 都是 0.1 的整数倍、且源数据本身是 0.1 离散点，
往返转换零误差 —— 4.1m 永远对应 cell 190 并转回 4.1m。

从 load_normalized 的元数据构造 (原生地图和源文件都带 _resolution_m / _origin_offset_m)。
"""
from __future__ import annotations


class MapFrame:
    def __init__(self, map_data):
        self.res = float(map_data.get("_resolution_m", 0.1))
        self.ox = float(map_data.get("_origin_offset_m", [0.0, 0.0])[0])
        self.oy = float(map_data.get("_origin_offset_m", [0.0, 0.0])[1])

    # ── cell → meter ──
    def to_meters(self, cx, cy):
        return (cx * self.res + self.ox, cy * self.res + self.oy)

    def cx2m(self, cx):
        return cx * self.res + self.ox

    def cy2m(self, cy):
        return cy * self.res + self.oy

    # ── meter → cell ──
    def to_cell(self, mx, my):
        return (int(round((mx - self.ox) / self.res)),
                int(round((my - self.oy) / self.res)))

    def mx2c(self, mx):
        return int(round((mx - self.ox) / self.res))

    def my2c(self, my):
        return int(round((my - self.oy) / self.res))

    # ── 米制半径 → 格子半径 (ceil, 至少 1 格) ──
    def cell_radius(self, meters):
        return max(1, int(round(meters / self.res)))

    # ── 格子占据的物理矩形 [x0, x1]×[y0, y1] (格子中心 ± res/2) ──
    def cell_footprint(self, cx, cy):
        return (self.cx2m(cx) - self.res / 2.0, self.cx2m(cx) + self.res / 2.0,
                self.cy2m(cy) - self.res / 2.0, self.cy2m(cy) + self.res / 2.0)

    def __repr__(self):
        return f"MapFrame(res={self.res}, offset=({self.ox},{self.oy}))"