"""
agv_map_convert.py — 把商用 AGV 地图导出文件（如 wangting...workflow2.json）转换为
本仓库 agv_simulation-main 原生 map_*.json 格式；并支持把 plan 轨迹导出为真实米制浮点。

所有解析/修复/量化逻辑复用 agv_map_common.load_normalized，保证与引擎入口行为一致。

用法：
  python src/agv_map_convert.py maps/wangting...json --res 0.1 --out map_wangting.json
  python src/agv_map_convert.py --to-meters plan_xxx.json           # plan 格子 → 米制浮点
"""
import argparse
import datetime
import json
import os

from agv_map_common import load_normalized


def convert(map_file, res, out_name, validate, n_landmarks=4, seed=None):
    if seed is not None:
        import random
        random.seed(seed)
    data = load_normalized(map_file, res=res, n_landmarks=n_landmarks)

    map_data = {
        "width": data["width"],
        "height": data["height"],
        "obstacles": data["obstacles"],
        "landmarks": data["landmarks"],
        "_source": os.path.basename(map_file),
        "_resolution_m": data["_resolution_m"],
        "_origin_offset_m": data["_origin_offset_m"],
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(map_data, f, indent=2)
    print(f"[out] {out_path}")

    if validate:
        from agv_planner import plan_paths
        plan_paths(out_path, num_cars=2)
        print("[validate] A* planning OK")
    return out_path


def to_meters(plan_file):
    """把 plan 轨迹从格子帧换算成真实米制浮点坐标（x*res + offset）。"""
    with open(plan_file, "r", encoding="utf-8") as f:
        plan = json.load(f)

    map_file = plan.get("map_file", "")
    if not map_file or not os.path.exists(map_file):
        raise SystemExit(f"Cannot resolve map file for plan: {map_file}")
    mdata = load_normalized(map_file)
    res = mdata["_resolution_m"]
    ox, oy = mdata["_origin_offset_m"]

    out = dict(plan)
    out["coordinate_system"] = "meters"
    out["_resolution_m"] = res
    out["_origin_offset_m"] = [ox, oy]
    for car in out.get("cars", []):
        for p in car.get("trajectory", []):
            p["x"] = round(p["x"] * res + ox, 4)
            p["y"] = round(p["y"] * res + oy, 4)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(plan_file))[0]
    out_path = os.path.join(os.path.dirname(plan_file), f"{base}_meters_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[to-meters] {out_path}  (res={res}m, offset=({ox},{oy})m)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert external AGV map to native format")
    parser.add_argument("map_file", nargs="?", help="source map JSON (e.g. maps/wangting...json)")
    parser.add_argument("--res", type=float, default=0.1, help="grid resolution in m/cell (default 0.1)")
    parser.add_argument("--out", default="map_wangting.json", help="output filename in maps/")
    parser.add_argument("--no-validate", action="store_true", help="skip A* validation")
    parser.add_argument("--landmarks", type=int, default=4,
                        help="auto-generate N random landmarks if map has none (default 4)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for landmarks")
    parser.add_argument("--to-meters", metavar="PLAN_JSON", help="convert a plan's trajectory to meters")
    args = parser.parse_args()

    if args.to_meters:
        to_meters(args.to_meters)
    elif args.map_file:
        convert(args.map_file, res=args.res, out_name=args.out,
                validate=not args.no_validate, n_landmarks=args.landmarks,
                seed=args.seed)
    else:
        parser.error("provide map_file or --to-meters PLAN_JSON")