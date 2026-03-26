#!/usr/bin/env python3
import argparse
import csv
import math
import os
from typing import Dict, List, Optional, Tuple

import yaml


def read_camera_info_yaml(path: str) -> Tuple[float, float, float, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "/**" in data and "ros__parameters" in data["/**"]:
        data = data["/**"]["ros__parameters"]

    K = None
    if "camera_matrix" in data:
        cm = data["camera_matrix"]
        if isinstance(cm, dict) and "data" in cm:
            K = cm["data"]
    if K is None and "K" in data:
        K = data["K"]
    if K is None or len(K) < 9:
        raise RuntimeError(f"Cannot find valid camera matrix in {path}")

    fx = float(K[0])
    fy = float(K[4])
    cx = float(K[2])
    cy = float(K[5])
    return fx, fy, cx, cy


def to_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> List[List[float]]:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def project_point_cam_axispose(p: Tuple[float, float, float], fx: float, fy: float, cx: float, cy: float) -> Optional[Tuple[float, float]]:
    x, y, z = p
    if x <= 1e-9:
        return None
    u = -(y / x) * fx + cx
    v = -(z / x) * fy + cy
    return (u, v)


def project_point_cam_opencv(p: Tuple[float, float, float], fx: float, fy: float, cx: float, cy: float) -> Optional[Tuple[float, float]]:
    x, y, z = p
    if z <= 1e-9:
        return None
    u = (x / z) * fx + cx
    v = (y / z) * fy + cy
    return (u, v)


def choose_projection_model(pose_rows: List[Dict[str, str]], pose_prefix: str, user_choice: str) -> str:
    if user_choice in ("axispose", "opencv"):
        return user_choice

    tx_vals: List[float] = []
    tz_vals: List[float] = []
    for r in pose_rows:
        tx = to_float(r.get(f"{pose_prefix}tx", ""))
        tz = to_float(r.get(f"{pose_prefix}tz", ""))
        if tx is not None:
            tx_vals.append(tx)
        if tz is not None:
            tz_vals.append(tz)

    if not tx_vals or not tz_vals:
        return "axispose"

    pos_tx = sum(1 for v in tx_vals if v > 1e-6) / max(1, len(tx_vals))
    pos_tz = sum(1 for v in tz_vals if v > 1e-6) / max(1, len(tz_vals))
    med_abs_tx = sorted(abs(v) for v in tx_vals)[len(tx_vals) // 2]
    med_abs_tz = sorted(abs(v) for v in tz_vals)[len(tz_vals) // 2]

    # Heuristic: OpenCV camera frame usually has positive Z depth and |tz| much larger than |tx|.
    if pos_tz > 0.8 and med_abs_tz > med_abs_tx * 1.5:
        return "opencv"
    if pos_tx > 0.8:
        return "axispose"
    # Fallback to OpenCV for GT-like data where x may change sign around zero.
    return "opencv"


def line_from_2pts(p1: Tuple[float, float], p2: Tuple[float, float]) -> Optional[Tuple[float, float, float]]:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return None
    dx /= n
    dy /= n
    a = -dy
    b = dx
    c = -(a * x1 + b * y1)
    return (a, b, c)


def intersect_lines(l1: Tuple[float, float, float], l2: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    u = (b1 * c2 - b2 * c1) / det
    v = (a2 * c1 - a1 * c2) / det
    return (u, v)


def point_line_residual(p: Tuple[float, float], l: Tuple[float, float, float]) -> float:
    u, v = p
    a, b, c = l
    den = math.hypot(a, b)
    if den < 1e-12:
        return float("nan")
    return abs(a * u + b * v + c) / den


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute upper/lower-line intersection and Tag-Y-line residual from line2d metrics.")
    ap.add_argument("--line2d", required=True, help="line2d_metrics.csv with upper/lower line coefficients")
    ap.add_argument("--aligned", required=True, help="aligned.csv containing tag pose (default: gt_* columns)")
    ap.add_argument("--camera-info", required=True, help="camera_info yaml")
    ap.add_argument("--output", default="statistics/line2d_tag_residual.csv", help="output csv path")
    ap.add_argument("--axis-length", type=float, default=0.10, help="Tag local Y-axis length in meters for projection")
    ap.add_argument("--pose-prefix", default="gt_", help="pose prefix in aligned csv, e.g. gt_ or est_")
    ap.add_argument(
        "--projection-model",
        default="auto",
        choices=["auto", "axispose", "opencv"],
        help="Projection convention for pose columns: axispose(x-forward), opencv(z-forward), or auto",
    )
    args = ap.parse_args()

    fx, fy, cx, cy = read_camera_info_yaml(args.camera_info)
    line_rows = read_csv_rows(args.line2d)
    pose_rows = read_csv_rows(args.aligned)
    proj_model = choose_projection_model(pose_rows, args.pose_prefix, args.projection_model)
    proj_fn = project_point_cam_axispose if proj_model == "axispose" else project_point_cam_opencv

    print(f"Projection model: {proj_model}")

    out_rows: List[List[object]] = []
    valid_residuals: List[float] = []

    for r in line_rows:
        frame_idx = int(float(r.get("frame_idx", "-1")))
        if frame_idx < 0 or frame_idx >= len(pose_rows):
            continue

        ua = to_float(r.get("upper_a", ""))
        ub = to_float(r.get("upper_b", ""))
        uc = to_float(r.get("upper_c", ""))
        la = to_float(r.get("lower_a", ""))
        lb = to_float(r.get("lower_b", ""))
        lc = to_float(r.get("lower_c", ""))

        uv = None
        if None not in (ua, ub, uc, la, lb, lc):
            uv = intersect_lines((ua, ub, uc), (la, lb, lc))

        prow = pose_rows[frame_idx]
        pfx = args.pose_prefix
        tx = to_float(prow.get(f"{pfx}tx", ""))
        ty = to_float(prow.get(f"{pfx}ty", ""))
        tz = to_float(prow.get(f"{pfx}tz", ""))
        qx = to_float(prow.get(f"{pfx}qx", ""))
        qy = to_float(prow.get(f"{pfx}qy", ""))
        qz = to_float(prow.get(f"{pfx}qz", ""))
        qw = to_float(prow.get(f"{pfx}qw", ""))

        tag_line = None
        if None not in (tx, ty, tz, qx, qy, qz, qw):
            R = quat_to_rot(qx, qy, qz, qw)
            y_dir = (R[0][1], R[1][1], R[2][1])
            p0 = (tx, ty, tz)
            p1 = (
                tx + y_dir[0] * args.axis_length,
                ty + y_dir[1] * args.axis_length,
                tz + y_dir[2] * args.axis_length,
            )
            uv0 = proj_fn(p0, fx, fy, cx, cy)
            uv1 = proj_fn(p1, fx, fy, cx, cy)
            if uv0 is not None and uv1 is not None:
                tag_line = line_from_2pts(uv0, uv1)

        residual = float("nan")
        if uv is not None and tag_line is not None:
            residual = point_line_residual(uv, tag_line)
            if not math.isnan(residual):
                valid_residuals.append(residual)

        out_rows.append([
            frame_idx,
            r.get("timestamp_sec", ""),
            r.get("timestamp_nsec", ""),
            "" if uv is None else uv[0],
            "" if uv is None else uv[1],
            "" if tag_line is None else tag_line[0],
            "" if tag_line is None else tag_line[1],
            "" if tag_line is None else tag_line[2],
            "" if math.isnan(residual) else residual,
        ])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx",
            "timestamp_sec",
            "timestamp_nsec",
            "intersect_u",
            "intersect_v",
            "tagy_a",
            "tagy_b",
            "tagy_c",
            "tagy_residual_px",
        ])
        w.writerows(out_rows)

    print(f"Rows written: {len(out_rows)}")
    if valid_residuals:
        avg = sum(valid_residuals) / len(valid_residuals)
        med = sorted(valid_residuals)[len(valid_residuals) // 2]
        mx = max(valid_residuals)
        print(f"Valid residual count: {len(valid_residuals)}")
        print(f"Residual mean px: {avg:.6f}")
        print(f"Residual median px: {med:.6f}")
        print(f"Residual max px: {mx:.6f}")
    else:
        print("Valid residual count: 0")


if __name__ == "__main__":
    main()
