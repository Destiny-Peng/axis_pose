#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def to_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def read_csv_by_key(path: Path, key_cols: Tuple[str, ...]) -> Dict[Tuple[str, ...], Dict[str, str]]:
    out: Dict[Tuple[str, ...], Dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = tuple(row.get(k, "") for k in key_cols)
            out[key] = row
    return out


def parse_vis_key(name: str) -> Optional[Tuple[str, ...]]:
    # Example: vis_20260326_173941_459_459965190.png
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    nsec = parts[-1]
    return (nsec,)


def line_segment_from_abc(a: float, b: float, c: float, w: int, h: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    pts: List[Tuple[float, float]] = []

    # x = 0 and x = w-1
    if abs(b) > 1e-12:
        y0 = -(a * 0.0 + c) / b
        yw = -(a * float(w - 1) + c) / b
        if 0.0 <= y0 <= float(h - 1):
            pts.append((0.0, y0))
        if 0.0 <= yw <= float(h - 1):
            pts.append((float(w - 1), yw))

    # y = 0 and y = h-1
    if abs(a) > 1e-12:
        x0 = -(b * 0.0 + c) / a
        xh = -(b * float(h - 1) + c) / a
        if 0.0 <= x0 <= float(w - 1):
            pts.append((x0, 0.0))
        if 0.0 <= xh <= float(w - 1):
            pts.append((xh, float(h - 1)))

    if len(pts) < 2:
        return None

    # deduplicate and pick farthest pair
    uniq: List[Tuple[float, float]] = []
    for p in pts:
        keep = True
        for q in uniq:
            if math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-3:
                keep = False
                break
        if keep:
            uniq.append(p)
    if len(uniq) < 2:
        return None

    best = (uniq[0], uniq[1])
    best_d = -1.0
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            d = math.hypot(uniq[i][0] - uniq[j][0], uniq[i][1] - uniq[j][1])
            if d > best_d:
                best_d = d
                best = (uniq[i], uniq[j])

    p1 = (int(round(best[0][0])), int(round(best[0][1])))
    p2 = (int(round(best[1][0])), int(round(best[1][1])))
    return p1, p2


def choose_power2_scale(u: Optional[float], v: Optional[float], src_w: int, src_h: int, dst_w: int, dst_h: int) -> int:
    if u is None or v is None or not math.isfinite(u) or not math.isfinite(v):
        return 1

    # Scale around source center into destination center.
    cx_s = 0.5 * (src_w - 1)
    cy_s = 0.5 * (src_h - 1)

    dx = abs(u - cx_s)
    dy = abs(v - cy_s)

    # Keep intersection inside 90% of destination extent.
    sx = dx / max(1.0, 0.45 * dst_w)
    sy = dy / max(1.0, 0.45 * dst_h)
    req = max(1.0, sx, sy)

    s = 1
    while s < req:
        s *= 2
    return s


def transform_line_to_dst(a: float, b: float, c: float, scale: int, src_w: int, src_h: int, dst_w: int, dst_h: int) -> Tuple[float, float, float]:
    cx_s = 0.5 * (src_w - 1)
    cy_s = 0.5 * (src_h - 1)
    cx_d = 0.5 * (dst_w - 1)
    cy_d = 0.5 * (dst_h - 1)

    # x' = cx_d + (x - cx_s)/s, y' = cy_d + (y - cy_s)/s
    # line in dst: A' x' + B' y' + C' = 0
    ap = a * scale
    bp = b * scale
    cp = a * (cx_s - scale * cx_d) + b * (cy_s - scale * cy_d) + c
    return ap, bp, cp


def warp_to_canvas(src: np.ndarray, scale: int, dst_w: int, dst_h: int) -> np.ndarray:
    src_h, src_w = src.shape[:2]
    cx_s = 0.5 * (src_w - 1)
    cy_s = 0.5 * (src_h - 1)
    cx_d = 0.5 * (dst_w - 1)
    cy_d = 0.5 * (dst_h - 1)

    m = np.array(
        [
            [1.0 / float(scale), 0.0, cx_d - cx_s / float(scale)],
            [0.0, 1.0 / float(scale), cy_d - cy_s / float(scale)],
        ],
        dtype=np.float32,
    )
    out = cv2.warpAffine(src, m, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw upper/lower/tagY lines on vis images with optional integer power-of-two scaling.")
    ap.add_argument("--vis-dir", default="statistics/vis", help="Input vis directory")
    ap.add_argument("--line2d", default="statistics/line2d_metrics.csv", help="Input line2d_metrics.csv")
    ap.add_argument("--residual", default="statistics/line2d_tag_residual.csv", help="Input line2d_tag_residual.csv")
    ap.add_argument("--output-dir", default="statistics/vis_three_lines", help="Output directory")
    ap.add_argument("--width", type=int, default=1280, help="Output canvas width")
    ap.add_argument("--height", type=int, default=720, help="Output canvas height")
    ap.add_argument("--scale-threshold", type=float, default=10000.0, help="Enable scaling when |u| or |v| exceeds this threshold")
    args = ap.parse_args()

    vis_dir = Path(args.vis_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    l2 = read_csv_by_key(Path(args.line2d), ("timestamp_nsec",))
    lr = read_csv_by_key(Path(args.residual), ("timestamp_nsec",))

    images = sorted([p for p in vis_dir.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")])

    written = 0
    for p in images:
        key = parse_vis_key(p.name)
        if key is None:
            continue
        if key not in l2 or key not in lr:
            continue

        row_l2 = l2[key]
        row_lr = lr[key]

        ua = to_float(row_l2.get("upper_a", ""))
        ub = to_float(row_l2.get("upper_b", ""))
        uc = to_float(row_l2.get("upper_c", ""))
        la = to_float(row_l2.get("lower_a", ""))
        lb = to_float(row_l2.get("lower_b", ""))
        lc = to_float(row_l2.get("lower_c", ""))

        ta = to_float(row_lr.get("tagy_a", ""))
        tb = to_float(row_lr.get("tagy_b", ""))
        tc = to_float(row_lr.get("tagy_c", ""))

        iu = to_float(row_lr.get("intersect_u", ""))
        iv = to_float(row_lr.get("intersect_v", ""))
        residual = to_float(row_lr.get("tagy_residual_px", ""))

        src = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if src is None:
            continue
        src_h, src_w = src.shape[:2]

        need_scale = False
        if iu is not None and iv is not None and math.isfinite(iu) and math.isfinite(iv):
            # Scale when intersection is far away OR simply outside the source image.
            out_of_src = (iu < 0.0 or iu > float(src_w - 1) or iv < 0.0 or iv > float(src_h - 1))
            too_far = (abs(iu) > args.scale_threshold or abs(iv) > args.scale_threshold)
            if out_of_src or too_far:
                need_scale = True

        scale = choose_power2_scale(iu, iv, src_w, src_h, args.width, args.height) if need_scale else 1
        canvas = warp_to_canvas(src, scale, args.width, args.height)

        def draw_one_line(a: Optional[float], b: Optional[float], c: Optional[float], color: Tuple[int, int, int], name: str) -> None:
            if a is None or b is None or c is None:
                return
            ap_, bp_, cp_ = transform_line_to_dst(a, b, c, scale, src_w, src_h, args.width, args.height)
            seg = line_segment_from_abc(ap_, bp_, cp_, args.width, args.height)
            if seg is None:
                return
            cv2.line(canvas, seg[0], seg[1], color, 2, cv2.LINE_AA)
            cv2.putText(canvas, name, (seg[0][0] + 6, seg[0][1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        draw_one_line(ua, ub, uc, (0, 255, 255), "upper")
        draw_one_line(la, lb, lc, (0, 255, 0), "lower")
        draw_one_line(ta, tb, tc, (0, 0, 255), "tagY")

        if iu is not None and iv is not None and math.isfinite(iu) and math.isfinite(iv):
            cx_s = 0.5 * (src_w - 1)
            cy_s = 0.5 * (src_h - 1)
            cx_d = 0.5 * (args.width - 1)
            cy_d = 0.5 * (args.height - 1)
            x = cx_d + (iu - cx_s) / float(scale)
            y = cy_d + (iv - cy_s) / float(scale)
            if 0 <= x < args.width and 0 <= y < args.height:
                cv2.circle(canvas, (int(round(x)), int(round(y))), 5, (255, 255, 255), -1)

        info = f"frame={row_l2.get('frame_idx','')} scale={scale}"
        if residual is not None and math.isfinite(residual):
            info += f" residual={residual:.3f}px"
        cv2.putText(canvas, info, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = out_dir / p.name
        cv2.imwrite(str(out_path), canvas)
        written += 1

    print(f"Input images: {len(images)}")
    print(f"Written images: {written}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
