#!/usr/bin/env python3
"""
Aggregate per-algorithm evaluation outputs into unified tables and optional plots.

Expected layout under --out-root:
  <out_root>/<algorithm>/metrics.csv
  <out_root>/<algorithm>/line2d_summary.csv (optional)
  <out_root>/<algorithm>/gt_eval/evaluation_summary.csv (optional)

Outputs in <output-dir> (default: <out_root>/summary):
  - summary_overall.csv
  - summary_overall.md
  - timing_by_stage.csv
    - static_stability_deltas.csv
  - total_elapsed_mean_ms.png (optional)
  - line2d_compare.png (optional)
  - gt_compare.png (optional)
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median
from datetime import datetime


def to_float(v):
    try:
        if v is None or v == "":
            return None
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def percentile(vals, p):
    if not vals:
        return None
    if p <= 0:
        return min(vals)
    if p >= 100:
        return max(vals)
    xs = sorted(vals)
    idx = (len(xs) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    w = idx - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def parse_iso_time(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


def vec_norm3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


def quat_norm(q):
    qx, qy, qz, qw = q
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n <= 1e-12:
        return None
    return (qx / n, qy / n, qz / n, qw / n)


def quat_angle_deg(q1, q2):
    a = quat_norm(q1)
    b = quat_norm(q2)
    if a is None or b is None:
        return None
    dot = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    dot = max(-1.0, min(1.0, dot))
    ang = 2.0 * math.acos(dot)
    return math.degrees(ang)


def robust_jump_threshold(values, absolute_threshold, mad_k):
    if not values:
        return absolute_threshold
    med = median(values)
    abs_dev = [abs(v - med) for v in values]
    mad = median(abs_dev)
    robust_sigma = 1.4826 * mad
    adaptive = med + mad_k * robust_sigma
    return max(absolute_threshold, adaptive)


def read_first_row_csv(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return None


def read_metrics(path: Path):
    if not path.exists():
        return [], [], None, [], 0
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    elapsed_all = []
    by_stage = {}
    by_run = {}
    pose_rows = []
    denoise_count = 0
    for r in rows:
        e = to_float(r.get("elapsed_ms"))
        if e is None:
            continue
        elapsed_all.append(e)

        stage = (r.get("name") or "unknown").strip()
        by_stage.setdefault(stage, []).append(e)
        if stage == "denoise":
            denoise_count += 1

        run_id = (r.get("run_id") or "").strip()
        if run_id:
            by_run[run_id] = by_run.get(run_id, 0.0) + e

        if stage.startswith("pose_"):
            px = to_float(r.get("pose_x"))
            py = to_float(r.get("pose_y"))
            pz = to_float(r.get("pose_z"))
            qx = to_float(r.get("qx"))
            qy = to_float(r.get("qy"))
            qz = to_float(r.get("qz"))
            qw = to_float(r.get("qw"))
            valid_points = to_float(r.get("valid_points"))
            if None not in (px, py, pz, qx, qy, qz, qw):
                pose_rows.append(
                    {
                        "run_id": run_id,
                        "timestamp_iso": r.get("timestamp_iso", ""),
                        "timestamp_s": parse_iso_time(r.get("timestamp_iso", "")),
                        "elapsed_ms": e,
                        "valid_points": valid_points,
                        "pose_x": px,
                        "pose_y": py,
                        "pose_z": pz,
                        "qx": qx,
                        "qy": qy,
                        "qz": qz,
                        "qw": qw,
                    }
                )

    total_mean = None
    if by_run:
        total_mean = mean(list(by_run.values()))
    elif elapsed_all:
        total_mean = mean(elapsed_all)

    stage_rows = []
    for stage, vals in sorted(by_stage.items()):
        stage_rows.append({
            "stage": stage,
            "count": len(vals),
            "mean_ms": mean(vals) if vals else "",
            "median_ms": median(vals) if vals else "",
            "min_ms": min(vals) if vals else "",
            "max_ms": max(vals) if vals else "",
        })

    return rows, stage_rows, total_mean, pose_rows, denoise_count


def evaluate_static_stability(pose_rows, denoise_count, pos_jump_mm, ang_jump_deg, mad_k):
    pose_count = len(pose_rows)
    summary = {
        "pose_count": pose_count,
        "denoise_count": denoise_count,
        "pose_success_rate": (float(pose_count) / float(denoise_count)) if denoise_count > 0 else (1.0 if pose_count > 0 else 0.0),
        "cadence_hz": "",
        "pose_elapsed_mean_ms": "",
        "pose_elapsed_p95_ms": "",
        "pos_delta_mean_mm": "",
        "pos_delta_p95_mm": "",
        "pos_delta_max_mm": "",
        "ang_delta_mean_deg": "",
        "ang_delta_p95_deg": "",
        "ang_delta_max_deg": "",
        "jump_pos_threshold_mm": "",
        "jump_ang_threshold_deg": "",
        "jump_count": 0,
        "jump_rate": "",
        "stable_ratio": "",
        "valid_points_mean": "",
        "valid_points_min": "",
        "valid_points_max": "",
    }

    if pose_count == 0:
        return summary, []

    elapsed_vals = [r["elapsed_ms"] for r in pose_rows if r.get("elapsed_ms") is not None]
    if elapsed_vals:
        summary["pose_elapsed_mean_ms"] = mean(elapsed_vals)
        summary["pose_elapsed_p95_ms"] = percentile(elapsed_vals, 95)

    vp = [r["valid_points"] for r in pose_rows if r.get("valid_points") is not None]
    if vp:
        summary["valid_points_mean"] = mean(vp)
        summary["valid_points_min"] = min(vp)
        summary["valid_points_max"] = max(vp)

    dt_vals = []
    pos_deltas_mm = []
    ang_deltas_deg = []
    per_frame = []

    for i in range(pose_count):
        row = pose_rows[i]
        dt_s = None
        pos_delta_mm = None
        ang_delta = None
        if i > 0:
            prev = pose_rows[i - 1]
            if row["timestamp_s"] is not None and prev["timestamp_s"] is not None:
                dt_s = row["timestamp_s"] - prev["timestamp_s"]
                if dt_s > 1e-6:
                    dt_vals.append(dt_s)

            dx = row["pose_x"] - prev["pose_x"]
            dy = row["pose_y"] - prev["pose_y"]
            dz = row["pose_z"] - prev["pose_z"]
            pos_delta_mm = vec_norm3(dx, dy, dz) * 1000.0
            pos_deltas_mm.append(pos_delta_mm)

            ang_delta = quat_angle_deg(
                (row["qx"], row["qy"], row["qz"], row["qw"]),
                (prev["qx"], prev["qy"], prev["qz"], prev["qw"]),
            )
            if ang_delta is not None:
                ang_deltas_deg.append(ang_delta)

        per_frame.append(
            {
                "index": i,
                "run_id": row["run_id"],
                "timestamp_iso": row["timestamp_iso"],
                "dt_s": dt_s if dt_s is not None else "",
                "valid_points": row["valid_points"] if row["valid_points"] is not None else "",
                "pose_x": row["pose_x"],
                "pose_y": row["pose_y"],
                "pose_z": row["pose_z"],
                "qx": row["qx"],
                "qy": row["qy"],
                "qz": row["qz"],
                "qw": row["qw"],
                "pos_delta_mm": pos_delta_mm if pos_delta_mm is not None else "",
                "ang_delta_deg": ang_delta if ang_delta is not None else "",
                "is_jump_pos": 0,
                "is_jump_ang": 0,
                "is_jump": 0,
            }
        )

    if dt_vals:
        med_dt = median(dt_vals)
        if med_dt > 1e-6:
            summary["cadence_hz"] = 1.0 / med_dt

    if pos_deltas_mm:
        summary["pos_delta_mean_mm"] = mean(pos_deltas_mm)
        summary["pos_delta_p95_mm"] = percentile(pos_deltas_mm, 95)
        summary["pos_delta_max_mm"] = max(pos_deltas_mm)

    if ang_deltas_deg:
        summary["ang_delta_mean_deg"] = mean(ang_deltas_deg)
        summary["ang_delta_p95_deg"] = percentile(ang_deltas_deg, 95)
        summary["ang_delta_max_deg"] = max(ang_deltas_deg)

    pos_threshold = robust_jump_threshold(pos_deltas_mm, pos_jump_mm, mad_k)
    ang_threshold = robust_jump_threshold(ang_deltas_deg, ang_jump_deg, mad_k)
    summary["jump_pos_threshold_mm"] = pos_threshold
    summary["jump_ang_threshold_deg"] = ang_threshold

    jump_count = 0
    for i in range(1, len(per_frame)):
        row = per_frame[i]
        is_jump_pos = int(row["pos_delta_mm"] != "" and row["pos_delta_mm"] > pos_threshold)
        is_jump_ang = int(row["ang_delta_deg"] != "" and row["ang_delta_deg"] > ang_threshold)
        is_jump = int(is_jump_pos or is_jump_ang)
        row["is_jump_pos"] = is_jump_pos
        row["is_jump_ang"] = is_jump_ang
        row["is_jump"] = is_jump
        jump_count += is_jump

    summary["jump_count"] = jump_count
    denom = max(1, pose_count - 1)
    summary["jump_rate"] = float(jump_count) / float(denom)
    summary["stable_ratio"] = 1.0 - summary["jump_rate"]

    return summary, per_frame


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def fmt(v, nd=4):
    if v is None or v == "":
        return ""
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def write_markdown_table(path: Path, rows):
    headers = [
        "algorithm",
        "metrics_rows",
        "total_elapsed_mean_ms",
        "pose_success_rate",
        "jump_rate",
        "pos_delta_p95_mm",
        "ang_delta_p95_deg",
        "line2d_count",
        "line2d_angle_mean_deg",
        "line2d_offset_mean_px",
        "gt_count",
        "gt_trans_rmse",
        "gt_ang_mean_deg",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Aggregated Evaluation Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            vals = [
                r.get("algorithm", ""),
                fmt(to_float(r.get("metrics_rows")), 0),
                fmt(to_float(r.get("total_elapsed_mean_ms")), 3),
                fmt(to_float(r.get("pose_success_rate")), 4),
                fmt(to_float(r.get("jump_rate")), 4),
                fmt(to_float(r.get("pos_delta_p95_mm")), 3),
                fmt(to_float(r.get("ang_delta_p95_deg")), 3),
                fmt(to_float(r.get("line2d_count")), 0),
                fmt(to_float(r.get("line2d_angle_mean_deg")), 4),
                fmt(to_float(r.get("line2d_offset_mean_px")), 4),
                fmt(to_float(r.get("gt_count")), 0),
                fmt(to_float(r.get("gt_trans_rmse")), 4),
                fmt(to_float(r.get("gt_ang_mean_deg")), 4),
            ]
            f.write("| " + " | ".join(vals) + " |\n")

        f.write("\n\n## 指标说明\n\n")
        f.write("| 指标 | 含义 | 计算公式 | 单位 |\n")
        f.write("|---|---|---|---|\n")
        f.write("| metrics_rows | 该算法 metrics.csv 记录总行数 | 行计数 | 条 |\n")
        f.write("| total_elapsed_mean_ms | 每个 run_id 的阶段耗时求和后再取均值 | mean(sum(elapsed_ms by run_id)) | ms |\n")
        f.write("| pose_success_rate | 位姿阶段成功率（pose记录数/denoise记录数） | pose_count / denoise_count | 比例(0-1) |\n")
        f.write("| jump_rate | 跳变率 | jump_count / max(1, pose_count-1) | 比例(0-1) |\n")
        f.write("| stable_ratio | 稳定率 | 1 - jump_rate | 比例(0-1) |\n")
        f.write("| pos_delta_p95_mm | 相邻帧位置增量的 P95 | P95( sqrt(dx^2+dy^2+dz^2 ) * 1000 ) | mm |\n")
        f.write("| ang_delta_p95_deg | 相邻帧姿态角增量的 P95 | P95( 2*acos(|q_t·q_{t-1}|) * 180/pi ) | deg |\n")
        f.write("| line2d_count | line2d 评估样本数 | count | 条 |\n")
        f.write("| line2d_angle_mean_deg | line2d 平均角误差 | mean(angle_error) | deg |\n")
        f.write("| line2d_offset_mean_px | line2d 平均偏移误差 | mean(offset_error) | px |\n")
        f.write("| gt_count | GT 对齐有效样本数 | count | 条 |\n")
        f.write("| gt_trans_rmse | 平移 RMSE | sqrt(mean(trans_error^2)) | m |\n")
        f.write("| gt_ang_mean_deg | 角误差均值 | mean(angle_error) | deg |\n")
        f.write("\n跳变判定规则：`is_jump = (pos_delta_mm > jump_pos_threshold_mm) OR (ang_delta_deg > jump_ang_threshold_deg)`，其中阈值为“绝对阈值”和“MAD自适应阈值”取较大者。\n")


def _row_float(row, key):
    return to_float(row.get(key))


def _mean_quaternion(rows):
    qs = []
    for r in rows:
        qx = _row_float(r, "qx")
        qy = _row_float(r, "qy")
        qz = _row_float(r, "qz")
        qw = _row_float(r, "qw")
        if None in (qx, qy, qz, qw):
            continue
        q = quat_norm((qx, qy, qz, qw))
        if q is not None:
            qs.append(q)
    if not qs:
        return None
    mx = mean([q[0] for q in qs])
    my = mean([q[1] for q in qs])
    mz = mean([q[2] for q in qs])
    mw = mean([q[3] for q in qs])
    return quat_norm((mx, my, mz, mw))


def _quat_axis_x(qx, qy, qz, qw):
    q = quat_norm((qx, qy, qz, qw))
    if q is None:
        return None
    x, y, z, w = q
    # World-frame direction of local +X axis.
    dx = 1.0 - 2.0 * (y * y + z * z)
    dy = 2.0 * (x * y + z * w)
    dz = 2.0 * (x * z - y * w)
    n = vec_norm3(dx, dy, dz)
    if n <= 1e-12:
        return None
    return (dx / n, dy / n, dz / n)


def _alpha_by_time(index_value, idx_min, idx_max, a_min=0.12, a_max=0.95):
    if idx_min is None or idx_max is None or idx_max <= idx_min:
        return a_max
    t = (index_value - idx_min) / float(idx_max - idx_min)
    t = max(0.0, min(1.0, t))
    return a_min + (a_max - a_min) * t


def _collect_pose_entries(rows):
    entries = []
    for r in rows:
        idx = _row_float(r, "index")
        x = _row_float(r, "pose_x")
        y = _row_float(r, "pose_y")
        z = _row_float(r, "pose_z")
        qx = _row_float(r, "qx")
        qy = _row_float(r, "qy")
        qz = _row_float(r, "qz")
        qw = _row_float(r, "qw")
        if None in (idx, x, y, z, qx, qy, qz, qw):
            continue
        axis = _quat_axis_x(qx, qy, qz, qw)
        if axis is None:
            continue
        entries.append(
            {
                "index": int(idx),
                "x": x,
                "y": y,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
                "axis": axis,
                "is_jump": int(to_float(r.get("is_jump")) or 0),
            }
        )
    entries.sort(key=lambda t: t["index"])
    return entries


def _arrow_length_from_positions(entries):
    if not entries:
        return 0.06
    xs = [e["x"] for e in entries]
    ys = [e["y"] for e in entries]
    zs = [e["z"] for e in entries]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    diag = vec_norm3(dx, dy, dz)
    if diag <= 1e-9:
        return 0.06
    return max(0.03, min(0.20, diag * 0.06))


def _plot_algorithm_attitude(alg, alg_rows, out_dir: Path, plt):
    idx = []
    qxv, qyv, qzv, qwv = [], [], [], []
    jump_idx = []
    for r in alg_rows:
        i = _row_float(r, "index")
        qx = _row_float(r, "qx")
        qy = _row_float(r, "qy")
        qz = _row_float(r, "qz")
        qw = _row_float(r, "qw")
        if None in (i, qx, qy, qz, qw):
            continue
        idx.append(int(i))
        qxv.append(qx)
        qyv.append(qy)
        qzv.append(qz)
        qwv.append(qw)
        if int(to_float(r.get("is_jump")) or 0) == 1:
            jump_idx.append(int(i))

    if not idx:
        return

    mq = _mean_quaternion(alg_rows)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(idx, qxv, lw=1.3, label="qx")
    ax.plot(idx, qyv, lw=1.3, label="qy")
    ax.plot(idx, qzv, lw=1.3, label="qz")
    ax.plot(idx, qwv, lw=1.3, label="qw")

    if jump_idx:
        ax.scatter(jump_idx, [qwv[min(max(i - idx[0], 0), len(qwv) - 1)] for i in jump_idx],
                   s=30, c="#dc2626", alpha=0.8, label="jump frames (on qw)")

    if mq is not None:
        ax.axhline(mq[0], ls="--", lw=1, color="#1d4ed8", alpha=0.8)
        ax.axhline(mq[1], ls="--", lw=1, color="#0f766e", alpha=0.8)
        ax.axhline(mq[2], ls="--", lw=1, color="#a16207", alpha=0.8)
        ax.axhline(mq[3], ls="--", lw=1, color="#7c2d12", alpha=0.8)

    ax.set_title(f"Jump attitude ({alg})")
    ax.set_xlabel("frame index")
    ax.set_ylabel("quaternion component")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    txt = ""
    if mq is not None:
        txt = f"mean q=({mq[0]:.4f}, {mq[1]:.4f}, {mq[2]:.4f}, {mq[3]:.4f})"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"jump_attitude_{alg}.png")
    plt.close(fig)


def _plot_algorithm_pose_view_single(alg, entries, mean_xyz, mean_dir, jump_rate, out_dir: Path, plt, view_name):
    if not entries:
        return
    mx, my, mz = mean_xyz
    arrow_len = _arrow_length_from_positions(entries)
    mean_len = arrow_len * 1.8
    idx_min = entries[0]["index"]
    idx_max = entries[-1]["index"]

    if view_name == "3d":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for e in entries:
            ux, uy, uz = e["axis"]
            alpha = _alpha_by_time(e["index"], idx_min, idx_max)
            is_jump = e["is_jump"] == 1
            color = (0.86, 0.15, 0.15, alpha) if is_jump else (0.10, 0.55, 0.20, alpha)
            width = 1.7 if is_jump else 0.9
            ax.quiver(e["x"], e["y"], e["z"], ux * arrow_len, uy * arrow_len, uz * arrow_len,
                      color=color, linewidth=width, arrow_length_ratio=0.25)

        ax.scatter([mx], [my], [mz], s=200, c="#1d4ed8", marker="*", label="mean pose")
        if mean_dir is not None:
            mdx, mdy, mdz = mean_dir
            ax.quiver(mx, my, mz, mdx * mean_len, mdy * mean_len, mdz * mean_len,
                      color="#1d4ed8", linewidth=3.0, arrow_length_ratio=0.35)
        t = f"{alg} pose arrows 3D"
        if jump_rate is not None:
            t += f"  jump_rate={jump_rate:.3f}"
        ax.set_title(t)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.text2D(0.02, 0.97, "normal=green jump=red, early=more transparent", transform=ax.transAxes, va="top")
        fig.tight_layout()
        fig.savefig(out_dir / "pose_arrows_3d.png")
        plt.close(fig)
        return

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    for e in entries:
        ux, uy, uz = e["axis"]
        alpha = _alpha_by_time(e["index"], idx_min, idx_max)
        is_jump = e["is_jump"] == 1
        color = (0.86, 0.15, 0.15, alpha) if is_jump else (0.10, 0.55, 0.20, alpha)
        width = 0.0035 if is_jump else 0.0025
        if view_name == "xy":
            ax.quiver(e["x"], e["y"], ux * arrow_len, uy * arrow_len, angles="xy", scale_units="xy", scale=1.0, color=color, width=width)
        elif view_name == "xz":
            ax.quiver(e["x"], e["z"], ux * arrow_len, uz * arrow_len, angles="xy", scale_units="xy", scale=1.0, color=color, width=width)
        elif view_name == "yz":
            ax.quiver(e["y"], e["z"], uy * arrow_len, uz * arrow_len, angles="xy", scale_units="xy", scale=1.0, color=color, width=width)

    ax.scatter([mx if view_name != "yz" else my], [my if view_name == "xy" else (mz if view_name in ("xz", "yz") else my)],
               s=180, c="#1d4ed8", marker="*")
    if mean_dir is not None:
        mdx, mdy, mdz = mean_dir
        if view_name == "xy":
            ax.quiver(mx, my, mdx * mean_len, mdy * mean_len, angles="xy", scale_units="xy", scale=1.0, color="#1d4ed8", width=0.006)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            fname = "pose_arrows_xy.png"
        elif view_name == "xz":
            ax.quiver(mx, mz, mdx * mean_len, mdz * mean_len, angles="xy", scale_units="xy", scale=1.0, color="#1d4ed8", width=0.006)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
            fname = "pose_arrows_xz.png"
        else:
            ax.quiver(my, mz, mdy * mean_len, mdz * mean_len, angles="xy", scale_units="xy", scale=1.0, color="#1d4ed8", width=0.006)
            ax.set_xlabel("y (m)")
            ax.set_ylabel("z (m)")
            fname = "pose_arrows_yz.png"
    else:
        if view_name == "xy":
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            fname = "pose_arrows_xy.png"
        elif view_name == "xz":
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
            fname = "pose_arrows_xz.png"
        else:
            ax.set_xlabel("y (m)")
            ax.set_ylabel("z (m)")
            fname = "pose_arrows_yz.png"

    title = f"{alg} pose arrows {view_name.upper()}"
    if jump_rate is not None:
        title += f"  jump_rate={jump_rate:.3f}"
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.97, "normal=green jump=red, early=more transparent", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(out_dir / fname)
    plt.close(fig)


def _plot_jump_poses(overall_rows, static_delta_rows, out_dir: Path, plt):
    if not static_delta_rows:
        return

    # Per-algorithm jump plots.
    algs = sorted({r.get("algorithm", "") for r in static_delta_rows if r.get("algorithm")})
    for alg in algs:
        alg_rows = [r for r in static_delta_rows if r.get("algorithm") == alg]
        if not alg_rows:
            continue

        entries = _collect_pose_entries(alg_rows)
        if not entries:
            continue

        xs = [e["x"] for e in entries]
        ys = [e["y"] for e in entries]
        zs = [e["z"] for e in entries]

        mean_x = mean(xs)
        mean_y = mean(ys)
        mean_z = mean(zs)
        mean_q = _mean_quaternion(alg_rows)
        mean_dir = None
        if mean_q is not None:
            mean_dir = _quat_axis_x(mean_q[0], mean_q[1], mean_q[2], mean_q[3])

        jump_rate = None
        for o in overall_rows:
            if o.get("algorithm") == alg:
                jump_rate = to_float(o.get("jump_rate"))
                break

        alg_plot_dir = out_dir / "jump_plots" / alg
        alg_plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_algorithm_pose_view_single(alg, entries, (mean_x, mean_y, mean_z), mean_dir, jump_rate, alg_plot_dir, plt, "3d")
        _plot_algorithm_pose_view_single(alg, entries, (mean_x, mean_y, mean_z), mean_dir, jump_rate, alg_plot_dir, plt, "xy")
        _plot_algorithm_pose_view_single(alg, entries, (mean_x, mean_y, mean_z), mean_dir, jump_rate, alg_plot_dir, plt, "xz")
        _plot_algorithm_pose_view_single(alg, entries, (mean_x, mean_y, mean_z), mean_dir, jump_rate, alg_plot_dir, plt, "yz")
        _plot_algorithm_attitude(alg, alg_rows, out_dir, plt)


def try_plot(overall_rows, static_delta_rows, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN: matplotlib unavailable, skip plots: {exc}")
        return

    algs = [r["algorithm"] for r in overall_rows]

    elapsed_vals = [to_float(r.get("total_elapsed_mean_ms")) for r in overall_rows]
    if any(v is not None for v in elapsed_vals):
        xs = []
        ys = []
        for a, v in zip(algs, elapsed_vals):
            if v is not None:
                xs.append(a)
                ys.append(v)
        if xs:
            plt.figure(figsize=(8, 4))
            plt.bar(xs, ys)
            plt.title("Total elapsed mean (ms)")
            plt.xlabel("algorithm")
            plt.ylabel("ms")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "total_elapsed_mean_ms.png")
            plt.close()

    line_angle = [to_float(r.get("line2d_angle_mean_deg")) for r in overall_rows]
    line_offset = [to_float(r.get("line2d_offset_mean_px")) for r in overall_rows]
    if any(v is not None for v in line_angle) or any(v is not None for v in line_offset):
        import numpy as np

        xs = np.arange(len(algs))
        width = 0.35
        angle_vals = [v if v is not None else 0.0 for v in line_angle]
        offset_vals = [v if v is not None else 0.0 for v in line_offset]

        plt.figure(figsize=(10, 4))
        plt.bar(xs - width / 2, angle_vals, width=width, label="angle_mean_deg")
        plt.bar(xs + width / 2, offset_vals, width=width, label="offset_mean_px")
        plt.xticks(xs, algs)
        plt.title("Line2D summary compare")
        plt.xlabel("algorithm")
        plt.ylabel("value")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "line2d_compare.png")
        plt.close()

    gt_trans = [to_float(r.get("gt_trans_rmse")) for r in overall_rows]
    gt_ang = [to_float(r.get("gt_ang_mean_deg")) for r in overall_rows]
    if any(v is not None for v in gt_trans) or any(v is not None for v in gt_ang):
        import numpy as np

        xs = np.arange(len(algs))
        width = 0.35
        trans_vals = [v if v is not None else 0.0 for v in gt_trans]
        ang_vals = [v if v is not None else 0.0 for v in gt_ang]

        plt.figure(figsize=(10, 4))
        plt.bar(xs - width / 2, trans_vals, width=width, label="trans_rmse")
        plt.bar(xs + width / 2, ang_vals, width=width, label="ang_mean_deg")
        plt.xticks(xs, algs)
        plt.title("Groundtruth summary compare")
        plt.xlabel("algorithm")
        plt.ylabel("value")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "gt_compare.png")
        plt.close()

    jump_vals = [to_float(r.get("jump_rate")) for r in overall_rows]
    if any(v is not None for v in jump_vals):
        xs = []
        ys = []
        for a, v in zip(algs, jump_vals):
            if v is not None:
                xs.append(a)
                ys.append(v)
        if xs:
            plt.figure(figsize=(8, 4))
            plt.bar(xs, ys)
            plt.title("Static jump rate")
            plt.xlabel("algorithm")
            plt.ylabel("jump_rate")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "static_jump_rate.png")
            plt.close()

    _plot_jump_poses(overall_rows, static_delta_rows, out_dir, plt)


def main():
    ap = argparse.ArgumentParser(description="Aggregate eval outputs under one out_root")
    ap.add_argument("--out-root", required=True, help="Root directory produced by tools/eval_runner.py")
    ap.add_argument("--output-dir", default="", help="Directory for aggregated outputs (default: <out_root>/summary)")
    ap.add_argument("--no-plots", action="store_true", help="Disable summary plot generation")
    ap.add_argument("--pos-jump-mm", type=float, default=40.0, help="Absolute position jump threshold in mm")
    ap.add_argument("--ang-jump-deg", type=float, default=12.0, help="Absolute orientation jump threshold in deg")
    ap.add_argument("--jump-mad-k", type=float, default=6.0, help="MAD multiplier for adaptive jump threshold")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    if not out_root.exists() or not out_root.is_dir():
        print(f"ERROR: invalid out_root: {out_root}")
        return 2

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (out_root / "summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    algo_dirs = []
    for p in sorted(out_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "metrics.csv").exists() or (p / "line2d_summary.csv").exists() or (p / "gt_eval" / "evaluation_summary.csv").exists():
            algo_dirs.append(p)

    if not algo_dirs:
        print(f"ERROR: no algorithm result dirs found under: {out_root}")
        return 3

    overall_rows = []
    stage_rows = []
    static_delta_rows = []

    for ad in algo_dirs:
        alg = ad.name

        metrics_path = ad / "metrics.csv"
        metrics_rows, stage_info, total_mean, pose_rows, denoise_count = read_metrics(metrics_path)
        for s in stage_info:
            row = {"algorithm": alg}
            row.update(s)
            stage_rows.append(row)

        static_summary, per_frame = evaluate_static_stability(
            pose_rows,
            denoise_count,
            args.pos_jump_mm,
            args.ang_jump_deg,
            args.jump_mad_k,
        )
        for pf in per_frame:
            item = {"algorithm": alg}
            item.update(pf)
            static_delta_rows.append(item)

        line2d = read_first_row_csv(ad / "line2d_summary.csv") or {}
        gt = read_first_row_csv(ad / "gt_eval" / "evaluation_summary.csv") or {}

        out_item = {
            "algorithm": alg,
            "metrics_rows": len(metrics_rows),
            "total_elapsed_mean_ms": total_mean if total_mean is not None else "",
            "pose_count": static_summary.get("pose_count", ""),
            "denoise_count": static_summary.get("denoise_count", ""),
            "pose_success_rate": static_summary.get("pose_success_rate", ""),
            "cadence_hz": static_summary.get("cadence_hz", ""),
            "pose_elapsed_mean_ms": static_summary.get("pose_elapsed_mean_ms", ""),
            "pose_elapsed_p95_ms": static_summary.get("pose_elapsed_p95_ms", ""),
            "pos_delta_mean_mm": static_summary.get("pos_delta_mean_mm", ""),
            "pos_delta_p95_mm": static_summary.get("pos_delta_p95_mm", ""),
            "pos_delta_max_mm": static_summary.get("pos_delta_max_mm", ""),
            "ang_delta_mean_deg": static_summary.get("ang_delta_mean_deg", ""),
            "ang_delta_p95_deg": static_summary.get("ang_delta_p95_deg", ""),
            "ang_delta_max_deg": static_summary.get("ang_delta_max_deg", ""),
            "jump_pos_threshold_mm": static_summary.get("jump_pos_threshold_mm", ""),
            "jump_ang_threshold_deg": static_summary.get("jump_ang_threshold_deg", ""),
            "jump_count": static_summary.get("jump_count", ""),
            "jump_rate": static_summary.get("jump_rate", ""),
            "stable_ratio": static_summary.get("stable_ratio", ""),
            "valid_points_mean": static_summary.get("valid_points_mean", ""),
            "valid_points_min": static_summary.get("valid_points_min", ""),
            "valid_points_max": static_summary.get("valid_points_max", ""),
            "line2d_count": line2d.get("count", ""),
            "line2d_angle_mean_deg": line2d.get("angle_mean_deg", ""),
            "line2d_offset_mean_px": line2d.get("offset_mean_px", ""),
            "gt_count": gt.get("count", ""),
            "gt_trans_rmse": gt.get("trans_rmse", ""),
            "gt_ang_mean_deg": gt.get("ang_mean_deg", ""),
        }
        overall_rows.append(out_item)

    overall_rows.sort(key=lambda x: x["algorithm"])
    stage_rows.sort(key=lambda x: (x["algorithm"], x["stage"]))

    write_csv(
        output_dir / "summary_overall.csv",
        [
            "algorithm",
            "metrics_rows",
            "total_elapsed_mean_ms",
            "pose_count",
            "denoise_count",
            "pose_success_rate",
            "cadence_hz",
            "pose_elapsed_mean_ms",
            "pose_elapsed_p95_ms",
            "pos_delta_mean_mm",
            "pos_delta_p95_mm",
            "pos_delta_max_mm",
            "ang_delta_mean_deg",
            "ang_delta_p95_deg",
            "ang_delta_max_deg",
            "jump_pos_threshold_mm",
            "jump_ang_threshold_deg",
            "jump_count",
            "jump_rate",
            "stable_ratio",
            "valid_points_mean",
            "valid_points_min",
            "valid_points_max",
            "line2d_count",
            "line2d_angle_mean_deg",
            "line2d_offset_mean_px",
            "gt_count",
            "gt_trans_rmse",
            "gt_ang_mean_deg",
        ],
        overall_rows,
    )

    write_csv(
        output_dir / "timing_by_stage.csv",
        ["algorithm", "stage", "count", "mean_ms", "median_ms", "min_ms", "max_ms"],
        stage_rows,
    )

    write_csv(
        output_dir / "static_stability_deltas.csv",
        [
            "algorithm",
            "index",
            "run_id",
            "timestamp_iso",
            "dt_s",
            "valid_points",
            "pose_x",
            "pose_y",
            "pose_z",
            "qx",
            "qy",
            "qz",
            "qw",
            "pos_delta_mm",
            "ang_delta_deg",
            "is_jump_pos",
            "is_jump_ang",
            "is_jump",
        ],
        static_delta_rows,
    )

    write_markdown_table(output_dir / "summary_overall.md", overall_rows)

    if not args.no_plots:
        try_plot(overall_rows, static_delta_rows, output_dir)

    print(f"Aggregated summary generated in: {output_dir}")
    print(f"  - {output_dir / 'summary_overall.csv'}")
    print(f"  - {output_dir / 'summary_overall.md'}")
    print(f"  - {output_dir / 'timing_by_stage.csv'}")
    print(f"  - {output_dir / 'static_stability_deltas.csv'}")
    if not args.no_plots:
        print(f"  - {output_dir / 'total_elapsed_mean_ms.png'} (if data available)")
        print(f"  - {output_dir / 'line2d_compare.png'} (if data available)")
        print(f"  - {output_dir / 'gt_compare.png'} (if data available)")
        print(f"  - {output_dir / 'static_jump_rate.png'} (if data available)")
        print(f"  - {output_dir / 'jump_attitude_<algorithm>.png'} (if data available)")
        print(f"  - {output_dir / 'jump_plots/<algorithm>/pose_arrows_3d.png'} (if data available)")
        print(f"  - {output_dir / 'jump_plots/<algorithm>/pose_arrows_xy.png'} (if data available)")
        print(f"  - {output_dir / 'jump_plots/<algorithm>/pose_arrows_xz.png'} (if data available)")
        print(f"  - {output_dir / 'jump_plots/<algorithm>/pose_arrows_yz.png'} (if data available)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
