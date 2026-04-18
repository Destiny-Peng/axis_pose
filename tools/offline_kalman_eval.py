#!/usr/bin/env python3
"""
Offline Kalman post-filter for pose rows in metrics.csv.

Single-file mode:
    python3 tools/offline_kalman_eval.py \
        --input statistics/axispose_eval/ceres/metrics.csv \
        --output-dir statistics/kalman_offline/ceres

Batch mode (recommended):
    python3 tools/offline_kalman_eval.py \
        --out-root statistics/eval_unified_20260326_210444
"""

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


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
    return math.degrees(2.0 * math.acos(dot))


def robust_jump_threshold(values, absolute_threshold, mad_k):
    if not values:
        return absolute_threshold
    med = median(values)
    mad = median([abs(v - med) for v in values])
    robust_sigma = 1.4826 * mad
    adaptive = med + mad_k * robust_sigma
    return max(absolute_threshold, adaptive)


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


def axis_from_quat(qx, qy, qz, qw):
    q = quat_norm((qx, qy, qz, qw))
    if q is None:
        return None
    x, y, z, w = q
    dx = 1.0 - 2.0 * (y * y + z * z)
    dy = 2.0 * (x * y + z * w)
    dz = 2.0 * (x * z - y * w)
    n = vec_norm3(dx, dy, dz)
    if n <= 1e-12:
        return None
    return (dx / n, dy / n, dz / n)


def quat_from_rot(R):
    r00, r01, r02 = R[0]
    r10, r11, r12 = R[1]
    r20, r21, r22 = R[2]
    tr = r00 + r11 + r22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif (r00 > r11) and (r00 > r22):
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s
    q = quat_norm((qx, qy, qz, qw))
    return q


def quat_from_axis_x(axis):
    ax, ay, az = axis
    up = (0.0, 1.0, 0.0)
    if abs(ax * up[0] + ay * up[1] + az * up[2]) > 0.99:
        up = (0.0, 0.0, 1.0)
    dot_up = up[0] * ax + up[1] * ay + up[2] * az
    yx = up[0] - dot_up * ax
    yy = up[1] - dot_up * ay
    yz = up[2] - dot_up * az
    yn = vec_norm3(yx, yy, yz)
    if yn <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    yx, yy, yz = yx / yn, yy / yn, yz / yn
    zx = ay * yz - az * yy
    zy = az * yx - ax * yz
    zz = ax * yy - ay * yx
    zn = vec_norm3(zx, zy, zz)
    if zn <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    zx, zy, zz = zx / zn, zy / zn, zz / zn
    q = quat_from_rot(((ax, yx, zx), (ay, yy, zy), (az, yz, zz)))
    if q is None:
        return (0.0, 0.0, 0.0, 1.0)
    return q


def split_sessions(rows, gap_sec):
    if not rows:
        return []
    sessions = []
    cur = [rows[0]]
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        now = rows[i]
        if prev["timestamp_s"] is None or now["timestamp_s"] is None:
            cur.append(now)
            continue
        dt = now["timestamp_s"] - prev["timestamp_s"]
        if dt > gap_sec:
            sessions.append(cur)
            cur = [now]
        else:
            cur.append(now)
    sessions.append(cur)
    return sessions


@dataclass
class KF1D:
    x: float
    v: float
    p00: float
    p01: float
    p10: float
    p11: float


def kf_predict_update_1d(st: KF1D, z, dt, q, r):
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    x_pred = st.x + dt * st.v
    v_pred = st.v

    p00 = st.p00 + dt * (st.p10 + st.p01) + dt2 * st.p11 + 0.25 * dt4 * q
    p01 = st.p01 + dt * st.p11 + 0.5 * dt3 * q
    p10 = st.p10 + dt * st.p11 + 0.5 * dt3 * q
    p11 = st.p11 + dt2 * q

    S = p00 + r
    if S <= 1e-12:
        st.x, st.v = x_pred, v_pred
        st.p00, st.p01, st.p10, st.p11 = p00, p01, p10, p11
        return

    k0 = p00 / S
    k1 = p10 / S
    innovation = z - x_pred

    st.x = x_pred + k0 * innovation
    st.v = v_pred + k1 * innovation

    st.p00 = (1.0 - k0) * p00
    st.p01 = (1.0 - k0) * p01
    st.p10 = p10 - k1 * p00
    st.p11 = p11 - k1 * p01


def evaluate_jumps(rows, pos_jump_mm, ang_jump_deg, mad_k):
    pos_deltas = []
    ang_deltas = []
    per_step = []

    for i in range(1, len(rows)):
        a = rows[i - 1]
        b = rows[i]
        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        dz = b["z"] - a["z"]
        pos_mm = vec_norm3(dx, dy, dz) * 1000.0
        ang_deg = quat_angle_deg(a["q"], b["q"])
        if ang_deg is not None:
            ang_deltas.append(ang_deg)
        pos_deltas.append(pos_mm)
        per_step.append((pos_mm, ang_deg))

    pos_thr = robust_jump_threshold(pos_deltas, pos_jump_mm, mad_k)
    ang_thr = robust_jump_threshold(ang_deltas, ang_jump_deg, mad_k)

    jump_count = 0
    jump_flags = []
    for pos_mm, ang_deg in per_step:
        is_jump_pos = int(pos_mm > pos_thr)
        is_jump_ang = int((ang_deg is not None) and (ang_deg > ang_thr))
        is_jump = int(is_jump_pos or is_jump_ang)
        jump_count += is_jump
        jump_flags.append((is_jump_pos, is_jump_ang, is_jump))

    denom = max(1, len(rows) - 1)
    return {
        "pose_count": len(rows),
        "jump_count": jump_count,
        "jump_rate": float(jump_count) / float(denom),
        "stable_ratio": 1.0 - float(jump_count) / float(denom),
        "pos_delta_mean_mm": mean(pos_deltas) if pos_deltas else None,
        "pos_delta_max_mm": max(pos_deltas) if pos_deltas else None,
        "pos_delta_p95_mm": percentile(pos_deltas, 95) if pos_deltas else None,
        "ang_delta_mean_deg": mean(ang_deltas) if ang_deltas else None,
        "ang_delta_max_deg": max(ang_deltas) if ang_deltas else None,
        "ang_delta_p95_deg": percentile(ang_deltas, 95) if ang_deltas else None,
        "jump_pos_threshold_mm": pos_thr,
        "jump_ang_threshold_deg": ang_thr,
        "per_step": per_step,
        "jump_flags": jump_flags,
    }


def write_step_delta_csv(path, raw_stat, filt_stat):
    rows = []
    n = min(len(raw_stat["per_step"]), len(filt_stat["per_step"]))
    for i in range(n):
        raw_pos, raw_ang = raw_stat["per_step"][i]
        kf_pos, kf_ang = filt_stat["per_step"][i]
        raw_jump_pos, raw_jump_ang, raw_jump = raw_stat["jump_flags"][i]
        kf_jump_pos, kf_jump_ang, kf_jump = filt_stat["jump_flags"][i]
        rows.append(
            {
                "step_index": i + 1,
                "raw_pos_delta_mm": raw_pos,
                "kf_pos_delta_mm": kf_pos,
                "raw_ang_delta_deg": raw_ang,
                "kf_ang_delta_deg": kf_ang,
                "raw_jump_pos": raw_jump_pos,
                "raw_jump_ang": raw_jump_ang,
                "raw_jump": raw_jump,
                "kf_jump_pos": kf_jump_pos,
                "kf_jump_ang": kf_jump_ang,
                "kf_jump": kf_jump,
            }
        )

    write_csv(
        path,
        [
            "step_index",
            "raw_pos_delta_mm",
            "kf_pos_delta_mm",
            "raw_ang_delta_deg",
            "kf_ang_delta_deg",
            "raw_jump_pos",
            "raw_jump_ang",
            "raw_jump",
            "kf_jump_pos",
            "kf_jump_ang",
            "kf_jump",
        ],
        rows,
    )


def write_plots(plot_dir, raw_stat, filt_stat):
    plot_dir.mkdir(parents=True, exist_ok=True)

    raw_pos = [v[0] for v in raw_stat["per_step"]]
    kf_pos = [v[0] for v in filt_stat["per_step"]]
    raw_ang = [v[1] for v in raw_stat["per_step"] if v[1] is not None]
    kf_ang = [v[1] for v in filt_stat["per_step"] if v[1] is not None]

    xs = list(range(1, len(raw_pos) + 1))
    plt.figure(figsize=(11, 4.5))
    plt.plot(xs, raw_pos, label="raw", linewidth=1.2)
    plt.plot(xs, kf_pos, label="kalman", linewidth=1.2)
    plt.axhline(raw_stat["jump_pos_threshold_mm"], linestyle="--", linewidth=1.0, label="raw_thr")
    plt.axhline(filt_stat["jump_pos_threshold_mm"], linestyle=":", linewidth=1.0, label="kf_thr")
    plt.title("Position Delta per Step (mm)")
    plt.xlabel("step index")
    plt.ylabel("mm")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "pos_delta_series.png", dpi=150)
    plt.close()

    xs_ang = list(range(1, len(raw_ang) + 1))
    plt.figure(figsize=(11, 4.5))
    plt.plot(xs_ang, raw_ang, label="raw", linewidth=1.2)
    plt.plot(xs_ang, kf_ang, label="kalman", linewidth=1.2)
    plt.axhline(raw_stat["jump_ang_threshold_deg"], linestyle="--", linewidth=1.0, label="raw_thr")
    plt.axhline(filt_stat["jump_ang_threshold_deg"], linestyle=":", linewidth=1.0, label="kf_thr")
    plt.title("Angular Delta per Step (deg)")
    plt.xlabel("step index")
    plt.ylabel("deg")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "ang_delta_series.png", dpi=150)
    plt.close()

    labels = ["pos_mean", "pos_p95", "pos_max", "ang_mean", "ang_p95", "ang_max"]
    raw_bar = [
        raw_stat["pos_delta_mean_mm"],
        raw_stat["pos_delta_p95_mm"],
        raw_stat["pos_delta_max_mm"],
        raw_stat["ang_delta_mean_deg"],
        raw_stat["ang_delta_p95_deg"],
        raw_stat["ang_delta_max_deg"],
    ]
    kf_bar = [
        filt_stat["pos_delta_mean_mm"],
        filt_stat["pos_delta_p95_mm"],
        filt_stat["pos_delta_max_mm"],
        filt_stat["ang_delta_mean_deg"],
        filt_stat["ang_delta_p95_deg"],
        filt_stat["ang_delta_max_deg"],
    ]
    x = list(range(len(labels)))
    w = 0.4
    plt.figure(figsize=(10, 4.8))
    plt.bar([i - w / 2 for i in x], raw_bar, width=w, label="raw")
    plt.bar([i + w / 2 for i in x], kf_bar, width=w, label="kalman")
    plt.xticks(x, labels)
    plt.title("Raw vs Kalman Summary Metrics")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "raw_vs_kf_summary_bars.png", dpi=150)
    plt.close()


def read_pose_rows(metrics_csv):
    rows = []
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            name = (r.get("name") or "").strip()
            if not name.startswith("pose_"):
                continue
            x = to_float(r.get("pose_x"))
            y = to_float(r.get("pose_y"))
            z = to_float(r.get("pose_z"))
            qx = to_float(r.get("qx"))
            qy = to_float(r.get("qy"))
            qz = to_float(r.get("qz"))
            qw = to_float(r.get("qw"))
            if None in (x, y, z, qx, qy, qz, qw):
                continue
            rows.append(
                {
                    "run_id": r.get("run_id", ""),
                    "name": name,
                    "timestamp_iso": r.get("timestamp_iso", ""),
                    "timestamp_s": parse_iso_time(r.get("timestamp_iso", "")),
                    "x": x,
                    "y": y,
                    "z": z,
                    "q": quat_norm((qx, qy, qz, qw)),
                }
            )
    rows = [r for r in rows if r["q"] is not None]
    rows.sort(key=lambda t: (t["timestamp_s"] is None, t["timestamp_s"], t["run_id"]))
    return rows


def run_offline_kalman(rows, args):
    if not rows:
        return [], {"rejected_count": 0, "accepted_count": 0}

    pos_states = [
        KF1D(rows[0]["x"], 0.0, args.kalman_initial_covariance, 0.0, 0.0, args.kalman_initial_covariance),
        KF1D(rows[0]["y"], 0.0, args.kalman_initial_covariance, 0.0, 0.0, args.kalman_initial_covariance),
        KF1D(rows[0]["z"], 0.0, args.kalman_initial_covariance, 0.0, 0.0, args.kalman_initial_covariance),
    ]
    axis0 = axis_from_quat(*rows[0]["q"])
    if axis0 is None:
        axis0 = (1.0, 0.0, 0.0)
    axis_states = [
        KF1D(axis0[0], 0.0, args.kalman_initial_covariance, 0.0, 0.0, args.kalman_initial_covariance),
        KF1D(axis0[1], 0.0, args.kalman_initial_covariance, 0.0, 0.0, args.kalman_initial_covariance),
        KF1D(axis0[2], 0.0, args.kalman_initial_covariance, 0.0, 0.0, args.kalman_initial_covariance),
    ]

    out = []
    prev_ts = rows[0]["timestamp_s"]
    prev_axis = axis0
    prev_kept_raw_q = rows[0]["q"]
    rejected_count = 0
    accepted_count = 1

    for idx, r in enumerate(rows):
        if idx == 0:
            qf = r["q"]
            out.append(
                {
                    **r,
                    "xf": r["x"],
                    "yf": r["y"],
                    "zf": r["z"],
                    "qf": qf,
                    "is_rejected": 0,
                    "reject_reason": "",
                    "raw_ang_to_prev_kept_deg": 0.0,
                }
            )
            continue

        raw_ang_to_prev = quat_angle_deg(prev_kept_raw_q, r["q"])
        if (
            args.hard_reject_ang_deg > 0.0
            and raw_ang_to_prev is not None
            and raw_ang_to_prev > args.hard_reject_ang_deg
        ):
            rejected_count += 1
            prev_out = out[-1]
            out.append(
                {
                    **r,
                    "xf": prev_out["xf"],
                    "yf": prev_out["yf"],
                    "zf": prev_out["zf"],
                    "qf": prev_out["qf"],
                    "is_rejected": 1,
                    "reject_reason": "ang_hard_gate",
                    "raw_ang_to_prev_kept_deg": raw_ang_to_prev,
                }
            )
            continue

        now_ts = r["timestamp_s"]
        dt = args.kalman_min_dt
        if now_ts is not None and prev_ts is not None:
            dt = now_ts - prev_ts
        if (not math.isfinite(dt)) or dt <= 0.0:
            dt = args.kalman_min_dt
        dt = max(args.kalman_min_dt, min(args.kalman_max_dt, dt))

        axis_m = axis_from_quat(*r["q"])
        if axis_m is None:
            axis_m = prev_axis
        if axis_m[0] * prev_axis[0] + axis_m[1] * prev_axis[1] + axis_m[2] * prev_axis[2] < 0.0:
            axis_m = (-axis_m[0], -axis_m[1], -axis_m[2])

        kf_predict_update_1d(pos_states[0], r["x"], dt, args.kalman_position_process_noise, args.kalman_position_measurement_noise)
        kf_predict_update_1d(pos_states[1], r["y"], dt, args.kalman_position_process_noise, args.kalman_position_measurement_noise)
        kf_predict_update_1d(pos_states[2], r["z"], dt, args.kalman_position_process_noise, args.kalman_position_measurement_noise)

        kf_predict_update_1d(axis_states[0], axis_m[0], dt, args.kalman_axis_process_noise, args.kalman_axis_measurement_noise)
        kf_predict_update_1d(axis_states[1], axis_m[1], dt, args.kalman_axis_process_noise, args.kalman_axis_measurement_noise)
        kf_predict_update_1d(axis_states[2], axis_m[2], dt, args.kalman_axis_process_noise, args.kalman_axis_measurement_noise)

        af = (axis_states[0].x, axis_states[1].x, axis_states[2].x)
        an = vec_norm3(*af)
        if an <= 1e-12:
            af = prev_axis
        else:
            af = (af[0] / an, af[1] / an, af[2] / an)
        prev_axis = af

        qf = quat_from_axis_x(af)
        out.append(
            {
                **r,
                "xf": pos_states[0].x,
                "yf": pos_states[1].x,
                "zf": pos_states[2].x,
                "qf": qf,
                "is_rejected": 0,
                "reject_reason": "",
                "raw_ang_to_prev_kept_deg": raw_ang_to_prev,
            }
        )
        prev_ts = now_ts
        prev_kept_raw_q = r["q"]
        accepted_count += 1

    return out, {"rejected_count": rejected_count, "accepted_count": accepted_count}


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown_table(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            vals = []
            for h in headers:
                v = r.get(h, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")


def process_metrics_csv(metrics_csv, output_dir, args, algorithm_name=""):
    metrics_csv = Path(metrics_csv).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pose = read_pose_rows(metrics_csv)
    if not all_pose:
        raise RuntimeError(f"no pose_* rows found in metrics.csv: {metrics_csv}")

    sessions = split_sessions(all_pose, args.session_gap_sec)
    selected = sessions[-1] if args.session_mode == "latest" else all_pose

    raw_rows = [{"x": r["x"], "y": r["y"], "z": r["z"], "q": r["q"]} for r in selected]
    filtered, reject_stat = run_offline_kalman(selected, args)
    filt_rows = [{"x": r["xf"], "y": r["yf"], "z": r["zf"], "q": r["qf"]} for r in filtered]

    raw_stat = evaluate_jumps(raw_rows, args.pos_jump_mm, args.ang_jump_deg, args.jump_mad_k)
    filt_stat = evaluate_jumps(filt_rows, args.pos_jump_mm, args.ang_jump_deg, args.jump_mad_k)

    pose_out = []
    for i, r in enumerate(filtered):
        qx, qy, qz, qw = r["q"]
        qfx, qfy, qfz, qfw = r["qf"]
        pose_out.append(
            {
                "index": i,
                "timestamp_iso": r["timestamp_iso"],
                "run_id": r["run_id"],
                "pose_x_raw": r["x"],
                "pose_y_raw": r["y"],
                "pose_z_raw": r["z"],
                "qx_raw": qx,
                "qy_raw": qy,
                "qz_raw": qz,
                "qw_raw": qw,
                "pose_x_kf": r["xf"],
                "pose_y_kf": r["yf"],
                "pose_z_kf": r["zf"],
                "qx_kf": qfx,
                "qy_kf": qfy,
                "qz_kf": qfz,
                "qw_kf": qfw,
                "is_rejected": r.get("is_rejected", 0),
                "reject_reason": r.get("reject_reason", ""),
                "raw_ang_to_prev_kept_deg": r.get("raw_ang_to_prev_kept_deg", ""),
            }
        )

    write_csv(
        output_dir / "filtered_pose.csv",
        [
            "index", "timestamp_iso", "run_id",
            "pose_x_raw", "pose_y_raw", "pose_z_raw", "qx_raw", "qy_raw", "qz_raw", "qw_raw",
            "pose_x_kf", "pose_y_kf", "pose_z_kf", "qx_kf", "qy_kf", "qz_kf", "qw_kf",
            "is_rejected", "reject_reason", "raw_ang_to_prev_kept_deg",
        ],
        pose_out,
    )

    write_step_delta_csv(output_dir / "step_deltas.csv", raw_stat, filt_stat)

    summary = {
        "algorithm": algorithm_name,
        "metrics_csv": str(metrics_csv),
        "output_dir": str(output_dir),
        "session_mode": args.session_mode,
        "session_count_detected": len(sessions),
        "pose_count_used": len(selected),
        "hard_reject_ang_deg": args.hard_reject_ang_deg,
        "rejected_count": reject_stat["rejected_count"],
        "rejected_ratio": (float(reject_stat["rejected_count"]) / float(max(1, len(selected) - 1))),
        "raw_jump_rate": raw_stat["jump_rate"],
        "kf_jump_rate": filt_stat["jump_rate"],
        "raw_stable_ratio": raw_stat["stable_ratio"],
        "kf_stable_ratio": filt_stat["stable_ratio"],
        "raw_pos_delta_mean_mm": raw_stat["pos_delta_mean_mm"],
        "kf_pos_delta_mean_mm": filt_stat["pos_delta_mean_mm"],
        "raw_pos_delta_p95_mm": raw_stat["pos_delta_p95_mm"],
        "kf_pos_delta_p95_mm": filt_stat["pos_delta_p95_mm"],
        "raw_pos_delta_max_mm": raw_stat["pos_delta_max_mm"],
        "kf_pos_delta_max_mm": filt_stat["pos_delta_max_mm"],
        "raw_ang_delta_mean_deg": raw_stat["ang_delta_mean_deg"],
        "kf_ang_delta_mean_deg": filt_stat["ang_delta_mean_deg"],
        "raw_ang_delta_p95_deg": raw_stat["ang_delta_p95_deg"],
        "kf_ang_delta_p95_deg": filt_stat["ang_delta_p95_deg"],
        "raw_ang_delta_max_deg": raw_stat["ang_delta_max_deg"],
        "kf_ang_delta_max_deg": filt_stat["ang_delta_max_deg"],
        "raw_jump_pos_threshold_mm": raw_stat["jump_pos_threshold_mm"],
        "kf_jump_pos_threshold_mm": filt_stat["jump_pos_threshold_mm"],
        "raw_jump_ang_threshold_deg": raw_stat["jump_ang_threshold_deg"],
        "kf_jump_ang_threshold_deg": filt_stat["jump_ang_threshold_deg"],
    }

    write_csv(output_dir / "summary.csv", list(summary.keys()), [summary])
    write_plots(output_dir / "plots", raw_stat, filt_stat)

    prefix = f"[{algorithm_name}] " if algorithm_name else ""
    print(f"{prefix}Input: {metrics_csv}")
    print(f"{prefix}Session mode: {args.session_mode} (detected sessions={len(sessions)})")
    print(f"{prefix}Pose rows used: {len(selected)}")
    print(f"{prefix}Hard reject ang threshold (deg): {args.hard_reject_ang_deg}")
    print(f"{prefix}Rejected frames: {reject_stat['rejected_count']} / {max(0, len(selected) - 1)}")
    print(f"{prefix}Raw jump_rate: {raw_stat['jump_rate']:.4f}")
    print(f"{prefix}KF  jump_rate: {filt_stat['jump_rate']:.4f}")
    print(f"{prefix}Saved: {output_dir / 'summary.csv'}")
    return summary


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    write_csv(path, fieldnames, rows)


def merge_into_summary_csv(out_root, kalman_rows):
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "summary_overall.csv"

    base_rows = read_csv_rows(summary_csv)
    km = {r.get("algorithm", ""): r for r in kalman_rows}

    add_cols = [
        "kalman_hard_reject_ang_deg",
        "kalman_rejected_ratio",
        "kalman_kf_jump_rate",
        "kalman_kf_stable_ratio",
        "kalman_kf_pos_delta_mean_mm",
        "kalman_kf_pos_delta_p95_mm",
        "kalman_kf_pos_delta_max_mm",
        "kalman_kf_ang_delta_mean_deg",
        "kalman_kf_ang_delta_p95_deg",
        "kalman_kf_ang_delta_max_deg",
    ]

    if not base_rows:
        base_rows = [{"algorithm": r.get("algorithm", "")} for r in kalman_rows]

    existing_algs = set()
    for row in base_rows:
        alg = row.get("algorithm", "")
        existing_algs.add(alg)
        k = km.get(alg)
        for c in add_cols:
            row.setdefault(c, "")
        if k is not None:
            row["kalman_hard_reject_ang_deg"] = k.get("hard_reject_ang_deg", "")
            row["kalman_rejected_ratio"] = k.get("rejected_ratio", "")
            row["kalman_kf_jump_rate"] = k.get("kf_jump_rate", "")
            row["kalman_kf_stable_ratio"] = k.get("kf_stable_ratio", "")
            row["kalman_kf_pos_delta_mean_mm"] = k.get("kf_pos_delta_mean_mm", "")
            row["kalman_kf_pos_delta_p95_mm"] = k.get("kf_pos_delta_p95_mm", "")
            row["kalman_kf_pos_delta_max_mm"] = k.get("kf_pos_delta_max_mm", "")
            row["kalman_kf_ang_delta_mean_deg"] = k.get("kf_ang_delta_mean_deg", "")
            row["kalman_kf_ang_delta_p95_deg"] = k.get("kf_ang_delta_p95_deg", "")
            row["kalman_kf_ang_delta_max_deg"] = k.get("kf_ang_delta_max_deg", "")

    for alg, k in km.items():
        if alg in existing_algs:
            continue
        row = {
            "algorithm": alg,
            "kalman_hard_reject_ang_deg": k.get("hard_reject_ang_deg", ""),
            "kalman_rejected_ratio": k.get("rejected_ratio", ""),
            "kalman_kf_jump_rate": k.get("kf_jump_rate", ""),
            "kalman_kf_stable_ratio": k.get("kf_stable_ratio", ""),
            "kalman_kf_pos_delta_mean_mm": k.get("kf_pos_delta_mean_mm", ""),
            "kalman_kf_pos_delta_p95_mm": k.get("kf_pos_delta_p95_mm", ""),
            "kalman_kf_pos_delta_max_mm": k.get("kf_pos_delta_max_mm", ""),
            "kalman_kf_ang_delta_mean_deg": k.get("kf_ang_delta_mean_deg", ""),
            "kalman_kf_ang_delta_p95_deg": k.get("kf_ang_delta_p95_deg", ""),
            "kalman_kf_ang_delta_max_deg": k.get("kf_ang_delta_max_deg", ""),
        }
        base_rows.append(row)

    base_rows.sort(key=lambda r: (r.get("algorithm", "")))
    write_csv_rows(summary_csv, base_rows)


def merge_into_summary_md(out_root, kalman_rows):
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_md = summary_dir / "summary_overall.md"
    marker = "## Offline Kalman Summary"

    headers = [
        "algorithm",
        "hard_reject_ang_deg",
        "rejected_ratio",
        "kf_jump_rate",
        "kf_stable_ratio",
        "kf_pos_delta_mean_mm",
        "kf_pos_delta_p95_mm",
        "kf_ang_delta_mean_deg",
        "kf_ang_delta_p95_deg",
    ]

    table_lines = []
    table_lines.append(marker)
    table_lines.append("")
    table_lines.append("| " + " | ".join(headers) + " |")
    table_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in sorted(kalman_rows, key=lambda x: x.get("algorithm", "")):
        vals = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        table_lines.append("| " + " | ".join(vals) + " |")
    table_text = "\n".join(table_lines) + "\n"

    if summary_md.exists():
        content = summary_md.read_text(encoding="utf-8")
    else:
        content = "# Aggregated Evaluation Summary\n\n"

    if marker in content:
        content = content.split(marker)[0].rstrip() + "\n\n"
    else:
        content = content.rstrip() + "\n\n"
    content += table_text
    summary_md.write_text(content, encoding="utf-8")

    # Also write a dedicated kalman-only markdown for quick view.
    kalman_md = summary_dir / "kalman_offline_overall.md"
    write_markdown_table(kalman_md, headers, sorted(kalman_rows, key=lambda x: x.get("algorithm", "")))


def parse_algorithm_filter(s):
    if s is None:
        return set()
    parts = [x.strip() for x in s.split(",") if x.strip()]
    return set(parts)


def main():
    ap = argparse.ArgumentParser(description="Offline Kalman filter over metrics.csv (single file or eval out_root batch)")
    ap.add_argument("--input", default="", help="Path to one metrics.csv")
    ap.add_argument("--out-root", default="", help="Eval root dir containing algorithm subdirs with metrics.csv")
    ap.add_argument("--output-dir", default="", help="Single-file mode: output dir; batch mode: output root for per-algorithm results")
    ap.add_argument("--algorithms", default="pca,ransac,gaussian,ceres", help="Batch mode algorithm filter, comma separated")
    ap.add_argument("--session-mode", choices=["latest", "all"], default="latest", help="Use latest session or all sessions")
    ap.add_argument("--session-gap-sec", type=float, default=30.0, help="Gap threshold to split sessions")

    ap.add_argument("--kalman-position-process-noise", type=float, default=0.001)
    ap.add_argument("--kalman-position-measurement-noise", type=float, default=0.0004)
    ap.add_argument("--kalman-axis-process-noise", type=float, default=0.04)
    ap.add_argument("--kalman-axis-measurement-noise", type=float, default=0.008)
    ap.add_argument("--kalman-initial-covariance", type=float, default=1.0)
    ap.add_argument("--kalman-min-dt", type=float, default=0.001)
    ap.add_argument("--kalman-max-dt", type=float, default=0.2)

    ap.add_argument("--pos-jump-mm", type=float, default=40.0)
    ap.add_argument("--ang-jump-deg", type=float, default=12.0)
    ap.add_argument("--jump-mad-k", type=float, default=6.0)
    ap.add_argument(
        "--hard-reject-ang-deg",
        type=float,
        default=45.0,
        help="Reject current frame when raw angular delta to previous kept frame exceeds threshold (<=0 disables)",
    )

    args = ap.parse_args()

    has_input = bool(args.input.strip())
    has_out_root = bool(args.out_root.strip())
    if has_input == has_out_root:
        print("ERROR: exactly one of --input or --out-root must be provided")
        return 2

    if has_input:
        metrics_csv = Path(args.input).resolve()
        if not metrics_csv.exists():
            print(f"ERROR: input not found: {metrics_csv}")
            return 2
        output_dir = Path(args.output_dir).resolve() if args.output_dir else (metrics_csv.parent / "kalman_offline")
        process_metrics_csv(metrics_csv, output_dir, args, algorithm_name="")
        print(f"Saved: {output_dir / 'filtered_pose.csv'}")
        print(f"Saved: {output_dir / 'step_deltas.csv'}")
        print(f"Saved: {output_dir / 'summary.csv'}")
        print(f"Saved: {output_dir / 'plots'}")
        return 0

    out_root = Path(args.out_root).resolve()
    if not out_root.exists() or not out_root.is_dir():
        print(f"ERROR: invalid out_root: {out_root}")
        return 2

    alg_filter = parse_algorithm_filter(args.algorithms)
    algo_dirs = []
    for p in sorted(out_root.iterdir()):
        if not p.is_dir():
            continue
        if alg_filter and p.name not in alg_filter:
            continue
        if (p / "metrics.csv").exists():
            algo_dirs.append(p)

    if not algo_dirs:
        print(f"ERROR: no algorithm result dirs with metrics.csv under: {out_root}")
        return 3

    all_rows = []
    for ad in algo_dirs:
        alg = ad.name
        metrics_csv = ad / "metrics.csv"
        if args.output_dir:
            out_dir = Path(args.output_dir).resolve() / alg
        else:
            out_dir = ad / "kalman_offline"
        try:
            row = process_metrics_csv(metrics_csv, out_dir, args, algorithm_name=alg)
            all_rows.append(row)
        except Exception as exc:
            print(f"[{alg}] ERROR: {exc}")

    if not all_rows:
        print("ERROR: all algorithms failed")
        return 4

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    kalman_csv = summary_dir / "kalman_offline_overall.csv"
    write_csv_rows(kalman_csv, sorted(all_rows, key=lambda x: x.get("algorithm", "")))
    merge_into_summary_csv(out_root, all_rows)
    merge_into_summary_md(out_root, all_rows)

    print(f"Saved: {kalman_csv}")
    print(f"Updated: {summary_dir / 'summary_overall.csv'}")
    print(f"Updated: {summary_dir / 'summary_overall.md'}")
    print(f"Saved: {summary_dir / 'kalman_offline_overall.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
