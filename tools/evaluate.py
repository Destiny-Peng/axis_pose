#!/usr/bin/env python3
"""
Evaluate aligned estimates vs ground-truth.
Inputs: aligned CSV with columns containing est_tx/est_ty/est_tz, gt_tx/gt_ty/gt_tz, est_qx..qw, gt_qx..qw
Outputs: per-frame errors CSV and a summary CSV + optional plots.
"""
import argparse
import csv
import math
import os
from statistics import mean, stdev, median

import numpy as np
import matplotlib.pyplot as plt


def quat_multiply(a, b):
    # a,b as [x,y,z,w]
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    w = aw*bw - ax*bx - ay*by - az*bz
    return [x, y, z, w]


def quat_conjugate(q):
    x,y,z,w = q
    return [-x, -y, -z, w]


def quat_normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return [0.0, 0.0, 0.0, 1.0]
    return (q / n).tolist()


def angular_error_deg(q_est, q_gt):
    # compute minimal rotation angle between two quaternions
    # use dot product: angle = 2 * acos(|dot(q_est, q_gt)|)
    q_est = quat_normalize(q_est)
    q_gt = quat_normalize(q_gt)
    # dot = q_est.x*q_gt.x + q_est.y*q_gt.y + q_est.z*q_gt.z + q_est.w*q_gt.w
    dot = float(q_est[0]*q_gt[0] + q_est[1]*q_gt[1] + q_est[2]*q_gt[2] + q_est[3]*q_gt[3])
    dot = max(-1.0, min(1.0, dot))
    dot = abs(dot)
    angle = 2.0 * math.acos(dot)
    return math.degrees(angle)


def rotate_vector_by_quat(v, q):
    # rotate vector v by quaternion q (v: [x,y,z], q: [x,y,z,w])
    qn = quat_normalize(q)
    qv = [v[0], v[1], v[2], 0.0]
    tmp = quat_multiply(qn, qv)
    qn_conj = quat_conjugate(qn)
    res = quat_multiply(tmp, qn_conj)
    return [res[0], res[1], res[2]]


def axis_angle_deg(q_est, q_gt, axis=(1.0, 0.0, 0.0)):
    # compare a chosen local axis (default X) rotated by each quaternion
    # lines are direction-only, so take absolute dot product and acos
    v_est = np.array(rotate_vector_by_quat(axis, q_est), dtype=float)
    v_gt = np.array(rotate_vector_by_quat(axis, q_gt), dtype=float)
    # normalize
    ne = np.linalg.norm(v_est)
    ng = np.linalg.norm(v_gt)
    if ne == 0 or ng == 0:
        return float('nan')
    v_est /= ne
    v_gt /= ng
    dot = float(np.dot(v_est, v_gt))
    dot = max(-1.0, min(1.0, abs(dot)))
    angle = math.acos(dot)
    return math.degrees(angle)


def axis_angle_yz_deg(q_est, q_gt, axis=(1.0, 0.0, 0.0)):
    # project rotated axes to YZ plane and compute angle between directions
    v_est = np.array(rotate_vector_by_quat(axis, q_est), dtype=float)
    v_gt = np.array(rotate_vector_by_quat(axis, q_gt), dtype=float)
    # ignore X
    v_est[0] = 0.0
    v_gt[0] = 0.0
    ne = np.linalg.norm(v_est)
    ng = np.linalg.norm(v_gt)
    if ne == 0 or ng == 0:
        return float('nan')
    v_est /= ne
    v_gt /= ng
    dot = float(np.dot(v_est, v_gt))
    dot = max(-1.0, min(1.0, abs(dot)))
    angle = math.acos(dot)
    return math.degrees(angle)


def read_aligned(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # skip rows with empty GT
            if not r.get('gt_tx'):
                continue
            try:
                est_t = [float(r.get('est_tx', 0.0)), float(r.get('est_ty', 0.0)), float(r.get('est_tz', 0.0))]
                gt_t = [float(r.get('gt_tx', 0.0)), float(r.get('gt_ty', 0.0)), float(r.get('gt_tz', 0.0))]
                est_q = [float(r.get('est_qx', 0.0)), float(r.get('est_qy', 0.0)), float(r.get('est_qz', 0.0)), float(r.get('est_qw', 1.0))]
                gt_q = [float(r.get('gt_qx', 0.0)), float(r.get('gt_qy', 0.0)), float(r.get('gt_qz', 0.0)), float(r.get('gt_qw', 1.0))]
                rows.append({'filename': r.get('filename', ''), 'timestamp_est': r.get('timestamp_est', ''), 'est_t': est_t, 'gt_t': gt_t, 'est_q': est_q, 'gt_q': gt_q})
            except Exception:
                continue
    return rows


def evaluate(rows):
    trans_errs = []
    ang_errs = []
    per = []
    for r in rows:
        est = np.array(r['est_t'], dtype=float)
        gt = np.array(r['gt_t'], dtype=float)
        tr_err = float(np.linalg.norm(est - gt))
        # compute axis-based angle between tag X-axis and estimated main axis
        ang_err = axis_angle_deg(r['est_q'], r['gt_q'])
        ang_err_yz = axis_angle_yz_deg(r['est_q'], r['gt_q'])
        trans_errs.append(tr_err)
        ang_errs.append(ang_err)
        per.append({'filename': r['filename'], 'timestamp_est': r['timestamp_est'], 'trans_err': tr_err, 'ang_err_deg': ang_err, 'ang_err_yz_deg': ang_err_yz,
                    'est_tx': est[0], 'est_ty': est[1], 'est_tz': est[2], 'gt_tx': gt[0], 'gt_ty': gt[1], 'gt_tz': gt[2]})
    stats = {}
    if trans_errs:
        stats['count'] = len(trans_errs)
        stats['trans_rmse'] = math.sqrt(sum([e*e for e in trans_errs]) / len(trans_errs))
        stats['trans_mean'] = mean(trans_errs)
        stats['trans_std'] = stdev(trans_errs) if len(trans_errs) > 1 else 0.0
        stats['trans_median'] = median(trans_errs)
        stats['ang_mean_deg'] = mean(ang_errs)
        stats['ang_std_deg'] = stdev(ang_errs) if len(ang_errs) > 1 else 0.0
        stats['ang_median_deg'] = median(ang_errs)
        # also compute XY-projected angle stats
        ang_xy = [p['ang_err_yz_deg'] for p in per if not math.isnan(p.get('ang_err_yz_deg', float('nan')))]
        if ang_xy:
            stats['ang_xy_mean_deg'] = mean(ang_xy)
            stats['ang_xy_std_deg'] = stdev(ang_xy) if len(ang_xy) > 1 else 0.0
            stats['ang_xy_median_deg'] = median(ang_xy)
        else:
            stats['ang_xy_mean_deg'] = ''
            stats['ang_xy_std_deg'] = ''
            stats['ang_xy_median_deg'] = ''
    else:
        stats['count'] = 0
    return per, stats


def save_perframe(per, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        fieldnames = ['filename', 'timestamp_est', 'trans_err', 'ang_err_deg', 'ang_err_yz_deg', 'est_tx', 'est_ty', 'est_tz', 'gt_tx', 'gt_ty', 'gt_tz']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per:
            writer.writerow(r)


def save_summary(stats, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['count', 'trans_rmse', 'trans_mean', 'trans_std', 'trans_median', 'ang_mean_deg', 'ang_std_deg', 'ang_median_deg', 'ang_xy_mean_deg', 'ang_xy_std_deg', 'ang_xy_median_deg'])
        writer.writerow([stats.get('count', 0), stats.get('trans_rmse', ''), stats.get('trans_mean', ''), stats.get('trans_std', ''), stats.get('trans_median', ''), stats.get('ang_mean_deg', ''), stats.get('ang_std_deg', ''), stats.get('ang_median_deg', ''), stats.get('ang_xy_mean_deg', ''), stats.get('ang_xy_std_deg', ''), stats.get('ang_xy_median_deg', '')])


def plot_hist(per, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    trans = [p['trans_err'] for p in per]
    ang = [p['ang_err_deg'] for p in per]
    if trans:
        plt.figure()
        plt.hist(trans, bins=40)
        plt.title('Translation error (m)')
        plt.xlabel('m')
        plt.ylabel('count')
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, 'trans_error_hist.png'))
        plt.close()
    if ang:
        plt.figure()
        plt.hist(ang, bins=40)
        plt.title('Angular error (deg)')
        plt.xlabel('deg')
        plt.ylabel('count')
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, 'ang_error_hist.png'))
        plt.close()


def plot_poses(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    xs_e = []
    ys_e = []
    zs_e = []
    xs_g = []
    ys_g = []
    zs_g = []
    ve = []
    vg = []
    for r in rows:
        est = np.array(r['est_t'], dtype=float)
        gt = np.array(r['gt_t'], dtype=float)
        xs_e.append(est[0]); ys_e.append(est[1]); zs_e.append(est[2])
        xs_g.append(gt[0]); ys_g.append(gt[1]); zs_g.append(gt[2])
        ve.append(np.array(rotate_vector_by_quat((1.0,0.0,0.0), r['est_q']), dtype=float))
        vg.append(np.array(rotate_vector_by_quat((1.0,0.0,0.0), r['gt_q']), dtype=float))

    if not xs_e:
        return

    # 3D plot
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs_e, ys_e, zs_e, c='r', marker='o', label='est')
    ax.scatter(xs_g, ys_g, zs_g, c='b', marker='^', label='gt')
    # draw direction vectors (quivers)
    scale = max(1e-3, np.max(np.abs(np.concatenate([xs_e, ys_e, zs_e, xs_g, ys_g, zs_g]))))
    L = scale * 0.2
    for (x,y,z), v in zip(zip(xs_e, ys_e, zs_e), ve):
        ax.quiver(x, y, z, v[0], v[1], v[2], length=L, color='r')
    for (x,y,z), v in zip(zip(xs_g, ys_g, zs_g), vg):
        ax.quiver(x, y, z, v[0], v[1], v[2], length=L, color='b')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Estimated (red) vs Ground-truth (blue) poses')
    plt.savefig(os.path.join(out_dir, 'poses_3d.png'))
    plt.show()
    plt.close()

    # top-down XY
    plt.figure(figsize=(6,6))
    plt.scatter(xs_e, ys_e, c='r', label='est')
    plt.scatter(xs_g, ys_g, c='b', label='gt')
    for (x,y), v in zip(zip(xs_e, ys_e), ve):
        plt.arrow(x, y, v[0]*L, v[1]*L, color='r', head_width=L*0.05)
    for (x,y), v in zip(zip(xs_g, ys_g), vg):
        plt.arrow(x, y, v[0]*L, v[1]*L, color='b', head_width=L*0.05)
    plt.xlabel('X'); plt.ylabel('Y'); plt.axis('equal'); plt.legend(); plt.grid(True)
    plt.title('Top-down view')
    plt.savefig(os.path.join(out_dir, 'poses_xy.png'))
    plt.close()

    # XZ side view
    plt.figure(figsize=(6,6))
    plt.scatter(xs_e, zs_e, c='r', label='est')
    plt.scatter(xs_g, zs_g, c='b', label='gt')
    for (x,z), v in zip(zip(xs_e, zs_e), ve):
        plt.arrow(x, z, v[0]*L, v[2]*L, color='r', head_width=L*0.05)
    for (x,z), v in zip(zip(xs_g, zs_g), vg):
        plt.arrow(x, z, v[0]*L, v[2]*L, color='b', head_width=L*0.05)
    plt.xlabel('X'); plt.ylabel('Z'); plt.axis('equal'); plt.legend(); plt.grid(True)
    plt.title('Side view (XZ)')
    plt.savefig(os.path.join(out_dir, 'poses_xz.png'))
    plt.close()

    # YZ front view
    plt.figure(figsize=(6,6))
    plt.scatter(ys_e, zs_e, c='r', label='est')
    plt.scatter(ys_g, zs_g, c='b', label='gt')
    for (y,z), v in zip(zip(ys_e, zs_e), ve):
        plt.arrow(y, z, v[1]*L, v[2]*L, color='r', head_width=L*0.05)
    for (y,z), v in zip(zip(ys_g, zs_g), vg):
        plt.arrow(y, z, v[1]*L, v[2]*L, color='b', head_width=L*0.05)
    plt.xlabel('Y'); plt.ylabel('Z'); plt.axis('equal'); plt.legend(); plt.grid(True)
    plt.title('Front view (YZ)')
    plt.savefig(os.path.join(out_dir, 'poses_yz.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Aligned CSV (aligned.csv)')
    parser.add_argument('--out', required=True, help='Per-frame errors CSV')
    parser.add_argument('--summary', required=True, help='Summary CSV')
    parser.add_argument('--plots', default='', help='Directory to save plots (optional)')
    args = parser.parse_args()

    rows = read_aligned(args.input)
    per, stats = evaluate(rows)
    save_perframe(per, args.out)
    save_summary(stats, args.summary)
    if args.plots:
        plot_hist(per, args.plots)
        plot_poses(rows, args.plots)
    print('Evaluated', stats.get('count', 0), 'frames')
    print('Trans RMSE:', stats.get('trans_rmse', 'n/a'))
    print('Ang mean (deg):', stats.get('ang_mean_deg', 'n/a'))

if __name__ == '__main__':
    main()
