#!/usr/bin/env python3
"""
GT ingestion utilities
- average: read raw detect CSV (filename,timestamp,tag_id,tx,ty,tz,qx,qy,qz,qw,score)
  and produce one averaged pose per image (weighted by score)
- align: align estimate CSV and averaged GT CSV by filename or nearest timestamp

Outputs:
  groundtruth_averaged.csv with columns: filename,timestamp,tx,ty,tz,qx,qy,qz,qw,score_mean,num_tags
  aligned CSV: merged rows with estimate and gt columns (see --output)

Usage examples:
  python3 tools/gt_ingest.py average --input statistics/groundtruth.csv --output statistics/groundtruth_averaged.csv
  python3 tools/gt_ingest.py align --est estimates.csv --gt statistics/groundtruth_averaged.csv --output statistics/aligned.csv --tol 0.5

"""
import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np


def quaternion_average(quaternions, weights=None):
    # quaternions: Nx4 list-like (x,y,z,w)
    Q = np.array(quaternions, dtype=float)
    if Q.size == 0:
        return None
    if weights is None:
        weights = np.ones((Q.shape[0],), dtype=float)
    w = np.array(weights, dtype=float)
    if np.allclose(w, 0):
        w = np.ones_like(w)
    # ensure quaternion sign consistency relative to first
    ref = Q[0]
    for i in range(Q.shape[0]):
        if np.dot(ref, Q[i]) < 0:
            Q[i] = -Q[i]
    # build symmetric accumulator
    A = np.zeros((4, 4), dtype=float)
    for qi, wi in zip(Q, w):
        q = qi.reshape(4, 1)
        A += wi * (q @ q.T)
    # eigenvector with largest eigenvalue
    vals, vecs = np.linalg.eigh(A)
    q_avg = vecs[:, np.argmax(vals)]
    # normalize to (x,y,z,w)
    q_avg = q_avg / np.linalg.norm(q_avg)
    return [float(q_avg[0]), float(q_avg[1]), float(q_avg[2]), float(q_avg[3])]


def weighted_translation_average(ts, weights=None):
    T = np.array(ts, dtype=float)
    if T.size == 0:
        return None
    if weights is None:
        weights = np.ones((T.shape[0],), dtype=float)
    w = np.array(weights, dtype=float)
    if np.allclose(w, 0):
        w = np.ones_like(w)
    wsum = np.sum(w)
    if wsum == 0:
        return [float(x) for x in np.mean(T, axis=0)]
    tavg = np.sum(T * w.reshape(-1, 1), axis=0) / wsum
    return [float(tavg[0]), float(tavg[1]), float(tavg[2])]


def read_raw_gt(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # expected columns: filename,timestamp,tag_id,tx,ty,tz,qx,qy,qz,qw,score
            try:
                row = {
                    'filename': r.get('filename', ''),
                    'timestamp': r.get('timestamp', ''),
                    'tag_id': int(r.get('tag_id', -1)) if r.get('tag_id', '') != '' else -1,
                    't': [float(r.get('tx', 0.0)), float(r.get('ty', 0.0)), float(r.get('tz', 0.0))],
                    'q': [float(r.get('qx', 0.0)), float(r.get('qy', 0.0)), float(r.get('qz', 0.0)), float(r.get('qw', 1.0))],
                    'score': float(r.get('score', 1.0)) if r.get('score', '') != '' else 1.0,
                }
                rows.append(row)
            except Exception:
                continue
    return rows


def average_gt_per_image(input_csv, output_csv):
    rows = read_raw_gt(input_csv)
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['filename']].append(r)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw', 'score_mean', 'num_tags'])
        for fname, items in sorted(grouped.items()):
            scores = [it['score'] for it in items]
            ts_vals = [it['timestamp'] for it in items if it['timestamp']!='']
            # prefer numeric timestamps if available; else blank
            ts_out = ''
            if ts_vals:
                # try to pick the most common or mean
                try:
                    nums = [float(x) for x in ts_vals]
                    ts_out = str(int(np.mean(nums)))
                except Exception:
                    ts_out = ts_vals[0]
            trans = [it['t'] for it in items]
            quats = [it['q'] for it in items]
            tavg = weighted_translation_average(trans, weights=scores)
            qavg = quaternion_average(quats, weights=scores)
            score_mean = float(np.mean(scores)) if scores else 0.0
            num_tags = len(items)
            writer.writerow([fname, ts_out, tavg[0], tavg[1], tavg[2], qavg[0], qavg[1], qavg[2], qavg[3], score_mean, num_tags])
    return output_csv


def read_estimates(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # try to accept common variants
            try:
                # support multiple possible column names for translation and rotation
                def get_float_any(d, keys, default=0.0):
                    for k in keys:
                        if k in d and d[k] != '':
                            try:
                                return float(d[k])
                            except Exception:
                                continue
                    return float(default)

                tx = get_float_any(r, ['tx', 'est_tx', 'pose_x', 'x'])
                ty = get_float_any(r, ['ty', 'est_ty', 'pose_y', 'y'])
                tz = get_float_any(r, ['tz', 'est_tz', 'pose_z', 'z'])
                qx = get_float_any(r, ['qx', 'est_qx', 'q_x'])
                qy = get_float_any(r, ['qy', 'est_qy', 'q_y'])
                qz = get_float_any(r, ['qz', 'est_qz', 'q_z'])
                qw = get_float_any(r, ['qw', 'est_qw', 'q_w', 'w'], default=1.0)
                row = {
                    'filename': r.get('filename', ''),
                    'timestamp': r.get('timestamp', ''),
                    't': [tx, ty, tz],
                    'q': [qx, qy, qz, qw],
                    'raw': r,
                }
                rows.append(row)
            except Exception:
                continue
    return rows


def align_estimates_to_gt(est_csv, gt_csv, out_csv, tol_s=0.5, by_sequence=True):
    # load estimates CSV but only keep rows where 'name' starts with 'pose' (if present).
    est = []
    with open(est_csv, newline='') as f_est:
        reader = csv.DictReader(f_est)
        rows_all = list(reader)
        # find name column case-insensitive
        name_key = None
        if reader.fieldnames:
            for k in reader.fieldnames:
                if k and k.lower() == 'name':
                    name_key = k
                    break
        # filter rows: if name column exists, select only rows where name startswith 'pose'
        if name_key:
            rows = [r for r in rows_all if r.get(name_key, '').lower().startswith('pose')]
        else:
            rows = rows_all

        # parse selected rows into est list (support multiple possible column names)
        for r in rows:
            try:
                def get_float_any(d, keys, default=0.0):
                    for k in keys:
                        if k in d and d[k] != '':
                            try:
                                return float(d[k])
                            except Exception:
                                continue
                    return float(default)

                tx = get_float_any(r, ['tx', 'est_tx', 'pose_x', 'x'])
                ty = get_float_any(r, ['ty', 'est_ty', 'pose_y', 'y'])
                tz = get_float_any(r, ['tz', 'est_tz', 'pose_z', 'z'])
                qx = get_float_any(r, ['qx', 'est_qx', 'q_x'])
                qy = get_float_any(r, ['qy', 'est_qy', 'q_y'])
                qz = get_float_any(r, ['qz', 'est_qz', 'q_z'])
                qw = get_float_any(r, ['qw', 'est_qw', 'q_w', 'w'], default=1.0)
                est.append({
                    'filename': r.get('filename', ''),
                    'timestamp': r.get('timestamp', ''),
                    't': [tx, ty, tz],
                    'q': [qx, qy, qz, qw],
                    'raw': r,
                })
            except Exception:
                continue
    gt = []
    with open(gt_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            gt.append({
                'filename': r.get('filename', ''),
                'timestamp': r.get('timestamp', ''),
                't': [float(r.get('tx', 0.0)), float(r.get('ty', 0.0)), float(r.get('tz', 0.0))],
                'q': [float(r.get('qx', 0.0)), float(r.get('qy', 0.0)), float(r.get('qz', 0.0)), float(r.get('qw', 1.0))],
                'raw': r,
            })
    # index gt by filename
    gt_by_fname = {g['filename']: g for g in gt}
    # prepare timestamp index if needed
    gt_timestamps = []
    for g in gt:
        if g['timestamp'] != '':
            try:
                gt_timestamps.append((float(g['timestamp']), g))
            except Exception:
                pass
    gt_timestamps.sort(key=lambda x: x[0])

    def find_nearest_gt_by_time(ts):
        if not gt_timestamps:
            return None
        try:
            t = float(ts)
        except Exception:
            return None
        arr = [x[0] for x in gt_timestamps]
        idx = np.searchsorted(arr, t)
        candidates = []
        if idx < len(arr):
            candidates.append(gt_timestamps[idx])
        if idx > 0:
            candidates.append(gt_timestamps[idx-1])
        best = None
        best_dt = None
        for ct, g in candidates:
            dt = abs(ct - t)
            if best is None or dt < best_dt:
                best = g
                best_dt = dt
        if best is None or best_dt is None or best_dt > tol_s:
            return None
        return best

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or '.', exist_ok=True)
    # produce merged CSV header
    with open(out_csv, 'w', newline='') as f_out:
        fieldnames = [
            'filename', 'timestamp_est', 'est_tx', 'est_ty', 'est_tz', 'est_qx', 'est_qy', 'est_qz', 'est_qw',
            'gt_timestamp', 'gt_tx', 'gt_ty', 'gt_tz', 'gt_qx', 'gt_qy', 'gt_qz', 'gt_qw', 'gt_score_mean', 'gt_num_tags'
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        # matching strategy: by sequence (index) or by filename/timestamp
        for idx, e in enumerate(est):
            fname = e['filename']
            chosen = None
            if by_sequence:
                if idx < len(gt):
                    chosen = gt[idx]
            else:
                if fname and fname in gt_by_fname:
                    chosen = gt_by_fname[fname]
                else:
                    # try nearest timestamp
                    ts = e.get('timestamp', '')
                    candidate = find_nearest_gt_by_time(ts)
                    if candidate is not None:
                        chosen = candidate
            if chosen is None:
                # write with empty GT
                row = {
                    'filename': fname,
                    'timestamp_est': e.get('timestamp', ''),
                    'est_tx': e['t'][0], 'est_ty': e['t'][1], 'est_tz': e['t'][2],
                    'est_qx': e['q'][0], 'est_qy': e['q'][1], 'est_qz': e['q'][2], 'est_qw': e['q'][3],
                    'gt_timestamp': '', 'gt_tx': '', 'gt_ty': '', 'gt_tz': '', 'gt_qx': '', 'gt_qy': '', 'gt_qz': '', 'gt_qw': '', 'gt_score_mean': '', 'gt_num_tags': ''
                }
            else:
                row = {
                    'filename': fname,
                    'timestamp_est': e.get('timestamp', ''),
                    'est_tx': e['t'][0], 'est_ty': e['t'][1], 'est_tz': e['t'][2],
                    'est_qx': e['q'][0], 'est_qy': e['q'][1], 'est_qz': e['q'][2], 'est_qw': e['q'][3],
                    'gt_timestamp': chosen.get('timestamp', ''), 'gt_tx': chosen['t'][0], 'gt_ty': chosen['t'][1], 'gt_tz': chosen['t'][2],
                    'gt_qx': chosen['q'][0], 'gt_qy': chosen['q'][1], 'gt_qz': chosen['q'][2], 'gt_qw': chosen['q'][3],
                    'gt_score_mean': chosen['raw'].get('score_mean', ''), 'gt_num_tags': chosen['raw'].get('num_tags', '')
                }
            writer.writerow(row)
    return out_csv


def cli():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    p_avg = sub.add_parser('average', help='Average multiple tag detections per image')
    p_avg.add_argument('--input', required=True)
    p_avg.add_argument('--output', required=True)
    p_align = sub.add_parser('align', help='Align estimate CSV to averaged GT CSV')
    p_align.add_argument('--est', required=True)
    p_align.add_argument('--gt', required=True)
    p_align.add_argument('--output', required=True)
    p_align.add_argument('--tol', type=float, default=0.5, help='timestamp tolerance in seconds')
    p_align.add_argument('--by-sequence', action='store_true', help='Align GT to estimates by sequence order instead of timestamps')

    args = parser.parse_args()
    if args.cmd == 'average':
        out = average_gt_per_image(args.input, args.output)
        print('Wrote averaged GT to', out)
    elif args.cmd == 'align':
        out = align_estimates_to_gt(args.est, args.gt, args.output, tol_s=args.tol, by_sequence=args.by_sequence)
        print('Wrote aligned CSV to', out)
    else:
        parser.print_help()


if __name__ == '__main__':
    cli()
