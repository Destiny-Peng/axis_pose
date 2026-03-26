#!/usr/bin/env python3
"""
Extract AprilTag bundle (board) ground-truth poses from an images directory.

It estimates ONE pose per image by jointly using all detected tags in a bundle
layout file (e.g. 36 tags) and then reprojects the full board to the image.
"""
import argparse
import os
import sys
import glob
import csv
import math
import re
import numpy as np
import cv2
import yaml

import apriltag as at
DETECTOR_LIB = 'apriltag'



def read_camera_info_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    # Support common camera_info layouts
    # either top-level keys or under '/**': {ros__parameters: {...}}
    if '/**' in data and 'ros__parameters' in data['/**']:
        data = data['/**']['ros__parameters']
    # camera_matrix.data or camera_matrix: {data: [...]} or K
    K = None
    D = None
    if 'camera_matrix' in data:
        cm = data['camera_matrix']
        if isinstance(cm, dict) and 'data' in cm:
            K = np.array(cm['data']).reshape((3,3))
    if K is None and 'K' in data:
        K = np.array(data['K']).reshape((3,3))
    if 'distortion_coefficients' in data:
        dc = data['distortion_coefficients']
        if isinstance(dc, dict) and 'data' in dc:
            D = np.array(dc['data'])
    if D is None and 'D' in data:
        D = np.array(data['D'])
    if K is None:
        raise RuntimeError(f"Cannot find camera matrix in {path}")
    if D is None:
        D = np.zeros((5,))
    return K, D


def rotation_matrix_to_quaternion(R):
    # R: 3x3 numpy
    m = R
    trace = m[0,0] + m[1,1] + m[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2,1] - m[1,2]) * s
        y = (m[0,2] - m[2,0]) * s
        z = (m[1,0] - m[0,1]) * s
    else:
        if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            s = 2.0 * math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = 2.0 * math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
    return [x, y, z, w]


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    # Normalize to avoid scale issues from malformed configs.
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x = qx / n
    y = qy / n
    z = qz / n
    w = qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], dtype=np.float64)


def read_layout_yaml(path):
    with open(path, 'r') as f:
        raw = f.read()

    data = None
    try:
        data = yaml.safe_load(raw)
    except Exception:
        data = None

    if not data or 'layout' not in data:
        # Fallback for custom non-standard format:
        # [id: 0, size: 0.06, x: ..., ...]
        entries = re.findall(r'\[(.*?)\]', raw, flags=re.DOTALL)
        parsed = []
        for e in entries:
            if 'id:' not in e or 'size:' not in e:
                continue
            item = {}
            parts = [p.strip() for p in e.split(',') if p.strip()]
            for p in parts:
                if ':' not in p:
                    continue
                k, v = p.split(':', 1)
                item[k.strip()] = v.strip()
            if 'id' in item and 'size' in item and 'x' in item and 'y' in item and 'z' in item:
                parsed.append(item)
        data = {'layout': parsed}

    if not data or 'layout' not in data or not data['layout']:
        raise RuntimeError(f"Cannot parse layout entries from {path}")

    layout = {}
    for item in data['layout']:
        tid = int(item['id'])
        size = float(item['size'])
        tx = float(item['x'])
        ty = float(item['y'])
        tz = float(item['z'])
        qw = float(item.get('qw', 1.0))
        qx = float(item.get('qx', 0.0))
        qy = float(item.get('qy', 0.0))
        qz = float(item.get('qz', 0.0))

        rot = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        trans = np.array([tx, ty, tz], dtype=np.float64)

        # Local tag-frame corners order must match detector corner order.
        s = size
        local_corners = np.array([
            [-s / 2.0, s / 2.0, 0.0],
            [s / 2.0, s / 2.0, 0.0],
            [s / 2.0, -s / 2.0, 0.0],
            [-s / 2.0, -s / 2.0, 0.0]
        ], dtype=np.float64)

        world_corners = (rot @ local_corners.T).T + trans
        layout[tid] = {
            'size': size,
            'center': trans,
            'corners3d': world_corners
        }
    return layout


def detect_tags(img, detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = []
    if DETECTOR_LIB == 'apriltag':
        dets = detector.detect(gray)
        for d in dets:
            tid = int(d.tag_id) if hasattr(d, 'tag_id') else int(d['id'])
            corners = np.array(d.corners) if hasattr(d, 'corners') else np.array(d['corners'])
            score = float(d.hamming) if hasattr(d, 'hamming') else float(d.get('decision_margin', 0.0))
            detections.append((tid, corners, score))
    elif DETECTOR_LIB == 'pupil_apriltags':
        dets = detector.detect(gray, estimate_tag_pose=False)
        for d in dets:
            tid = int(d.tag_id)
            corners = np.array(d.corners)
            score = float(d.decision_margin) if hasattr(d, 'decision_margin') else 0.0
            detections.append((tid, corners, score))
    else:
        raise RuntimeError('No apriltag detector available. Install apriltag or pupil_apriltags')
    return detections


def estimate_bundle_pose(img_path, detector, K, D, layout, min_tags):
    img = cv2.imread(img_path)
    if img is None:
        return None

    detections = detect_tags(img, detector)
    obj_points = []
    img_points = []
    used_tag_ids = []
    used_scores = []
    detected_corners = {}

    for tid, corners, score in detections:
        if tid not in layout:
            continue
        if corners.shape[0] != 4:
            continue
        obj_points.extend(layout[tid]['corners3d'])
        img_points.extend(corners.astype(np.float64))
        used_tag_ids.append(tid)
        used_scores.append(score)
        detected_corners[tid] = corners

    unique_tags = sorted(set(used_tag_ids))
    if len(unique_tags) < int(min_tags):
        return {
            'ok': False,
            'img': img,
            'num_tags': len(unique_tags),
            'num_points': len(obj_points),
            'tag_ids': unique_tags,
            'message': f'not enough tags: {len(unique_tags)} < {min_tags}'
        }

    obj_points = np.asarray(obj_points, dtype=np.float64)
    img_points = np.asarray(img_points, dtype=np.float64)

    # ok, rvec, tvec, inliers = cv2.solvePnPRansac(
    #     obj_points,
    #     img_points,
    #     K,
    #     D,
    #     iterationsCount=500,
    #     reprojectionError=3.0,
    #     confidence=0.9,
    #     flags=cv2.SOLVEPNP_ITERATIVE
    # )
    # use solvepnp directly
    ok, rvec, tvec = cv2.solvePnP(
        obj_points,
        img_points,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    inliers = np.arange(len(img_points))  # all inliers since no RANSAC
    if not ok:
        return {
            'ok': False,
            'img': img,
            'num_tags': len(unique_tags),
            'num_points': len(obj_points),
            'tag_ids': unique_tags,
            'message': 'solvePnPRansac failed'
        }

    proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    residuals = np.linalg.norm(img_points - proj, axis=1)
    mean_err = float(np.mean(residuals)) if residuals.size > 0 else 0.0

    R, _ = cv2.Rodrigues(rvec)
    q = rotation_matrix_to_quaternion(R)
    tx, ty, tz = float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])
    return {
        'ok': True,
        'img': img,
        'rvec': rvec,
        'tvec': tvec,
        't': (tx, ty, tz),
        'q': q,
        'num_tags': len(unique_tags),
        'num_points': int(obj_points.shape[0]),
        'tag_ids': unique_tags,
        'score': float(np.mean(used_scores)) if used_scores else 0.0,
        'mean_reproj_err': mean_err,
        'inliers': int(inliers.shape[0]) if inliers is not None else 0,
        'detected_corners': detected_corners
    }


def draw_bundle_reprojection(img, result, layout, K, D):
    out = img.copy()

    # Draw observed corners (yellow) for debugging.
    for tid, corners in result.get('detected_corners', {}).items():
        for p in corners:
            cv2.circle(out, tuple(np.int32(np.round(p))), 2, (0, 255, 255), -1)
        c = np.mean(corners, axis=0).astype(int)
        cv2.putText(out, f'det:{tid}', (int(c[0]) + 4, int(c[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # Reproject full 36-tag layout.
    for tid, info in layout.items():
        c3d = info['corners3d'].astype(np.float64)
        imgpts, _ = cv2.projectPoints(c3d, result['rvec'], result['tvec'], K, D)
        imgpts = imgpts.reshape(-1, 2)
        poly = np.int32(np.round(imgpts)).reshape((-1, 1, 2))
        cv2.polylines(out, [poly], True, (0, 255, 0), 1)

        center3d = info['center'].reshape(1, 3).astype(np.float64)
        center2d, _ = cv2.projectPoints(center3d, result['rvec'], result['tvec'], K, D)
        c = tuple(np.int32(np.round(center2d.reshape(2))))
        cv2.putText(out, str(tid), (c[0] + 2, c[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

    axis_len = 0.85
    obj_axes = np.array([
        [0.0, 0.0, 0.0],
        [axis_len, 0.0, 0.0],
        [0.0, axis_len, 0.0],
        [0.0, 0.0, axis_len]
    ], dtype=np.float64)
    axis2d, _ = cv2.projectPoints(obj_axes, result['rvec'], result['tvec'], K, D)
    axis2d = np.int32(np.round(axis2d.reshape(-1, 2)))
    o = tuple(axis2d[0])
    cv2.line(out, o, tuple(axis2d[1]), (0, 0, 255), 2)
    cv2.line(out, o, tuple(axis2d[2]), (0, 255, 0), 2)
    cv2.line(out, o, tuple(axis2d[3]), (255, 0, 0), 2)

    msg = f"tags:{result['num_tags']} inliers:{result['inliers']} err:{result['mean_reproj_err']:.2f}px"
    cv2.putText(out, msg, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help='Directory with images (jpg/png)')
    parser.add_argument('--camera-info', required=True, help='camera_info YAML file')
    parser.add_argument('--layout', required=True, help='AprilTag bundle layout YAML file (contains ids, size and 3D positions)')
    parser.add_argument('--output', default='groundtruth.csv', help='Output CSV file')
    parser.add_argument('--save-annotated', default='', help='Optional dir to save annotated images')
    parser.add_argument('--ext', default='png', help='Primary image extension to scan (png/jpg/jpeg)')
    parser.add_argument('--min-tags', type=int, default=4, help='Minimum detected tags required for bundle pose')
    parser.add_argument('--visualize', action='store_true', help='Show annotations in a window')
    args = parser.parse_args()

    if DETECTOR_LIB is None:
        print('Please install apriltag or pupil_apriltags: pip install apriltag OR pip install pupil-apriltags')
        sys.exit(1)

    K, D = read_camera_info_yaml(args.camera_info)
    layout = read_layout_yaml(args.layout)

    # create detector
    if DETECTOR_LIB == 'apriltag':
        detector = at.Detector()
    else:
        detector = APDetector(families='tag36h11')

    ext = args.ext.lower().lstrip('.')
    images = sorted(glob.glob(os.path.join(args.images, f'*.{ext}')))
    if ext != 'jpg':
        images += sorted(glob.glob(os.path.join(args.images, '*.jpg')))
    if ext != 'jpeg':
        images += sorted(glob.glob(os.path.join(args.images, '*.jpeg')))
    if ext != 'png':
        images += sorted(glob.glob(os.path.join(args.images, '*.png')))
    images = sorted(set(images))
    if not images:
        print('No images found in', args.images)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.save_annotated:
        os.makedirs(args.save_annotated, exist_ok=True)

    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'timestamp', 'bundle', 'num_tags', 'num_points', 'inliers', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw', 'mean_reproj_err_px', 'score'])
        for img_path in images:
            print('Processing', img_path)
            # derive timestamp from filename if possible (ISO or numbers), else use file mtime
            fname = os.path.basename(img_path)
            ts = ''
            # try parse YYYYMMDD... or unix timestamp in name
            try:
                # try mtime first
                ts = str(int(os.path.getmtime(img_path)))
            except Exception:
                ts = ''

            result = estimate_bundle_pose(img_path, detector, K, D, layout, args.min_tags)
            if result is None:
                continue
            if not result['ok']:
                print(f"  skip: {result['message']}")
                continue

            tx, ty, tz = result['t']
            qx, qy, qz, qw = result['q']
            writer.writerow([
                fname,
                ts,
                'bundle',
                result['num_tags'],
                result['num_points'],
                result['inliers'],
                tx,
                ty,
                tz,
                qx,
                qy,
                qz,
                qw,
                result['mean_reproj_err'],
                result['score']
            ])

            anno = draw_bundle_reprojection(result['img'], result, layout, K, D)
            if args.save_annotated:
                outp = os.path.join(args.save_annotated, fname)
                cv2.imwrite(outp, anno)
            if args.visualize:
                cv2.imshow('annot', anno)
                key = cv2.waitKey(1)
                if key == 27:
                    print('User cancelled')
                    return
    print('Done. Wrote', args.output)

if __name__ == '__main__':
    main()
