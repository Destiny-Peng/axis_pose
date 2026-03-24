#!/usr/bin/env python3
"""
Extract AprilTag-based ground-truth poses from an images directory.
Outputs a CSV with: filename,timestamp,tag_id,tx,ty,tz,qx,qy,qz,qw,score

Usage:
  python3 tools/extract_groundtruth.py --images /path/to/images --camera-info config/d457_color.yaml --tag-size 0.05 --output gt.csv --save-annotated out/annot

Dependencies:
  pip install opencv-python numpy pyyaml pandas apriltag
  or for pupil_apriltags: pip install pupil-apriltags

The script will try to import either `apriltag` or `pupil_apriltags`.
"""
import argparse
import os
import sys
import glob
import csv
import math
import numpy as np
import cv2
import yaml

try:
    import apriltag as at
    DETECTOR_LIB = 'apriltag'
except Exception:
    try:
        from pupil_apriltags import Detector as APDetector
        DETECTOR_LIB = 'pupil_apriltags'
    except Exception:
        DETECTOR_LIB = None


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


def detect_tags_and_pose(img_path, detector, K, D, tag_size_m):
    img = cv2.imread(img_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = []
    if DETECTOR_LIB == 'apriltag':
        dets = detector.detect(gray)
        for d in dets:
            # apriltag python returns 'id', 'center', 'corners', 'hamming', 'decision_margin'
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

    results = []
    for tid, corners, score in detections:
        # object points in tag frame: corners assumed order corresponds to image corners
        s = float(tag_size_m)
        obj_pts = np.array([
            [-s/2.0,  s/2.0, 0.0],
            [ s/2.0,  s/2.0, 0.0],
            [ s/2.0, -s/2.0, 0.0],
            [-s/2.0, -s/2.0, 0.0]
        ], dtype=np.float32)
        img_pts = corners.astype(np.float32)
        # ensure correct shape
        if img_pts.shape[0] != 4:
            continue
        # solvePnP
        try:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            q = rotation_matrix_to_quaternion(R)
            tx, ty, tz = float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])
            results.append({'tag_id': tid, 't': (tx, ty, tz), 'q': q, 'score': score, 'corners': corners, 'rvec': rvec, 'tvec': tvec})
        except Exception as e:
            # fallback: skip
            print('solvePnP failed for', img_path, 'tag', tid, e)
            continue
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help='Directory with images (jpg/png)')
    parser.add_argument('--camera-info', required=True, help='camera_info YAML file')
    parser.add_argument('--tag-size', required=True, type=float, help='AprilTag size in meters (side length)')
    parser.add_argument('--output', default='groundtruth.csv', help='Output CSV file')
    parser.add_argument('--save-annotated', default='', help='Optional dir to save annotated images')
    parser.add_argument('--ext', default='png', help='Image extension to scan (png/jpg)')
    parser.add_argument('--visualize', action='store_true', help='Show annotations in a window')
    args = parser.parse_args()

    if DETECTOR_LIB is None:
        print('Please install apriltag or pupil_apriltags: pip install apriltag OR pip install pupil-apriltags')
        sys.exit(1)

    K, D = read_camera_info_yaml(args.camera_info)

    # create detector
    if DETECTOR_LIB == 'apriltag':
        detector = at.Detector()
    else:
        detector = APDetector(families='tag36h11')

    images = sorted(glob.glob(os.path.join(args.images, f'*.{args.ext}'))) + sorted(glob.glob(os.path.join(args.images, f'*.jpg')))
    if not images:
        print('No images found in', args.images)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.save_annotated:
        os.makedirs(args.save_annotated, exist_ok=True)

    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'timestamp', 'tag_id', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw', 'score'])
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

            results = detect_tags_and_pose(img_path, detector, K, D, args.tag_size)
            if results:
                img = cv2.imread(img_path)
                for r in results:
                    tx,ty,tz = r['t']
                    qx,qy,qz,qw = r['q']
                    writer.writerow([fname, ts, r['tag_id'], tx, ty, tz, qx, qy, qz, qw, r['score']])
                    if args.save_annotated:
                        # draw smaller polygon and tag coordinate axes
                        corners = r['corners'].astype(int)
                        cv2.polylines(img, [corners.reshape((-1,1,2))], True, (0,255,0), 1)
                        # draw tag coordinate axes using projectPoints
                        axis_len = float(args.tag_size) * 0.5
                        obj_axes = np.array([[0.0,0.0,0.0],[axis_len,0.0,0.0],[0.0,axis_len,0.0],[0.0,0.0,axis_len]], dtype=np.float32)
                        try:
                            imgpts, _ = cv2.projectPoints(obj_axes, r['rvec'], r['tvec'], K, D)
                            imgpts = imgpts.reshape(-1,2).astype(int)
                            origin = tuple(imgpts[0])
                            # x - red, y - green, z - blue
                            cv2.circle(img, origin, 3, (0,0,255), -1)
                            cv2.line(img, origin, tuple(imgpts[1]), (0,0,255), 2)
                            cv2.line(img, origin, tuple(imgpts[2]), (0,255,0), 2)
                            cv2.line(img, origin, tuple(imgpts[3]), (255,0,0), 2)
                            cv2.putText(img, f"id:{r['tag_id']}", (origin[0]+6, origin[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                        except Exception as e:
                            # fallback to drawing center
                            c = np.mean(corners, axis=0).astype(int)
                            cv2.circle(img, tuple(c), 3, (0,0,255), -1)
                if args.save_annotated:
                    outp = os.path.join(args.save_annotated, fname)
                    cv2.imwrite(outp, img)
                if args.visualize:
                    cv2.imshow('annot', img)
                    key = cv2.waitKey(1)
                    if key == 27:
                        print('User cancelled')
                        return
            else:
                # write empty line? skip
                pass
    print('Done. Wrote', args.output)

if __name__ == '__main__':
    main()
