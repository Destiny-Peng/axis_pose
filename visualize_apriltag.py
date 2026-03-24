#!/usr/bin/env python3
"""
将 AprilTag 的结果反投影到图像上进行可视化。
读取 groundtruth.csv 中的 3D 坐标（相机坐标系），投影到对应图像上。
"""

import csv
import os
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from collections import defaultdict


def load_camera_info_yaml(yaml_path):
    """从 YAML 文件加载相机内参"""
    import yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # 解析相机矩阵
    cam_matrix_data = data['camera_matrix']['data']
    K = np.array(cam_matrix_data, dtype=np.float32).reshape(3, 3)
    
    # 解析畸变系数
    dist_data = data['distortion_coefficients']['data']
    D = np.array(dist_data, dtype=np.float32)
    
    width = data['image_width']
    height = data['image_height']
    
    return K, D, width, height


def project_point(point_3d, K):
    """
    将 3D 点（相机坐标系）投影到图像平面
    point_3d: [tx, ty, tz] - 相机坐标系下的 3D 点
    K: 相机内参矩阵 (3x3)
    返回: [x, y] - 图像坐标（像素）
    """
    tx, ty, tz = point_3d
    
    # 如果深度 <= 0，无法投影
    if tz <= 0:
        return None
    
    # 投影公式：x = K @ [tx, ty, tz]^T / tz
    x = (K[0, 0] * tx / tz) + K[0, 2]
    y = (K[1, 1] * ty / tz) + K[1, 2]
    
    return (x, y)


def intersect_ray_with_image(ox, oy, dx, dy, img_w, img_h):
    """求从 (ox,oy) 沿方向 (dx,dy) 的射线与图像边界的交点，返回 (ix,iy) 或 None。"""
    candidates = []
    eps = 1e-9
    # 检查与垂直边 x=0 和 x=img_w-1 的交点
    if abs(dx) > eps:
        for xb in (0, img_w - 1):
            t = (xb - ox) / dx
            if t > 0:
                yb = oy + t * dy
                if 0 <= yb <= img_h - 1:
                    candidates.append((t, xb, yb))
    # 检查与水平边 y=0 和 y=img_h-1 的交点
    if abs(dy) > eps:
        for yb in (0, img_h - 1):
            t = (yb - oy) / dy
            if t > 0:
                xb = ox + t * dx
                if 0 <= xb <= img_w - 1:
                    candidates.append((t, xb, yb))

    if not candidates:
        return None
    tmin = min(candidates, key=lambda x: x[0])
    return (int(round(tmin[1])), int(round(tmin[2])))


def load_averaged_groundtruth(csv_path):
    """
    从 groundtruth_averaged.csv 加载数据并保留顺序。
    返回: list of rows(dict) 顺序与文件一致，每行包含 tx,ty,tz,qx,qy,qz,qw 等字段。
    """
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 解析数值字段
            try:
                row_parsed = {
                    'filename': row.get('filename', ''),
                    'timestamp': row.get('timestamp', ''),
                    'tx': float(row.get('tx', 0.0)),
                    'ty': float(row.get('ty', 0.0)),
                    'tz': float(row.get('tz', 0.0)),
                    'qx': float(row.get('qx', 0.0)),
                    'qy': float(row.get('qy', 0.0)),
                    'qz': float(row.get('qz', 0.0)),
                    'qw': float(row.get('qw', 0.0)),
                    'score_mean': float(row.get('score_mean', 0.0)) if row.get('score_mean') else 0.0,
                    'num_tags': int(row.get('num_tags', 0)) if row.get('num_tags') else 0,
                }
            except Exception:
                # 如果解析失败，仍保留原始行但数值设为0
                row_parsed = {
                    'filename': row.get('filename', ''),
                    'timestamp': row.get('timestamp', ''),
                    'tx': 0.0,
                    'ty': 0.0,
                    'tz': 0.0,
                    'qx': 0.0,
                    'qy': 0.0,
                    'qz': 0.0,
                    'qw': 0.0,
                    'score_mean': 0.0,
                    'num_tags': 0,
                }
            rows.append(row_parsed)
    return rows


def visualize_apriltags(
    image_dir,
    groundtruth_csv,
    camera_yaml,
    output_dir,
    draw_tag_id=True,
    draw_coordinate_frame=False
):
    """
    对图像进行 AprilTag 可视化
    
    Args:
        image_dir: 图像所在目录
        groundtruth_csv: groundtruth.csv 文件路径
        camera_yaml: 相机参数 YAML 文件路径
        output_dir: 输出目录
        draw_tag_id: 是否绘制 tag ID
        draw_coordinate_frame: 是否绘制坐标系（会增加复杂度）
    """
    
    # 加载相机参数
    K, D, img_width, img_height = load_camera_info_yaml(camera_yaml)
    print(f"相机参数: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    
    # 加载按序的 averaged groundtruth 数据
    averaged_rows = load_averaged_groundtruth(groundtruth_csv)
    print(f"加载了 {len(averaged_rows)} 条 averaged groundtruth 记录")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_dir = Path(image_dir)
    
    # 获取 vis 目录中的图像文件并按名称排序，按顺序一一对应 averaged_rows
    img_paths = []
    for ext in ('.png', '.jpg', '.jpeg'):
        img_paths.extend(sorted(image_dir.glob(f'*{ext}')))
    img_paths = sorted(img_paths)

    if len(img_paths) == 0:
        print(f"未在 {image_dir} 中找到图像文件")
        return

    n_pairs = min(len(img_paths), len(averaged_rows))
    print(f"将处理 {n_pairs} 张图像（按顺序配对）")

    for i in range(n_pairs):
        img_path = img_paths[i]
        row = averaged_rows[i]

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  无法读取图像: {img_path}")
            continue

        tx, ty, tz = row['tx'], row['ty'], row['tz']
        qx, qy, qz, qw = row['qx'], row['qy'], row['qz'], row['qw']

        # 将 tag 坐标系的三个轴（以米为单位的向量）旋转到相机坐标系并投影
        proj_origin = project_point([tx, ty, tz], K)
        if proj_origin is None:
            print(f"  ⚠️  第 {i} 行投影深度无效，跳过")
            continue
        ox, oy = proj_origin

        # default axis length（米）由命令行参数传入全局变量 axis_length_m
        try:
            axis_length_m = visualize_apriltags.axis_length_m
        except Exception:
            axis_length_m = 0.2

        # 使用 scipy Rotation（四元数顺序为 x,y,z,w）
        rot = R.from_quat([qx, qy, qz, qw])
        axes = {
            'x': np.array([axis_length_m, 0.0, 0.0], dtype=np.float64),
            'y': np.array([0.0, axis_length_m, 0.0], dtype=np.float64),
            'z': np.array([0.0, 0.0, axis_length_m], dtype=np.float64),
        }

        # 投影并绘制三轴（红:X, 绿:Y, 蓝:Z）
        colors = {'x': (0, 0, 255), 'y': (0, 255, 0), 'z': (255, 0, 0)}
        thickness = 2
        tip_length = 0.08

        for key, vec in axes.items():
            p_cam = np.array([tx, ty, tz], dtype=np.float64) + rot.apply(vec)
            proj_p = project_point(p_cam, K)
            if proj_p is None:
                continue
            px_f, py_f = proj_p[0], proj_p[1]
            px, py = int(px_f), int(py_f)
            ox_i, oy_i = int(ox), int(oy)
            # 对于 Y 轴，延伸至图像边界
            if key == 'y':
                dx = px_f - ox
                dy = py_f - oy
                end_pt = intersect_ray_with_image(ox, oy, dx, dy, img_width, img_height)
                if end_pt is not None:
                    ex, ey = end_pt
                    cv2.line(img, (ox_i, oy_i), (ex, ey), colors[key], thickness)
                    # 在末端画一个小箭头
                    # 计算箭头短端点用于绘制箭头head
                    # 使用 cv2.arrowedLine 效果不好跨长线，先画线再画三角
                    # 画箭头三角
                    vec_e = np.array([ex - ox_i, ey - oy_i], dtype=np.float32)
                    norm = np.linalg.norm(vec_e)
                    if norm > 1e-6:
                        ux, uy = vec_e / norm
                        # 两个侧点
                        arrow_size = max(10, int(0.03 * max(img_width, img_height)))
                        left = (int(ex - ux * 0 - uy * arrow_size), int(ey - uy * 0 + ux * arrow_size))
                        right = (int(ex - ux * 0 + uy * arrow_size), int(ey - uy * 0 - ux * arrow_size))
                        cv2.fillConvexPoly(img, np.array([ (ex,ey), left, right ], dtype=np.int32), colors[key])
                else:
                    cv2.arrowedLine(img, (ox_i, oy_i), (px, py), colors[key], thickness, tipLength=tip_length)
            else:
                # 使用箭头绘制轴线
                cv2.arrowedLine(img, (ox_i, oy_i), (px, py), colors[key], thickness, tipLength=tip_length)

        # 可选绘制标签信息
        if draw_tag_id:
            label = f"avg #{i+1} n={row.get('num_tags', 0)}"
            cv2.putText(img, label, (int(ox) + 10, int(oy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # 如果投影超出图像范围或无效，已经在前面通过条件跳过或提示

        out_filename = f"vis_{img_path.stem}.png"
        out_path = output_path / out_filename
        cv2.imwrite(str(out_path), img)
        print(f"  ✓  已保存: {out_path}")
    
    print(f"\n✓ 可视化完成！输出目录: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化 AprilTag 检测结果')
    parser.add_argument(
        '--image-dir',
        default='/home/jacy/project/final_design/axispose/statistics/vis',
        help='图像目录'
    )
    parser.add_argument(
        '--groundtruth-csv',
        default='/home/jacy/project/final_design/axispose/statistics/groundtruth_averaged.csv',
        help='groundtruth.csv 文件路径'
    )
    parser.add_argument(
        '--camera-yaml',
        default='/home/jacy/project/final_design/axispose/config/d457_color.yaml',
        help='相机参数 YAML 文件路径'
    )
    parser.add_argument(
        '--output-dir',
        default='/home/jacy/project/final_design/axispose/statistics/vis_with_apriltag',
        help='输出目录'
    )
    parser.add_argument(
        '--draw-tag-id',
        action='store_true',
        default=True,
        help='是否绘制 tag ID'
    )
    parser.add_argument(
        '--draw-coordinate-frame',
        action='store_true',
        default=False,
        help='是否绘制坐标系'
    )
    parser.add_argument(
        '--axis-length',
        type=float,
        default=0.2,
        help='坐标轴长度（米），例如 0.2'
    )
    
    args = parser.parse_args()
    # 将 axis length 绑定到函数属性，便于内部访问
    visualize_apriltags.axis_length_m = args.axis_length

    visualize_apriltags(
        image_dir=args.image_dir,
        groundtruth_csv=args.groundtruth_csv,
        camera_yaml=args.camera_yaml,
        output_dir=args.output_dir,
        draw_tag_id=args.draw_tag_id,
        draw_coordinate_frame=args.draw_coordinate_frame
    )
