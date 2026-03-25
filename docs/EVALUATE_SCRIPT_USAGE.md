# 四算法集成评估脚本使用说明

## 脚本
- `evaluate_4_methods.sh`

## 功能
- 在相同输入下依次运行四种算法：`pca/ransac/gaussian/ceres`
- 每种算法通过 launch 启动完整流程
- 输出：
  - `metrics.csv`（耗时与位姿统计）
  - `line2d_metrics.csv`（每帧2D线误差）
  - `line2d_summary.csv`（2D线误差汇总）

## 用法
```bash
cd /home/jacy/project/final_design/axispose
bash ./evaluate_4_methods.sh
```

可选参数：
```bash
bash ./evaluate_4_methods.sh <rgb_dir> <depth_dir> <out_root>
```
- `rgb_dir` 默认：`axispose/image1/rgb`
- `depth_dir` 默认：`axispose/image1/depth`
- `out_root` 默认：`axispose/statistics/eval4_时间戳`

## 结果目录结构（示例）
```text
statistics/eval4_20260325_190000/
  pca/
    metrics.csv
    line2d_metrics.csv
    line2d_summary.csv
  ransac/
    ...
  gaussian/
    ...
  ceres/
    ...
```

## 结果解读建议
- `metrics.csv`：关注 `pose_*` 行的 `elapsed_ms`
- `line2d_summary.csv`：
  - `angle_mean_deg` 越小越好
  - `offset_mean_px` 越小越好
