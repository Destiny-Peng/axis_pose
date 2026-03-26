# 四算法集成评估脚本使用说明

## 脚本
- `evaluate_4_methods.sh`（主线，推荐保留）
- `evaluate_3_methods.sh`（历史脚本，三算法 + GT流水线）
- `evaluate_just_run.sh`（历史快速脚本，gaussian/ceres 两算法）

## 功能
- 在相同输入下依次运行四种算法：`pca/ransac/gaussian/ceres`
- 每种算法通过切换 `pose_plugin` 启动独立算法节点
- 输出：
  - `metrics.csv`（耗时与位姿统计）
  - `line2d_metrics.csv`（每帧2D线误差）
  - `line2d_summary.csv`（2D线误差汇总）

## 参数来源
- 主要运行参数来自：`config/param.yaml`
- 评测超时默认读取：`evaluation_timeout_seconds`（当前建议 65 秒）
- 脚本参数仅用于覆盖输入目录与输出目录

## eval3 / evaljustrun 与 eval4 的关系
- `eval4` 是当前标准入口，覆盖四算法统一比较。
- `eval3` 主要历史价值是内置调用 `run_full_eval.sh` 进行 AprilTag GT 评估。
- `evaljustrun` 是早期临时脚本（仅 2 算法 + 固定数据路径），与现有流程高度重叠。

建议：
1. 若你后续都使用 `eval4`（含本文档下方 GT 接口），`evaljustrun` 可删除。
2. `eval3` 可暂留作历史对照；若团队统一迁移到 `eval4 + GT`，也可删除。

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

## GroundTruth 评估接口（已预留）

### 启用方式 A：脚本参数直接传入
```bash
bash ./evaluate_4_methods.sh <rgb_dir> <depth_dir> <out_root> <gt_images_dir> <gt_camera_info_yaml> <gt_tag_size_m>
```

### 启用方式 B：`param.yaml` 配置
在 `config/param.yaml` 设置：
- `groundtruth_eval_enabled: true`
- `groundtruth_images_dir: <你的GT图像目录>`
- `groundtruth_camera_info_file: config/d457_color.yaml`（或绝对路径）
- `groundtruth_tag_size_m: 0.05`

启用后，每个算法目录会新增：
- `gt_eval/groundtruth.csv`
- `gt_eval/aligned.csv`
- `gt_eval/evaluation_summary.csv`

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
