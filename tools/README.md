# tools 使用说明（整理后）

本目录已整理为“单一评估入口 + 可编辑 YAML 配置”的结构。

## 1. 单一评估入口

- 入口脚本：`tools/eval_runner.py`
- 配置文件：`tools/eval_config.yaml`

你只需要改 `eval_config.yaml`，然后执行：

```bash
cd /home/jacy/project/final_design/axispose
python3 tools/eval_runner.py
```

可选：

```bash
python3 tools/eval_runner.py --config tools/eval_config.yaml
python3 tools/eval_runner.py --out-root statistics/custom_eval_001
python3 tools/eval_runner.py --dry-run
```

## 2. 支持的评估模式

通过 `tools/eval_config.yaml` 控制：

1. 仅运行算法、只记录时延和位姿（不做额外评估）
设置：
- `line2d_eval_enabled: false`
- `groundtruth_eval_enabled: false`

2. 纯 Line2D 评估（无 AprilTag）
设置：
- `line2d_eval_enabled: true`
- `groundtruth_eval_enabled: false`

3. 启用 AprilTag GroundTruth 评估
设置：
- `groundtruth_eval_enabled: true`
- 并补齐 `groundtruth_images_dir`、`groundtruth_camera_info_file`、`groundtruth_layout_file`

## 3. 输出结构

`out_root` 下每个算法目录（pca/ransac/gaussian/ceres）会包含：

- `metrics.csv`（算法耗时与位姿输出）
- `line2d_metrics.csv`（来自可视化节点）
- `line2d_summary.csv`（当 `line2d_eval_enabled=true`）
- `gt_eval/`（当 `groundtruth_eval_enabled=true`）
  - `groundtruth.csv`
  - `groundtruth_averaged.csv`
  - `aligned.csv`
  - `per_frame_errors.csv`
  - `evaluation_summary.csv`
  - `plots/`

## 4. 配置文件关键字段

`tools/eval_config.yaml`：

- `rgb_dir` / `depth_dir`：输入图像目录
- `algorithms`：算法列表（pca/ransac/gaussian/ceres）
- `timeout_seconds`：单算法运行超时
- `line2d_eval_enabled`：是否计算 line2d summary
- `groundtruth_eval_enabled`：是否执行 AprilTag GT 全流程
- `groundtruth_images_dir`：GT 图像目录
- `groundtruth_camera_info_file`：相机内参 YAML
- `groundtruth_layout_file`：AprilTag 布局文件（如 `tools/apriltag.yaml`）

## 5. 其他脚本说明

以下脚本保留为“子工具”，由 `eval_runner.py` 统一调度或供独立调试：

- `extract_groundtruth.py`
- `gt_ingest.py`
- `evaluate.py`
- `evaluate_line2d.py`
- `plot_metrics.py`
- `merge_metrics.py`
- `experiment_runner.py`
- `run.py`（通用工具入口，`run_full_eval` 已标记 deprecated）

## 6. 已清理的重复入口

为避免重复与混淆，以下旧评估入口已移除：

- 根目录：`evaluate_3_methods.sh`
- 根目录：`evaluate_4_methods.sh`
- 根目录：`evaluate_just_run.sh`
- tools：`run_full_eval.sh`
