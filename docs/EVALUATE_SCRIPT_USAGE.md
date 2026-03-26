# 四算法评估使用说明（统一入口）

## 入口
- 统一评估脚本：`tools/eval_runner.py`
- 统一配置文件：`tools/eval_config.yaml`

## 设计目标
- 单一入口，避免历史脚本重复。
- 支持三种评估模式：
  1. 仅记录算法输出与耗时（不做额外评估）
  2. 纯 line2d 评估（无需 AprilTag）
  3. line2d + GroundTruth（AprilTag）联合评估

## 快速使用
```bash
cd /home/jacy/project/final_design/axispose
python3 tools/eval_runner.py
```

可选参数：
```bash
python3 tools/eval_runner.py --config tools/eval_config.yaml
python3 tools/eval_runner.py --out-root statistics/my_eval
python3 tools/eval_runner.py --dry-run
```

## 配置方式（推荐）
主要参数都在 `tools/eval_config.yaml` 中维护：
- 输入目录：`rgb_dir` / `depth_dir`
- 算法列表：`algorithms`
- 超时：`timeout_seconds`
- 模式开关：`line2d_eval_enabled` / `groundtruth_eval_enabled`
- GT 参数：`groundtruth_images_dir` / `groundtruth_camera_info_file` / `groundtruth_layout_file`

## 输出
每个算法输出目录包含：
- `metrics.csv`
- `line2d_metrics.csv`
- `line2d_summary.csv`（当 line2d 开启）
- `gt_eval/*`（当 GT 开启）

## 说明
详细使用示例与字段解释见：`tools/README.md`。
