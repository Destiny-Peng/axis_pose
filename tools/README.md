# tools 使用说明（完整参数版）

本目录采用“综合入口优先 + 子工具分工明确”的方式组织。

## 1. 推荐先用综合入口

- 主入口脚本：`tools/eval_runner.py`
- 主配置文件：`tools/eval_config.yaml`

标准执行：

```bash
cd /home/jacy/project/final_design/axispose
python3 tools/eval_runner.py
```

常用变体：

```bash
python3 tools/eval_runner.py --config tools/eval_config.yaml
python3 tools/eval_runner.py --out-root statistics/custom_eval_001
python3 tools/eval_runner.py --dry-run
```

`eval_runner.py` 做的事情：

1. 逐算法启动 ROS2 launch，输出每个算法目录下的 `metrics.csv`。
2. 可选执行 `evaluate_line2d.py` 产出 `line2d_summary.csv`。
3. 可选执行 GT 全流程（`extract_groundtruth.py` + `gt_ingest.py` + `evaluate.py`）。
4. 可选执行 `summarize_eval_results.py` 生成 `summary/` 总览。

## 2. 典型产物目录

在 `out_root` 下，算法目录（`pca/ransac/gaussian/ceres`）通常包含：

- `metrics.csv`
- `line2d_metrics.csv`
- `line2d_summary.csv`（启用 line2d 时）
- `gt_eval/`（启用 groundtruth 时）
  - `groundtruth.csv`
  - `groundtruth_averaged.csv`
  - `aligned.csv`
  - `per_frame_errors.csv`
  - `evaluation_summary.csv`
  - `plots/`

聚合目录 `summary/`（由 `summarize_eval_results.py` 生成）：

- `summary_overall.csv`
- `summary_overall.md`
- `timing_by_stage.csv`
- `static_stability_deltas.csv`
- `total_elapsed_mean_ms.png`
- `line2d_compare.png`
- `gt_compare.png`
- `static_jump_rate.png`
- `jump_attitude_<algorithm>.png`
- `jump_plots/<algorithm>/pose_arrows_3d.png`
- `jump_plots/<algorithm>/pose_arrows_xy.png`
- `jump_plots/<algorithm>/pose_arrows_xz.png`
- `jump_plots/<algorithm>/pose_arrows_yz.png`

## 3. 离线 Kalman（支持直接输入评估目录）

脚本：`tools/offline_kalman_eval.py`

### 3.1 推荐：批处理评估目录

```bash
cd /home/jacy/project/final_design/axispose
python3 tools/offline_kalman_eval.py \
  --out-root statistics/eval_unified_20260326_210444
```

行为：

1. 自动遍历 `out_root` 下有 `metrics.csv` 的算法目录。
2. 在每个算法目录生成 `kalman_offline/` 子目录结果。
3. 生成 `summary/kalman_offline_overall.csv` 与 `summary/kalman_offline_overall.md`。
4. 回写 `summary/summary_overall.csv`（新增 kalman 统计列）。
5. 回写 `summary/summary_overall.md`（新增 `Offline Kalman Summary` 章节）。

### 3.2 单文件模式

```bash
python3 tools/offline_kalman_eval.py \
  --input statistics/axispose_eval/ceres/metrics.csv \
  --output-dir statistics/kalman_offline/ceres
```

单算法输出：

- `filtered_pose.csv`
- `step_deltas.csv`
- `summary.csv`
- `plots/pos_delta_series.png`
- `plots/ang_delta_series.png`
- `plots/raw_vs_kf_summary_bars.png`

### 3.3 角度硬门限机制

- 默认启用 `--hard-reject-ang-deg 45`
- 规则：当前 raw 姿态相对“上一帧保留姿态”的角差超过阈值，则丢弃当前帧并复用上一帧 KF 输出。
- 逐帧导出字段：`is_rejected`、`reject_reason`、`raw_ang_to_prev_kept_deg`。

## 4. 各脚本参数总览（每个 py 单独章节）

## 4.1 tools/eval_runner.py

用途：统一评估入口，按 YAML 批量跑算法并串联后处理。

参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--config` | 否 | `tools/eval_config.yaml` | 评估配置 YAML 路径 |
| `--out-root` | 否 | 空 | 覆盖输出根目录 |
| `--dry-run` | 否 | false | 仅打印执行计划，不运行 |

## 4.2 tools/eval_config.yaml（配置文件字段）

根键：`evaluation`

| 字段 | 默认值 | 说明 |
|---|---|---|
| `setup_bash` | `install/setup.bash` | ROS2 环境脚本 |
| `launch_file` | `launch/launch.py` | 启动文件 |
| `rgb_dir` | - | RGB 输入目录 |
| `depth_dir` | - | Depth 输入目录 |
| `out_root` | `statistics/eval_unified_%Y%m%d_%H%M%S` | 输出根目录模板（支持 strftime） |
| `algorithms` | `[pca,ransac,gaussian,ceres]` | 算法列表 |
| `timeout_seconds` | `65` | 每算法运行超时 |
| `line2d_eval_enabled` | `true` | 是否生成 line2d_summary |
| `groundtruth_eval_enabled` | `false` | 是否执行 GT 全流程 |
| `aggregate_summary_enabled` | `true` | 是否最终聚合 |
| `aggregate_summary_plots_enabled` | `true` | 聚合时是否画图 |
| `groundtruth_images_dir` | 空 | GT 图像目录 |
| `groundtruth_camera_info_file` | `config/d457_color.yaml` | GT 相机内参 |
| `groundtruth_layout_file` | `tools/apriltag.yaml` | AprilTag 布局 |
| `groundtruth_min_tags` | `4` | GT 最小可用标签数 |

## 4.3 tools/summarize_eval_results.py

用途：汇总 `out_root` 多算法结果，生成 CSV/MD/图。

参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--out-root` | 是 | 无 | 由 eval_runner 产出的根目录 |
| `--output-dir` | 否 | `<out_root>/summary` | 汇总输出目录 |
| `--no-plots` | 否 | false | 不生成图 |
| `--pos-jump-mm` | 否 | `40.0` | 位置跳变绝对阈值（mm） |
| `--ang-jump-deg` | 否 | `12.0` | 姿态跳变绝对阈值（deg） |
| `--jump-mad-k` | 否 | `6.0` | MAD 自适应阈值系数 |

## 4.4 tools/offline_kalman_eval.py

用途：对 `metrics.csv` 的位姿行做离线 Kalman 后处理，支持单文件和评估目录批处理，并可回写 `summary_overall`。

参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--input` | 条件必选 | 空 | 单文件模式输入 `metrics.csv` |
| `--out-root` | 条件必选 | 空 | 批处理模式输入评估根目录 |
| `--output-dir` | 否 | 空 | 单文件模式为输出目录；批处理模式为“所有算法输出根目录” |
| `--algorithms` | 否 | `pca,ransac,gaussian,ceres` | 批处理算法过滤列表 |
| `--session-mode` | 否 | `latest` | 会话选择：`latest` 或 `all` |
| `--session-gap-sec` | 否 | `30.0` | 会话切分时间阈值（秒） |
| `--kalman-position-process-noise` | 否 | `0.001` | 位置过程噪声 |
| `--kalman-position-measurement-noise` | 否 | `0.0004` | 位置测量噪声 |
| `--kalman-axis-process-noise` | 否 | `0.04` | 姿态轴过程噪声 |
| `--kalman-axis-measurement-noise` | 否 | `0.008` | 姿态轴测量噪声 |
| `--kalman-initial-covariance` | 否 | `1.0` | 初始协方差 |
| `--kalman-min-dt` | 否 | `0.001` | 最小时间步长 |
| `--kalman-max-dt` | 否 | `0.2` | 最大时间步长 |
| `--pos-jump-mm` | 否 | `40.0` | 位置跳变阈值（mm） |
| `--ang-jump-deg` | 否 | `12.0` | 姿态跳变阈值（deg） |
| `--jump-mad-k` | 否 | `6.0` | MAD 系数 |
| `--hard-reject-ang-deg` | 否 | `45.0` | 硬门限丢帧角度阈值，`<=0` 关闭 |

## 4.5 tools/evaluate_line2d.py

用途：统计 line2d 误差均值/方差/中位数。

参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--input` | 是 | 无 | `line2d_metrics.csv` |
| `--output` | 是 | 无 | 输出 `line2d_summary.csv` |

## 4.6 tools/extract_groundtruth.py

用途：从图像提取 AprilTag bundle 位姿并生成 groundtruth CSV，可选保存可视化。

参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--images` | 是 | 无 | 输入图像目录 |
| `--camera-info` | 是 | 无 | 相机内参 YAML |
| `--layout` | 是 | 无 | AprilTag 布局文件 |
| `--output` | 否 | `groundtruth.csv` | 输出 CSV |
| `--save-annotated` | 否 | 空 | 保存标注图目录 |
| `--ext` | 否 | `png` | 主扫描扩展名（`png/jpg/jpeg`） |
| `--min-tags` | 否 | `4` | bundle 求解最小标签数 |
| `--visualize` | 否 | false | 弹窗显示标注 |

## 4.7 tools/gt_ingest.py

用途：GT 数据后处理，包含两个子命令。

子命令 `average` 参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--input` | 是 | 无 | 原始 `groundtruth.csv` |
| `--output` | 是 | 无 | 聚合后 `groundtruth_averaged.csv` |

子命令 `align` 参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--est` | 是 | 无 | 估计结果 CSV（通常 `metrics.csv`） |
| `--gt` | 是 | 无 | 平均后 GT CSV |
| `--output` | 是 | 无 | 对齐输出 CSV |
| `--tol` | 否 | `0.5` | 按时间戳对齐容忍度（秒） |
| `--by-sequence` | 否 | false | 按序号对齐而不是时间戳 |

## 4.8 tools/evaluate.py

用途：基于 `aligned.csv` 计算每帧误差与汇总统计，可选绘图。

参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--input` | 是 | 无 | `aligned.csv` |
| `--out` | 是 | 无 | 每帧误差 CSV |
| `--summary` | 是 | 无 | 汇总 CSV |
| `--plots` | 否 | 空 | 绘图输出目录 |

## 4.9 tools/line2d_tag_residual.py

用途：计算上下边线交点、Tag-Y 投影线与残差。

参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--line2d` | 是 | 无 | 含 upper/lower 线系数的 `line2d_metrics.csv` |
| `--aligned` | 是 | 无 | 含位姿列的 `aligned.csv` |
| `--camera-info` | 是 | 无 | 相机内参 YAML |
| `--output` | 否 | `statistics/line2d_tag_residual.csv` | 输出残差 CSV |
| `--axis-length` | 否 | `0.10` | Tag 局部 Y 轴投影长度（m） |
| `--pose-prefix` | 否 | `gt_` | 位姿列前缀（如 `gt_`/`est_`） |
| `--projection-model` | 否 | `auto` | 投影模型：`auto`/`axispose`/`opencv` |

## 4.10 tools/visualize_three_lines.py

用途：把 upper/lower/tagY 三条线绘制在 vis 图上，支持自动缩放。

参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--vis-dir` | 否 | `statistics/vis` | 输入可视化图目录 |
| `--line2d` | 否 | `statistics/line2d_metrics.csv` | line2d 指标 CSV |
| `--residual` | 否 | `statistics/line2d_tag_residual.csv` | 残差 CSV |
| `--output-dir` | 否 | `statistics/vis_three_lines` | 输出目录 |
| `--width` | 否 | `1280` | 输出画布宽度 |
| `--height` | 否 | `720` | 输出画布高度 |
| `--scale-threshold` | 否 | `10000.0` | 交点超阈值时触发缩放 |

## 4.11 tools/experiment_runner.py

用途：按 experiments YAML 做参数网格实验与重复运行。

参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `config` | 是 | 无 | experiments YAML 路径 |

说明：该脚本主要通过 YAML 字段控制行为（`launch_cmd`、`sweeps`、`repeats`、`run_duration_sec` 等）。

## 4.12 tools/plot_metrics.py

用途：画 `metrics.csv` 的按阶段时序图和直方图。

参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `path` | 是 | 无 | 一个 `metrics.csv` 或包含多个 run 的目录 |
| `--out` | 否 | `plots` | 图输出目录 |

## 4.13 tools/merge_metrics.py

用途：递归合并多个 `metrics.csv`，新增 `run_name` 列。

参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `path` | 是 | 无 | 统计根目录 |

## 4.14 tools/run.py（兼容封装入口）

用途：历史兼容封装，内部通过 subprocess 调用其他脚本。

子命令 `run_full_eval` 参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--config` | 否 | `tools/eval_config.yaml` | 传给 `eval_runner.py` |
| `--out-dir` | 否 | 空 | 传给 `eval_runner.py --out-root` |

子命令 `extract_gt` 参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--images` | 是 | 无 | 图像目录 |
| `--camera-info` | 是 | 无 | 相机内参 YAML |
| `--tag-size` | 是 | 无 | 兼容参数（注意：目标脚本当前按 `--layout` 工作） |
| `--output` | 是 | 无 | 输出 CSV |
| `--save-annotated` | 否 | 空 | 标注图目录 |

子命令 `gt_average` 参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--input` | 是 | 无 | 输入 GT CSV |
| `--output` | 是 | 无 | 输出平均 GT CSV |

子命令 `gt_align` 参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--est` | 是 | 无 | 估计 CSV |
| `--gt` | 是 | 无 | GT CSV |
| `--output` | 是 | 无 | 输出对齐 CSV |
| `--by-sequence` | 否 | false | 按序号对齐 |

子命令 `evaluate` 参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--input` | 是 | 无 | 对齐 CSV |
| `--out` | 是 | 无 | 每帧误差 CSV |
| `--summary` | 是 | 无 | 汇总 CSV |
| `--plots` | 否 | 空 | 绘图目录 |

子命令 `plot_metrics` 参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `path` | 是 | 无 | 输入路径 |
| `--out` | 否 | `plots` | 输出目录 |

子命令 `merge_metrics` 参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `path` | 是 | 无 | 统计根目录 |

子命令 `experiment` 参数：

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `config` | 是 | 无 | 实验配置 YAML |

## 5. 已移除的旧入口

以下旧脚本已清理，避免重复流程：

- 根目录 `evaluate_3_methods.sh`
- 根目录 `evaluate_4_methods.sh`
- 根目录 `evaluate_just_run.sh`
- `tools/run_full_eval.sh`
