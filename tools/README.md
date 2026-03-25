# tools - 使用说明

此目录包含若干用于评估、可视化及实验管理的脚本。已提供一个统一入口 `tools/run.py`，建议优先使用该命令行工具来运行常见流程。

前提依赖
- Python 3
- 推荐安装：`numpy opencv-python pyyaml pandas matplotlib scipy` 以及 AprilTag 实现之一：`apriltag` 或 `pupil-apriltags`。

主要脚本
- `extract_groundtruth.py`：从图像中检测 AprilTag 并通过 `solvePnP` 导出每个 tag 的位姿（CSV）。
- `gt_ingest.py`：GT 处理工具，支持 `average`（对同一图像多 tag 取平均）和 `align`（将估计与 GT 对齐生成 aligned.csv）。
- `evaluate.py`：对齐后的结果计算位置 / 角度误差并保存 per-frame 与 summary，还可生成若干可视化图表。
- `plot_metrics.py`：绘制 `metrics.csv` 的时间序列与直方图。
- `merge_metrics.py`：合并多个 run 的 `metrics.csv` 为单个 CSV（添加 `run_name` 列）。
- `experiment_runner.py`：基于实验配置自动生成参数并运行 ROS2 launch（用于批量试验）。
- `run_full_eval.sh`：原有的 Bash 一键脚本（等效功能已整合到 `tools/run.py`）。
- `run.py`：新的统一 Python CLI（包装并依次调用上面的脚本），优先使用。

统一入口：`tools/run.py`

示例：完整评估流程（提取 GT -> 平均 -> 对齐 -> 评估）

```bash
cd /path/to/project/axispose
python3 tools/run.py run_full_eval \
  --images statistics/vis \
  --camera-info config/d457_color.yaml \
  --tag-size 0.05 \
  --est-csv statistics/metrics.csv \
  --out-dir statistics/run01
```

独立子命令示例

- 提取 GT（只运行 detect + save annotated）：
```bash
python3 tools/run.py extract_gt --images /path/to/images --camera-info config/d457_color.yaml --tag-size 0.05 --output stats/gt.csv --save-annotated stats/gt_vis
```
- 对 GT 取平均：
```bash
python3 tools/run.py gt_average --input stats/gt.csv --output stats/gt_avg.csv
```
- 对齐估计与 GT：
```bash
python3 tools/run.py gt_align --est estimates.csv --gt stats/gt_avg.csv --output stats/aligned.csv --by-sequence
```
- 评估 aligned 结果并绘图：
```bash
python3 tools/run.py evaluate --input stats/aligned.csv --out stats/per_frame.csv --summary stats/summary.csv --plots stats/plots
```

其他工具
- 合并 metrics：`python3 tools/run.py merge_metrics /path/to/statistics`
- 绘制 metrics：`python3 tools/run.py plot_metrics /path/to/statistics --out plots`
- 运行实验 sweep：`python3 tools/run.py experiment examples/experiments.yaml`

说明与下一步
- `tools/run.py` 当前通过 `subprocess` 调用现有脚本以保证行为一致；若想进一步整合（直接在 Python 层调用、共享函数、改进错误处理与日志），我可以把这些脚本的核心函数重构成可导入模块并在 `run.py` 中直接调用。
- 若需我现在把 `run_full_eval.sh` 删除或替换为快捷软链接，我也可以处理。

如需我把 README 翻译成英文或补充更多示例（例如 `visualize_apriltag.py` 的用法与参数），请告诉我。
