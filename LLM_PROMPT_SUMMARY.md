# Axispose Long-Term Project Summary

本文件用于给后续 Agent 提供长期稳定上下文，减少重复探索成本。

## 1) 项目目标与主链路
- 目标：从 RGB-D + 分割 mask 估计轴体 3D 位姿，并在 2D 投影域评估主轴拟合效果。
- 标准链路：`CameraDriver -> SegmentNode -> PoseEstimate* -> Visualization`。
- 核心输出：
	- `/shaft/pose`
	- `metrics.csv`
	- `line2d_metrics.csv`
	- `line2d_summary.csv`

## 2) 当前关键架构决策（必须遵守）
- Pose 估计已经按 OOP+ROS 组件解耦：
	- 公共基类：`PoseEstimateBase`
	- 独立算法节点：`PoseEstimatePCA`、`PoseEstimateRANSAC`、`PoseEstimateGaussian`、`PoseEstimateCeres`
- 算法切换不再依赖单节点内部 `if/else` 路由，改为 launch 层切换 `pose_plugin`。
- 评测脚本默认超时为 65 秒（并可从 `config/param.yaml` 的 `evaluation_timeout_seconds` 读取）。

## 3) 目录重点
- 核心源码：`src/`、`include/axispose/`
- 配置：`config/param.yaml`
- 启动：`launch/launch.py` + `launch/launch_*.py`
- 评测：`evaluate_4_methods.sh`、`tools/evaluate_line2d.py`
- 文档：`docs/`
- 历史/可能冗余：`src_1/`、旧 `statistics/` 结果

## 4) 推荐阅读顺序（新 Agent 首次接手）
1. `docs/ARCHITECTURE_AND_DEPENDENCIES.md`
2. `docs/ALGORITHMS_IMPLEMENTATION.md`
3. `docs/EVALUATE_SCRIPT_USAGE.md`
4. `config/param.yaml`
5. `src/poseEstimate.cpp` 与 `include/axispose/poseEstimate.hpp`

## 5) 文档索引（可直接引用）
- 体系结构与依赖：`docs/ARCHITECTURE_AND_DEPENDENCIES.md`
- 算法原理与实现：`docs/ALGORITHMS_IMPLEMENTATION.md`
- 评测脚本说明：`docs/EVALUATE_SCRIPT_USAGE.md`
- 协作进度板：`docs/AGENT_TODO_PROGRESS.md`

## 6) 协作约定
- 新增算法优先采用“新增独立组件节点”方式，不破坏已有节点。
- 参数优先放入 `config/param.yaml`，脚本仅做最小编排。
- 评估比较统一使用 `evaluate_4_methods.sh`，避免多套口径并行。
