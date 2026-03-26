# Agent Todo And Progress Board

本文件用于多 Agent 协作交接，记录待办、进行中、已完成和风险。

## Current Todo
- [ ] 继续优化 Gaussian 方向稳定性，目标 angle_mean_deg < 0.5
- [ ] 补充单元/回归测试，覆盖四算法节点组件加载与基础输出
- [ ] 清理 `src_1/` 与历史脚本，降低维护复杂度
- [ ] 跨模块统一“数据表达 + 职责边界”：继续推进 Driver/Seg（Visualization 已完成首轮）

## In Progress
- [ ] 观察 Ceres 在不同数据集上的偏移稳定性（当前仅 image1 全量验证）

## Completed
- [x] PoseEstimate 重构为 OOP：`PoseEstimateBase` + 四独立算法组件节点
- [x] launch 支持 `pose_plugin` 选择算法组件
- [x] 新增便捷 launch：`launch_pca.py`、`launch_ransac.py`、`launch_gaussian.py`、`launch_ceres.py`
- [x] 评测脚本改为插件切换逻辑，超时统一为 65 秒
- [x] 评测超时支持从 `config/param.yaml` 读取 `evaluation_timeout_seconds`
- [x] 更新长期知识文档：`LLM_PROMPT_SUMMARY.md`
- [x] 更新架构/算法/评测说明文档
- [x] PoseEstimate 内参传递统一为 camera matrix（cv::Mat），移除散落 fx/fy/cx/cy 主通路
- [x] DepthAligner 改为无状态 API（由调用方传入 depth/color camera matrix）
- [x] PointCloudProcessor 接管完整去噪流程（voxel + SOR + cluster 筛选），PoseEstimate 仅做编排
- [x] Visualization 投影链路改为 camera matrix 表达，去除回调内分散标量内参解析
- [x] 评估工具链统一为单入口：`tools/eval_runner.py` + `tools/eval_config.yaml`，支持仅计时 / 纯line2d / GT联合评估
- [x] 清理重复评估入口脚本（evaluate_3/4/just_run、tools/run_full_eval）

## Code Quality Pattern Summary
- 问题模式：
  - 抽象层存在但核心逻辑仍在调用方堆叠，导致“名义解耦、实际耦合”。
  - 同一类数据在多种表达之间反复转换（如标量内参与矩阵内参并存）。
  - 工具类只承接局部功能，业务节点仍保留重复流程。
- 本轮已落地策略：
  - 统一数据表达：内参统一为 camera matrix。
  - 统一职责边界：处理链路下沉到工具类，节点负责编排。
  - 统一扩展接口：算法节点继承基类，避免 if/else 扩散。

## Latest Verified Result Snapshot
- 结果目录：`statistics/eval4_20260325_192159`
- line2d 有效帧：四算法均为 50
- line2d 指标（mean）：
  - PCA: angle 0.298975858 deg, offset 2.52594544 px
  - RANSAC: angle 0.442568782 deg, offset 3.087508308 px
  - Gaussian: angle 2.053701876 deg, offset 3.716609272 px
  - Ceres: angle 0.021122266 deg, offset 2.04929412 px

## Handover Notes
- 后续新增算法建议直接新增 `PoseEstimateXXX` 组件类，复用 `PoseEstimateBase`。
- 任何评测结论请附输出目录路径，避免口头结论无法复现。
