# 项目架构、作业流程与依赖文件清单

## 一、整体架构
### 1. 节点级流程
1. `CameraDriver`：读取离线图像目录并发布 RGB/Depth/CameraInfo
2. `SegmentNode`：YOLO 分割，输出 mask
3. `PoseEstimate`：点云构建、去噪、四算法位姿求解、输出 `/shaft/pose`
4. `Visualization`：叠加可视化、保存图片、记录 2D 线评估 CSV

### 2. launch 入口
- `launch/launch.py`
- 通过 `algorithm_type` 切换 `pca/ransac/gaussian/ceres`

## 二、作业流程（推荐）
1. 准备输入目录：`image1/rgb` + `image1/depth`
2. 执行四算法脚本：`evaluate_4_methods.sh`
3. 查看每算法目录中的：
   - `metrics.csv`
   - `line2d_metrics.csv`
   - `line2d_summary.csv`
4. 对比耗时与 2D 误差指标

## 三、项目实际使用的关键源码（可用于清理无效部分）

### C++ 源文件（核心）
- `src/driver.cpp`
- `src/seg.cpp`
- `src/poseEstimate.cpp`
- `src/visualization.cpp`
- `src/depth_aligner.cpp`
- `src/point_cloud_processor.cpp`
- `src/gaussian_map_solver.cpp`
- `src/ceres_joint_optimizer.cpp`

### C++ 头文件（核心）
- `include/axispose/driver.hpp`
- `include/axispose/seg.hpp`
- `include/axispose/poseEstimate.hpp`
- `include/axispose/visualization.hpp`
- `include/axispose/depth_aligner.hpp`
- `include/axispose/point_cloud_processor.hpp`
- `include/axispose/gaussian_map_solver.hpp`
- `include/axispose/ceres_joint_optimizer.hpp`
- `include/axispose/benchmark.hpp`
- `include/axispose/debug_manager.hpp`

### Python 工具（评估相关）
- `tools/evaluate_line2d.py`
- `tools/run_full_eval.sh`
- `tools/extract_groundtruth.py`
- `tools/gt_ingest.py`
- `tools/evaluate.py`

## 四、构建系统中参与编译的目标（来自 CMake）
- `CameraDriver`（组件库）
- `SegmentNode`（组件库）
- `PoseEstimate`（组件库）
- `Visualization`（组件库）
- `evaluate_pipelines`（测试可执行）

## 五、可优先检查的潜在冗余区域
- `src_1/` 目录（历史副本）
- 旧统计目录与历史脚本（不影响运行但体积大）
- 非当前评估链路使用的临时脚本
