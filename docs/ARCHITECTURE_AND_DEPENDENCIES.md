# 项目架构、作业流程与依赖文件清单

## 一、整体架构

### 1. 节点级流程
1. `CameraDriver`：读取离线目录并发布 RGB/Depth/CameraInfo。
2. `SegmentNode`：运行分割模型并发布 mask。
3. `PoseEstimate*`：点云构建、去噪、位姿求解并发布 `/shaft/pose`。
4. `Visualization`：叠加可视化、保存图像、记录 `line2d_metrics.csv`。

### 2. PoseEstimate 的 OOP 解耦（当前版本）
- 公共基类：`PoseEstimateBase`（统一订阅、预处理、评测记录、发布）。
- 独立算法节点：
   - `PoseEstimatePCA`
   - `PoseEstimateRANSAC`
   - `PoseEstimateGaussian`
   - `PoseEstimateCeres`
- 每个算法节点作为独立 ROS2 组件注册，可在 launch 中直接更换组件插件，不再依赖运行时 `if/else` 切分算法。

### 3. launch 入口
- 通用入口：`launch/launch.py`
   - 通过 `pose_plugin` 选择算法组件插件。
- 便捷入口：
   - `launch/launch_pca.py`
   - `launch/launch_ransac.py`
   - `launch/launch_gaussian.py`
   - `launch/launch_ceres.py`

## 二、参数管理策略

- 主要运行参数统一在 `config/param.yaml`。
- 脚本层只保留必要的运行编排参数（输入目录、输出目录、插件映射）。
- 评测超时默认读取 `config/param.yaml` 中 `evaluation_timeout_seconds`（当前值 65）。

## 三、作业流程（推荐）

1. 准备输入目录：`image1/rgb` 与 `image1/depth`。
2. 运行四算法脚本：`evaluate_4_methods.sh`。
3. 查看每算法目录中的：
    - `metrics.csv`
    - `line2d_metrics.csv`
    - `line2d_summary.csv`
4. 横向比较耗时与 2D 指标（角度、偏移）。

## 四、项目核心源码清单

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

### 评估工具
- `evaluate_4_methods.sh`
- `tools/evaluate_line2d.py`

## 五、构建目标（CMake）

- `CameraDriver`（组件库）
- `SegmentNode`（组件库）
- `PoseEstimate`（组件库，包含 4 个算法组件插件）
- `Visualization`（组件库）

## 六、优先可清理区域

- `src_1/` 历史副本目录
- 历史统计输出目录（体积大）
- 已淘汰评估脚本与临时脚本
