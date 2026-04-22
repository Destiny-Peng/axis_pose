# Axispose 四算法原理与实现

## OOP 结构（当前）
- 公共基类节点：`PoseEstimateBase`
- 独立算法节点：`PoseEstimatePCA` / `PoseEstimateRANSAC` / `PoseEstimateGaussian` / `PoseEstimateCeres`
- 统一接口：`computePoseByAlgorithm(...)` 与 `benchmarkLabel()`
- 统一公共流程：订阅、点云预处理、benchmark记录、pose发布由基类完成

## 1. PCA（基线）
- 原理：对分割点云做协方差分解，最大特征值对应主轴方向。
- 适用：速度快、实现简单，作为稳定基线。
- 节点类：`PoseEstimatePCA`
- 实现入口：`PoseEstimateBase::computePoseFromCloud`
- 代码位置：
  - `src/poseEstimate.cpp`

## 2. RANSAC（线模型）
- 原理：在点云中随机采样拟合直线模型（SACMODEL_LINE），通过内点最大化获得主轴。
- 适用：有离群点时比纯PCA更鲁棒。
- 节点类：`PoseEstimateRANSAC`
- 实现入口：`PoseEstimateBase::computePoseFromSACLine`
- 代码位置：
  - `src/poseEstimate.cpp`
  - 点云预处理中 cluster+rasa c内点逻辑：`PoseEstimate::denoisePointCloud`

## 4. Ceres（3D+2D联合优化）
- 原理：
  - 3D 约束：点到线残差（Plücker 表达）
  - 2D 约束：mask 拟合中心线 `A*u + B*v + C = 0`，约束反投影线落在中心线上
  - 位置先验：约束线位置靠近初始轴点，避免“平行不重合”
- 提速策略：
  - 点采样（限制参与优化点数）
  - 降低迭代次数
  - 使用轻量线残差替代距离变换插值残差
- 节点类：`PoseEstimateCeres`
- 实现入口：`PoseEstimateBase::computePoseCeres`
- 代码位置：
  - `src/ceres_joint_optimizer.cpp`
  - `include/axispose/ceres_joint_optimizer.hpp`
  - `src/poseEstimate.cpp`

## 算法切换统一规则
- 推荐方式：在 launch 中切换 `pose_plugin`（组件插件名）
- 插件值：
  - `axispose::PoseEstimatePCA`
  - `axispose::PoseEstimateRANSAC`
  - `axispose::PoseEstimateGaussian`
  - `axispose::PoseEstimateCeres`
