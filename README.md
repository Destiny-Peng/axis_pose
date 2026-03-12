# axispose

驱动与位姿估计组件（ROS 2）

本包包含用于仿真/回放场景的相机驱动、位姿估计与可视化组件。主要功能：
- 从磁盘发布 RGB / 深度 图像流（`CameraDriver`）。
- 基于深度点云与 mask 估计轴线位姿（`PoseEstimate`）。
- 在彩色图像上可视化位姿与 mask（`Visualization`）。

本文档包含构建、运行与常见用法示例。

## 构建

在工作区根或包含 `axispose` 的 colcon 工作区运行：

```bash
colcon build --packages-select axispose
source install/setup.bash
```

## 启动（Launch）

包内已提供可组合组件启动脚本：

```bash
ros2 launch axispose launch.py
```

可通过命令行覆盖常用参数，例如：

```bash
ros2 launch axispose launch.py rgb_dir:=/path/to/rgb depth_dir:=/path/to/depth \
  color_camera_info_file:=/path/to/color_camera_info.yaml \
  depth_camera_info_file:=/path/to/depth_camera_info.yaml
```

常用 launch 参数（在 `launch/launch.py` 可声明覆盖）：
- `rgb_dir`：RGB 图像目录（默认包内示例目录）。
- `depth_dir`：深度图像目录（默认包内示例目录）。
- `color_camera_info_file`：彩色相机 `camera_info` YAML 文件路径。
- `depth_camera_info_file`：深度相机 `camera_info` YAML 文件路径。
- `engine` / `statistic_directory`：与其他组件相关的参数（见 `config/param.yaml`）。

## 重要运行细节

- `CameraDriver`：
  - 现在将 `camera_info` 以 `transient_local`（latched）QoS 发布一次，晚启动的订阅者也能拿到相机参数。
  - 图像消息的 `header.frame_id` 会使用 `frame_id` 参数派生为 `frame_id_color` 与 `frame_id_depth`。
  - 预加载图像可通过 `rgb_dir` / `depth_dir` 指定目录加载。若数据量很大，可考虑禁用或限制预加载（后续改进）。

- `PoseEstimate`：
  - 订阅深度图与 mask，同步后将深度转换为点云并估计轴线位姿。
  - 如果提供了 color camera_info，节点会把深度对齐到彩色图像分辨率再应用 mask（精确对齐）；否则回退到将 mask 缩放到深度分辨率的旧逻辑。

- `Visualization`：默认订阅 `camera/color/camera_info`（用于投影），请确保 `CameraDriver` 发布的 color camera_info topic 名称一致。

## camera_info YAML 示例

支持常见 `camera_info` 格式（`image_width`/`image_height`、`camera_matrix`/`K`、`distortion_coefficients`/`D`、`projection_matrix`/`P` 等）。示例最小结构：

```yaml
image_width: 640
image_height: 480
camera_matrix:
  rows: 3
  cols: 3
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
distortion_coefficients:
  rows: 1
  cols: 5
  data: [k1, k2, p1, p2, k3]
projection_matrix:
  rows: 3
  cols: 4
  data: [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
```

注意：若深度为 `uint16`，应以毫米为单位（本代码将其视作 mm 并转换为米）；若使用 `float32`，请确保深度以米为单位。

## 使用录制的 bag 测试

1. 播放 bag（假设 bag 中包含 RGB/Depth/CameraInfo/Mask）：

```bash
ros2 bag play my_capture.db3 --loop
```

2. 启动本包并指向合适目录或 camera_info：

```bash
ros2 launch axispose launch.py rgb_dir:=/path/to/rgb depth_dir:=/path/to/depth \
  color_camera_info_file:=/path/to/color_camera_info.yaml depth_camera_info_file:=/path/to/depth_camera_info.yaml
```

或直接仅启动节点组合（在 launch 中已包含）：

```bash
ros2 launch axispose launch.py
```

## 故障排查

- 若未看到 camera_info：确认 `color_camera_info_file`/`depth_camera_info_file` 路径正确，或检查 `ros2 topic echo /camera/color/camera_info`。
- 若深度对齐有偏移：确认 `camera_info` 中的内参（K）对应正确的相机，并提供 color 与 depth 的 camera_info 以获得更精确对齐。
- 若节点崩溃或退出：查看终端日志与 `ros2 run rclcpp_components component_container_mt` 的输出；检查是否由于不支持的深度图格式导致（建议使用 `uint16` mm 或 `float32` m）。

## 后续改进建议

- 支持 camera_info 的 `transient_local` 外加一次性发布，避免周期性发布（已实现）。
- 支持读取 color↔depth 外参（YAML 或 TF），并使用 OpenCV 的 rectification / `rgbd::registerDepth` 进行更精确对齐。
- 增加单元测试与 CI，文档化更多示例与常见问题。

---

如果你需要，我可以：
- 帮你把 README 转成包级别的 `package.xml` 描述或更详细的 `USAGE.md`；
- 添加示例 camera_info YAML 文件到 `config/`；
- 或现在运行一次编译/launch 验证 README 中的命令。
