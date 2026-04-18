# AxisPose ROS1 (Noetic) Deploy Notes

## 1. Build

```bash
cd ~/catkin_ws
catkin_make --pkg axispose
source devel/setup.bash
```

## 2. Real-Hardware Launch (Jetson Orin)

By default this launch uses real camera topics and does not start mock image driver.

```bash
roslaunch axispose ros1_node.launch \
  use_mock_driver:=false \
  run_segmentation:=true \
  run_visualization:=false \
  pose_algorithm:=pca
```

Useful args:
- `pose_algorithm:=pca|ransac|gaussian|ceres`
- `run_segmentation:=false` when mask is provided by another node.
- `enable_preload:=true` only if `/usr/local/lib/libcustom_plugins.so` exists.
- `color_topic:=... depth_topic:=...` for camera remap.
- `align_depth_to_color:=false` in pose node params (default in launch).

If you run built-in segmentation, make sure `engine/best.engine` exists.

## 3. Run All Algorithms (real topics)

```bash
rosrun axispose run_pose_algorithms_ros1.sh --seconds 60 --stats-root /tmp/axispose_eval --no-vis
```

This runs `pca/ransac/gaussian/ceres` sequentially and writes `metrics.csv` to:
- `/tmp/axispose_eval/pca/metrics.csv`
- `/tmp/axispose_eval/ransac/metrics.csv`
- `/tmp/axispose_eval/gaussian/metrics.csv`
- `/tmp/axispose_eval/ceres/metrics.csv`

## 4. Online Success Monitor

```bash
rosrun axispose pose_success_monitor.py _window_sec:=10.0
```

It reports mask/pose counts and success ratio in a sliding window.

## 5. Deployment Checklist

- Verify TensorRT engine path: `$(find axispose)/engine/best.engine`
- Verify camera topics are correct for your driver.
- Verify both camera-info topics are published.
- Start without visualization first (`run_visualization:=false`) to reduce load.
- Use `pose_algorithm:=pca` as first bring-up baseline.
