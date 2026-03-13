#!/usr/bin/env python3
"""
Composable launch for axispose camera driver.
- 将 CameraDriver 作为可组合组件加载到 component_container_mt 中
- rgb_dir 和 depth_dir 默认指向本包内的 image/rgb 和 image/depth 目录
- 其它参数从 config/param.yaml 加载
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('axispose')
    pkg_directory = os.path.join(pkg_share, '..', '..', '..', '..')

    # 默认图片目录在包内 image/rgb 和 image/depth
    default_rgb = os.path.join(pkg_directory, 'image1', 'rgb')
    default_depth = os.path.join(pkg_directory, 'image1', 'depth')
    default_engine = os.path.join(
        pkg_directory, 'engine', 'occlusion.engine')
    default_statistic = os.path.join(
        pkg_directory, 'statistics')

    # 参数文件
    param_file = os.path.join(pkg_share, 'config', 'param.yaml')
    # /home/jacy/project/final_design/axispose/image/depth
    # /home/jacy/Downloads/captured_img/captured_rgb
    # /home/jacy/Downloads/images_20251206/img
    # 声明可覆盖的 launch 参数（仍然允许用户通过命令行指定）
    rgb_dir_arg = DeclareLaunchArgument(
        'rgb_dir', default_value=default_rgb, description='路径到 RGB 图像目录')
    depth_dir_arg = DeclareLaunchArgument(
        'depth_dir', default_value=default_depth, description='路径到 深度 图像目录')
    # color_camera_info_arg = DeclareLaunchArgument('color_camera_info_file', default_value=os.path.join(
    #     pkg_share, 'config', 'd457_color.yaml'), description='color camera_info YAML 文件路径')
    # depth_camera_info_arg = DeclareLaunchArgument('depth_camera_info_file', default_value=os.path.join(
    #     pkg_share, 'config', 'd457_depth.yaml'), description='depth camera_info YAML 文件路径')
    engine_arg = DeclareLaunchArgument(
        'engine', default_value=default_engine, description='engine文件路径')
    statistic_directory_arg = DeclareLaunchArgument(
        'statistic_directory', default_value=default_statistic, description='统计信息文件路径')
    # Composable node 描述
    container = ComposableNodeContainer(
        name='axispose_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='axispose',
                plugin='axispose::SegmentNode',
                name='segment_node_component',
                parameters=[param_file, {
                    'engine': LaunchConfiguration('engine')
                }],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='axispose',
                plugin='axispose::PoseEstimate',
                name='pose_estimate_component',
                parameters=[param_file, {
                    'statistics_directory_path': LaunchConfiguration('statistic_directory')
                }],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='axispose',
                plugin='axispose::Visualization',
                name='visualization_component',
                parameters=[param_file],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='axispose',
                plugin='axispose::CameraDriver',
                name='camera_driver_component',
                parameters=[param_file, {
                    'rgb_dir': LaunchConfiguration('rgb_dir'),
                    'depth_dir': LaunchConfiguration('depth_dir'),
                    # 'color_camera_info_file': LaunchConfiguration('color_camera_info_file'),
                    # 'depth_camera_info_file': LaunchConfiguration('depth_camera_info_file')
                }],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
        ],
        output='screen'
    )

    return LaunchDescription([
        rgb_dir_arg,
        depth_dir_arg,
        engine_arg,
        statistic_directory_arg,
        LogInfo(
            msg=["axispose: starting camera_driver component with params from ", param_file]),
        container
    ])
