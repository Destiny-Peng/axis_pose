cmake_minimum_required(VERSION 3.10)
project(axispose)

# 标准配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()



# 依赖查找
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(TensorRT-YOLO REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Ceres REQUIRED)

ament_auto_find_build_dependencies()

include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${TensorRT-YOLO_INCLUDE_DIRS}
)
# 组件库定义
add_library(CameraDriver SHARED
  src/driver.cpp
)
target_link_libraries(CameraDriver
  ${OpenCV_LIBS}
)
ament_target_dependencies(CameraDriver
  rclcpp
  rclcpp_components
  tf2
  tf2_ros
  tf2_eigen
  geometry_msgs
  visualization_msgs
  OpenCV
  Eigen3
  sensor_msgs
  cv_bridge
  image_transport
  camera_info_manager
  ament_index_cpp
)
rclcpp_components_register_nodes(CameraDriver PLUGIN "axispose::CameraDriver" EXECUTABLE camera_driver_node)

#next one
add_library(SegmentNode SHARED
  src/seg.cpp
)

target_link_libraries(SegmentNode
  ${OpenCV_LIBS}
  ${TensorRT-YOLO_LIBs}
)
ament_target_dependencies(SegmentNode
  rclcpp
  rclcpp_components
  tf2
  tf2_ros
  tf2_eigen
  visualization_msgs
  geometry_msgs
  OpenCV
  Eigen3
  cv_bridge
  image_transport
)
rclcpp_components_register_nodes(SegmentNode PLUGIN "axispose::SegmentNode" EXECUTABLE SegmentNode_node)

# next one
add_library(PoseEstimate SHARED
  src/poseEstimate.cpp
  src/depth_aligner.cpp
  src/point_cloud_processor.cpp
  src/gaussian_map_solver.cpp
  src/ceres_joint_optimizer.cpp
)
target_link_libraries(PoseEstimate
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  Ceres::ceres
)
ament_target_dependencies(PoseEstimate
  rclcpp
  rclcpp_components
  tf2
  tf2_ros
  tf2_eigen
  geometry_msgs
  visualization_msgs
  OpenCV
  Eigen3
  cv_bridge
  pcl_conversions
  image_transport
)
rclcpp_components_register_nodes(PoseEstimate PLUGIN "axispose::PoseEstimate" EXECUTABLE PoseEstimate_node)

# next one
add_library(Visualization SHARED
  src/visualization.cpp
)
target_link_libraries(Visualization
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
ament_target_dependencies(Visualization
  rclcpp
  rclcpp_components
  tf2
  tf2_ros
  tf2_eigen
  geometry_msgs
  visualization_msgs
  OpenCV
  Eigen3
  cv_bridge
  pcl_conversions
  image_transport
)
rclcpp_components_register_nodes(Visualization PLUGIN "axispose::Visualization" EXECUTABLE Visualization_node)

# 安装规则
ament_export_targets(export_CameraDriver)
install(TARGETS CameraDriver
        EXPORT export_CameraDriver
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)
ament_export_targets(export_SegmentNode)
install(TARGETS SegmentNode
        EXPORT export_SegmentNode
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)
ament_export_targets(export_PoseEstimate)
install(TARGETS PoseEstimate
        EXPORT export_PoseEstimate
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)
ament_export_targets(export_Visualization)
install(TARGETS Visualization
        EXPORT export_Visualization
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

add_executable(evaluate_pipelines 
  src/evaluate_pipelines.cpp
  src/gaussian_map_solver.cpp
  src/ceres_joint_optimizer.cpp
)
target_link_libraries(evaluate_pipelines
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  Ceres::ceres
)
ament_target_dependencies(evaluate_pipelines
  Eigen3
)
install(TARGETS evaluate_pipelines
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
