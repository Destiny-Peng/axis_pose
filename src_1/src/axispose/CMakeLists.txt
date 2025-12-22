cmake_minimum_required(VERSION 3.0.2)
project(axispose)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/trtyolo")
message(STATUS "cmake: ${CMAKE_PREFIX_PATH}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  camera_info_manager
  message_filters
  tf
  pcl_ros
  pcl_conversions
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TensorRT-YOLO REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES axispose
  CATKIN_DEPENDS roscpp sensor_msgs geometry_msgs cv_bridge image_transport camera_info_manager message_filters pcl_ros pcl_conversions
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${TensorRT-YOLO_INCLUDE_DIRS}

)
message(STATUS "TRTYOLO: ${TensorRT-YOLO_INCLUDE_DIRS}")

set(SRC_DIR src)

add_executable(camera_driver_node
  ${SRC_DIR}/driver.cpp
)
add_dependencies(camera_driver_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(camera_driver_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(pose_estimate_node
  ${SRC_DIR}/poseEstimate.cpp
)
add_dependencies(pose_estimate_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pose_estimate_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(segment_node
  ${SRC_DIR}/seg.cpp
)
add_dependencies(segment_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(segment_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${TensorRT-YOLO_LIBs})

add_executable(visualization_node
  ${SRC_DIR}/visualization.cpp
)
add_dependencies(visualization_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(visualization_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
