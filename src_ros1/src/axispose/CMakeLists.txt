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
find_package(TensorRT-YOLO QUIET)
find_package(Ceres QUIET)

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

)
if(TensorRT-YOLO_FOUND)
  include_directories(${TensorRT-YOLO_INCLUDE_DIRS})
  message(STATUS "TRTYOLO: ${TensorRT-YOLO_INCLUDE_DIRS}")
else()
  message(WARNING "axispose ROS1: TensorRT-YOLO not found, segment_node will not be built")
endif()

set(SRC_DIR src)

set(POSE_SOURCES
  ${SRC_DIR}/poseEstimate.cpp
  ${SRC_DIR}/point_cloud_processor.cpp
  ${SRC_DIR}/depth_aligner.cpp
  ${SRC_DIR}/gaussian_map_solver.cpp
)

if(Ceres_FOUND)
  list(APPEND POSE_SOURCES ${SRC_DIR}/ceres_joint_optimizer.cpp)
  add_definitions(-DAXISPOSE_HAS_CERES=1)
  message(STATUS "axispose ROS1: Ceres found, ceres algorithm enabled")
else()
  message(WARNING "axispose ROS1: Ceres not found, ceres algorithm will fallback to gaussian")
endif()

add_executable(camera_driver_node
  ${SRC_DIR}/driver.cpp
)
add_dependencies(camera_driver_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(camera_driver_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(pose_estimate_node
  ${POSE_SOURCES}
)
add_dependencies(pose_estimate_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pose_estimate_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
if(Ceres_FOUND)
  target_include_directories(pose_estimate_node PRIVATE ${CERES_INCLUDE_DIRS})
  target_link_libraries(pose_estimate_node ${CERES_LIBRARIES})
endif()

if(TensorRT-YOLO_FOUND)
  add_executable(segment_node
    ${SRC_DIR}/seg.cpp
  )
  add_dependencies(segment_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(segment_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${TensorRT-YOLO_LIBs})
endif()

add_executable(visualization_node
  ${SRC_DIR}/visualization.cpp
)
add_dependencies(visualization_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(visualization_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

install(TARGETS
  camera_driver_node
  pose_estimate_node
  visualization_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(TensorRT-YOLO_FOUND)
  install(TARGETS
    segment_node
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
endif()

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(PROGRAMS
  scripts/run_pose_algorithms_ros1.sh
  scripts/pose_success_monitor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
