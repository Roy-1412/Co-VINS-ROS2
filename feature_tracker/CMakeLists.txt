cmake_minimum_required(VERSION 3.5)
project(feature_tracker)

# -------------------- C++ Standard --------------------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------- Find Dependencies --------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ament_index_cpp REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
find_package(Ceres REQUIRED)
find_package(camera_model REQUIRED)

# -------------------- Locate Camodocal Header --------------------
# Ensure we can find CameraFactory.h under camera_model's install include
find_path(CAMODAL_INCLUDE_DIR
  NAMES camodocal/camera_models/CameraFactory.h
  PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../install/camera_model/include
    $ENV{COLCON_PREFIX_PATH}/camera_model/include
  NO_DEFAULT_PATH
)
if(NOT CAMODAL_INCLUDE_DIR)
  message(FATAL_ERROR "Cannot find camodocal/camera_models/CameraFactory.h; ensure camera_model is built and installed.")
endif()

# -------------------- Executable --------------------
add_executable(feature_tracker_node
  src/feature_tracker_node.cpp
  src/feature_tracker.cpp
  src/parameters.cpp
)

# -------------------- Include Directories --------------------
target_include_directories(feature_tracker_node
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CAMODAL_INCLUDE_DIR}
)

# -------------------- ROS2 Package Dependencies --------------------
ament_target_dependencies(feature_tracker_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  tf2_ros
  ament_index_cpp
)

# -------------------- Link Libraries --------------------
target_link_libraries(feature_tracker_node
  camera_model::camera_model
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Boost::filesystem
  Boost::program_options
  Boost::system
  Ceres::ceres
)

# -------------------- Install --------------------
install(TARGETS feature_tracker_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/../config
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml" PATTERN "*.json"
)

ament_package()
