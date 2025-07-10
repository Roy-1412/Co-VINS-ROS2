cmake_minimum_required(VERSION 3.5)
project(vins_estimator LANGUAGES CXX)

# -------------------- C++ Standard --------------------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------- Find Dependencies --------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp         REQUIRED)
find_package(std_msgs       REQUIRED)
find_package(geometry_msgs  REQUIRED)
find_package(nav_msgs       REQUIRED)
find_package(tf2_ros        REQUIRED)
find_package(tf2            REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge      REQUIRED)
find_package(camera_model   REQUIRED)
find_package(agent_msg      REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV         REQUIRED)
find_package(Ceres          REQUIRED)
find_package(Eigen3         REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)

# -------------------- Locate Camodocal Header --------------------
find_path(CAMODAL_INCLUDE_DIR
  NAMES camodocal/camera_models/CameraFactory.h
  PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../install/camera_model/include
    $ENV{COLCON_PREFIX_PATH}/camera_model/include
  NO_DEFAULT_PATH
)
if(NOT CAMODAL_INCLUDE_DIR)
  message(FATAL_ERROR "Cannot find camodocal/camera_models/CameraFactory.h; please build & install camera_model first.")
endif()

# -------------------- Executable --------------------
add_executable(vins_estimator
  src/estimator_node.cpp
  src/parameters.cpp
  src/estimator.cpp
  src/feature_manager.cpp
  src/factor/pose_local_parameterization.cpp
  src/factor/projection_factor.cpp
  src/factor/projection_td_factor.cpp
  src/factor/marginalization_factor.cpp
  src/utility/utility.cpp
  src/utility/visualization.cpp
  src/utility/CameraPoseVisualization.cpp
  src/initial/solve_5pts.cpp
  src/initial/initial_aligment.cpp
  src/initial/initial_sfm.cpp
  src/initial/initial_ex_rotation.cpp
  src/ThirdParty/DUtils/Random.cpp
  src/ThirdParty/DUtils/Timestamp.cpp
  src/ThirdParty/DVision/BRIEF.cpp
)

# -------------------- Include Directories --------------------
target_include_directories(vins_estimator
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/ThirdParty>
  PRIVATE
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CAMODAL_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

# -------------------- ROS2 Package Dependencies --------------------
ament_target_dependencies(vins_estimator
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2
  tf2_geometry_msgs
  cv_bridge
  agent_msg
  visualization_msgs
)

# -------------------- Link Libraries --------------------
target_link_libraries(vins_estimator
  camera_model::camera_model
  ${CERES_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
  Boost::filesystem
  Boost::program_options
  Boost::system 
  rclcpp::rclcpp             
)

# -------------------- Install --------------------
install(TARGETS vins_estimator
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY support_files/
  DESTINATION share/${PROJECT_NAME}/support_files
)

ament_package()
