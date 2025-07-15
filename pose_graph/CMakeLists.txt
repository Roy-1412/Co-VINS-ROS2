cmake_minimum_required(VERSION 3.5)
project(pose_graph)


# 默认使用 C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ament build 系统
find_package(ament_cmake REQUIRED)

# ROS 2 & TF2 相关
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(agent_msg REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# camera_model 包（提供 camodocal 头和库）
find_package(camera_model REQUIRED)

# 第三方库
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)

# 定位 CameraFactory.h
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

# 可执行文件及其源码
add_executable(pose_graph
  src/pose_graph_node.cpp
  src/pose_graph.cpp
  src/keyframe.cpp
  src/utility/CameraPoseVisualization.cpp
  src/ThirdParty/DBoW/BowVector.cpp
  src/ThirdParty/DBoW/FBrief.cpp
  src/ThirdParty/DBoW/FeatureVector.cpp
  src/ThirdParty/DBoW/QueryResults.cpp
  src/ThirdParty/DBoW/ScoringObject.cpp
  src/ThirdParty/DUtils/Random.cpp
  src/ThirdParty/DUtils/Timestamp.cpp
  src/ThirdParty/DVision/BRIEF.cpp
  src/ThirdParty/VocabularyBinary.cpp
)

# 包含目录
target_include_directories(pose_graph
  PRIVATE
    include
    src/utility
    src/ThirdParty/DBoW
    src/ThirdParty/DUtils
    src/ThirdParty/DVision
    src
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${tf2_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CAMODAL_INCLUDE_DIR}
)

# ROS2 包依赖（不直接包含 camera_model，链接库手动添加）
ament_target_dependencies(pose_graph
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  cv_bridge
  agent_msg
  tf2
  tf2_ros
  tf2_geometry_msgs
  rclcpp
)

# 链接第三方库
target_link_libraries(pose_graph
  camera_model::camera_model
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  Boost::filesystem
  Boost::program_options
  Boost::system
  tf2::tf2
  tf2_geometry_msgs::tf2_geometry_msgs
  rclcpp::rclcpp            
)
# 安装
install(TARGETS pose_graph
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch meshes
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
