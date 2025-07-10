cmake_minimum_required(VERSION 3.5)
project(camera_model)

# --- C++ settings ---
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

# --- Include directories ---
include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# --- Source files ---
set(SRC_FILES
  src/chessboard/Chessboard.cc
  src/calib/CameraCalibration.cc
  src/camera_models/Camera.cc
  src/camera_models/CameraFactory.cc
  src/camera_models/CostFunctionFactory.cc
  src/camera_models/PinholeCamera.cc
  src/camera_models/CataCamera.cc
  src/camera_models/EquidistantCamera.cc
  src/camera_models/ScaramuzzaCamera.cc
  src/sparse_graph/Transform.cc
  src/gpl/gpl.cc
  src/gpl/EigenQuaternionParameterization.cc
)

# --- Library target ---
add_library(camera_model ${SRC_FILES})

# Optional alias target
add_library(camodocal::camera_model ALIAS camera_model)

# --- Link libraries ---
target_link_libraries(camera_model
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  Eigen3::Eigen
)

# --- Export dependencies ---
ament_target_dependencies(camera_model
  rclcpp
  std_msgs
)

# --- Executable (for calibration use) ---
add_executable(calibration_node src/intrinsic_calib.cc)

target_link_libraries(calibration_node
  camera_model
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  Eigen3::Eigen
)

ament_target_dependencies(calibration_node
  rclcpp
  std_msgs
)

# --- Install headers ---
install(DIRECTORY include/ DESTINATION include)

# --- Install targets ---
install(TARGETS camera_model calibration_node
  EXPORT camera_model_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# --- Export for downstream use ---
ament_export_include_directories(include)
ament_export_libraries(camera_model)
ament_export_targets(camera_model_targets HAS_LIBRARY_TARGET)

ament_package()
