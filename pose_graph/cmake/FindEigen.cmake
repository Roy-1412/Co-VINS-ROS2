# cmake/FindEigen.cmake - Simplified for ROS 2 / Ament using built-in FindEigen3

# Use CMake's built-in module to locate Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Provide compatibility variables for existing CMakeLists
set(EIGEN_FOUND TRUE)
set(EIGEN_INCLUDE_DIR ${Eigen3_INCLUDE_DIRS})
set(EIGEN_INCLUDE_DIRS ${Eigen3_INCLUDE_DIRS})
set(EIGEN_VERSION ${Eigen3_VERSION})

# Mark include directory advanced to hide in GUI
mark_as_advanced(FORCE EIGEN_INCLUDE_DIR)

# Standard args handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen3 DEFAULT_MSG EIGEN_INCLUDE_DIR)
