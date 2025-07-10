cmake_minimum_required(VERSION 3.5)
project(agent_msg)

# Find ament and ROSIDL generators
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Find message dependencies
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Generate ROS 2 interface definitions
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AgentMsg.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Export runtime dependencies for downstream packages
ament_export_dependencies(rosidl_default_runtime)

# Declare this as an ament package
ament_package()
