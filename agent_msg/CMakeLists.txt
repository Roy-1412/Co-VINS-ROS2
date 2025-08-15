cmake_minimum_required(VERSION 3.5)
project(agent_msg)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# 1. 指定需要生成的消息接口
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AgentMsg.msg"
  DEPENDENCIES
    std_msgs
    geometry_msgs
)

# 2. 安装生成的消息接口
ament_export_dependencies(rosidl_default_runtime)

ament_package()
