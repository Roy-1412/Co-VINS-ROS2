#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import GroupAction, ExecuteProcess, LogInfo
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()
    
    config_path = os.path.join(
        get_package_share_directory("vins"), 
        "config/config",
        "A3",
        "A3_config.yaml"
      
    )

    # agent 1
    ld.add_action(GroupAction([
        PushRosNamespace("vins_1"),
        Node(package="vins", 
        executable="vins_node", 
        name="vins_node", 
        output="screen",
            arguments=["--remap", "/vins_1/agent_frame:=/agent_frame",],
            parameters=[{"agent_num": 1, "config_file": config_path,}],
            ),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", "/bags/1bag/1.db3", 
            "--loop",
            "--remap", "hcfox_1/image:=/vins_1/hcfox_1/image",],
            output="screen", name="rosbag_play_vins_1"),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", "/bags/1bag/1.db3", 
            "--loop",
            "--remap", "dji_sdk_1/dji_sdk/imu:=/vins_1/dji_sdk_1/dji_sdk/imu",],
            output="screen", name="rosbag_play_vins_1"),
        LogInfo(msg="[vins_1] started with agent_num=1"),
        ]))

 
    # agent 2
    # ld.add_action(GroupAction([
    #     PushRosNamespace("vins_2"),
    #     Node(package="vins", 
    #     executable="vins_node", 
    #     name="vins_node", 
    #     output="screen",
    #         arguments=["--remap", "/vins_2/agent_frame:=/agent_frame",],
    #         parameters=[{"agent_num": 2, "config_file": config_path,}],
    #         ),
    #     ExecuteProcess(
    #         cmd=["ros2", "bag", "play", "/bags/2bag/2.db3", 
    #         "--remap", "hcfox_1/image:=/vins_2/hcfox_1/image",],
    #         output="screen", name="rosbag_play_vins_2"),
    #     ExecuteProcess(
    #         cmd=["ros2", "bag", "play", "/bags/2bag/2.db3", 
    #         "--remap", "dji_sdk_1/dji_sdk/imu:=/vins_2/dji_sdk_1/dji_sdk/imu",],
    #         output="screen", name="rosbag_play_vins_2"),
    #     LogInfo(msg="[vins_2] started with agent_num=2"),
    #     ]))

    # # agent 3
    # ld.add_action(GroupAction([
    #     PushRosNamespace("vins_3"),
    #     Node(package="vins", 
    #     executable="vins_node", 
    #     name="vins_node", 
    #     output="screen",
    #         arguments=["--remap", "/vins_3/agent_frame:=/agent_frame",],
    #         parameters=[{"agent_num": 3, "config_file": config_path,}],
    #         ),
    #     ExecuteProcess(
    #         cmd=["ros2", "bag", "play", "/bags/3bag/3.db3", 
    #         "--remap", "hcfox_1/image:=/vins_3/hcfox_1/image",],
    #         output="screen", name="rosbag_play_vins_3"),
    #     ExecuteProcess(
    #         cmd=["ros2", "bag", "play", "/bags/3bag/3.db3", 
    #         "--remap", "dji_sdk_1/dji_sdk/imu:=/vins_3/dji_sdk_1/dji_sdk/imu",],
    #         output="screen", name="rosbag_play_vins_3"),
    #     LogInfo(msg="[vins_3] started with agent_num=3"),
    #     ]))

    # # agent 4
    # ld.add_action(GroupAction([
    #     PushRosNamespace("vins_4"),
    #     Node(package="vins", 
    #     executable="vins_node", 
    #     name="vins_node", 
    #     output="screen",
    #         arguments=["--remap", "/vins_4/agent_frame:=/agent_frame",],
    #         parameters=[{"agent_num": 4, "config_file": config_path,}],
    #         ),
    #     ExecuteProcess(
    #         cmd=["ros2", "bag", "play", "/bags/4bag/4.db3", 
    #         "--remap", "hcfox_1/image:=/vins_4/hcfox_1/image",],
    #         output="screen", name="rosbag_play_vins_4"),
    #     ExecuteProcess(
    #         cmd=["ros2", "bag", "play", "/bags/4bag/4.db3", 
    #         "--remap", "dji_sdk_1/dji_sdk/imu:=/vins_4/dji_sdk_1/dji_sdk/imu",],
    #         output="screen", name="rosbag_play_vins_4"),
    #     LogInfo(msg="[vins_4] started with agent_num=4"),
    #     ]))
   

    ld.add_action(Node(package='loop_fusion', executable='loop_fusion_node', name='loop_fusion_node', output='screen',))

    rviz_conf = os.path.join(get_package_share_directory('vins'), 'config', 'multi_agent_rviz.rviz')

    ld.add_action(Node(package='rviz2', executable='rviz2', name='rvizvisualisation', output='screen', arguments=['-d', rviz_conf],))

    return ld

