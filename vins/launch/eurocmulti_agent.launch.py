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
        "euroc",
        "euroc_mono_imu_config.yaml"
        # "euroc",
        # "euroc_mono_imu_config.yaml",
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
            cmd=["ros2", "bag", "play", "/bags/MH1/MH1.db3", 
            "--remap", "cam0/image_raw:=/vins_1/cam0/image_raw",],
            output="screen", name="rosbag_play_vins_1"),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", "/bags/MH1/MH1.db3", 
            "--remap", "imu0:=/vins_1/imu0",],
            output="screen", name="rosbag_play_vins_1"),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", "/bags/MH1/MH1.db3", 
            "--remap", "leica/position:=/vins_1/leica/position",],
            output="screen", name="rosbag_play_vins_1"),
        LogInfo(msg="[vins_1] started with agent_num=1"),
        ]))

   # agent 2
    ld.add_action(GroupAction([
        PushRosNamespace("vins_2"),
        Node(package="vins", 
        executable="vins_node", 
        name="vins_node", 
        output="screen",
            arguments=["--remap", "/vins_2/agent_frame:=/agent_frame",],
            parameters=[{"agent_num": 2, "config_file": config_path,}],
            ),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", "/bags/MH2/MH2.db3", 
            "--remap", "cam0/image_raw:=/vins_2/cam0/image_raw",],
            output="screen", name="rosbag_play_vins_2"),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", "/bags/MH2/MH2.db3", 
            "--remap", "imu0:=/vins_2/imu0",],
            output="screen", name="rosbag_play_vins_2"),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", "/bags/MH2/MH2.db3", 
            "--remap", "leica/position:=/vins_2/leica/position",],
            output="screen", name="rosbag_play_vins_2"),
        LogInfo(msg="[vins_2] started with agent_num=2"),
        ]))

    # agent 3
    ld.add_action(GroupAction([
        PushRosNamespace("vins_3"),
        Node(package="vins", 
        executable="vins_node", 
        name="vins_node", 
        output="screen",
            arguments=["--remap", "/vins_3/agent_frame:=/agent_frame",],
            parameters=[{"agent_num": 3, "config_file": config_path,}],
            ),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", "/bags/MH3/MH3.db3", 
            "--remap", "cam0/image_raw:=/vins_3/cam0/image_raw",],
            output="screen", name="rosbag_play_vins_3"),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", "/bags/MH3/MH3.db3", 
            "--remap", "imu0:=/vins_3/imu0",],
            output="screen", name="rosbag_play_vins_3"),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", "/bags/MH3/MH3.db3", 
            "--remap", "leica/position:=/vins_3/leica/position",],
            output="screen", name="rosbag_play_vins_3"),
        LogInfo(msg="[vins_3] started with agent_num=3"),
        ]))
   

    ld.add_action(Node(package='loop_fusion', executable='loop_fusion_node', name='loop_fusion_node', output='screen',))

    rviz_conf = os.path.join(get_package_share_directory('vins'), 'config', 'multi_agent_rviz.rviz')

    ld.add_action(Node(package='rviz2', executable='rviz2', name='rvizvisualisation', output='screen', arguments=['-d', rviz_conf],))

    return ld

