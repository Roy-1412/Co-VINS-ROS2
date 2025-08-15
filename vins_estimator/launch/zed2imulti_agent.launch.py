#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, GroupAction, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()

    config_pkg_path = get_package_share_directory('config_pkg')
    config_path = PathJoinSubstitution([
        config_pkg_path,
        'config/zed2i/zed2i_multi_config.yaml'
    ])
    vins_path = PathJoinSubstitution([
        config_pkg_path,
        'config/../'
    ])
    support_path = PathJoinSubstitution([
        config_pkg_path,
        'support_files'
    ])

    # agent 1
    ld.add_action(GroupAction([
        Node(
            package='feature_tracker',
            executable='feature_tracker',
            name='feature_tracker',
            namespace='vins_1',
            output='screen',
            parameters=[{'config_file': config_path, 'vins_folder': vins_path}]
        ),
        Node(
            package='vins_estimator',
            executable='vins_estimator',
            name='vins_estimator',
            namespace='vins_1',
            output='screen',
            arguments=["--remap", "/vins_1/agent_frame:=/agent_frame"],
            parameters=[{"agent_num": 1, "config_file": config_path, "vins_folder": vins_path}]
        ),

        ExecuteProcess(
            cmd=["ros2", "bag", "play", "/bags/ROY_0.db3", 
            "--remap", "zed/zed_node/imu/data_raw:=/vins_1/zed/zed_node/imu/data_raw",],
            output="screen", name="rosbag_play_vins_1"),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", "/bags/ROY_0.db3", 
            "--remap", "zed/zed_node/left/image_rect_color:=/vins_1/zed/zed_node/left/image_rect_color",],
            output="screen", name="rosbag_play_vins_1"),
        LogInfo(msg="[vins_1] started with agent_num=1"),
    ]))

   # agent 2
    ld.add_action(GroupAction([
    Node(
    package='feature_tracker',
    executable='feature_tracker',
    name='feature_tracker',
    namespace='vins_2',
    output='screen',
    parameters=[{'config_file': config_path, 'vins_folder': vins_path}]),

    Node(package='vins_estimator',
    executable='vins_estimator',
    name='vins_estimator',
    namespace='vins_2',
    output='screen',
    arguments=["--remap", "/vins_2/agent_frame:=/agent_frame",],
    parameters=[{"agent_num": 2, "config_file": config_path, "vins_folder": vins_path}],),
    # ExecuteProcess(
    #     cmd=["ros2", "bag", "play", "/bags/ROY_0.db3", 
    #     "--remap", "zed/zed_node/imu/data_raw:=/vins_2/zed/zed_node/imu/data_raw",],
    #     output="screen", name="rosbag_play_vins_2"),
    # ExecuteProcess(
    #     cmd=["ros2", "bag", "play", "/bags/ROY_0.db3", 
    #     "--remap", "zed/zed_node/left/image_rect_color:=/vins_2/zed/zed_node/left/image_rect_color",],
    #     output="screen", name="rosbag_play_vins_2"),
    LogInfo(msg="[vins_2] started with agent_num=2"),
    ]))

#     # agent 3
#     ld.add_action(GroupAction([
#     Node(
#     package='feature_tracker',
#     executable='feature_tracker',
#     name='feature_tracker',
#     namespace='vins_3',
#     output='screen',
#     parameters=[{'config_file': config_path, 'vins_folder': vins_path}]),

#     Node(package='vins_estimator',
#     executable='vins_estimator',
#     name='vins_estimator',
#     namespace='vins_3',
#     output='screen',
#     arguments=["--remap", "/vins_3/agent_frame:=/agent_frame",],
#     parameters=[{"agent_num": 3, "config_file": config_path, "vins_folder": vins_path}],),

#     ExecuteProcess(
#         cmd=["ros2", "bag", "play", "/bags/ROY_0.db3", 
#         "--remap", "zed/zed_node/imu/data_raw:=/vins_3/zed/zed_node/imu/data_raw",],
#         output="screen", name="rosbag_play_vins_3"),
#     ExecuteProcess(
#         cmd=["ros2", "bag", "play", "/bags/ROY_0.db3", 
#         "--remap", "zed/zed_node/imu/data_raw:=/vins_3/zed/zed_node/imu/data_raw",],
#         output="screen", name="rosbag_play_vins_3"),
#     LogInfo(msg="[vins_3] started with agent_num=3"),
#     ]))
   

    ld.add_action(Node(
        package='pose_graph',
        executable='pose_graph',
        name='pose_graph',
        output='screen'
    ))

    rviz_config_path = PathJoinSubstitution([config_pkg_path, 'config/euroc_multi.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )
    ld.add_action(rviz_node)

    return ld

