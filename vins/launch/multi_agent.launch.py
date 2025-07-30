#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
#    # Bag 文件路径参数
#     bag_arg = DeclareLaunchArgument(
#         'bag_file',
#         default_value='/bags/test.db3',
#         description='要播放的 ros2 bag 路径'
#     )

#     # 配置文件参数（绝对路径写到你源树中的文件）
#     config_arg = DeclareLaunchArgument(
#         'config_file',
#         default_value='/root/ros2_ws/src/VINS-Fusion-ROS2-no_cuda/config/euroc/euroc_mono_imu_config.yaml',
#         description='VINS 节点的配置文件'
#     )

#     ld = LaunchDescription([bag_arg, config_arg])

#     # 命名空间 vins_1
#     ld.add_action(PushRosNamespace('vins_1'))

#     # 启动 vins_node，把 config_file 当作 argv[1]
#     ld.add_action(
#         Node(
#             package='vins',
#             executable='vins_node',
#             name='vins_node',
#             output='screen',
#             parameters=[{
#             'config_file': LaunchConfiguration('config_file'),
#             'agent_num': 1
#   }],
#         )
#     )

#     # 播放 ros2 bag
#     ld.add_action(
#         ExecuteProcess(
#             cmd=[
#                 'ros2', 'bag', 'play',
#                 LaunchConfiguration('bag_file'),
#                 '--remap', '/imu0:=imu0',
#                 '--remap', '/cam0/image_raw:=cam0/image_raw',
#             ],
#             output='screen',
#         )
#     )

    # loop_fusion_node (原 pose_graph_node)，覆盖 mesh_resource
    ld = LaunchDescription()
    mesh_pkg = get_package_share_directory('loop_fusion')
    mesh_default = PathJoinSubstitution([mesh_pkg, 'meshes', 'hummingbird.mesh'])
    ld.add_action(
        Node(
            package='loop_fusion',
            executable='loop_fusion_node',
            name='loop_fusion_node',
            output='screen',
            parameters=[
                {'visualization_shift_x': 0},
                {'visualization_shift_y': 0},
                {'skip_cnt': 0},
                {'skip_dis': 0.0},
                {'pose_graph_save_path': '/root/ros2_ws/'},
                {'pose_graph_result_path': '/root/ros2_ws/'},
                {'mesh_resource': mesh_default},
            ],
        )
    )

    # RViz2 单独节点
    rviz_conf = os.path.join(
        get_package_share_directory('vins'),
        'config', 'multi_agent_rviz.rviz'
    )
    ld.add_action(
        Node(
            package='rviz2',
            executable='rviz2',
            name='rvizvisualisation',
            output='screen',
            arguments=['-d', rviz_conf],
        )
    )

    return ld
