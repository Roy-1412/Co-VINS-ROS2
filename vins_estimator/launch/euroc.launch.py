from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():

    config_pkg_path = get_package_share_directory('config_pkg')

    config_path = PathJoinSubstitution([
        config_pkg_path,
        'config/euroc/euroc_config.yaml'
    ])

    vins_path = PathJoinSubstitution([
        config_pkg_path,
        'config/../'
    ])

    support_path = PathJoinSubstitution([
        config_pkg_path,
        'support_files'
    ])

    
    # Define the vins_estimator node
    # vins_estimator_node = Node(
    #     package='vins_estimator',
    #     executable='vins_estimator',
    #     name='vins_estimator',
    #     namespace='vins_estimator',
    #     output='screen',
    #     parameters=[{
    #         'config_file': config_path,
    #         'vins_folder': vins_path
    #     }]
    # )

    # Define the pose_graph node
   

    return LaunchDescription([
        LogInfo(msg=['[vins estimator launch] config path: ', config_path]),
        LogInfo(msg=['[vins estimator launch] vins path: ', vins_path]),
        LogInfo(msg=['[vins estimator launch] support path: ', support_path]),
        # vins_estimator_node,
        pose_graph_node
    ])