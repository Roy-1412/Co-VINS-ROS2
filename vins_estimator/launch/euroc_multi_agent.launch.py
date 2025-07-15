import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable

def generate_launch_description():

    
    # Declare bag file path as launch argument
    sequence_1_arg = DeclareLaunchArgument(
        'sequence_1',
        default_value='/root/test/test.db3',
        #default_value='/root/ros/ros.db3',
        description='Path to the ROS2 bag file'
    )

    # Paths for config_file and vins_folder
    feature_tracker_share = get_package_share_directory('feature_tracker')
    config_file_path = '/root/Co-VINS-ROS2_ws/config/euroc/euroc_config.yaml'
    #config_file_path = '/root/Co-VINS-ROS2_ws/config/zed2i/zed2i_imu_config.yaml'
    vins_folder_path = os.path.join(
        feature_tracker_share,
        '../config/../'
    )

    # Feature Tracker Node (no --params-file for ROS2 parameters)
    feature_tracker_node = Node(
        package='feature_tracker',
        executable='feature_tracker_node',
        name='feature_tracker',
        output='screen',
        parameters=[
            {'config_file': config_file_path},  # algorithm config path
            {'vins_folder': vins_folder_path},
            {'agent_num': 1}
        ]
    )
    
    
    # VINS Estimator Node
    vins_estimator = Node(
        package='vins_estimator',
        executable='vins_estimator',
        name='vins_estimator',
        output='screen',
        
        parameters=[
            {'config_file': config_file_path},
            {'vins_folder': vins_folder_path},
            {'agent_num': 1}
        ]
    )

    # ros2 bag play process
    bag_play = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', LaunchConfiguration('sequence_1'),'--loop'
        ],
        output='screen'
    )
    
    # Pose Graph Node
    pose_graph_node = Node(
        package='pose_graph',
        executable='pose_graph',
        name='pose_graph',
        output='screen',
        parameters=[
            {'visualization_shift_x': 0},
            {'visualization_shift_y': 0},
            {'skip_cnt': 0},
            {'skip_dis': 0.0},
            {'pose_graph_save_path': '/home/ros/output/pose_graph/'},
            {'pose_graph_result_path': '/home/ros/output/'}
        ]
    )

    # RViz Visualization Node
    rviz_config = os.path.join(
        get_package_share_directory('vins_estimator'),
        '../config/multi_agent_rviz.rviz'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rvizvisualisation',
        output='screen',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        
        sequence_1_arg,
        feature_tracker_node,
        
        vins_estimator,
        bag_play,
        pose_graph_node,
        rviz_node
    ])
