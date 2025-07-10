# agent_msg/AgentMsg.msg

# Standard ROS header
std_msgs/Header header

# Sequence number
int16 seq

# IMU pose and extrinsics
geometry_msgs/Point position_imu
geometry_msgs/Quaternion orientation_imu
geometry_msgs/Point tic
geometry_msgs/Quaternion ric

# 3D and 2D feature points
geometry_msgs/Point32[] point_3d
geometry_msgs/Point32[] point_uv
int64[] point_des
geometry_msgs/Point32[] feature_2d
int64[] feature_des

# Notes:
# points refer to window features, containing 3D positions
# feature refers to FAST features in the whole image
# point_uv and feature_uv can be removed in the future
# TODO: implement distance resampling
# TODO: handle tic and ric when loading/saving
