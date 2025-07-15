#include "visualization.h"
#include "rclcpp/rclcpp.hpp"
#include <rclcpp/logging.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/bool.hpp"
#include "agent_msg/msg/agent_msg.hpp"
#include <eigen3/Eigen/Dense>
#include <sensor_msgs/msg/point_field.hpp>
#include "CameraPoseVisualization.h"
#include <builtin_interfaces/msg/duration.hpp>

#include "estimator.h"
using namespace rclcpp;
using namespace Eigen;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry, pub_latest_odometry;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path, pub_relo_path;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_point_cloud2, pub_margin_cloud;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_relo_relative_pose;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_camera_pose;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_keyframe_point;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_extrinsic;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_keyframe_point2;
rclcpp::Publisher<agent_msg::msg::AgentMsg>::SharedPtr pub_agent_frame;
// 全局消息对象
nav_msgs::msg::Path path;
nav_msgs::msg::Path relo_path;

// 全局发布器
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr relo_path_pub;
 
    static CameraPoseVisualization cameraposevisual(0,1,0,1);
    static CameraPoseVisualization keyframebasevisual(0.0,0.0,1.0,1.0);
    static double sum_of_path = 0;
    static Vector3d last_path(0.0, 0.0, 0.0);

std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ = nullptr;

void registerPub(rclcpp::Node::SharedPtr node)
{
    pub_latest_odometry = node->create_publisher<nav_msgs::msg::Odometry>("vins_estimator/imu_propagate", 1000);
    path_pub = node->create_publisher<nav_msgs::msg::Path>("path", 1000);
    relo_path_pub = node->create_publisher<nav_msgs::msg::Path>("relo_path", 1000);
    pub_odometry = node->create_publisher<nav_msgs::msg::Odometry>("vins_estimator/odometry", 1000);
    pub_margin_cloud = node->create_publisher<sensor_msgs::msg::PointCloud2>("vins_estimator/history_cloud", 1000);
    pub_key_poses = node->create_publisher<visualization_msgs::msg::Marker>("vins_estimator/key_poses", 1000);
    pub_camera_pose = node->create_publisher<nav_msgs::msg::Odometry>("vins_estimator/camera_pose", 1000);
    pub_camera_pose_visual = node->create_publisher<visualization_msgs::msg::MarkerArray>("vins_estimator/camera_pose_visual", 1000);
    pub_keyframe_pose = node->create_publisher<nav_msgs::msg::Odometry>("vins_estimator/keyframe_pose", 1000);
    pub_keyframe_point = node->create_publisher<sensor_msgs::msg::PointCloud2>("vins_estimator/keyframe_point", 1000);
    pub_keyframe_point2 = node->create_publisher<sensor_msgs::msg::PointCloud2>("vins_estimator/keyframe_point2", 1000);
    pub_extrinsic = node->create_publisher<nav_msgs::msg::Odometry>("vins_estimator/extrinsic", 1000);
    pub_relo_relative_pose = node->create_publisher<nav_msgs::msg::Odometry>("vins_estimator/relo_relative_pose", 1000);
    pub_agent_frame = node->create_publisher<agent_msg::msg::AgentMsg>("/agent_frame", 1000);
    pub_point_cloud2 = node->create_publisher<sensor_msgs::msg::PointCloud2>("vins_estimator/point_cloud2", rclcpp::QoS(rclcpp::KeepLast(1000)));
    
    cameraposevisual.setScale(1);
    cameraposevisual.setLineWidth(0.05);
    keyframebasevisual.setScale(0.1);
    keyframebasevisual.setLineWidth(0.01);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::msg::Header &header)
{
    Eigen::Quaterniond quadrotor_Q = Q;

    nav_msgs::msg::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = quadrotor_Q.x();
    odometry.pose.pose.orientation.y = quadrotor_Q.y();
    odometry.pose.pose.orientation.z = quadrotor_Q.z();
    odometry.pose.pose.orientation.w = quadrotor_Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry->publish(odometry);
}


void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;

    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("statistics"), "position: " << estimator.Ps[WINDOW_SIZE].transpose());
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("statistics"), "orientation: " << estimator.Vs[WINDOW_SIZE].transpose());

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        RCLCPP_DEBUG_STREAM(rclcpp::get_logger("statistics"), "extrinsic tic: " << estimator.tic[i].transpose());
        RCLCPP_DEBUG_STREAM(rclcpp::get_logger("statistics"), "extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());

        if (ESTIMATE_EXTRINSIC)
        {
            cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
            Eigen::Matrix3d eigen_R = estimator.ric[i];
            Eigen::Vector3d eigen_T = estimator.tic[i];
            cv::Mat cv_R, cv_T;
            cv::eigen2cv(eigen_R, cv_R);
            cv::eigen2cv(eigen_T, cv_T);
            fs << "extrinsicRotation" << cv_R << "extrinsicTranslation" << cv_T;
            fs.release();
        }
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    RCLCPP_DEBUG(rclcpp::get_logger("statistics"), "vo solver costs: %f ms", t);
    RCLCPP_DEBUG(rclcpp::get_logger("statistics"), "average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    RCLCPP_DEBUG(rclcpp::get_logger("statistics"), "sum of path %f", sum_of_path);
    if (ESTIMATE_TD)
        RCLCPP_DEBUG(rclcpp::get_logger("statistics"), "td %f", estimator.td);
}

void pubOdometry(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::msg::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = std::to_string(AGENT_NUM);
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(estimator.Rs[WINDOW_SIZE]);
        odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
        odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
        odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE].x();
        odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE].y();
        odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE].z();
        pub_odometry->publish(odometry);

        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        path_pub->publish(path);

        Vector3d correct_t;
        Vector3d correct_v;
        Quaterniond correct_q;
        correct_t = estimator.drift_correct_r * estimator.Ps[WINDOW_SIZE] + estimator.drift_correct_t;
        correct_q = estimator.drift_correct_r * estimator.Rs[WINDOW_SIZE];
        odometry.pose.pose.position.x = correct_t.x();
        odometry.pose.pose.position.y = correct_t.y();
        odometry.pose.pose.position.z = correct_t.z();
        odometry.pose.pose.orientation.x = correct_q.x();
        odometry.pose.pose.orientation.y = correct_q.y();
        odometry.pose.pose.orientation.z = correct_q.z();
        odometry.pose.pose.orientation.w = correct_q.w();

        pose_stamped.pose = odometry.pose.pose;
        relo_path.header = header;
        relo_path.header.frame_id = "world";
        relo_path.poses.push_back(pose_stamped);
        relo_path_pub->publish(relo_path);

        // write result to file
        std::ofstream foutC(VINS_RESULT_PATH, std::ios::app);
        foutC.setf(std::ios::fixed, std::ios::floatfield);
        foutC.precision(0);
        foutC << header.stamp.sec * 1e9 << ",";
        foutC.precision(5);
        foutC << estimator.Ps[WINDOW_SIZE].x() << ","
              << estimator.Ps[WINDOW_SIZE].y() << ","
              << estimator.Ps[WINDOW_SIZE].z() << ","
              << tmp_Q.w() << ","
              << tmp_Q.x() << ","
              << tmp_Q.y() << ","
              << tmp_Q.z() << ","
              << estimator.Vs[WINDOW_SIZE].x() << ","
              << estimator.Vs[WINDOW_SIZE].y() << ","
              << estimator.Vs[WINDOW_SIZE].z() << "," << std::endl;
        foutC.close();
    }
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::msg::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::msg::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = builtin_interfaces::msg::Duration();

    key_poses.id = 0;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::msg::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses->publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

        nav_msgs::msg::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        pub_camera_pose->publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}



void pubPointCloud2(const Estimator &estimator, const std_msgs::msg::Header &header) {
    // 1) 收集所有符合条件的三维点到一个临时数组
    std::vector<Eigen::Vector3d> world_points;
    world_points.reserve(estimator.f_manager.feature.size());
    for (auto &it_per_id : estimator.f_manager.feature) {
        int used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1) continue;
        int imu_i = it_per_id.start_frame;
        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Eigen::Vector3d w_pts_i =
            estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
            + estimator.Ps[imu_i];
        world_points.push_back(w_pts_i);
    }

    // 2) 构造 PointCloud2 消息
    sensor_msgs::msg::PointCloud2 cloud2;
    cloud2.header       = header;
    cloud2.height       = 1;  // 单行点云
    cloud2.width        = static_cast<uint32_t>(world_points.size());
    cloud2.is_dense     = true;   // 假设没有 NaN
    cloud2.is_bigendian = false;  // 小端模式

    // 3) 定义 x, y, z 三个字段
    cloud2.fields.clear();
    cloud2.fields.reserve(3);
    {
        using PF = sensor_msgs::msg::PointField;
        PF pf;
        pf.datatype = PF::FLOAT32;
        pf.count    = 1;

        // x
        pf.name   = "x";
        pf.offset = 0;
        cloud2.fields.push_back(pf);

        // y
        pf.name   = "y";
        pf.offset = 4;  // 紧跟 x
        cloud2.fields.push_back(pf);

        // z
        pf.name   = "z";
        pf.offset = 8;  // 紧跟 y
        cloud2.fields.push_back(pf);
    }

    // 4) 计算 step 并分配 buffer
    cloud2.point_step = 3 * sizeof(float);                   // 每个点 12 字节
    cloud2.row_step   = cloud2.point_step * cloud2.width;    // 每行
    cloud2.data.resize(cloud2.row_step * cloud2.height);     // 总大小

    // 5) memcpy 拷贝所有点到 data
    uint8_t *ptr = cloud2.data.data();
    for (auto &P : world_points) {
        float x = static_cast<float>(P.x());
        float y = static_cast<float>(P.y());
        float z = static_cast<float>(P.z());
        std::memcpy(ptr + 0, &x, sizeof(float));
        std::memcpy(ptr + 4, &y, sizeof(float));
        std::memcpy(ptr + 8, &z, sizeof(float));
        ptr += cloud2.point_step;
    }

    // 6) 发布到订阅者
    pub_point_cloud2->publish(cloud2);
}



void pubTF(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;

    // 1) world -> body
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header = header;
    tf_msg.header.frame_id    = "world";
    tf_msg.child_frame_id     = "body";

    const auto &body_p = estimator.Ps[WINDOW_SIZE];
    tf_msg.transform.translation.x = body_p.x();
    tf_msg.transform.translation.y = body_p.y();
    tf_msg.transform.translation.z = body_p.z();
    {
      auto q = Quaterniond(estimator.Rs[WINDOW_SIZE]);
      tf_msg.transform.rotation.x = q.x();
      tf_msg.transform.rotation.y = q.y();
      tf_msg.transform.rotation.z = q.z();
      tf_msg.transform.rotation.w = q.w();
    }
    RCLCPP_DEBUG(
        rclcpp::get_logger("vins_visualization"),
        "broadcast %s→%s @ %.9f",
        tf_msg.header.frame_id.c_str(),
        tf_msg.child_frame_id.c_str(),
        tf_msg.header.stamp.sec + tf_msg.header.stamp.nanosec * 1e-9
        );
    tf_broadcaster_->sendTransform(tf_msg);

    // 2) body -> camera
    tf_msg.header.frame_id    = "body";
    tf_msg.child_frame_id     = "camera";

    const auto &cam_t = estimator.tic[0];
    tf_msg.transform.translation.x = cam_t.x();
    tf_msg.transform.translation.y = cam_t.y();
    tf_msg.transform.translation.z = cam_t.z();
    {
      auto qc = Quaterniond(estimator.ric[0]);
      tf_msg.transform.rotation.x = qc.x();
      tf_msg.transform.rotation.y = qc.y();
      tf_msg.transform.rotation.z = qc.z();
      tf_msg.transform.rotation.w = qc.w();
    }
    RCLCPP_DEBUG(
        rclcpp::get_logger("vins_visualization"),
        "broadcast %s→%s @ %.9f",
        tf_msg.header.frame_id.c_str(),
        tf_msg.child_frame_id.c_str(),
        tf_msg.header.stamp.sec + tf_msg.header.stamp.nanosec * 1e-9
        );
    tf_broadcaster_->sendTransform(tf_msg);

    // 3) 如果还要发布 extrinsic odom
    nav_msgs::msg::Odometry odom;
    odom.header = header;
    odom.header.frame_id = "world";
    odom.pose.pose.position.x    = cam_t.x();
    odom.pose.pose.position.y    = cam_t.y();
    odom.pose.pose.position.z    = cam_t.z();
    {
      auto qc = Quaterniond(estimator.ric[0]);
      odom.pose.pose.orientation.x = qc.x();
      odom.pose.pose.orientation.y = qc.y();
      odom.pose.pose.orientation.z = qc.z();
      odom.pose.pose.orientation.w = qc.w();
    }
    pub_extrinsic->publish(odom);
}



void pubKeyframe(const Estimator &estimator)
{
  // 1) 先发里程计位姿
  if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR
      && estimator.marginalization_flag == Estimator::MarginalizationFlag::MARGIN_OLD)
  {
    int idx = WINDOW_SIZE - 2;
    // 发布位姿
    {
      Vector3d P = estimator.Ps[idx];
      Quaterniond Q(estimator.Rs[idx]);

      nav_msgs::msg::Odometry odom;
      odom.header           = estimator.Headers[idx];
      odom.header.frame_id  = "world";
      odom.pose.pose.position.x    = P.x();
      odom.pose.pose.position.y    = P.y();
      odom.pose.pose.position.z    = P.z();
      odom.pose.pose.orientation.x = Q.x();
      odom.pose.pose.orientation.y = Q.y();
      odom.pose.pose.orientation.z = Q.z();
      odom.pose.pose.orientation.w = Q.w();

      pub_keyframe_pose->publish(odom);
    }

    // 2) 收集符合条件的 3D 点
    std::vector<Eigen::Vector3d> world_pts;
    world_pts.reserve(estimator.f_manager.feature.size());
    for (auto &f : estimator.f_manager.feature)
    {
      int sz = f.feature_per_frame.size();
      if (f.start_frame < idx
          && f.start_frame + sz - 1 >= idx
          && f.solve_flag == 1)
      {
        int imu_i = f.start_frame;
        // 相机坐标系下重建的点
        Eigen::Vector3d cam_p = f.feature_per_frame[0].point * f.estimated_depth;
        // 投到世界
        Eigen::Vector3d w = estimator.Rs[imu_i] * (estimator.ric[0] * cam_p + estimator.tic[0])
                          + estimator.Ps[imu_i];
        world_pts.push_back(w);
      }
    }

    // 3) 构造 PointCloud2
    sensor_msgs::msg::PointCloud2 cloud2;
    cloud2.header        = estimator.Headers[idx];
    cloud2.header.frame_id = "world";
    cloud2.height        = 1;
    cloud2.width         = static_cast<uint32_t>(world_pts.size());
    cloud2.is_dense      = true;
    cloud2.is_bigendian  = false;

    // 只要 x,y,z
    cloud2.fields.clear();
    cloud2.fields.reserve(3);
    {
      using PF = sensor_msgs::msg::PointField;
      PF pf;
      pf.datatype = PF::FLOAT32;
      pf.count    = 1;

      pf.name   = "x"; pf.offset =  0; cloud2.fields.push_back(pf);
      pf.name   = "y"; pf.offset =  4; cloud2.fields.push_back(pf);
      pf.name   = "z"; pf.offset =  8; cloud2.fields.push_back(pf);
    }

    cloud2.point_step  = 3 * sizeof(float);  // 12 bytes
    cloud2.row_step    = cloud2.point_step * cloud2.width;
    cloud2.data.resize(cloud2.row_step * cloud2.height);

    // 4) 填充 data
    uint8_t *ptr = cloud2.data.data();
    for (auto &p : world_pts)
    {
      float x = float(p.x()), y = float(p.y()), z = float(p.z());
      std::memcpy(ptr + 0, &x, sizeof(float));
      std::memcpy(ptr + 4, &y, sizeof(float));
      std::memcpy(ptr + 8, &z, sizeof(float));
      ptr += cloud2.point_step;
    }

    // 5) 发布
    pub_keyframe_point2->publish(cloud2); 
  }
}


void pubRelocalization(const Estimator &estimator)
{
    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = rclcpp::Time(estimator.relo_frame_stamp);
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.relo_relative_t.x();
    odometry.pose.pose.position.y = estimator.relo_relative_t.y();
    odometry.pose.pose.position.z = estimator.relo_relative_t.z();
    odometry.pose.pose.orientation.x = estimator.relo_relative_q.x();
    odometry.pose.pose.orientation.y = estimator.relo_relative_q.y();
    odometry.pose.pose.orientation.z = estimator.relo_relative_q.z();
    odometry.pose.pose.orientation.w = estimator.relo_relative_q.w();
    odometry.twist.twist.linear.x = estimator.relo_relative_yaw;
    odometry.twist.twist.linear.y = estimator.relo_frame_index;

    pub_relo_relative_pose->publish(odometry);
}


void preprocessAgentFrame(const Estimator &estimator, agent_msg::msg::AgentMsg &agent_frame_msg)
{
    agent_frame_msg.header = estimator.Headers[WINDOW_SIZE - 2];
    agent_frame_msg.seq = AGENT_NUM;

    int i = WINDOW_SIZE - 2;
    Vector3d P = estimator.Ps[i];
    Quaterniond R = Quaterniond(estimator.Rs[i]);
    agent_frame_msg.position_imu.x = P.x();
    agent_frame_msg.position_imu.y = P.y();
    agent_frame_msg.position_imu.z = P.z();
    agent_frame_msg.orientation_imu.x = R.x();
    agent_frame_msg.orientation_imu.y = R.y();
    agent_frame_msg.orientation_imu.z = R.z();
    agent_frame_msg.orientation_imu.w = R.w();

    agent_frame_msg.tic.x = estimator.tic[0].x();
    agent_frame_msg.tic.y = estimator.tic[0].y();
    agent_frame_msg.tic.z = estimator.tic[0].z();
    Quaterniond tmpQ(estimator.ric[0]);
    agent_frame_msg.ric.x = tmpQ.x();
    agent_frame_msg.ric.y = tmpQ.y();
    agent_frame_msg.ric.z = tmpQ.z();
    agent_frame_msg.ric.w = tmpQ.w();

    vector<Eigen::Vector3d> window_points_3d;
    vector<Eigen::Vector2d> window_points_uv;
    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int frame_size = it_per_id.feature_per_frame.size();
        if(it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                                  + estimator.Ps[imu_i];
            window_points_3d.push_back(w_pts_i);
            int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
            window_points_uv.push_back(it_per_id.feature_per_frame[imu_j].uv);
        }
    }

        // 将 3D 点填入 agent_frame_msg.point_3d
    for (size_t i = 0; i < window_points_3d.size(); ++i) {
        const auto &wp = window_points_3d[i];
        geometry_msgs::msg::Point32 p;
        p.x = static_cast<float>(wp.x());
        p.y = static_cast<float>(wp.y());
        p.z = static_cast<float>(wp.z());
        agent_frame_msg.point_3d.push_back(p);
    }

    // 将像素坐标（uv）填入 agent_frame_msg.point_uv
    for (size_t i = 0; i < window_points_uv.size(); ++i) {
        const auto &uv = window_points_uv[i];
        geometry_msgs::msg::Point32 p;
        p.x = static_cast<float>(uv.x());
        p.y = static_cast<float>(uv.y());
        p.z = 1.0f;  // 或者你希望存储的深度值
        agent_frame_msg.point_uv.push_back(p);
    }

}

void pubAgentFrame(agent_msg::msg::AgentMsg &agent_frame_msg, const cv::Mat &image, camodocal::CameraPtr m_camera)
{
    if (!m_camera) 
    {
        RCLCPP_ERROR(rclcpp::get_logger("statistics"), "visualization: camera model is null!");
        return;
    }
    BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
    const int fast_th = 20; // corner detector response threshold
    vector<cv::KeyPoint> keypoints_uv, window_keypoints_uv;
    vector<cv::KeyPoint> keypoints_2d;
    if(0)
        cv::FAST(image, keypoints_uv, fast_th, true);
    else
    {
        vector<cv::Point2f> tmp_pts;
        cv::goodFeaturesToTrack(image, tmp_pts, 1000, 0.01, 5);
        for(int i = 0; i < (int)tmp_pts.size(); i++)
        {
            cv::KeyPoint key;
            key.pt = tmp_pts[i];
            keypoints_uv.push_back(key);
        }
    }
    std::vector<boost::dynamic_bitset<>> brief_descriptors;
    std::vector<boost::dynamic_bitset<>> window_brief_descriptors;

    // 然后调用 extractor：
    extractor(image, keypoints_uv, brief_descriptors);
    extractor(image, window_keypoints_uv, window_brief_descriptors);
    for (int i = 0; i < (int)keypoints_uv.size(); i++)
    {
        /*if (!m_camera) 
        {
        RCLCPP_ERROR(rclcpp::get_logger("statistics"), "visualization: camera model is null before projecting keypoints");
        return;
        }
        */
        Eigen::Vector3d tmp_p;
        m_camera->liftProjective(Eigen::Vector2d(keypoints_uv[i].pt.x, keypoints_uv[i].pt.y), tmp_p);
        cv::KeyPoint tmp_norm;
        tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
        keypoints_2d.push_back(tmp_norm);
    }

    for(int i = 0; i < (int)agent_frame_msg.point_uv.size(); i++)
    {
        cv::KeyPoint key;
        key.pt = cv::Point2f(agent_frame_msg.point_uv[i].x,agent_frame_msg.point_uv[i].y);
        window_keypoints_uv.push_back(key);
    }
    extractor(image, window_keypoints_uv, window_brief_descriptors);

    for(int i = 0 ; i < (int)keypoints_2d.size(); i++)
    {
        geometry_msgs::msg::Point32 p;
        p.x = keypoints_2d[i].pt.x;
        p.y = keypoints_2d[i].pt.y;
        p.z = 1;
        agent_frame_msg.feature_2d.push_back(p);
    }

    for (int i = 0; i < (int)window_brief_descriptors.size();i++)
    {
        for (int k = 0; k < 4; k++)
        {
            unsigned long long int tmp_int = 0;
            for (int j = 255 - 64 * k; j > 255 - 64 * k - 64; j--)
            {
                tmp_int <<= 1;
                tmp_int += window_brief_descriptors[i][j];
            }
            agent_frame_msg.point_des.push_back(tmp_int);
        }
    }

    for (int i = 0; i < (int)brief_descriptors.size();i++)
    {
        for (int k = 0; k < 4; k++)
        {
            unsigned long long int tmp_int = 0;
            for (int j = 255 - 64 * k; j > 255 - 64 * k - 64; j--)
            {
                tmp_int <<= 1;
                tmp_int += brief_descriptors[i][j];
            }
            agent_frame_msg.feature_des.push_back(tmp_int);
        }
    }
    pub_agent_frame->publish(agent_frame_msg);
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
    // Load the BRIEF pattern from file (compatible with DVision::BRIEF)
    cv::FileStorage fs(pattern_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Could not open pattern file: " + pattern_file);
    }
    std::vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["y1"] >> y1;
    fs["x2"] >> x2;
    fs["y2"] >> y2;

    m_brief.importPairs(x1, y1, x2, y2);
}

void BriefExtractor::operator()(
    const cv::Mat &im,
    std::vector<cv::KeyPoint> &keys,
    std::vector<boost::dynamic_bitset<>> &descriptors
) const
{
    m_brief.compute(im, keys, descriptors);
}








