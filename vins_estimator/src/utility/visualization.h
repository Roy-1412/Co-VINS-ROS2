#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <boost/dynamic_bitset.hpp>
#include "../estimator.h"
#include "../parameters.h"
#include <agent_msg/msg/agent_msg.hpp>
#include "../ThirdParty/DUtils/DUtils.h"
#include "../ThirdParty/DVision/DVision.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

// -----------------------------------------------------------------------------
// BriefExtractor: 只负责载入 BRIEF 模板并计算描述子
// -----------------------------------------------------------------------------
class BriefExtractor
{
public:
  // 根据 pattern_file 构造
  explicit BriefExtractor(const std::string &pattern_file);

  // 注意签名必须和实现完全一致
  virtual void operator()(
    const cv::Mat &im,
    std::vector<cv::KeyPoint> &keys,
    std::vector<boost::dynamic_bitset<>> &descriptors
  ) const;

private:
  DVision::BRIEF m_brief;
};

// -----------------------------------------------------------------------------
// Visualizer: 负责 TF 广播、各种可视化消息发布
// -----------------------------------------------------------------------------
class Visualizer
{
public:
  explicit Visualizer(rclcpp::Node::SharedPtr node)
    : node_(node)
  {
    
  }

  // 把自由函数 pubTF 拆到这里
  void pubTF(
    const Estimator &estimator,
    const std_msgs::msg::Header &header
  );

  // 你还可以把 pubKeyframe、pubPointCloud2 等都改成成员函数
  // void pubPointCloud2(...);

private:
  rclcpp::Node::SharedPtr        node_;
  
};

// -----------------------------------------------------------------------------
// 全局 extern 发布器指针
// -----------------------------------------------------------------------------
extern rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_point_cloud2;
extern rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_keyframe_point2;
extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr        pub_odometry;
extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr        pub_latest_odometry;
extern rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr            pub_path, pub_relo_path;
extern rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr  pub_cloud, pub_margin_cloud;
extern rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr        pub_camera_pose;
extern rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr        pub_keyframe_pose;
extern rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr  pub_keyframe_point;
extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr        pub_extrinsic;
extern rclcpp::Publisher<agent_msg::msg::AgentMsg>::SharedPtr       pub_agent_frame;
extern std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
// path 缓存
extern rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
extern rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr relo_path_pub;

// 注册所有 publisher
void registerPub(rclcpp::Node::SharedPtr node);

// 各种发布函数
void pubLatestOdometry(const Eigen::Vector3d &P,
                       const Eigen::Quaterniond &Q,
                       const Eigen::Vector3d &V,
                       const std_msgs::msg::Header &header);

void printStatistics(const Estimator &estimator, double t);

void pubOdometry(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubKeyPoses(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubCameraPose(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubPointCloud2(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubTF(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubKeyframe(const Estimator &estimator);

void pubRelocalization(const Estimator &estimator);

void preprocessAgentFrame(const Estimator &estimator,
                          agent_msg::msg::AgentMsg &agent_frame_msg);

void pubAgentFrame(agent_msg::msg::AgentMsg &agent_frame_msg,
                   const cv::Mat &image,
                   camodocal::CameraPtr m_camera);


