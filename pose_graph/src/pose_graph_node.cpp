#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>

#include <rclcpp/rclcpp.hpp>
#include <agent_msg/msg/agent_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph.h"
#include "parameters.h"
#include "ThirdParty/DVision/DVision.h"

using namespace std;
int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
class PoseGraphNode : public rclcpp::Node {
public:
  PoseGraphNode()
    : Node("pose_graph"),
      tf_buffer_(this->get_clock()),
      tf_listener_(std::make_shared<tf2_ros::TransformListener>(tf_buffer_)),
      posegraph_(this)
  {
    // 参数声明与读取
    this->get_parameter("visualization_shift_x", VISUALIZATION_SHIFT_X);
    this->get_parameter("visualization_shift_y", VISUALIZATION_SHIFT_Y);
    this->get_parameter("skip_dis", SKIP_DIS);
    this->get_parameter("mesh_resource", mesh_resource);
    this->get_parameter("pose_graph_save_path", POSE_GRAPH_SAVE_PATH);
    this->get_parameter("pose_graph_result_path", VINS_RESULT_PATH);

    // 加载词典
    auto pkg_share = ament_index_cpp::get_package_share_directory("pose_graph");
    string vocabulary_file = pkg_share + "/../support_files/brief_k10L6.bin";
    RCLCPP_INFO(this->get_logger(), "vocabulary_file: %s", vocabulary_file.c_str());
    posegraph_.loadVocabulary(vocabulary_file);

    BRIEF_PATTERN_FILE = pkg_share + "/../support_files/brief_pattern.yml";
    RCLCPP_INFO(this->get_logger(), "BRIEF_PATTERN_FILE: %s", BRIEF_PATTERN_FILE.c_str());

    // 准备输出文件
    string pose_graph_path = VINS_RESULT_PATH + "/pose_graph_path.csv";
    ofstream loop_path_file_tmp(pose_graph_path);
    loop_path_file_tmp.close();

    // 话题订阅
    agent_sub_ = this->create_subscription<agent_msg::msg::AgentMsg>(
      "/agent_frame", 2000,
      std::bind(&PoseGraphNode::agent_callback, this, std::placeholders::_1)
    );
    odom_sub1_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/vins_1/vins_estimator/odometry", 100,
      std::bind(&PoseGraphNode::odom_callback, this, std::placeholders::_1)
    );
    odom_sub2_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/vins_2/vins_estimator/odometry", 100,
      std::bind(&PoseGraphNode::odom_callback, this, std::placeholders::_1)
    );
    odom_sub3_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/vins_3/vins_estimator/odometry", 100,
      std::bind(&PoseGraphNode::odom_callback, this, std::placeholders::_1)
    );
    odom_sub4_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/vins_4/vins_estimator/odometry", 100,
      std::bind(&PoseGraphNode::odom_callback, this, std::placeholders::_1)
    );

    // 发布器
    mesh_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("robot", 100);

    // 给 PoseGraph 注册发布器
    posegraph_.registerPublishers(this);

    // 启动后台线程
    agent_thread_   = std::thread(&PoseGraphNode::agent_process, this);
    command_thread_ = std::thread(&PoseGraphNode::command, this);
  }

  ~PoseGraphNode() {
    if (agent_thread_.joinable())   agent_thread_.join();
    if (command_thread_.joinable()) command_thread_.join();
  }

private:
  void agent_callback(const agent_msg::msg::AgentMsg::SharedPtr msg) {
    if (start_flag) {
      std::lock_guard<std::mutex> lock(m_agent_msg_buf);
      agent_msg_buf.push(msg);
    }
  }

  void agent_process() {
    using namespace std::chrono_literals;
    while (rclcpp::ok()) {
      agent_msg::msg::AgentMsg::SharedPtr msg;
      {
        std::lock_guard<std::mutex> lock(m_agent_msg_buf);
        if (!agent_msg_buf.empty()) {
          msg = agent_msg_buf.front();
          agent_msg_buf.pop();
        }
      }
      if (msg) {
        TicToc t_addframe;
        // 提取关键帧信息
        double time_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        int sequence     = msg->seq;
        Eigen::Vector3d T(msg->position_imu.x,
                          msg->position_imu.y,
                          msg->position_imu.z);
        Eigen::Matrix3d R = Eigen::Quaterniond(msg->orientation_imu.w,
                                               msg->orientation_imu.x,
                                               msg->orientation_imu.y,
                                               msg->orientation_imu.z)
                                  .toRotationMatrix();
        Eigen::Vector3d tic(msg->tic.x,
                            msg->tic.y,
                            msg->tic.z);
        Eigen::Matrix3d ric = Eigen::Quaterniond(msg->ric.w,
                                                 msg->ric.x,
                                                 msg->ric.y,
                                                 msg->ric.z)
                                  .toRotationMatrix();
        std::vector<cv::Point3f> point_3d;
        point_3d.reserve(msg->point_3d.size());
        for (auto &p : msg->point_3d)
          point_3d.emplace_back(p.x, p.y, p.z);
        std::vector<cv::Point2f> feature_2d;
        feature_2d.reserve(msg->feature_2d.size());
        for (auto &p : msg->feature_2d)
          feature_2d.emplace_back(p.x, p.y);
        std::vector<BRIEF::bitset> point_descriptors;
        point_descriptors.reserve(msg->point_des.size());
        for (auto &d : msg->point_des)
          point_descriptors.emplace_back(/*...*/);
        std::vector<BRIEF::bitset> feature_descriptors;
        feature_descriptors.reserve(msg->feature_des.size());
        for (auto &d : msg->feature_des)
          feature_descriptors.emplace_back(/*...*/);

        KeyFrame *kf = new KeyFrame(
          sequence, time_stamp,
          T, R, tic, ric,
          point_3d, feature_2d,
          point_descriptors, feature_descriptors
        );
        {
          std::lock_guard<std::mutex> lock(m_process);
          posegraph_.addAgentFrame(kf);
        }
      }
      std::this_thread::sleep_for(5ms);
    }
  }

  void command() {
    using namespace std::chrono_literals;
    while (rclcpp::ok()) {
      char c = std::getchar();
      if (c == 's') {
        std::lock_guard<std::mutex> lock(m_process);
        posegraph_.savePoseGraph();
        RCLCPP_INFO(this->get_logger(), "Pose graph saved");
      } else if (c == 'l') {
        std::lock_guard<std::mutex> lock(m_process);
        posegraph_.loadPoseGraph();
        RCLCPP_INFO(this->get_logger(), "Pose graph loaded");
      } else if (c == 'b') {
        start_flag = true;
        RCLCPP_INFO(this->get_logger(), "Begin receiving agent messages");
      }
      std::this_thread::sleep_for(5ms);
    }
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    int seq = std::stoi(msg->child_frame_id);
    // 提取姿态和平移
    Eigen::Quaterniond q(msg->pose.pose.orientation.w,
                         msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y,
                         msg->pose.pose.orientation.z);
    Eigen::Vector3d t(msg->pose.pose.position.x,
                      msg->pose.pose.position.y,
                      msg->pose.pose.position.z);
    try {
      auto tf_stamped = tf_buffer_.lookupTransform(
        "/global",
        "/drone_" + std::to_string(seq),
        tf2::TimePointZero
      );
      // 直接从消息里读数值，不再用 tf2::fromMsg
      auto &gtrans = tf_stamped.transform.translation;
      auto &grot   = tf_stamped.transform.rotation;
      Eigen::Vector3d w_T_local(gtrans.x, gtrans.y, gtrans.z);
      Eigen::Quaterniond w_Q_local(grot.w, grot.x, grot.y, grot.z);

      q = w_Q_local * q;
      t = w_Q_local * t + w_T_local;
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Transform unavailable: %s", ex.what());
    }

    // 发布可视化网格
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "/world";
    marker.header.stamp    = this->now();
    marker.ns              = "mesh";
    marker.id              = seq;
    marker.type            = visualization_msgs::msg::Marker::MESH_RESOURCE;
    marker.action          = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = t.x();
    marker.pose.position.y = t.y();
    marker.pose.position.z = t.z();
    marker.pose.orientation.w = 1.0;
    marker.scale.x = marker.scale.y = marker.scale.z = 1.0;
    marker.color.a  = 1.0;
    marker.mesh_resource = mesh_resource;
    mesh_pub_->publish(marker);
  }

  // 成员变量
  rclcpp::Subscription<agent_msg::msg::AgentMsg>::SharedPtr  agent_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr  odom_sub1_, odom_sub2_, odom_sub3_, odom_sub4_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mesh_pub_;
  tf2_ros::Buffer tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::mutex  m_agent_msg_buf;
  std::queue<agent_msg::msg::AgentMsg::SharedPtr> agent_msg_buf;
  std::mutex  m_process;
  std::thread agent_thread_, command_thread_;

  PoseGraph  posegraph_;

  
  double SKIP_DIS;
  string mesh_resource;
  string POSE_GRAPH_SAVE_PATH;
  string VINS_RESULT_PATH;
  string BRIEF_PATTERN_FILE;

  bool start_flag = true;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PoseGraphNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
