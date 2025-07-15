// parameters.cpp (ROS 2 version)

#include "parameters.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/opencv.hpp>

// 全局变量定义
std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

void readParameters(rclcpp::Node * node)
{
  // 1. 声明并读取参数 "config_file"
  node->declare_parameter<std::string>("config_file", "");
  std::string config_file;
  if (!node->get_parameter("config_file", config_file) || config_file.empty()) {
    RCLCPP_ERROR(node->get_logger(), "Parameter 'config_file' not provided or empty.");
    rclcpp::shutdown();
    return;
  }
  RCLCPP_INFO(node->get_logger(), "Loaded config_file: %s", config_file.c_str());

  // 2. 拼接配置文件的完整路径：
  //    如果用户传入的是绝对路径，就直接用；否则从 share/feature_tracker/config/ 下找
  std::string full_config_path;
  if (config_file.front() == '/') {
    full_config_path = config_file;
  } else {
    auto share_dir = ament_index_cpp::get_package_share_directory("feature_tracker");
    full_config_path = share_dir + "/config/" + config_file;
  }

  // 3. 尝试打开 YAML 设置
  cv::FileStorage fsSettings(full_config_path, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    RCLCPP_ERROR(node->get_logger(), "Failed to open settings file: %s", full_config_path.c_str());
    rclcpp::shutdown();
    return;
  }

  // 4. 从 YAML 中读取话题名称
  fsSettings["image_topic"] >> IMAGE_TOPIC;
  fsSettings["imu_topic"] >> IMU_TOPIC;
  RCLCPP_INFO(node->get_logger(), "YAML image_topic: %s", IMAGE_TOPIC.c_str());
  RCLCPP_INFO(node->get_logger(), "YAML imu_topic: %s", IMU_TOPIC.c_str());

  // 5. 从 YAML 中读取数值参数
  MAX_CNT     = static_cast<int>(fsSettings["max_cnt"]);
  MIN_DIST    = static_cast<int>(fsSettings["min_dist"]);
  ROW         = static_cast<int>(fsSettings["image_height"]);
  COL         = static_cast<int>(fsSettings["image_width"]);
  FREQ        = static_cast<int>(fsSettings["freq"]);
  F_THRESHOLD = static_cast<double>(fsSettings["F_threshold"]);
  SHOW_TRACK  = static_cast<int>(fsSettings["show_track"]);
  EQUALIZE    = static_cast<int>(fsSettings["equalize"]);
  FISHEYE     = static_cast<int>(fsSettings["fisheye"]);

  RCLCPP_INFO(node->get_logger(), 
              "max_cnt: %d, min_dist: %d, image size: %dx%d", 
              MAX_CNT, MIN_DIST, COL, ROW);
  RCLCPP_INFO(node->get_logger(), 
              "freq: %d, F_threshold: %.3f, show_track: %d, equalize: %d, fisheye: %d",
              FREQ, F_THRESHOLD, SHOW_TRACK, EQUALIZE, FISHEYE);

  // 6. 如果启用鱼眼，从 share/feature_tracker/config 下读取 mask 文件
  if (FISHEYE == 1) {
    std::string share_dir = ament_index_cpp::get_package_share_directory("feature_tracker");
    std::string mask_filename;
    fsSettings["fisheye_mask_filename"] >> mask_filename;
    FISHEYE_MASK = share_dir + "/config/" + mask_filename;
    RCLCPP_INFO(node->get_logger(), "Fisheye mask: %s", FISHEYE_MASK.c_str());
  }

  // 7. 相机标定文件列表，这里只放一个 config_file 路径
  CAM_NAMES.clear();
  CAM_NAMES.push_back(full_config_path);
  RCLCPP_INFO(node->get_logger(), "Camera YAML: %s", CAM_NAMES[0].c_str());

  // 8. 为其余参数设默认值
  WINDOW_SIZE    = 20;
  STEREO_TRACK   = false;
  FOCAL_LENGTH   = 460;
  PUB_THIS_FRAME = false;

  if (FREQ == 0) {
    FREQ = 100;
    RCLCPP_WARN(node->get_logger(), "freq was 0 in YAML; overriding to 100");
  }

  fsSettings.release();
}
