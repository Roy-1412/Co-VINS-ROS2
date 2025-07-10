// parameters.h

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <rclcpp/rclcpp.hpp>

/**
 * 全局配置参数声明
 */
extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::vector<std::string> CAM_NAMES;
extern std::string FISHEYE_MASK;
extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int STEREO_TRACK;
extern int EQUALIZE;
extern int ROW;
extern int COL;
extern int FOCAL_LENGTH;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;
const int NUM_OF_CAM = 1;
/**
 * @brief 读取配置文件并填充上述全局变量
 *
 * 参数说明：
 *  - config_file：YAML 配置文件名或绝对路径
 * 如果指定的是相对路径，则从安装后的 share/feature_tracker/config/ 下加载
 */
void readParameters(rclcpp::Node * node);
