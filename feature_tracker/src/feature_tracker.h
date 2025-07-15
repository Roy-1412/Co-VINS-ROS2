// feature_tracker.h

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"


#include "parameters.h"
#include "tic_toc.h"
using namespace camodocal;

/**
 * @brief 检查一个点是否在图像有效边界内
 */
static inline bool inBorder(const cv::Point2f &pt);


/**
 * @brief 使用 status 掩码收缩 vector<cv::Point2f>
 */
static inline void reduceVector(std::vector<cv::Point2f> &v, const std::vector<uchar> &status);

/**
 * @brief 使用 status 掩码收缩 vector<int>
 */
static inline void reduceVector(std::vector<int> &v, const std::vector<uchar> &status);

/**
 * @brief FeatureTracker 类，用于在 ROS 2 中提取并跟踪特征
 */
class FeatureTracker
{
public:
    FeatureTracker();

    /**
     * @brief 处理新来的图像帧
     * @param img       输入（灰度）图像
     * @param cur_time  时间戳（秒）
     */
    void readImage(const cv::Mat &img, double cur_time);

    /**
     * @brief 构建/更新 mask，用于挑选新特征，优先保留跟踪时间长的点
     */
    void setMask();

    /**
     * @brief 将新检测到的点添加到跟踪列表
     */
    void addPoints();

    /**
     * @brief 若索引 i 处的特征未分配 ID，就分配一个唯一 ID
     * @param i  特征索引
     * @return   如果已分配或分配成功，返回 true
     */
    bool updateID(unsigned int i);

    /**
     * @brief 从 YAML 文件加载相机内参（Camodocal 功能已注释）
     * @param calib_file  相机标定文件路径
     */
    void readIntrinsicParameter(const std::string &calib_file);

    /**
     * @brief 可视化去畸过程，将像素映射到无畸变画布
     * @param name  窗口名
     */
    //void showUndistortion(const string &name);
    void showUndistortion(const std::string &name);
    /**
     * @brief 使用基础矩阵 RANSAC 拒绝外点
     */
    void rejectWithF();

    /**
     * @brief 去畸并计算特征速度
     */
    void undistortedPoints();

    // ---------------------------------------
    // 以下为公有成员变量（由 ROS 1 迁移）
    // ---------------------------------------

    cv::Mat mask;                        ///< 当前用于特征选择的 mask
    cv::Mat fisheye_mask;                ///< 鱼眼 mask（如果 FISHEYE 为真）
    cv::Mat prev_img, cur_img, forw_img; ///< 上一帧、当前帧、下一帧图像

    std::vector<cv::Point2f> n_pts;         ///< 新检测到的点
    std::vector<cv::Point2f> prev_pts;      ///< 上帧跟踪到的点
    std::vector<cv::Point2f> cur_pts;       ///< 当前跟踪到的点
    std::vector<cv::Point2f> forw_pts;      ///< 向前跟踪到的点
    std::vector<cv::Point2f> prev_un_pts;   ///< 上帧去畸后的点
    std::vector<cv::Point2f> cur_un_pts;    ///< 当前帧去畸后的点
    std::vector<cv::Point2f> pts_velocity;  ///< 特征点速度

    std::vector<int> ids;              ///< 每个特征的唯一 ID
    std::vector<int> track_cnt;        ///< 每个特征被跟踪的帧数

    std::map<int, cv::Point2f> cur_un_pts_map;   ///< ID → 当前帧去畸点
    std::map<int, cv::Point2f> prev_un_pts_map;  ///< ID → 上一帧去畸点

    camodocal::CameraPtr m_camera;  ///< Camodocal 相机模型（已注释）

    double cur_time;  ///< 当前帧时间戳
    double prev_time; ///< 上一帧时间戳

    static int n_id; ///< 静态计数器，用于给新特征分配唯一 ID
};

/**
 * @brief 全局标志，用于控制是否发布当前帧
 */
extern bool PUB_THIS_FRAME;
