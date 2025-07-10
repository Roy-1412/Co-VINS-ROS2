#pragma once

#include <map>
#include <vector>
#include <utility>

#include <Eigen/Dense>

#include "factor/imu_factor.h"
#include "utility/utility.h"
#include "feature_manager.h"

namespace vins_estimator {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::VectorXd;

/**
 * @brief 帧数据结构，包含图像特征观测以及 IMU 预积分等信息
 */
class ImageFrame
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImageFrame() = default;

  /**
   * @param _points 特征观测：
   *        map<feature_id, vector<pair<cam_id, [x,y,z, u,v, vu,vv]>>>
   * @param _t      时间戳（秒）
   */
  ImageFrame(
    const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& _points,
    double _t)
    : points(_points),
      t(_t),
      R(Matrix3d::Identity()),
      T(Vector3d::Zero()),
      pre_integration(nullptr),
      is_key_frame(false)
  {}

  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points;  ///< 特征观测
  double t;                   ///< 帧时间戳
  Matrix3d R;                 ///< 视觉位姿：旋转
  Vector3d T;                 ///< 视觉位姿：平移
  IntegrationBase *pre_integration;  ///< 从上一帧到本帧的 IMU 预积分
  bool is_key_frame;          ///< 标记关键帧
};

/**
 * @brief 视觉–IMU 初始对齐
 *
 * 在多帧图像和对应 IMU 偏置的基础上，估计尺度、速度、重力向量等初始量
 *
 * @param[in,out] all_image_frame  所有已收到的 ImageFrame，key 为时间戳
 * @param[in]     Bgs              各帧陀螺仪偏置数组（长度与帧数相同）
 * @param[out]    g                输出的重力向量
 * @param[out]    x                输出优化变量（速度、尺度等）
 * @return                         对齐是否成功
 */
bool VisualIMUAlignment(
  std::map<double, ImageFrame> &all_image_frame,
  Vector3d *                    Bgs,
  Vector3d &                    g,
  VectorXd &                    x);

}  // namespace vins_estimator
