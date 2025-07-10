#pragma once
#include "parameters.h"
#include <cassert>
#include <ceres/ceres.h>
#include <Eigen/Dense>

/**
 * @brief 带时间延迟的投影因子
 *
 * 参数块顺序：
 *   - pose_i (7)
 *   - pose_j (7)
 *   - extrinsic (7)
 *   - feature depth (1)
 *   - time delay td (1)
 */
class ProjectionTdFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // 构造函数
  ProjectionTdFactor(
    const Eigen::Vector3d& pts_i,
    const Eigen::Vector3d& pts_j,
    const Eigen::Vector2d& velocity_i,
    const Eigen::Vector2d& velocity_j,
    double td_i,
    double td_j,
    double row_i,
    double row_j);

  // Ceres 接口
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override;

  // 数值雅可比检查
  void check(double** parameters) const;

  // ——— 以下全都是 public 成员 ———

  // 观测量
  Eigen::Vector3d pts_i;
  Eigen::Vector3d pts_j;
  Eigen::Vector3d velocity_i;  // 已扩到 3D
  Eigen::Vector3d velocity_j;
  double td_i;
  double td_j;
  double row_i;
  double row_j;
  Eigen::Matrix<double, 2, 3> tangent_base;  // 用于 UNIT_SPHERE_ERROR

  // 静态成员：残差加权信息 & 调试累计时间
  static Eigen::Matrix2d sqrt_info;
  static double sum_t;
};
