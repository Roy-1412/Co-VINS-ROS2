#pragma once

#include <rclcpp/rclcpp.hpp>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"



class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Constructor
   * @param pts_i 3D point in the first camera frame
   * @param pts_j 3D point in the second camera frame
   */
  ProjectionFactor(const Eigen::Vector3d &pts_i, const Eigen::Vector3d &pts_j);

  /**
   * @brief Ceres evaluate interface
   */
  bool Evaluate(double const *const *parameters,
                double *residuals,
                double **jacobians) const override;

  /**
   * @brief Optional debug helper
   */
  void check(double **parameters) const;

  // Static information matrix for measurement noise (initialized in .cpp)
  static Eigen::Matrix2d sqrt_info;
  static double sum_t;

  // Observations
  Eigen::Vector3d pts_i;
  Eigen::Vector3d pts_j;
  Eigen::Matrix<double, 2, 3> tangent_base_;
};


