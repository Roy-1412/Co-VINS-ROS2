#pragma once

#include <vector>
#include "parameters.h"

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <rclcpp/rclcpp.hpp>


namespace vins_estimator {
/**
 * @brief Calibrate the extrinsic rotation between IMU and camera when the extrinsic
 *        parameters are initially unknown.
 */
class InitialEXRotation
{
public:
  InitialEXRotation();

  /**
   * @brief Estimate the rotation from IMU frame to camera frame.
   * @param corres        Vector of corresponding bearing vectors in camera frame.
   * @param delta_q_imu   Measured relative rotation from IMU preintegration.
   * @param[out] calib_ric_result  Estimated camera-to-IMU rotation matrix.
   * @return True if calibration succeeded.
   */
  bool CalibrationExRotation(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& corres,
    const Eigen::Quaterniond&                                      delta_q_imu,
    Eigen::Matrix3d&                                              calib_ric_result);

private:
  /// Solve for relative rotation from correspondences.
  Eigen::Matrix3d solveRelativeR(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& corres);

  /// Triangulation quality check (used internally).
  double testTriangulation(
    const std::vector<cv::Point2f>& l,
    const std::vector<cv::Point2f>& r,
    const cv::Mat_<double>&          R,
    const cv::Mat_<double>&          t);

  /// Decompose essential matrix into two rotations and two translations.
  void decomposeE(
    const cv::Mat&         E,
    cv::Mat_<double>&      R1,
    cv::Mat_<double>&      R2,
    cv::Mat_<double>&      t1,
    cv::Mat_<double>&      t2);

  int frame_count_;
  std::vector<Eigen::Matrix3d>       Rc;    ///< camera-frame rotations
  std::vector<Eigen::Matrix3d>       Rimu;  ///< imu-frame rotations
  std::vector<Eigen::Matrix3d>       Rc_g;  ///< guessed camera rotations
  Eigen::Matrix3d                    ric;   ///< current estimate of camera-to-imu rotation

  rclcpp::Logger                     logger_{rclcpp::get_logger("InitialEXRotation")};
};
}

