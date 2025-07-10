#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <rclcpp/rclcpp.hpp>  // ROS 2 logging

using namespace std;
using namespace Eigen;

class MotionEstimator
{
public:
  /**
   * @brief Estimate the relative rotation and translation between two views given correspondences.
   * @param corres  A vector of (bearing1, bearing2) pairs in normalized image coordinates.
   * @param R       Output rotation matrix mapping frame1 to frame2.
   * @param T       Output translation vector (up to scale) from frame1 to frame2.
   * @return        True if estimation succeeded, false otherwise.
   */
  bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres,
                       Matrix3d &R,
                       Vector3d &T);

private:
  /**
   * @brief Triangulate points and count how many lie in front of both cameras.
   * @param l  Points in the first view.
   * @param r  Points in the second view.
   * @param R  Rotation from view1 to view2.
   * @param t  Translation from view1 to view2.
   * @return   The fraction of points in front of both cameras.
   */
  double testTriangulation(const vector<cv::Point2f> &l,
                           const vector<cv::Point2f> &r,
                           const cv::Mat_<double> &R,
                           const cv::Mat_<double> &t);

  /**
   * @brief Decompose an essential matrix into possible [R|t] pairs.
   * @param E   The 3×3 essential matrix.
   * @param R1  First possible rotation.
   * @param R2  Second possible rotation.
   * @param t1  First possible translation (up to scale).
   * @param t2  Second possible translation.
   */
  void decomposeE(const cv::Mat &E,
                  cv::Mat_<double> &R1,
                  cv::Mat_<double> &R2,
                  cv::Mat_<double> &t1,
                  cv::Mat_<double> &t2);

                  
};
