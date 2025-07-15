#include "initial_ex_rotation.h"

#include <rclcpp/rclcpp.hpp>  // ROS 2 logging
#include <cmath>
#include <opencv2/calib3d.hpp> // for findFundamentalMat, triangulatePoints

using std::vector;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::JacobiSVD;
using Eigen::MatrixXd;

namespace vins_estimator {

InitialEXRotation::InitialEXRotation()
  : frame_count_(0),
    ric(Matrix3d::Identity())
{
  Rc.emplace_back(Matrix3d::Identity());
  Rc_g.emplace_back(Matrix3d::Identity());
  Rimu.emplace_back(Matrix3d::Identity());
}

bool InitialEXRotation::CalibrationExRotation(
    const vector<std::pair<Vector3d, Vector3d>> &corres,
    const Quaterniond &delta_q_imu,
    Matrix3d &calib_ric_result)
{
  frame_count_++;
  Rc.push_back(solveRelativeR(corres));
  Rimu.push_back(delta_q_imu.toRotationMatrix());
  Rc_g.push_back(ric.inverse() * delta_q_imu.toRotationMatrix() * ric);

  // build stacked linear system A * x = 0
  MatrixXd A(frame_count_ * 4, 4);
  A.setZero();

  for (int i = 1; i <= frame_count_; ++i) {
    Quaterniond r1(Rc[i]);
    Quaterniond r2(Rc_g[i]);

    double angular_distance =
      r1.angularDistance(r2) * 180.0 / M_PI;
    RCLCPP_DEBUG(
      rclcpp::get_logger("InitialEXRotation"),
      "frame %d angular distance: %f", i, angular_distance);

    double huber = (angular_distance > 5.0)
      ? (5.0 / angular_distance) : 1.0;

    // build left quaternion matrix L
    double w = r1.w();
    Vector3d q = r1.vec();
    Matrix4d L;
    L.block<3,3>(0,0) =  w * Matrix3d::Identity() + Utility::skewSymmetric(q);
    L.block<3,1>(0,3) = q;
    L.block<1,3>(3,0) = -q.transpose();
    L(3,3) = w;

    // build right quaternion matrix R
    Quaterniond r_imu(delta_q_imu);
    w = r_imu.w();
    q = r_imu.vec();
    Matrix4d R;
    R.block<3,3>(0,0) =  w * Matrix3d::Identity() - Utility::skewSymmetric(q);
    R.block<3,1>(0,3) = q;
    R.block<1,3>(3,0) = -q.transpose();
    R(3,3) = w;

    A.block<4,4>((i-1)*4, 0) = huber * (L - R);
  }

  // solve by SVD (Eigen enums in global namespace)
  JacobiSVD<MatrixXd> svd(A,
      Eigen::ComputeFullU |
      Eigen::ComputeFullV);
  Vector3d singular_vals = svd.singularValues().tail<3>();
  Eigen::Vector4d qvec = svd.matrixV().col(3);
  // Note Eigen quaternion takes (w,x,y,z)
  Quaterniond est_q(qvec[3], qvec[0], qvec[1], qvec[2]);
  ric = est_q.toRotationMatrix().inverse();

  if (frame_count_ >= WINDOW_SIZE && singular_vals(1) > 0.25) {
    calib_ric_result = ric;
    return true;
  } else {
    return false;
  }
}

Matrix3d InitialEXRotation::solveRelativeR(
    const vector<std::pair<Vector3d, Vector3d>> &corres)
{
  if (corres.size() < 9) {
    return Matrix3d::Identity();
  }

  vector<cv::Point2f> pts1, pts2;
  pts1.reserve(corres.size());
  pts2.reserve(corres.size());
  for (auto &c : corres) {
    pts1.emplace_back(c.first.x(),  c.first.y());
    pts2.emplace_back(c.second.x(), c.second.y());
  }

  // find essential via fundamental (assuming normalized coords)
  cv::Mat E = cv::findFundamentalMat(pts1, pts2);
  cv::Mat_<double> R1, R2, t1, t2;
  decomposeE(E, R1, R2, t1, t2);

  // ensure correct orientation
  if (cv::determinant(R1) + 1.0 < 1e-9) {
    E = -E;
    decomposeE(E, R1, R2, t1, t2);
  }

  // pick best by triangulation
  double score1 = std::max(testTriangulation(pts1, pts2, R1, t1),
                           testTriangulation(pts1, pts2, R1, t2));
  double score2 = std::max(testTriangulation(pts1, pts2, R2, t1),
                           testTriangulation(pts1, pts2, R2, t2));
  cv::Mat_<double> bestR = (score1 > score2) ? R1 : R2;

  // convert to Eigen
  Matrix3d Re;
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      Re(c,r) = bestR(r,c);

  return Re;
}

double InitialEXRotation::testTriangulation(
    const vector<cv::Point2f> &l,
    const vector<cv::Point2f> &r,
    const cv::Mat_<double> &R,
    const cv::Mat_<double> &t)
{
  cv::Mat pts4d;
  cv::Matx34f P0 = cv::Matx34f::eye();
  cv::Matx34f P1(R(0,0), R(0,1), R(0,2), t(0),
                 R(1,0), R(1,1), R(1,2), t(1),
                 R(2,0), R(2,1), R(2,2), t(2));

  cv::triangulatePoints(P0, P1, l, r, pts4d);
  int front_count = 0;
  for (int i = 0; i < pts4d.cols; i++) {
    float w = pts4d.at<float>(3,i);
    if (w == 0.0f) continue;
    cv::Vec3d p0(
      (P0(2,0)*pts4d.at<float>(0,i) + P0(2,1)*pts4d.at<float>(1,i) + P0(2,2)*pts4d.at<float>(2,i) + P0(2,3)*w) / w
    );
    cv::Vec3d p1(
      (P1(2,0)*pts4d.at<float>(0,i) + P1(2,1)*pts4d.at<float>(1,i) + P1(2,2)*pts4d.at<float>(2,i) + P1(2,3)*w) / w
    );
    if (p0[2] > 0 && p1[2] > 0) {
      ++front_count;
    }
  }

  double ratio = static_cast<double>(front_count) / pts4d.cols;
  RCLCPP_DEBUG(
    rclcpp::get_logger("InitialEXRotation"),
    "triangulation front ratio: %f", ratio);
  return ratio;
}

void InitialEXRotation::decomposeE(
    const cv::Mat &E,
    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
    cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
  cv::SVD svd(E, cv::SVD::MODIFY_A);
  cv::Matx33d W(0,-1,0, 1,0,0, 0,0,1);
  cv::Matx33d Wt(0,1,0, -1,0,0, 0,0,1);

  R1 = svd.u * cv::Mat(W)  * svd.vt;
  R2 = svd.u * cv::Mat(Wt) * svd.vt;
  t1 = svd.u.col(2);
  t2 = -svd.u.col(2);
}

}  // namespace vins_estimator
