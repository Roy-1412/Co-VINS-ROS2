#include "initial/initial_alignment.h"
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>

using namespace Eigen;
using std::map;
using std::next;

namespace vins_estimator {

static const rclcpp::Logger LOGGER = rclcpp::get_logger("initial_alignment");

void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    RCLCPP_INFO(LOGGER, ">>> Before solveGyroscopeBias: all_image_frame has %zu entries", all_image_frame.size());
    for (auto &kv : all_image_frame) {
  double t = kv.first;
  const auto &F = kv.second;
  //RCLCPP_INFO(LOGGER, "frame %.6f: pre_integration ptr = %p, sum_dt = %.6f",kv.first,static_cast<void*>(F.pre_integration),  F.pre_integration ? F.pre_integration->sum_dt : 0.0);
}
  Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;

    }
    delta_bg = A.ldlt().solve(b);
    RCLCPP_WARN_STREAM(LOGGER, "Gyroscope bias initial calibration: " << delta_bg.transpose());
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

MatrixXd TangentBasis(Vector3d &g0)
{
  Vector3d b, c;
  Vector3d a = g0.normalized();
  Vector3d tmp(0, 0, 1);
  if(a == tmp)
      tmp << 1, 0, 0;
  b = (tmp - a * (a.transpose() * tmp)).normalized();
  c = a.cross(b);
  MatrixXd bc(3, 2);
  bc.block<3, 1>(0, 0) = b;
  bc.block<3, 1>(0, 1) = c;
  return bc;
}

void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)
    {
      MatrixXd lxly(3, 2);
      lxly = TangentBasis(g0);
      int i = 0;
      for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
      {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 9);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;


        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
        tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
        b.tail<3>() += r_b.tail<3>();

        A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
        A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
      }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    VectorXd dg = x.segment<2>(n_state - 3);
    g0 = (g0 + lxly * dg).normalized() * G.norm();
  //double s = x(n_state - 1);
  }   
  g = g0;
}

/*bool LinearAlignment(
  map<double, ImageFrame> & all_image_frame,
  Vector3d &                g,
  VectorXd &                x)
{
  int N = static_cast<int>(all_image_frame.size());
  int state_size = N*3 + 3 + 1;

  MatrixXd A = MatrixXd::Zero(state_size, state_size);
  VectorXd b = VectorXd::Zero(state_size);

  int idx = 0;
  for (auto it = all_image_frame.begin(); next(it) != all_image_frame.end(); ++it, ++idx) {
    auto jt = next(it);
    double dt = jt->second.pre_integration->sum_dt;

    // build local system
    MatrixXd tmpA(6, state_size);
    tmpA.setZero();
    VectorXd tmpb(6);
    tmpb.setZero();

    // position
    tmpA.block<3,3>(0, idx*3) = -dt * Matrix3d::Identity();
    tmpA.block<3,3>(0, N*3 + 0) =
      it->second.R.transpose() * (dt*dt/2) * Matrix3d::Identity();
    tmpA.block<3,1>(0, state_size - 1) =
      it->second.R.transpose() * (jt->second.T - it->second.T) / 100.0;
    tmpb.segment<3>(0) =
      jt->second.pre_integration->delta_p
      + it->second.R.transpose() * jt->second.R * TIC[0]
      - TIC[0];

    // velocity
    tmpA.block<3,3>(3, idx*3) = -Matrix3d::Identity();
    tmpA.block<3,3>(3, idx*3 + 3) =
      it->second.R.transpose() * jt->second.R;
    tmpA.block<3,3>(3, N*3 + 0) =
      it->second.R.transpose() * dt * Matrix3d::Identity();
    tmpb.segment<3>(3) =
      jt->second.pre_integration->delta_v;

    Matrix<double,6,6> cov = Matrix<double,6,6>::Identity();
    MatrixXd rA = tmpA.transpose() * cov * tmpA;
    VectorXd rb = tmpA.transpose() * cov * tmpb;

    int base = idx*3;
    A.block(base, base, 6, 6)     += rA.topLeftCorner<6,6>();
    b.segment(base, 6)            += rb.head<6>();
    A.bottomRightCorner<4,4>()    += rA.bottomRightCorner<4,4>();
    b.tail<4>()                   += rb.tail<4>();
    A.block(base, state_size-4, 6, 4)       += rA.topRightCorner<6,4>();
    A.block(state_size-4, base, 4, 6)       += rA.bottomLeftCorner<4,6>();
  }

  A *= 1000; b *= 1000;
  x = A.ldlt().solve(b);

  double s = x(state_size - 1) / 100.0;
  g = x.segment<3>(state_size - 4);

  if (std::fabs(g.norm() - G.norm()) > 1.0 || s < 0) {
    return false;
  }

  // refine and re-scale
  RefineGravity(all_image_frame, g, x);
  s = x.tail<1>()(0) / 100.0;
  x.tail<1>()(0) = s;
  return s >= 0.0;
}
*/
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    RCLCPP_DEBUG(LOGGER, "estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    RCLCPP_DEBUG_STREAM(LOGGER, " result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    RCLCPP_DEBUG_STREAM(LOGGER, "refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

/*bool VisualIMUAlignment(
  map<double, ImageFrame> & all_image_frame,
  Vector3d *                Bgs,
  Vector3d &                g,
  VectorXd &                x)
{
  // 1) 校准陀螺仪偏置
  solveGyroscopeBias(all_image_frame, Bgs);
  // 2) 线性对齐（求重力向量和尺度）
  return LinearAlignment(all_image_frame, g, x);
}
*/
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(all_image_frame, Bgs);

    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}

}  // namespace vins_estimator
