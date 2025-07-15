#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>

#include <rclcpp/rclcpp.hpp>

#include "../utility/utility.h"
#include "../parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>

class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
{
public:
  IMUFactor() = delete;
  IMUFactor(IntegrationBase* _pre_integration)
    : pre_integration(_pre_integration)
  {}

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const override
  {
    using namespace Eigen;

    // unpack states
    Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
    Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

    Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
    Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
    Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

    // compute residual
    Map<Matrix<double, 15, 1>> resid_map(residuals);
    resid_map = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                          Pj, Qj, Vj, Baj, Bgj);

    // whiten
    Matrix<double, 15, 15> sqrt_info =
      LLT<Matrix<double,15,15>>(pre_integration->covariance.inverse())
        .matrixL()
        .transpose();
    resid_map = sqrt_info * resid_map;

    // optionally compute jacobians
    if (jacobians)
    {
      double sum_dt = pre_integration->sum_dt;
      const auto& J = pre_integration->jacobian;

      // check for numerical issues
      if (J.maxCoeff() > 1e8 || J.minCoeff() < -1e8)
      {
        RCLCPP_WARN(rclcpp::get_logger("IMUFactor"),
                    "numerical unstable in preintegration");
      }

      // ∂res/∂Pose_i
      if (jacobians[0])
      {
        Map<Matrix<double,15,7,RowMajor>> jac_Pi(jacobians[0]);
        jac_Pi.setZero();

        // position block
        jac_Pi.block<3,3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
        jac_Pi.block<3,3>(O_P, O_R) =
          Utility::skewSymmetric(
            Qi.inverse() *
            (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt)
          );

        // orientation block
        {
          Matrix3d dq_dbg =
            J.template block<3,3>(O_R, O_BG);
          Quaterniond corr_dq = pre_integration->delta_q *
                                Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
          jac_Pi.block<3,3>(O_R, O_R) =
            -(Utility::Qleft(Qj.inverse() * Qi) *
              Utility::Qright(corr_dq))
             .bottomRightCorner<3,3>();
        }

        // velocity block
        jac_Pi.block<3,3>(O_V, O_R) =
          Utility::skewSymmetric(
            Qi.inverse() * (G * sum_dt + Vj - Vi)
          );

        // whiten
        jac_Pi = sqrt_info * jac_Pi;

        if (jac_Pi.maxCoeff() > 1e8 || jac_Pi.minCoeff() < -1e8)
        {
          RCLCPP_WARN(rclcpp::get_logger("IMUFactor"),
                      "numerical unstable in jacobian_pose_i");
        }
      }

      // ∂res/∂SpeedBias_i
      if (jacobians[1])
      {
        Map<Matrix<double,15,9,RowMajor>> jac_Si(jacobians[1]);
        jac_Si.setZero();

        // blocks from preintegration jacobian
        Matrix3d dp_dba = J.template block<3,3>(O_P, O_BA);
        Matrix3d dp_dbg = J.template block<3,3>(O_P, O_BG);
        Matrix3d dv_dba = J.template block<3,3>(O_V, O_BA);
        Matrix3d dv_dbg = J.template block<3,3>(O_V, O_BG);
        Matrix3d dq_dbg = J.template block<3,3>(O_R, O_BG);

        jac_Si.block<3,3>(O_P, O_V - O_V)   = -Qi.inverse().toRotationMatrix() * sum_dt;
        jac_Si.block<3,3>(O_P, O_BA - O_V)  = -dp_dba;
        jac_Si.block<3,3>(O_P, O_BG - O_V)  = -dp_dbg;

        {
          Quaterniond corr_dq = pre_integration->delta_q *
                                Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
          jac_Si.block<3,3>(O_R, O_BG - O_V) =
            -Utility::Qleft(Qj.inverse() * Qi * corr_dq)
               .bottomRightCorner<3,3>() *
             dq_dbg;
        }

        jac_Si.block<3,3>(O_V, O_V - O_V)   = -Qi.inverse().toRotationMatrix();
        jac_Si.block<3,3>(O_V, O_BA - O_V)  = -dv_dba;
        jac_Si.block<3,3>(O_V, O_BG - O_V)  = -dv_dbg;

        // biases
        jac_Si.block<3,3>(O_BA, O_BA - O_V) = -Matrix3d::Identity();
        jac_Si.block<3,3>(O_BG, O_BG - O_V) = -Matrix3d::Identity();

        // whiten
        jac_Si = sqrt_info * jac_Si;
      }

      // ∂res/∂Pose_j
      if (jacobians[2])
      {
        Map<Matrix<double,15,7,RowMajor>> jac_Pj(jacobians[2]);
        jac_Pj.setZero();

        jac_Pj.block<3,3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

        {
          Matrix3d dq_dbg = J.template block<3,3>(O_R, O_BG);
          Quaterniond corr_dq = pre_integration->delta_q *
                                Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
          jac_Pj.block<3,3>(O_R, O_R) =
            Utility::Qleft(corr_dq.inverse() * Qi.inverse() * Qj)
              .bottomRightCorner<3,3>();
        }

        jac_Pj = sqrt_info * jac_Pj;
      }

      // ∂res/∂SpeedBias_j
      if (jacobians[3])
      {
        Map<Matrix<double,15,9,RowMajor>> jac_Sj(jacobians[3]);
        jac_Sj.setZero();

        jac_Sj.block<3,3>(O_V, O_V - O_V)  = Qi.inverse().toRotationMatrix();
        jac_Sj.block<3,3>(O_BA, O_BA - O_V) = Matrix3d::Identity();
        jac_Sj.block<3,3>(O_BG, O_BG - O_V) = Matrix3d::Identity();

        jac_Sj = sqrt_info * jac_Sj;
      }
    }

    return true;
  }

  IntegrationBase* pre_integration;
};
