#pragma once

#include <rclcpp/rclcpp.hpp>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

static constexpr int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction * cost_function,
                      ceres::LossFunction  * loss_function,
                      std::vector<double*>   parameter_blocks,
                      std::vector<int>       drop_set)
      : cost_function(cost_function)
      , loss_function(loss_function)
      , parameter_blocks(std::move(parameter_blocks))
      , drop_set(std::move(drop_set))
    {}

    /// Evaluate this residual block (fills jacobians and residuals).
    void Evaluate();

    ceres::CostFunction* cost_function;
    ceres::LossFunction* loss_function;
    std::vector<double*> parameter_blocks;
    std::vector<int>     drop_set;

    double** raw_jacobians = nullptr;
    std::vector<
      Eigen::Matrix<double,
                    Eigen::Dynamic,
                    Eigen::Dynamic,
                    Eigen::RowMajor>
    > jacobians;
    Eigen::VectorXd residuals;

    /// ROS-independent helper to map 7-dof poses → 6
    inline int localSize(int size) const
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo*> sub_factors;
    Eigen::MatrixXd                 A;
    Eigen::VectorXd                 b;
    std::unordered_map<long, int>   parameter_block_size; // global sizes
    std::unordered_map<long, int>   parameter_block_idx;  // local indices
};

class MarginalizationInfo
{
public:
    ~MarginalizationInfo();

    /// Convert block size in parameters[] to its local dimension
    int localSize(int size) const;
    /// Convert block size in parameters[] to its global dimension
    int globalSize(int size) const;

    /// Add a factor to marginalize out
    void addResidualBlockInfo(ResidualBlockInfo* info);

    /// Precompute jacobians/residuals of all factors
    void preMarginalize();

    /// Perform Schur complement / marginalization
    void marginalize();

    /// Return the list of remaining parameter blocks (after shifting addresses)
    std::vector<double*> getParameterBlocks(
      std::unordered_map<long, double*>& addr_shift);

    std::vector<ResidualBlockInfo*>              factors;
    int                                          m = 0, n = 0;
    std::unordered_map<long, int>                parameter_block_size; // global sizes
    int                                          sum_block_size = 0;
    std::unordered_map<long, int>                parameter_block_idx;  // local indices
    std::unordered_map<long, double*>            parameter_block_data;

    std::vector<int>                             keep_block_size; // global
    std::vector<int>                             keep_block_idx;  // local
    std::vector<double*>                         keep_block_data;

    Eigen::MatrixXd                              linearized_jacobians;
    Eigen::VectorXd                              linearized_residuals;
    static constexpr double                     eps = 1e-8;
};


class MarginalizationFactor : public ceres::CostFunction
{
public:
  /// 构造时需要提前调用 preMarginalize()/marginalize() 并设置残差维度和参数块大小
  explicit MarginalizationFactor(MarginalizationInfo *marginalization_info);

  /// Ceres 接口：填充 residuals 和 jacobians
  bool Evaluate(double const *const *parameters,
                double *residuals,
                double **jacobians) const override;
  MarginalizationInfo *marginalization_info;
};

