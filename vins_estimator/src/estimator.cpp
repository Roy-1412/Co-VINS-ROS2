#include "estimator.h"
#include "parameters.h"
#include "initial/initial_alignment.h"

#include <Eigen/Dense>
#include <map>
using namespace vins_estimator;
using vins_estimator::InitialEXRotation;
using vins_estimator::ImageFrame;

rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
/*double fx = 4.616e+02;
double fy = 4.603e+02;
double cx = 3.630e+02;
double cy = 2.481e+02;
*/
Estimator::Estimator() : f_manager{Rs}
{
    // 构造时只做最小初始化
    clearState();
    
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }// TIC/RIC 是全局定义好的常量外参
    // TIC[i] 表示第 i 个相机相对于 IMU 的平移向量，RIC[i] 表示第 i 个相机相对于 IMU 的旋转矩阵

    f_manager.setRic(ric);  // 传参给 FeatureManager
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;  // 相机–IMU 时间偏差，恢复到初始全局常量
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)  // 滑窗内的每一帧
    {
        Rs[i].setIdentity();  // 姿态
        Ps[i].setZero();  // 位置
        Vs[i].setZero();  // 速度
        Bas[i].setZero();  // 加速度偏置
        Bgs[i].setZero();  // 角速度偏置
        dt_buf[i].clear();  // 清空第 i 帧所有 IMU 时间增量缓存
        linear_acceleration_buf[i].clear();  // 清空第 i 帧所有线加速度缓存
        angular_velocity_buf[i].clear();  // 清空第 i 帧所有角速度缓存

        if (pre_integrations[i] != nullptr)  // 如果存在预积分对象，就删除
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;  // 置空指针
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();  // 相机平移外参 ← 零
        ric[i] = Matrix3d::Identity();  // 相机旋转外参 ← 单位矩阵
    }

    solver_flag = Estimator::SolverFlag::INITIAL;  // 求解器状态 ← 初始（只做结构化初始化阶段）
    first_imu = false;  // 是否收到第一条 IMU 的标志 ← 否
    sum_of_back = 0;  // “旧帧”边缘化计数 ← 清零
    sum_of_front = 0;  // “新帧”边缘化计数 ← 清零
    frame_count = 0;  // 当前已处理图像帧数 ← 0
    initial_timestamp = 0;  // 结构化初始化时最后一次时间戳 ← 0
    all_image_frame.clear();  // 清空保存的所有 ImageFrame（历史关键帧）
    td = TD;  // 相机–IMU 时间偏差 ← 恢复到初始全局常量

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;  // 删除当前临时预积分
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;  // 删除上一轮滑窗边缘化信息

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();  // 清空边缘化残差对应的参数块列表

    f_manager.clearState();  // 删除所有特征跟踪、深度、帧对应关系等前端缓存

    failure_occur = 0;  // 失败检测标志 ← 清零（无失败）
    relocalization_info = 0;  // 重定位触发标志 ← 清零（不处于回环重定位中）

    drift_correct_r = Matrix3d::Identity();  // 漂移校正旋转 ← 单位
    drift_correct_t = Vector3d::Zero();  // 漂移校正平移 ← 零（回环检测时用到）
}

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;  // 记录第一个收到的 IMU 读数，为后续的预积分、状态传播提供“上一时刻”参考值。
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    } // 只在第一次调用时执行一次，将 acc_0／gyr_0 置为这条测量

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    } // 为当前图像帧（索引 frame_count）分配一个新的 IntegrationBase（IMU 预积分器），带入初始的加速度、角速度和当前估计的偏置
    // 这一帧看到的 IMU 流都要累积到同一个预积分器里
    if (frame_count != 0) // 不是第一帧
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
        // 把当前这次测量及时间间隔 dt 累加进两个预积分器 pre_integrations[frame_count] 用于生成 IMU 因子
        // tmp_pre_integration 用于下一个图像帧的初始化
        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        // 1) 计算上一时刻与当前时刻的“去偏”加速度和角速度
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        // 2) 更新旋转：用平均角速度乘小角增量
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        // 3) 计算新的“去偏”加速度
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        // 4) 用梯形积分算增量加速度
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
        // 5) 位置与速度更新
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; // 保存“上一时刻”测量
}

void Estimator::processImage(
    const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
    const std_msgs::msg::Header &header)
{
    // 1) 取出 ROS 时间戳为 double
    double timestamp = rclcpp::Time(header.stamp).seconds();

    // 2) 日志输出
    std::cout << "New image coming ------------------------- at " 
              << timestamp << std::endl;
    std::cout << "Adding feature points: " << image.size() << std::endl;

    // 3) 判断是否为关键帧（设置边缘化策略）
    if (f_manager.addFeatureCheckParallax(frame_count, image, td)) 
    {
        marginalization_flag = Estimator::MarginalizationFlag::MARGIN_OLD;
    } 
    else 
    {
        marginalization_flag = Estimator::MarginalizationFlag::MARGIN_SECOND_NEW;
    }

    std::cout << "This frame is "
          << (marginalization_flag == Estimator::MarginalizationFlag::MARGIN_OLD
                 ? "reject"
                 : "accept")
          << std::endl;

    //std::cout << (marginalization_flag == MarginalizationFlag::MARGIN_OLD ? "Non-keyframe" : "Keyframe") << std::endl;
    RCLCPP_INFO( logger_,"%s", (marginalization_flag == MarginalizationFlag::MARGIN_OLD? "Non-keyframe": "Keyframe"));

    std::cout << "Solving: " << frame_count << std::endl;
    std::cout << "Number of features: " << f_manager.getFeatureCount() 
              << std::endl;

    RCLCPP_DEBUG(logger_,
    "This frame is %s",
    (marginalization_flag == Estimator::MarginalizationFlag::MARGIN_OLD
       ? "reject"
       : "accept"));

    RCLCPP_DEBUG(logger_,
        "%s",
        (marginalization_flag == Estimator::MarginalizationFlag::MARGIN_OLD
        ? "Non-keyframe"
        : "Keyframe"));

    RCLCPP_DEBUG(logger_,
        "Solving: %d",
        frame_count);

    RCLCPP_DEBUG(logger_,
        "Number of features: %d",
        f_manager.getFeatureCount());


    // 4) 记录时间戳、保留到 Headers（已将 Headers 改为 double[]）
    RCLCPP_INFO(logger_, "Insert frame %d → timestamp = %.9f", frame_count, timestamp);
    Headers.push_back(header);
    // 5) 存入 all_image_frame
    ImageFrame imageframe(image, timestamp);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.emplace(timestamp, std::move(imageframe));
    // 6) 为下一帧创建新的预积分器
    tmp_pre_integration = new IntegrationBase{
        acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]
    };

    // 7) 如果在外参标定模式，做旋转标定
    if (ESTIMATE_EXTRINSIC == 2) {
        std::cout << "Calibrating extrinsic param, rotation movement is needed" 
                  << std::endl;
        if (frame_count != 0) {
            auto corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(
                    corres,
                    pre_integrations[frame_count]->delta_q,
                    calib_ric))
            {
                std::cout << "Initial extrinsic rotation calib success" 
                          << std::endl;
                std::cout << "Initial extrinsic rotation:\n" 
                          << calib_ric << std::endl;
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // 8) 如果还在初始化阶段
    if (solver_flag == Estimator::SolverFlag::INITIAL) {
        if (frame_count == WINDOW_SIZE) {
             RCLCPP_INFO(logger_, "[InitCheck] frame_count = %d, WINDOW_SIZE = %d → %s",
                    frame_count, WINDOW_SIZE,
                    frame_count == WINDOW_SIZE ? "true" : "false");
            bool result = false;
            // 保证间隔够长才重建
            double dt = timestamp - initial_timestamp;
             RCLCPP_INFO(logger_, "[InitCheck] ESTIMATE_EXTRINSIC = %d (need !=2), dt_since_last = %.3f (need >0.1)",
                    ESTIMATE_EXTRINSIC, dt);
            if (ESTIMATE_EXTRINSIC != 2 &&
                (timestamp - initial_timestamp) > 0.1)
            {
                RCLCPP_INFO(logger_, "[InitCheck] attempting initialStructure()");
                result = initialStructure();
                initial_timestamp = timestamp;
                RCLCPP_INFO(logger_, "[InitCheck] initialStructure() returned %s",
                        result ? "true" : "false");
            }
            else 
            {
                RCLCPP_INFO(logger_, "[InitCheck] skip initialStructure() (mode/time not met)");
            }
            if (result) {
                RCLCPP_INFO(logger_, "[InitCheck] all conditions met → switch to NON_LINEAR");
                solver_flag = Estimator::SolverFlag::NON_LINEAR;
                solveOdometry();
                slideWindow();
                f_manager.removeFailures();
                std::cout << "Initialization finished!" << std::endl;
                last_R  = Rs[WINDOW_SIZE];
                last_P  = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
            } else {
                RCLCPP_INFO(logger_, "[InitCheck] not initialized → slideWindow()");
                slideWindow();
            }
        } else {
            RCLCPP_INFO(logger_, "[InitCheck] accumulating frames: %d/%d", frame_count, WINDOW_SIZE);
            frame_count++;
        }
    }
    // 9) 非线性优化阶段
    else {
        TicToc t_solve;
        solveOdometry();
        std::cout << "Solver costs: " << t_solve.toc() 
                  << "ms" << std::endl;

        if (failureDetection()) {
            std::cout << "Failure detection!" << std::endl;
            failure_occur = 1;
            clearState();
            setParameter();
            std::cout << "System reboot!" << std::endl;
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        std::cout << "Marginalization costs: " << t_margin.toc() 
                  << "ms" << std::endl;

        // 准备输出轨迹
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++) {
            key_poses.push_back(Ps[i]);
        }
        last_R  = Rs[WINDOW_SIZE];
        last_P  = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}


bool Estimator::initialStructure()
{
    RCLCPP_INFO(logger_, ">>> Enter initialStructure(), frame_count = %d", frame_count);
    TicToc t_sfm;

    // 检查 IMU 可观测性
    {
        std::map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        RCLCPP_INFO(logger_, "    start imu");
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        RCLCPP_INFO(logger_, "    start aver_g");
        // 计算每段预积分对应的平均“重力向量”
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); // 求出加速度向量的均值
        double var = 0;
        //for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        for (auto frame_it = std::next(all_image_frame.begin()); frame_it != all_image_frame.end(); ++frame_it)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        // 计算标准差，度量不同时间段里“重力向量”估计的分散程度
        if(var < 0.25)
        {
            RCLCPP_WARN(logger_, "IMU excitation not enough!");
        }
    }

    // global SFM
    Quaterniond Q[frame_count + 1];  // 存放一帧图像对应的旋转 四元数表示
    Vector3d T[frame_count + 1];  // 平移
    std::map<int, Vector3d> sfm_tracked_points;  // 特征点在某个参考坐标系下的 3D 坐标
    std::vector<SFMFeature> sfm_f;  // 特征结构体

    for (auto &it_per_id : f_manager.feature)  // 遍历 f_manager.feature 容器中的每一个元素
    {
        int imu_j = it_per_id.start_frame - 1;  // 用来在“内层循环”中给每次观测分配正确的帧索引
        SFMFeature tmp_feature;  // 新建一个临时结构体 tmp_feature，用来保存当前这个特征在不同帧上的所有观测
        tmp_feature.state = false;  // 未做三角化标志
        tmp_feature.id = it_per_id.feature_id;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;  // 取出当前观测里的坐标数据，存入一个 Vector3d 变量 pts_j
            tmp_feature.observation.push_back(std::make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    const int MIN_TRACK_LEN = 5;
    vector<SFMFeature> sfm_f_filtered;
    sfm_f_filtered.reserve(sfm_f.size());
    for (auto &f : sfm_f) {
        if ((int)f.observation.size() >= MIN_TRACK_LEN) {
            sfm_f_filtered.push_back(std::move(f));
        }
    }
    sfm_f.swap(sfm_f_filtered);
   // visualizeTracks(sfm_f);auto &f0 = sfm_f.front();

    Matrix3d relative_R;  // 存储两帧之间估计得到的相对旋转矩阵
    Vector3d relative_T;  // 存储两帧之间估计得到的相对平移向量
    int l;

    if (!relativePose(relative_R, relative_T, l))  // 从滑动窗口或关键帧集合中选取一对视差足够大、共视特征足够多的帧来估计它们之间的粗糙相对位姿
    {
        return false;
    }
    RCLCPP_INFO(logger_, "    relativePose succeeded: l = %d", l);

    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points))
    {
        Matrix3d R = relative_R;
        Matrix3d I_diff = R * R.transpose() - Matrix3d::Identity();
        double ortho_err = I_diff.norm();
        double det = R.determinant();
        RCLCPP_INFO(logger_, "[Check] relative_R ortho_err=%.6f, det=%.6f", ortho_err, det); //正交误差0 det=1

        RCLCPP_INFO(logger_,
        "[Check] relative_T = [%.3f, %.3f, %.3f], |T|=%.3f",
        relative_T.x(), relative_T.y(), relative_T.z(),
        relative_T.norm());//纯方向的平移估计  范数=1




        marginalization_flag = Estimator::MarginalizationFlag::MARGIN_OLD;

   
         //--- Basic statistics check ---
        {
            std::vector<int> lengths;
            for (auto &f : sfm_f) {
                lengths.push_back(static_cast<int>(f.observation.size()));
            }
            if (!lengths.empty()) {
                std::sort(lengths.begin(), lengths.end());
                int n = static_cast<int>(lengths.size());
                int min_len = lengths.front();
                int med_len = lengths[n / 2];
                int max_len = lengths.back();
                RCLCPP_INFO(logger_,
                    "[SFM] Track lengths: total=%d, min=%d, med=%d, max=%d",
                    n, min_len, med_len, max_len);
            }
            // Check for frame gaps
            for (auto &f : sfm_f) {
                for (int i = 1; i < static_cast<int>(f.observation.size()); ++i) {
                    int prev = f.observation[i-1].first;
                    int curr = f.observation[i].first;
                    if (curr != prev + 1) {
                        RCLCPP_WARN(logger_,
                            "[SFM] Feature %d frame gap: %d -> %d",
                            f.id, prev, curr);
                        break;
                    }
                }
            }
        }
        
        
        // --- 2D reprojection consistency check ---
        // Assume Q[i] (Quaterniond) and T[i] (Vector3d) hold initial poses and K (Eigen::Matrix3d) is intrinsics
    /*  {            
            Eigen::Matrix3d K;
            K << fx,   0,    cx,
                0,    fy,   cy,
                0,    0,    1;
            auto project = [&](const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                            const Eigen::Vector3d &X) {
                Eigen::Vector3d P = R * X + t;
                return Eigen::Vector2d(P.x() / P.z(), P.y() / P.z());
            };

            for (auto &f : sfm_f) {
                if (f.observation.size() < 2) continue;
                // Triangulate with first and last observation
                int i0 = f.observation.front().first;
                int i1 = f.observation.back().first;
                Eigen::Vector2d u0 = f.observation.front().second;
                Eigen::Vector2d u1 = f.observation.back().second;

                Eigen::Matrix<double,4,4> A;
                Eigen::Matrix<double,3,4> P0, P1;
                P0.leftCols<3>() = K * Q[i0].toRotationMatrix();
                P0.rightCols<1>()  = K * T[i0];
                P1.leftCols<3>() = K * Q[i1].toRotationMatrix();
                P1.rightCols<1>() = K * T[i1];

                A.row(0) = u0.x() * P0.row(2) - P0.row(0);
                A.row(1) = u0.y() * P0.row(2) - P0.row(1);
                A.row(2) = u1.x() * P1.row(2) - P1.row(0);
                A.row(3) = u1.y() * P1.row(2) - P1.row(1);

                Eigen::Vector4d Xh = A.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3);
                Eigen::Vector3d X  = Xh.head<3>() / Xh(3);

                double sum_err = 0;
                for (auto &obs : f.observation) {
                    int idx = obs.first;
                    Eigen::Vector2d uv = obs.second;
                    Eigen::Vector2d pr = project(
                        Q[idx].toRotationMatrix(), T[idx], X);
                    sum_err += (pr - uv).norm();
                }
                double mean_err = sum_err / f.observation.size();
                if (mean_err > 1.0) {
                    RCLCPP_WARN(logger_,
                        "[SFM] Feature %d reproj err=%.2f px (%d obs)",//单点平均误差
                        f.id, mean_err, static_cast<int>(f.observation.size()));
                }
            }
        }*/
        
        return false;
    }
    
   /* for (auto &f : sfm_f) 
    {
    

        std::ostringstream oss;
        oss << "[Track " << f.id << "]";

        for (auto &obs : f.observation) {
            int frame_idx = obs.first;
            double u_pix = obs.second.x();
            double v_pix = obs.second.y();
            oss << " (f" << frame_idx
                << ", u=" << std::fixed << std::setprecision(2) << u_pix
                << ", v=" << std::fixed << std::setprecision(2) << v_pix
                << ")";
        }

        RCLCPP_INFO(logger_, "[DEBUG] %s", oss.str().c_str());
    }
    
    
    // --- Basic statistics check ---
    {
        std::vector<int> lengths;
        for (auto &f : sfm_f) {
            lengths.push_back(static_cast<int>(f.observation.size()));
        }
        if (!lengths.empty()) {
            std::sort(lengths.begin(), lengths.end());
            int n = static_cast<int>(lengths.size());
            int min_len = lengths.front();
            int med_len = lengths[n / 2];
            int max_len = lengths.back();
            RCLCPP_INFO(logger_,
                "[SFM] Track lengths: total=%d, min=%d, med=%d, max=%d",
                n, min_len, med_len, max_len);
        }
        // Check for frame gaps
        for (auto &f : sfm_f) {
            for (int i = 1; i < static_cast<int>(f.observation.size()); ++i) {
                int prev = f.observation[i-1].first;
                int curr = f.observation[i].first;
                if (curr != prev + 1) {
                    RCLCPP_WARN(logger_,
                        "[SFM] Feature %d frame gap: %d -> %d",
                        f.id, prev, curr);
                    break;
                }
            }
        }
    }*/

    // --- 2D reprojection consistency check ---
    // Assume Q[i] (Quaterniond) and T[i] (Vector3d) hold initial poses and K (Eigen::Matrix3d) is intrinsics
   /* {
        //单点重投影误差
        Eigen::Matrix3d K;
            K << fx,   0,    cx,
                0,    fy,   cy,
                0,    0,    1;
        
        
        auto project = [&](const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                        const Eigen::Vector3d &X) {
            Eigen::Vector3d P = R * X + t;
            return Eigen::Vector2d(P.x() / P.z(), P.y() / P.z());
        };

        for (auto &f : sfm_f) {
            if (f.observation.size() < 2) continue;
            // Triangulate with first and last observation
            int i0 = f.observation.front().first;
            int i1 = f.observation.back().first;
            Eigen::Vector2d u0 = f.observation.front().second;
            Eigen::Vector2d u1 = f.observation.back().second;

            Eigen::Matrix<double,4,4> A;
            Eigen::Matrix<double,3,4> P0, P1;
            P0.leftCols<3>() = K * Q[i0].toRotationMatrix();
            P0.rightCols<1>()  = K * T[i0];
            P1.leftCols<3>() = K * Q[i1].toRotationMatrix();
            P1.rightCols<1>() = K * T[i1];

            A.row(0) = u0.x() * P0.row(2) - P0.row(0);
            A.row(1) = u0.y() * P0.row(2) - P0.row(1);
            A.row(2) = u1.x() * P1.row(2) - P1.row(0);
            A.row(3) = u1.y() * P1.row(2) - P1.row(1);

            Eigen::Vector4d Xh = A.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3);
            Eigen::Vector3d X  = Xh.head<3>() / Xh(3);

            double sum_err = 0;
            for (auto &obs : f.observation) {
                int idx = obs.first;
                Eigen::Vector2d uv = obs.second;
                Eigen::Vector2d pr = project(
                    Q[idx].toRotationMatrix(), T[idx], X);
                sum_err += (pr - uv).norm();
            }
            double mean_err = sum_err / f.observation.size();
            if (mean_err > 1.0) {
                RCLCPP_WARN(logger_,
                    "[SFM] Feature %d reproj err=%.2f px (%d obs)",
                    f.id, mean_err, static_cast<int>(f.observation.size()));
            }
        }
    }
*/





    RCLCPP_INFO(logger_, "    Global SFM succeeded");

    // solve PnP for all frames
    std::map<double, ImageFrame>::iterator frame_it;
    std::map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    RCLCPP_INFO(logger_, "    Solving PnP for non-key frames");
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        cv::Mat r, rvec, t, D, tmp_r;

        if ((frame_it->first) == rclcpp::Time( Headers[i].stamp ).seconds())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > rclcpp::Time( Headers[i].stamp ).seconds())
        {
            i++;
        }

        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        std::vector<cv::Point3f> pts_3_vector;
        std::vector<cv::Point2f> pts_2_vector;

        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }

        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

        if (pts_3_vector.size() < 6)
        {
            RCLCPP_ERROR(logger_, "  initialstructure failed in pnp  Not enough pts_3_vector");
            return false;
        }

        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            RCLCPP_ERROR(logger_, "  initialstructure failed in pnp  ");
            
            return false;
        }

        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    if (visualInitialAlign())
    {
        return true;
        RCLCPP_INFO(logger_, "  initialstructure successful!!!!!!!!!!!");
    }
    else
    {
        RCLCPP_ERROR(logger_, "  initialstructure failed in visualInitialAlign");
        return false;
    }
}

/*void Estimator::visualizeTracks(const std::vector<SFMFeature>& sfm_f) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "camera_frame";
    rclcpp::Clock ros_clock(RCL_ROS_TIME); m.header.stamp = ros_clock.now();
    m.ns = "sfm_tracks";
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::LINE_LIST;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = 0.005;
    m.color.a = 1.0;

    // Random color generator per feature
    for (const auto &feat : sfm_f) {
        std::mt19937 gen(feat.id);
        std::uniform_real_distribution<float> dis(0.0f,1.0f);
        std_msgs::msg::ColorRGBA c;
        c.r = dis(gen);
        c.g = dis(gen);
        c.b = dis(gen);
        c.a = 1.0f;

        for (size_t i = 1; i < feat.observation.size(); ++i) {
            // observation: pair<frame_idx, Vector2d>
            auto uv0 = feat.observation[i-1].second;
            auto uv1 = feat.observation[i].second;
            geometry_msgs::msg::Point p0, p1;
            p0.x = uv0.x(); p0.y = uv0.y(); p0.z = 1.0;
            p1.x = uv1.x(); p1.y = uv1.y(); p1.z = 1.0;
            m.points.push_back(p0);
            m.points.push_back(p1);
            m.colors.push_back(c);
            m.colors.push_back(c);
        }
    }
    marker_pub_->publish(m);
}
*/












bool Estimator::visualInitialAlign()
{
    RCLCPP_INFO(logger_, " Calling VisualIMUAlignment()");
    TicToc t_g;
    VectorXd x;

    // solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);

    if (!result)
    {
        RCLCPP_ERROR(logger_, "Solve g failed! Early return from initialStructure");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; ++i)
    {
        double ts = rclcpp::Time(Headers[i].stamp).seconds();
        auto &frame = all_image_frame[ts];
        Ps[i] = frame.T;
        Rs[i] = frame.R;
        frame.is_key_frame = true;
    }


    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    // triangulate on cam pose, no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, tic, ric);

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);

    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (auto frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    Matrix3d rot_diff = R0;

    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }

    RCLCPP_INFO(logger_, "visualInitialAlign succeeful");
    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // Find previous frame which contains enough correspondance and parallax with newest frame
    RCLCPP_INFO(logger_, "calling relativePose ");
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        std::vector<std::pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        const size_t num_corr = corres.size();
        RCLCPP_INFO(logger_,
            "relativePose: frame %d → %d, corres.size() = %zu",
            i, WINDOW_SIZE, corres.size());
        if (corres.size() > 20)// ??
        {      
            double sum_parallax = 0;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax += parallax;
            }
            double average_parallax;
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            double scaled_parallax  = average_parallax * 460.0;
            RCLCPP_INFO(logger_, "relativePose: frame %d → %d, average_parallax=%.4f, scaled_parallax=%.2f", i, WINDOW_SIZE, average_parallax, scaled_parallax);
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))//??
            {
                l = i;
                RCLCPP_INFO(logger_,
                    "relativePose SUCCESS: Average parallax*460 = %.2f, choose l = %d",
                    scaled_parallax, l);
                return true;
            }
        }
    }
    RCLCPP_ERROR(logger_,
        "relativePose FAILED: 窗口内无帧满足 corres>20 且 parallax*460>30");
    return false;
}
/*bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // Find previous frame which contains enough correspondance and parallax with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        std::vector<std::pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax += parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                std::cout << "Average parallax: " << average_parallax * 460 
                          << ", choose l: " << l << " and newest frame to triangulate the whole structure" << std::endl;
                return true;
            }
        }
    }
    return false;
}
*/
void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        std::cout << "Triangulation costs: " << t_tri.toc() << "ms" << std::endl;
        optimization();
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        RCLCPP_DEBUG(rclcpp::get_logger("Estimator"),"euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        std::cout << "Little feature: " << f_manager.last_track_num << std::endl;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        std::cout << "Big IMU acc bias estimation: " << Bas[WINDOW_SIZE].norm() << std::endl;
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        std::cout << "Big IMU gyr bias estimation: " << Bgs[WINDOW_SIZE].norm() << std::endl;
        return true;
    }

    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        std::cout << "Big translation" << std::endl;
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        std::cout << "Big Z translation" << std::endl;
        return true;
    }

    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        std::cout << "Big delta_angle" << std::endl;
        return true;
    }
    return false;
}


void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    // loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            std::cout << "Fix extrinsic param" << std::endl;
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
        {
            std::cout << "Estimate extrinsic param" << std::endl;
        }
    }

    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        // problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    vector2double();

    if (last_marginalization_info)
    {
        // Construct new marginalization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL, last_marginalization_parameter_blocks);
    }

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD)
            {
                ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    std::cout << "Visual measurement count: " << f_m_cnt << std::endl;
    std::cout << "Prepare for Ceres: " << t_prepare.toc() << std::endl;

    if (relocalization_info)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if (start <= relo_frame_local_index)
            {
                while ((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if ((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }
            }
        }
    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;

    if (marginalization_flag == Estimator::MarginalizationFlag::MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;

    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "Iterations: " << summary.iterations.size() << std::endl;
    std::cout << "Solver costs: " << t_solver.toc() << std::endl;

    double2vector();

    TicToc t_whole_marginalization;
    if (marginalization_flag == Estimator::MarginalizationFlag::MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }

            // construct new marginalization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                               vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                               vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        std::cout << "Pre marginalization: " << t_pre_margin.toc() << " ms" << std::endl;

        TicToc t_margin;
        marginalization_info->marginalize();
        std::cout << "Marginalization: " << t_margin.toc() << " ms" << std::endl;

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }

    std::cout << "Whole marginalization costs: " << t_whole_marginalization.toc() << std::endl;
    std::cout << "Whole time for Ceres: " << t_whole.toc() << std::endl;
}


void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == Estimator::MarginalizationFlag::MARGIN_OLD)
    {
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            double t_0 = rclcpp::Time( Headers[0].stamp ).seconds();
            auto it_0 = all_image_frame.find(t_0);
            delete it_0->second.pre_integration;
            all_image_frame.erase(all_image_frame.begin(), it_0);

            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// Real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

// Real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == Estimator::SolverFlag::NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        if (relo_frame_stamp == rclcpp::Time( Headers[i].stamp ).seconds())
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}
