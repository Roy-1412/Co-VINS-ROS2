#include "parameters.h"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

// Global variables
double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMU_TOPIC;
std::string IMAGE_TOPIC;
double ROW, COL;
double TD, TR;
int SWARM_AGENT;
int AGENT_NUM;
void readParameters(const std::shared_ptr<rclcpp::Node> &node)
{
     std::string config_file;
    if (!node->get_parameter("config_file", config_file) || config_file.empty()) {
        RCLCPP_ERROR(node->get_logger(),
                    "Parameter 'config_file' not provided or is empty!");
        rclcpp::shutdown();
        return;
    }

    // 2) 打开 YAML 文件
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        RCLCPP_ERROR(node->get_logger(),
                    "ERROR: Wrong path to settings: %s",
                    config_file.c_str());
        rclcpp::shutdown();
        return;
    }

    fsSettings["imu_topic"] >> IMU_TOPIC;
    fsSettings["image_topic"] >> IMAGE_TOPIC;

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = static_cast<int>(fsSettings["max_num_iterations"]);
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX /= FOCAL_LENGTH;

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
    RCLCPP_INFO(node->get_logger(), "result path: %s", VINS_RESULT_PATH.c_str());
    std::ofstream fout(VINS_RESULT_PATH);
    fout.close();

    ACC_N = fsSettings["acc_n"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    RCLCPP_INFO(node->get_logger(), "ROW: %f COL: %f", ROW, COL);

    ESTIMATE_EXTRINSIC = static_cast<int>(fsSettings["estimate_extrinsic"]);
    if (ESTIMATE_EXTRINSIC == 2) {
        RCLCPP_WARN(node->get_logger(), "No prior extrinsic, calibrate extrinsic parameters");
        RIC.emplace_back(Eigen::Matrix3d::Identity());
        TIC.emplace_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
    } else {
        if (ESTIMATE_EXTRINSIC == 1) {
            RCLCPP_WARN(node->get_logger(), "Optimize extrinsic parameters around initial guess");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        } else {
            RCLCPP_WARN(node->get_logger(), "Fix extrinsic parameters");
        }
        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
        RCLCPP_INFO_STREAM(node->get_logger(), "Extrinsic_R:\n" << eigen_R.format(Eigen::IOFormat()));
        RCLCPP_INFO(node->get_logger(), "Extrinsic_T: [%f, %f, %f]", eigen_T.x(), eigen_T.y(), eigen_T.z());
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = static_cast<int>(fsSettings["estimate_td"]);
    if (ESTIMATE_TD)
        RCLCPP_INFO(node->get_logger(), "Unsynced sensors, estimate time offset. initial td: %f", TD);
    else
        RCLCPP_INFO(node->get_logger(), "Synced sensors, fixed time offset: %f", TD);

    ROLLING_SHUTTER = static_cast<int>(fsSettings["rolling_shutter"]);
    if (ROLLING_SHUTTER) {
        TR = fsSettings["rolling_shutter_tr"];
        RCLCPP_INFO(node->get_logger(), "Rolling shutter camera, readout time per line: %f", TR);
    } else {
        TR = 0.0;
    }

    SWARM_AGENT = static_cast<int>(fsSettings["swarm_agent"]);
    if (SWARM_AGENT) {
        RCLCPP_INFO(node->get_logger(), "Swarm agent mode enabled");
        node->declare_parameter<int>("agent_num", 1);
        node->get_parameter("agent_num", AGENT_NUM);
    }

    fsSettings.release();
}
