/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "featureTracker/feature_tracker.h"
#include "utility/visualization.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "agent_msg/msg/agent_msg.hpp"
#include <cmath>  // for std::abs

Estimator estimator;

queue<sensor_msgs::msg::Imu::ConstPtr> imu_buf;
queue<sensor_msgs::msg::PointCloud::ConstPtr> feature_buf;
queue<sensor_msgs::msg::Image::ConstPtr> img0_buf;
queue<sensor_msgs::msg::Image::ConstPtr> img1_buf;
queue<sensor_msgs::msg::Image::ConstPtr> image_buf;
std::mutex m_buf;
camodocal::CameraPtr m_camera;

constexpr double kTentativeOffset = 1e-5;

// header: 1403715278
void img0_callback(const sensor_msgs::msg::Image::SharedPtr img_msg)
{
    m_buf.lock();
    // std::cout << "Left : " << img_msg->header.stamp.sec << "." << img_msg->header.stamp.nanosec << endl;
    img0_buf.push(img_msg);
    image_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::msg::Image::SharedPtr img_msg)
{
    m_buf.lock();
    // std::cout << "Right: " << img_msg->header.stamp.sec << "." << img_msg->header.stamp.nanosec << endl;
    img1_buf.push(img_msg);
    m_buf.unlock();
}


// cv::Mat getImageFromMsg(const sensor_msgs::msg::Image::SharedPtr img_msg)
cv::Mat getImageFromMsg(const sensor_msgs::msg::Image::ConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::msg::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// extract images with same timestamp from two topics
void sync_process()
{
    while(1)
    {
        if(STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::msg::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9);
                double time1 = img1_buf.front()->header.stamp.sec + img1_buf.front()->header.stamp.nanosec * (1e-9);

                // 0.003s sync tolerance
                if(time0 < time1 - 0.003)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if(time0 > time1 + 0.003)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9) + kTentativeOffset;
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                    //printf("find img0 and img1\n");

                    // std::cout << std::fixed << img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9) << std::endl;
                    // assert(0);
                    
                }
            }
            m_buf.unlock();
            if(!image0.empty())
                estimator.inputImage(time, image0, image1);
        }
        else
        {
            cv::Mat image;
            std_msgs::msg::Header header;
            double time = 0;
            m_buf.lock();
            if(!img0_buf.empty())
            {
                time = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9) + kTentativeOffset;
                header = img0_buf.front()->header;
                image = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
            }
            m_buf.unlock();
            if(!image.empty())
                estimator.inputImage(time, image);
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
{
    // std::cout << "IMU cb" << std::endl;

    double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * (1e-9);
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);

    // std::cout << "got t_imu: " << std::fixed << t << endl;
    estimator.inputIMU(t, acc, gyr);
    return;
}


void feature_callback(const sensor_msgs::msg::PointCloud::SharedPtr feature_msg)
{
    std::cout << "feature cb" << std::endl;
    std::cout << "Feature: " << feature_msg->points.size() << std::endl;


    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        assert(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.sec + feature_msg->header.stamp.nanosec * (1e-9);
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::msg::Bool::SharedPtr restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::msg::Bool::SharedPtr switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        //ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::msg::Bool::SharedPtr switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        //ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

void agent_process()
{
    while(1)
    {
        m_agent_msg_buf.lock();
        agent_msg::msg::AgentMsg tmp_msg;
        bool pub_flag = false;
        while (!agent_msg_buf.empty())
        {
            tmp_msg = agent_msg_buf.front();
            agent_msg_buf.pop();
            pub_flag = true;
        }
        m_agent_msg_buf.unlock();

        if (pub_flag)
        {
            TicToc pubAgentFrame_time;
            std::shared_ptr<const sensor_msgs::msg::Image> image_msg = nullptr;
            m_buf.lock();
            
            // if (image_buf.empty()) 
            // {
            // ROS_WARN("image_buf is empty before popping");
            // } 
            // // 2. 再判断队首帧时间戳是否小于目标时间戳
            // else 
            // {
            //     double front_t = rclcpp::Time(image_buf.front()->header.stamp).seconds();
            //     double tmp_t   = rclcpp::Time(tmp_msg.header.stamp).seconds();
            //     if (front_t < tmp_t) 
            //     {
            //         ROS_WARN("front timestamp %.9f < tmp timestamp %.9f: will pop", front_t, tmp_t);
            //     } 
            //     else 
            //     {
            //         ROS_INFO("front timestamp %.9f >= tmp timestamp %.9f: will not pop", front_t, tmp_t);
            //     }
            // }

            while(!image_buf.empty() && rclcpp::Time(image_buf.front()->header.stamp).seconds() < rclcpp::Time(tmp_msg.header.stamp).seconds())
                image_buf.pop();
            if (!image_buf.empty())
            {
                image_msg = image_buf.front();
            }
            m_buf.unlock();
            //if (image_msg == NULL || rclcpp::Time(image_msg->header.stamp).seconds() != rclcpp::Time(tmp_msg.header.stamp).seconds())
            if (image_msg == nullptr ||std::abs( rclcpp::Time(image_msg->header.stamp).seconds() - rclcpp::Time(tmp_msg.header.stamp).seconds()) > 0.1)
            {
                // // 先判断指针是否为空
                // if (!image_msg) 
                // {
                //     ROS_WARN("No image received: image_msg is NULL");
                // }
                // // 再判断时间戳是否匹配
                // else if (rclcpp::Time(image_msg->header.stamp).seconds() != rclcpp::Time(tmp_msg.header.stamp).seconds())
                // {
                //     ROS_WARN("Timestamp mismatch: image time = %.9f, expected time = %.9f",
                //     rclcpp::Time(image_msg->header.stamp).seconds(),
                //     rclcpp::Time(tmp_msg.header.stamp).seconds());
                // }
                
                ROS_WARN("can not find corresponding image");
            }
            else
            {
                cv_bridge::CvImageConstPtr ptr;
                if (image_msg->encoding == "8UC1")
                {
                    sensor_msgs::msg::Image img;
                    img.header = image_msg->header;
                    img.height = image_msg->height;
                    img.width = image_msg->width;
                    img.is_bigendian = image_msg->is_bigendian;
                    img.step = image_msg->step;
                    img.data = image_msg->data;
                    img.encoding = "mono8";
                    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
                }
                else
                    ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);

                cv::Mat img = ptr->image;
                pubAgentFrame(tmp_msg, img, m_camera);
            }
            //ROS_WARN("pub agent frame time %f", pubAgentFrame_time.toc());
        }
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
        
}





int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto n = rclcpp::Node::make_shared("vins_estimator");

    std::cout << "[DEBUG] entered vins_node main()" << std::endl;

    // 设置 BRIEF 描述子路径
    std::string pkg_path = ament_index_cpp::get_package_share_directory("vins");
    BRIEF_PATTERN_FILE = pkg_path + "/../support_files/brief_pattern.yml";

    // int FISHEYE = 0;
    // n->declare_parameter<int>("FISHEYE", 0);
    // n->get_parameter("FISHEYE", FISHEYE);

    //std::cout << "[DEBUG] FISHEYE = " << FISHEYE << std::endl;
    if(FISHEYE)
    {
        cout<<"calling fisheye_mask"<<endl;
        fisheye_mask = cv::imread(FISHEYE_MASK, /*flags=*/0);
        if (!fisheye_mask.data) 
        {
            ROS_INFO("load mask fail");
            std::abort(); 
        }
        else
            ROS_INFO("load mask success");
        // if (fisheye_mask.empty()) 
        // {
        //     std::cout << "load mask fail: empty Mat" << std::endl;
        //     std::abort();
        // } 
        // else 
        // {
        //     std::cout << "load mask success: "<< "size = " << fisheye_mask.cols << "x" << fisheye_mask.rows<< std::endl;
        // }
        
    }
    // 声明参数
    n->declare_parameter<std::string>("config_file", std::string(""));
    n->declare_parameter<int>("agent_num", 1);

    // 读取参数
    std::string config_file;
    int agent_num = 1;
    n->get_parameter("config_file", config_file);
    n->get_parameter("agent_num", agent_num);

    // fallback 到位置参数（仅当没通过 parameter 传）
    if (config_file.empty() && argc >= 2) {
        config_file = std::string(argv[1]);
    }

    if (config_file.empty()) {
        RCLCPP_ERROR(n->get_logger(),
            "No config file provided. Pass via ROS2 parameter 'config_file' or as first positional argument.");
        return 1;
    }

    RCLCPP_INFO(n->get_logger(), "Using config file: %s", config_file.c_str());
    RCLCPP_INFO(n->get_logger(), "Agent num: %d", agent_num);

    std::string cam0Path;
    readParameters(n, config_file, cam0Path);

    std::cout << "Using camera calib file: " << CAM_NAMES[0] << std::endl;

    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);


    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu = NULL;
    if(USE_IMU)
    {
        sub_imu = n->create_subscription<sensor_msgs::msg::Imu>(IMU_TOPIC, rclcpp::QoS(rclcpp::KeepLast(2000)), imu_callback);
    }
    auto sub_feature = n->create_subscription<sensor_msgs::msg::PointCloud>("feature_tracker/feature", rclcpp::QoS(rclcpp::KeepLast(2000)), feature_callback);
    auto sub_img0 = n->create_subscription<sensor_msgs::msg::Image>(IMAGE0_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)), img0_callback);
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img1 = NULL;
    if(STEREO)
    {
        sub_img1 = n->create_subscription<sensor_msgs::msg::Image>(IMAGE1_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)), img1_callback);
    }
    
    auto sub_restart = n->create_subscription<std_msgs::msg::Bool>("vins_restart", rclcpp::QoS(rclcpp::KeepLast(100)), restart_callback);
    auto sub_imu_switch = n->create_subscription<std_msgs::msg::Bool>("vins_imu_switch", rclcpp::QoS(rclcpp::KeepLast(100)), imu_switch_callback);
    auto sub_cam_switch = n->create_subscription<std_msgs::msg::Bool>("vins_cam_switch", rclcpp::QoS(rclcpp::KeepLast(100)), cam_switch_callback);

    std::thread sync_thread{sync_process};

    std::thread agent_process_thread;
    if (SWARM_AGENT)
    {
        ROS_INFO("start swarm mode");
        ROS_INFO("BRIEF_PATTERN_FILE = %s", BRIEF_PATTERN_FILE.c_str());



        m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0Path.c_str());
        
      

        
        std::cerr << "相机模型路径： " << config_file.c_str() << std::endl;
            
        
        
        std::string pkg_path = ament_index_cpp::get_package_share_directory("vins"); 
        BRIEF_PATTERN_FILE = pkg_path + "/support_files/brief_pattern.yml";
        agent_process_thread = std::thread(agent_process);
    }


    rclcpp::spin(n);

    return 0;
}
