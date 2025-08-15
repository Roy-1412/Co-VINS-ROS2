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

#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/image.hpp>
// #include <sensor_msgs/image_encodings.h>
#include "sensor_msgs/image_encodings.hpp"

#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
// #include <ros/package.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <mutex>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
// #include "camodocal/camera_models/CameraFactory.h"
#include "parameters.h"
#include <agent_msg/msg/agent_msg.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
 

// …



#define SKIP_FIRST_CNT 10
using namespace std;

static PoseGraph * posegraph = nullptr;

geometry_msgs::msg::TransformStamped transform;
queue<sensor_msgs::msg::Image::ConstPtr> image_buf;
queue<sensor_msgs::msg::PointCloud::ConstPtr> point_buf;
// queue<sensor_msgs::msg::PointCloud2::ConstPtr> point_buf2;
queue<nav_msgs::msg::Odometry::ConstPtr> pose_buf;
queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf;
std::mutex m_process;
int frame_index  = 0;
int sequence = 1;

int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = 0;
bool start_flag = 1;
double SKIP_DIS = 0;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
std::string mesh_resource;
int ROW;
int COL;
int DEBUG_IMAGE;
double t_agent = 0;
int frame_cnt = 0;



camodocal::CameraPtr m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;


std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;

visualization_msgs::msg::Marker meshROS;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;


std::queue<agent_msg::msg::AgentMsg::ConstSharedPtr> agent_msg_buf;
std::mutex m_agent_msg_buf;

rclcpp::Subscription<agent_msg::msg::AgentMsg>::SharedPtr sub_agent_msg;
rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom1;
rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom2;
rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom3;
rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom4;

rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr meshPub;



void agent_callback(const agent_msg::msg::AgentMsg::ConstSharedPtr & agent_msg)
{
    if(start_flag)
    {
        m_agent_msg_buf.lock();
        agent_msg_buf.push(agent_msg);
        m_agent_msg_buf.unlock();
    }
}

void agent_process()
{
    cout<<"calling agent process"<<endl;
    while (true)
    {
        agent_msg::msg::AgentMsg::ConstSharedPtr agent_msg = nullptr;
        m_agent_msg_buf.lock();
        //if ((int)agent_msg_buf.size() > 10)
        //    printf("agent_msg buf size %d\n", agent_msg_buf.size());
        if (!agent_msg_buf.empty())
        {
            agent_msg = agent_msg_buf.front();
            agent_msg_buf.pop();
        }
        m_agent_msg_buf.unlock();
        
        if(agent_msg != nullptr)
        {
            // build keyframe
            TicToc t_addframe;
            double time_stamp = agent_msg->header.stamp.sec + agent_msg->header.stamp.nanosec * 1e-9;
            int sequence = agent_msg->seq;
            Vector3d T = Vector3d(agent_msg->position_imu.x,
                                  agent_msg->position_imu.y,
                                  agent_msg->position_imu.z);
            Matrix3d R = Quaterniond(agent_msg->orientation_imu.w,
                                     agent_msg->orientation_imu.x,
                                     agent_msg->orientation_imu.y,
                                     agent_msg->orientation_imu.z).toRotationMatrix();

            Vector3d tic = Vector3d(agent_msg->tic.x,
                                    agent_msg->tic.y,
                                    agent_msg->tic.z);
            Matrix3d ric = Quaterniond(agent_msg->ric.w,
                                       agent_msg->ric.x,
                                       agent_msg->ric.y,
                                       agent_msg->ric.z).toRotationMatrix();

            vector<cv::Point3f> point_3d;  
            vector<cv::Point2f> feature_2d;
            vector<BRIEF::bitset> feature_descriptors, point_descriptors;

            
            for (unsigned int i = 0; i < agent_msg->point_3d.size(); i++)
            {
                cv::Point3f p_3d;
                p_3d.x = agent_msg->point_3d[i].x;
                p_3d.y = agent_msg->point_3d[i].y;
                p_3d.z = agent_msg->point_3d[i].z;
                point_3d.push_back(p_3d);
            }
            for (unsigned int i = 0; i < agent_msg->feature_2d.size(); i++)
            {
                cv::Point2f p_2d;
                p_2d.x = agent_msg->feature_2d[i].x;
                p_2d.y = agent_msg->feature_2d[i].y;
                feature_2d.push_back(p_2d);
            }

            for (unsigned int i = 0; i < agent_msg->point_des.size(); i = i + 4)
            {
                boost::dynamic_bitset<> tmp_brief(256);
                for (int k = 0; k < 4; k++)
                {
                    unsigned long long int tmp_int = agent_msg->point_des[i + k];
                    for (int j = 0; j < 64; ++j, tmp_int >>= 1)
                    {
                        tmp_brief[256 - 64 * (k + 1) + j] = (tmp_int & 1);
                    }
                } 
                point_descriptors.push_back(tmp_brief);
            } 

            for (unsigned int i = 0; i < agent_msg->feature_des.size(); i = i + 4)
            {
                boost::dynamic_bitset<> tmp_brief(256);
                for (int k = 0; k < 4; k++)
                {
                    unsigned long long int tmp_int = agent_msg->feature_des[i + k];
                    for (int j = 0; j < 64; ++j, tmp_int >>= 1)
                    {
                        tmp_brief[256 - 64 * (k + 1) + j] = (tmp_int & 1);
                    }
                } 
                feature_descriptors.push_back(tmp_brief);
                //cout << i / 4 << "  "<< tmp_brief << endl;
            } 

                    
            // std::cout << ">> 构造 KeyFrame 的参数：\n";
            // std::cout << " sequence       = " << sequence   << "\n";
            // std::cout << " time_stamp     = " << time_stamp << "\n";
            // std::cout << " T = [" 
            //         << T.x() << ", " << T.y() << ", " << T.z() << "]\n";
            // std::cout << " R = \n";
            // for (int i = 0; i < 3; ++i) {
            // std::cout << "    [" 
            //             << R(i,0) << ", "
            //             << R(i,1) << ", "
            //             << R(i,2) << "]\n";
            // }
            // std::cout << " tic = [" 
            //         << tic.x() << ", " << tic.y() << ", " << tic.z() << "]\n";
            // std::cout << " ric = \n";
            // for (int i = 0; i < 3; ++i) {
            // std::cout << "    [" 
            //             << ric(i,0) << ", "
            //             << ric(i,1) << ", "
            //             << ric(i,2) << "]\n";
            // }
            // std::cout << " #3D points = "    << point_3d.size()       << "\n";
            // std::cout << " #2D features = "  << feature_2d.size()     << "\n";
            // std::cout << " #3D descrs = "    << point_descriptors.size()   << "\n";
            // std::cout << " #2D descrs = "    << feature_descriptors.size() << "\n";


            KeyFrame* keyframe = new KeyFrame(sequence, time_stamp, T, R, tic, ric, point_3d, feature_2d, 
                                             point_descriptors, feature_descriptors);      
 

            m_process.lock();
            posegraph->addAgentFrame(keyframe);
            t_agent += t_addframe.toc();
            frame_cnt++;
            //printf("add agent frame time %f\n",t_agent / frame_cnt);
            m_process.unlock();
        }
        
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}


void command()
{
    while(1)
    {
        char c = getchar();
        if (c == 's')
        {
            m_process.lock();
            posegraph->savePoseGraph();
            m_process.unlock();
            printf("save pose graph finish\n you can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
            //printf("program shutting down...\n");
            //ros::shutdown();
        }
        if(c  == 'l')
        {
            printf("load pose graph\n");
            m_process.lock();
            posegraph->loadPoseGraph();
            m_process.unlock();
            printf("load pose graph finish\n");
        }
        if(c  == 'b')
        {
            printf("begin receive agent msg\n");
            start_flag = 1;
        }

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

void odom_callback(const nav_msgs::msg::Odometry::ConstPtr& msg)
{
    // Mesh model           
    //ROS_INFO("odometry callback");   
    int sequence = std::stoi(msg->child_frame_id);
    Quaterniond q(msg->pose.pose.orientation.w,
                    msg->pose.pose.orientation.x,
                    msg->pose.pose.orientation.y,
                    msg->pose.pose.orientation.z);
    Vector3d t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    auto buffer = posegraph->getBuffer();

    geometry_msgs::msg::TransformStamped transform_stamped;
    try {
    // 1) 从 buffer 拿到最新变换
    transform_stamped = buffer->lookupTransform("/global", "/drone_" + std::to_string(sequence), tf2::TimePointZero);

    // 2) 拆出平移和平移
    const auto &tr = transform_stamped.transform.translation;
    const auto &ro = transform_stamped.transform.rotation;
    Eigen::Vector3d trans(tr.x, tr.y, tr.z);
    Eigen::Quaterniond rot(ro.w, ro.x, ro.y, ro.z);

    // 3) 用 rot 和 trans 更新 q, t
    q = rot * q;
    t = rot * t + trans;
    }


    catch (const tf2::TransformException & ex) {
        //ROS_WARN("no %d transform yet", sequence);
        //return;
    }
    //ROS_WARN("read transform success!");

    Vector3d ypr = Utility::R2ypr(q.toRotationMatrix());
    ypr(0)    += 90.0*3.14159/180.0;
    q          = Utility::ypr2R(ypr); 
        
    meshROS.header.frame_id = string("/world");
    meshROS.header.stamp = msg->header.stamp; 
    meshROS.ns = "mesh";
    meshROS.id = sequence;
    meshROS.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
    meshROS.action = visualization_msgs::msg::Marker::ADD;
    meshROS.pose.position.x = t.x();
    meshROS.pose.position.y = t.y();
    meshROS.pose.position.z = t.z();
    meshROS.pose.orientation.w = 1;
    meshROS.pose.orientation.x = 0;
    meshROS.pose.orientation.y = 0;
    meshROS.pose.orientation.z = 0;
    meshROS.scale.x = 1;
    meshROS.scale.y = 1;
    meshROS.scale.z = 1;

    meshROS.color.a = 1.0;
    if (sequence == 1)
    {
        meshROS.color.r = 0.0;
        meshROS.color.g = 1.0;
        meshROS.color.b = 0.0;
    }
    else if(sequence == 2)
    {
        meshROS.color.r = 1.0;
        meshROS.color.g = 0.5;
        meshROS.color.b = 0.0;
    }
    else if(sequence == 3)
    {
        meshROS.color.r = 1.0;
        meshROS.color.g = 0.0;
        meshROS.color.b = 0.0;
    }
    else if(sequence == 4)
    {
        meshROS.color.r = 1.0;
        meshROS.color.g = 1.0;
        meshROS.color.b = 1.0;
    }
    else if(sequence == 5)
    {
        meshROS.color.r = 0.0;
        meshROS.color.g = 0.5;
        meshROS.color.b = 1.0;
    }

    meshROS.mesh_resource = mesh_resource;
    meshPub->publish(meshROS);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto n = std::make_shared<PoseGraph>();
    posegraph = n.get();
    posegraph->initROS();
    posegraph->registerPub(n);

    // read param
     // 1 声明参数（给个合理默认值）
    n->declare_parameter<int>("visualization_shift_x", 0);
    n->declare_parameter<int>("visualization_shift_y", 0);
    n->declare_parameter<double>("skip_dis", 1.0);
    n->declare_parameter<std::string>("pose_graph_save_path", std::string(""));
    n->declare_parameter<std::string>(
        "mesh_resource",
        std::string("package://pose_graph/meshes/hummingbird.mesh"));

    // 2 读取参数
    n->get_parameter("visualization_shift_x", VISUALIZATION_SHIFT_X);
    n->get_parameter("visualization_shift_y", VISUALIZATION_SHIFT_Y);
    n->get_parameter("skip_dis", SKIP_DIS);
    n->get_parameter("pose_graph_save_path", POSE_GRAPH_SAVE_PATH);
    n->get_parameter("mesh_resource", mesh_resource);

    // 3 打印确认
    //RCLCPP_INFO(n->get_logger(), "Params: shift_x=%d, shift_y=%d, skip_dis=%d", VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, SKIP_DIS);
    RCLCPP_INFO(n->get_logger(), "pose_graph_save_path='%s', mesh_resource='%s'", POSE_GRAPH_SAVE_PATH.c_str(), mesh_resource.c_str());

    std::string pkg_path = ament_index_cpp::get_package_share_directory("pose_graph");

    string vocabulary_file = pkg_path + "/support_files/brief_k10L6.bin";
    cout << "vocabulary_file" << vocabulary_file << endl;
    posegraph->loadVocabulary(vocabulary_file);

    BRIEF_PATTERN_FILE = pkg_path + "/support_files/brief_pattern.yml";
    cout << "BRIEF_PATTERN_FILE" << BRIEF_PATTERN_FILE << endl;

    n->declare_parameter<std::string>("pose_graph_result_path", "");
    n->get_parameter("pose_graph_result_path", VINS_RESULT_PATH);
    std::string pose_graph_path = VINS_RESULT_PATH + "/pose_graph_path.csv";
    ofstream loop_path_file_tmp(pose_graph_path, ios::out);
    loop_path_file_tmp.close();

    sub_agent_msg = n->create_subscription<agent_msg::msg::AgentMsg>("agent_frame", rclcpp::QoS(2000), std::bind(&agent_callback, std::placeholders::_1));
    sub_odom1 = n->create_subscription<nav_msgs::msg::Odometry>("/vins_1/odometry",  rclcpp::QoS(100),  odom_callback);
    sub_odom2 = n->create_subscription<nav_msgs::msg::Odometry>("/vins_2/odometry",  rclcpp::QoS(100),  odom_callback);
    sub_odom3 = n->create_subscription<nav_msgs::msg::Odometry>("/vins_3/odometry",  rclcpp::QoS(100),  odom_callback);
    sub_odom4 = n->create_subscription<nav_msgs::msg::Odometry>("/vins_4/odometry",  rclcpp::QoS(100),  odom_callback);
    meshPub = n->create_publisher<visualization_msgs::msg::Marker>("robot", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local());
    

    std::thread agent_frame_thread;
    agent_frame_thread = std::thread(agent_process);
    std::thread keyboard_command_process;
    keyboard_command_process = std::thread(command);

   
    rclcpp::spin(n);
    return 0;
}
