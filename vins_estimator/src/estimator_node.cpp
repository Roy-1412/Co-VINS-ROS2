#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>

// ROS 2 头文件
#include <cassert>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/bool.hpp>
#include <agent_msg/msg/agent_msg.hpp>

#include <std_msgs/msg/header.hpp> 
#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

using std::placeholders::_1;

Estimator estimator;

std::condition_variable con;
double current_time = -1;
std::queue<sensor_msgs::msg::Imu::ConstSharedPtr>      imu_buf;
std::queue<sensor_msgs::msg::PointCloud2::ConstSharedPtr> feature_buf;
std::queue<sensor_msgs::msg::PointCloud2::ConstSharedPtr> relo_buf;
std::queue<sensor_msgs::msg::Image::ConstSharedPtr>    image_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex m_estimator;
std::mutex m_image_buf;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = false;
bool init_imu     = true;
double last_imu_t = 0;

camodocal::CameraPtr m_camera;
std::string BRIEF_PATTERN_FILE;
Eigen::Vector3d last_agent_t = Eigen::Vector3d::Zero();
std::queue<agent_msg::msg::AgentMsg> agent_msg_buf;
std::mutex m_agent_msg_buf;
rclcpp::Node::SharedPtr node;




void predict(const sensor_msgs::msg::Imu::ConstSharedPtr &imu_msg)
{
    double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * 1e-9;
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    // 拷贝一份缓冲区，原缓冲区不受影响
    std::queue<sensor_msgs::msg::Imu::ConstSharedPtr> tmp_imu_buf = imu_buf;

    // 用 while 循环显式弹出队头并处理
    while (!tmp_imu_buf.empty()) {
        auto tmp_imu_msg = tmp_imu_buf.front();
        tmp_imu_buf.pop();
        predict(tmp_imu_msg);
    }
}


std::vector<std::pair<std::vector<sensor_msgs::msg::Imu::ConstSharedPtr>,sensor_msgs::msg::PointCloud2::ConstSharedPtr>>
getMeasurements()
{
    std::vector<std::pair<
        std::vector<sensor_msgs::msg::Imu::ConstSharedPtr>,
        sensor_msgs::msg::PointCloud2::ConstSharedPtr>> measurements;

    while (true)
    {
        // 1) 如果任一队列为空，直接返回
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
        // 2) 检查最新的 IMU 是否已经过了下一帧特征（加上时间偏移）
        double last_imu_t = imu_buf.back()->header.stamp.sec
                          + imu_buf.back()->header.stamp.nanosec * 1e-9;
        double first_img_t = feature_buf.front()->header.stamp.sec
                           + feature_buf.front()->header.stamp.nanosec * 1e-9;
        if (!(last_imu_t > first_img_t + estimator.td))
        {
            // 还要等更多 IMU 到来
            sum_of_wait++;
            return measurements;
        }

        // 3) 丢弃时间过早的特征帧
        double first_imu_t = imu_buf.front()->header.stamp.sec
                           + imu_buf.front()->header.stamp.nanosec * 1e-9;
        if (!(first_imu_t < first_img_t + estimator.td))
        {
            RCLCPP_WARN(node->get_logger(),
                        "throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }

        // 4) 正式取出这帧图像和对应的 IMU
        auto img_msg = feature_buf.front();
        feature_buf.pop();
        std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> IMUs;

        // 4.1) 先把所有时间小于 (img_t + td) 的 IMU 全部弹出
        double img_t = img_msg->header.stamp.sec
                     + img_msg->header.stamp.nanosec * 1e-9;
        while (!imu_buf.empty())
        {
            double t = imu_buf.front()->header.stamp.sec
                     + imu_buf.front()->header.stamp.nanosec * 1e-9;
            if (t < img_t + estimator.td)
            {
                IMUs.emplace_back(imu_buf.front());
                imu_buf.pop();
            }
            else
            {
                break;
            }
        }

        // 4.2) 再取一个最靠近图像时刻的 IMU（如果还有的话）
        if (!imu_buf.empty())
        {
            IMUs.emplace_back(imu_buf.front());
        }
        else
        {
            RCLCPP_WARN(node->get_logger(),
                        "no imu between two image");
        }

        // 5) 把这一对数据推到 measurements
        measurements.emplace_back(IMUs, img_msg);
        
    }

    // unreachable
    return measurements;
}



void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg)
{
  // 严格按时间戳顺序处理
  double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * 1e-9;
 
  if (t <= last_imu_t) {
    RCLCPP_WARN(rclcpp::get_logger("vins_node"), "imu message in disorder!");
    return;
  }
  last_imu_t = t;

  // 缓存 IMU 数据
  {
    std::lock_guard<std::mutex> lg(m_buf);
    imu_buf.push(imu_msg);
    

  }
  con.notify_one();

  // 状态预测并发布
  {
    std::lock_guard<std::mutex> lg(m_state);
    predict(imu_msg);

    // 直接修改消息头，而不是复制成 const 引用
    auto hdr = imu_msg->header;
    hdr.frame_id = "world";
    pubLatestOdometry(tmp_P, tmp_Q, tmp_V, hdr);
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
      pubLatestOdometry(tmp_P, tmp_Q, tmp_V, imu_msg->header);
    }
  }
  

}

void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
    double t = img_msg->header.stamp.sec +
               img_msg->header.stamp.nanosec * 1e-9;
    m_image_buf.lock();
    image_buf.push(img_msg);
    m_image_buf.unlock();
 
}

void feature_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(msg);
    m_buf.unlock();
    con.notify_one();
}

void restart_callback(const std_msgs::msg::Bool::ConstSharedPtr restart_msg)
{
    if (restart_msg->data == true)
    {
        RCLCPP_WARN(node->get_logger(),"restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
    }
    return;
}

void relocalization_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
void process()
{
    RCLCPP_INFO(node->get_logger(),
                        "process begin");
  while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::msg::Imu::ConstSharedPtr>, sensor_msgs::msg::PointCloud2::ConstSharedPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        m_estimator.lock();
        
        for (auto &measurement : measurements)
        {
            TicToc t_s;
            auto img_msg = measurement.second;
            // —— 1. 字段检查 —— 
            static const std::vector<std::string> required_fields = {
            "id","x","y","z","u","v","vx","vy"
            };
            bool ok = true;
            for (const auto &fld : required_fields) {
            bool found = false;
            for (const auto &f : img_msg->fields) {
                if (f.name == fld) {
                found = true;
                break;
                }
            }
            if (!found) {
                RCLCPP_WARN(node->get_logger(),
                            "PointCloud2 缺少字段 '%s'，跳过该帧", fld.c_str());
                ok = false;
                break;
            }
            }
            if (!ok) {
            continue;  // 字段不全，直接跳过这一帧，下面的 iterator 都不会执行
            }
            // —— 2. IMU 预测部分（保持原样） —— 
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t     = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * 1e-9;
                double img_t = img_msg->header.stamp.sec + img_msg->header.stamp.nanosec * 1e-9;
                if (t <= img_t)
                {
                    if (current_time < 0) current_time = t;
                    double dt = t - current_time; assert(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, {dx,dy,dz}, {rx,ry,rz});
                }
                else
                {
                    double dt1 = img_t - current_time;
                    double dt2 = t - img_t;
                    assert(dt1 >= 0 && dt2 >= 0 && dt1 + dt2 > 0);
                    double w1 = dt2 / (dt1 + dt2), w2 = dt1 / (dt1 + dt2);
                    dx = w1*dx + w2*imu_msg->linear_acceleration.x;
                    dy = w1*dy + w2*imu_msg->linear_acceleration.y;
                    dz = w1*dz + w2*imu_msg->linear_acceleration.z;
                    rx = w1*rx + w2*imu_msg->angular_velocity.x;
                    ry = w1*ry + w2*imu_msg->angular_velocity.y;
                    rz = w1*rz + w2*imu_msg->angular_velocity.z;
                    estimator.processIMU(dt1, {dx,dy,dz}, {rx,ry,rz});
                    current_time = img_t;
                }
            }
            RCLCPP_INFO(node->get_logger(),
                        "processIMU finish");
            // —— 3. Relocalization （保持原样） —— 
            sensor_msgs::msg::PointCloud2::ConstSharedPtr relo_msg = nullptr;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg)
            {
                RCLCPP_INFO(node->get_logger(),
                        "start Relocalization ");
              double frame_stamp =
                relo_msg->header.stamp.sec +
                relo_msg->header.stamp.nanosec * 1e-9;

                // 3.2 提取匹配点坐标 (x, y, z)
                std::vector<Eigen::Vector3d> match_points;
                // 用同一对 begin/end 来遍历，防止越界
                auto it_x_begin = sensor_msgs::PointCloud2ConstIterator<float>(*relo_msg, "x");
                auto it_x_end   = it_x_begin.end(); 
                auto it_y = sensor_msgs::PointCloud2ConstIterator<float>(*relo_msg, "y");
                auto it_z = sensor_msgs::PointCloud2ConstIterator<float>(*relo_msg, "z");
                for (auto it_x = it_x_begin; it_x != it_x_end; ++it_x, ++it_y, ++it_z)
                {
                match_points.emplace_back(*it_x, *it_y, *it_z);
                }

                // 3.3 提取重定位的平移 (tx, ty, tz)
                sensor_msgs::PointCloud2ConstIterator<float> it_tx(*relo_msg, "tx");
                sensor_msgs::PointCloud2ConstIterator<float> it_ty(*relo_msg, "ty");
                sensor_msgs::PointCloud2ConstIterator<float> it_tz(*relo_msg, "tz");
                Eigen::Vector3d relo_t(*it_tx, *it_ty, *it_tz);

                // 3.4 提取重定位的旋转四元数 (qx, qy, qz, qw)
                sensor_msgs::PointCloud2ConstIterator<float> it_qx(*relo_msg, "qx");
                sensor_msgs::PointCloud2ConstIterator<float> it_qy(*relo_msg, "qy");
                sensor_msgs::PointCloud2ConstIterator<float> it_qz(*relo_msg, "qz");
                sensor_msgs::PointCloud2ConstIterator<float> it_qw(*relo_msg, "qw");
                Eigen::Quaterniond relo_q(*it_qx, *it_qy, *it_qz, *it_qw);
                Eigen::Matrix3d relo_r = relo_q.toRotationMatrix();

                // 3.5 提取帧索引 (frame_index)
                sensor_msgs::PointCloud2ConstIterator<int> it_idx(*relo_msg, "frame_index");
                int frame_index = *it_idx;

                // 3.6 调用重定位接口
                estimator.setReloFrame(
                frame_stamp,
                frame_index,
                match_points,
                relo_t,
                relo_r
                );
            }

            // —— 4. 图像点云处理 （修正迭代器循环条件） —— 
            std::map<int, std::vector<std::pair<int, Eigen::Matrix<double,7,1>>>> image;
            auto it_id_begin = sensor_msgs::PointCloud2ConstIterator<int>(*img_msg, "id");
            auto it_id_end   = it_id_begin.end(); 
            auto it_x2 = sensor_msgs::PointCloud2ConstIterator<float>(*img_msg, "x");
            auto it_y2 = sensor_msgs::PointCloud2ConstIterator<float>(*img_msg, "y");
            auto it_z2 = sensor_msgs::PointCloud2ConstIterator<float>(*img_msg, "z");
            auto it_pu = sensor_msgs::PointCloud2ConstIterator<float>(*img_msg, "u");
            auto it_pv = sensor_msgs::PointCloud2ConstIterator<float>(*img_msg, "v");
            auto it_vx = sensor_msgs::PointCloud2ConstIterator<float>(*img_msg, "vx");
            auto it_vy = sensor_msgs::PointCloud2ConstIterator<float>(*img_msg, "vy");

            for (auto it_id = it_id_begin; it_id != it_id_end;
                 ++it_id, ++it_x2, ++it_y2, ++it_z2,
                 ++it_pu, ++it_pv, ++it_vx, ++it_vy)
            {
                int v = static_cast<int>(*it_id + 0.5f);
                int feature_id = v / NUM_OF_CAM;
                int camera_id  = v % NUM_OF_CAM;
                Eigen::Matrix<double,7,1> xyz_uv_vel;
                xyz_uv_vel << *it_x2, *it_y2, *it_z2, *it_pu, *it_pv, *it_vx, *it_vy;
                image[feature_id].emplace_back(camera_id, xyz_uv_vel);
            }

            estimator.processImage(image, img_msg->header);
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);

            // —— 5. 发布结果 （保持原样） —— 
            std_msgs::msg::Header header = img_msg->header;
            header.frame_id = "world";
            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud2(estimator, header);
            pubTF(estimator, header);
            RCLCPP_INFO(node->get_logger(),">>> pubTF at t = %.9f",header.stamp.sec + header.stamp.nanosec * 1e-9);
            pubKeyframe(estimator);
            if (relo_msg) pubRelocalization(estimator);

            if (SWARM_AGENT)
            {
                if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == Estimator::MarginalizationFlag::MARGIN_OLD)
                {
                    Vector3d tmp_agent_t = estimator.Ps[WINDOW_SIZE - 2];
                    if ((tmp_agent_t - last_agent_t).norm() > 0.05)
                    {
                        TicToc pubAgentFrame_time;
                        agent_msg::msg::AgentMsg agent_frame_msg;
                        preprocessAgentFrame(estimator, agent_frame_msg);
                        m_agent_msg_buf.lock();
                        agent_msg_buf.push(agent_frame_msg);
                        m_agent_msg_buf.unlock();
                        //RCLCPP_WARN(node->get_logger(),"preprocess agent frame time %f", pubAgentFrame_time.toc());
                        last_agent_t = tmp_agent_t;
                    }
                }
            }
            
        }
       
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
        std::chrono::milliseconds dura(1);
        std::this_thread::sleep_for(dura);
    }
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
    sensor_msgs::msg::Image::ConstSharedPtr image_msg = nullptr;

    // 计算 agent_msg 的时间戳
    double agent_t = tmp_msg.header.stamp.sec
                   + tmp_msg.header.stamp.nanosec * 1e-9;

    // 从缓存中找出与 agent_msg 同步的图像
    {
        std::lock_guard<std::mutex> lg(m_image_buf);
        // 丢弃时间早于 agent_t 的旧图像
        while (!image_buf.empty()) {
            auto &hdr = image_buf.front()->header;
            double img_t = hdr.stamp.sec + hdr.stamp.nanosec * 1e-9;
            if (img_t < agent_t) {
                image_buf.pop();
            } else {
                break;
            }
        }
        if (!image_buf.empty()) {
            image_msg = image_buf.front();
        }
    }

    // 再次判断时间是否完全匹配
    double img_t = 0;
    if (image_msg) {
        img_t = image_msg->header.stamp.sec
              + image_msg->header.stamp.nanosec * 1e-9;
    }
    if (!image_msg || img_t != agent_t)
    {
        RCLCPP_WARN(node->get_logger(), "cannot find corresponding image for agent_msg @ %.9f", agent_t);
    }
    else
    {
        // 转换并发布
        cv_bridge::CvImageConstPtr ptr;
        if (image_msg->encoding == "8UC1")
        {
            sensor_msgs::msg::Image img;
            img.header       = image_msg->header;
            img.height       = image_msg->height;
            img.width        = image_msg->width;
            img.is_bigendian = image_msg->is_bigendian;
            img.step         = image_msg->step;
            img.data         = image_msg->data;
            img.encoding     = "mono8";
            ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
        }
        else
        {
            ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
        }

        cv::Mat img = ptr->image;
        RCLCPP_INFO(node->get_logger(),
            ">> Before pubAgentFrame: m_camera ptr = %p", m_camera.get());

        pubAgentFrame(tmp_msg, img, m_camera);
    }
}

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}


int main(int argc, char **argv)
{
 
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;
  node = rclcpp::Node::make_shared("vins_estimator");
  
  node->declare_parameter<std::string>("config_file", "");

  // —— 2. 再 get 一次 —— 
  std::string config_file;
  node->get_parameter("config_file", config_file);

  // —— 3. 检查是否真的传了值 —— 
  if (config_file.empty()) {
    RCLCPP_ERROR(node->get_logger(),
                 "config_file 参数为空！请在 launch 中传入：\n"
                 "  -p config_file:=/path/to/your/camera.yaml");
    return 1;  // 直接退出
  }

  RCLCPP_INFO(node->get_logger(),
              "Loaded config_file: %s", config_file.c_str());

  executor.add_node(node);
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(
    node,
    rclcpp::QoS(rclcpp::KeepLast(10)),
    rclcpp::PublisherOptions());
  RCLCPP_INFO(node->get_logger(), "TF Broadcaster initialized");

  // 参数读取
  
  readParameters(node);      // 直接传 SharedPtr
  estimator.setParameter();

  RCLCPP_WARN(node->get_logger(), "waiting for image and imu...");

  // 注册发布器
  registerPub(node);         // 直接传 SharedPtr

  // QoS 推荐使用传感器数据 QoS
  auto sensor_qos = rclcpp::SensorDataQoS();

  auto callback_group_imu = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto callback_group_feature = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto callback_group_restart = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto callback_group_relo = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // 创建订阅选项并绑定回调组
  rclcpp::SubscriptionOptions opt_imu, opt_feature, opt_restart, opt_relo;
  opt_imu.callback_group = callback_group_imu;
  opt_feature.callback_group = callback_group_feature;
  opt_restart.callback_group = callback_group_restart;
  opt_relo.callback_group = callback_group_relo;
  // 订阅器
  // IMU
    auto sub_imu = node->create_subscription<sensor_msgs::msg::Imu>(
    IMU_TOPIC, sensor_qos,
    [](sensor_msgs::msg::Imu::ConstSharedPtr msg) {
      imu_callback(msg);
    },
    opt_imu);
  /*auto sub_imu = node->create_subscription<sensor_msgs::msg::Imu>(
  IMU_TOPIC,
  rclcpp::SensorDataQoS().reliable(),
  [](sensor_msgs::msg::Imu::ConstSharedPtr msg) {
    imu_callback(msg);
  }
 );
 */

  // 视觉特征
 auto sub_feature = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    "feature_tracker/feature", sensor_qos,
    [](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      feature_callback(msg);
    },
    opt_feature);
  /*auto sub_feature = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    "feature_tracker/feature", sensor_qos,
    [](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      feature_callback(msg);
    }
  );
 */
  
 // 重启信号，用深度 10 的 QoS
  auto sub_restart = node->create_subscription<std_msgs::msg::Bool>(
    "feature_tracker/restart", rclcpp::QoS(10),
    [](std_msgs::msg::Bool::ConstSharedPtr msg) {
      restart_callback(msg);
    },
    opt_restart);
    
  /*auto sub_restart = node->create_subscription<std_msgs::msg::Bool>(
    "feature_tracker/restart", rclcpp::QoS(10),
    [](std_msgs::msg::Bool::ConstSharedPtr msg) {
      restart_callback(msg);
    }
  );
*/
    auto relo_qos = rclcpp::QoS(10);
    relo_qos.best_effort();
  // 重定位点云
 auto sub_relo = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/pose_graph/match_points", relo_qos,
    [](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      relocalization_callback(msg);
    },
    opt_relo);
  /*auto sub_relo = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/pose_graph/match_points", sensor_qos,
    [](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      relocalization_callback(msg);
    }
  );
  */

  // 启动处理线程
  std::thread measurement_thread{process};

  // 如果启用 swarm agent，再订阅图像
  rclcpp::Subscription<sensor_msgs::msg::Image>::ConstSharedPtr sub_image;
  std::thread agent_thread;
  if (SWARM_AGENT) {
    RCLCPP_INFO(node->get_logger(), "start swarm mode");
    sub_image = node->create_subscription<sensor_msgs::msg::Image>(
      IMAGE_TOPIC, sensor_qos,
      [](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        image_callback(msg);
      }
    );
    RCLCPP_INFO(rclcpp::get_logger("FeatureTracker"), "Attempt to load camera YAML at: %s", config_file.c_str());
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(config_file.c_str());
    RCLCPP_INFO(rclcpp::get_logger("FeatureTracker"), " -> result ptr = %p", m_camera.get());
    std::string pkg_path =
      ament_index_cpp::get_package_share_directory("vins_estimator");
    BRIEF_PATTERN_FILE = pkg_path + "/../support_files/brief_pattern.yml";
    agent_thread = std::thread(agent_process);
  }

 executor.spin();

  rclcpp::shutdown();

  if (measurement_thread.joinable()) measurement_thread.join();
  if (agent_thread.joinable()) agent_thread.join();
  return 0;

  //rclcpp::spin(node);
  //rclcpp::shutdown();
}
