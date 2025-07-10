// feature_tracker_node_ros2.cpp
#include "feature_tracker.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.h>
#include "parameters.h"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>


#define SHOW_UNDISTORTION 0

using std::placeholders::_1;

namespace feature_tracker_ros2
{

// Global vectors and variables migrated from ROS 1 version
static std::vector<uchar> r_status;
static std::vector<float> r_err;
static std::queue<sensor_msgs::msg::Image::ConstSharedPtr> img_buf;
static bool first_image_flag = true;
static double first_image_time = 0.0;
static double last_image_time = 0.0;
static int pub_count = 1;
static bool init_pub = false;


FeatureTracker trackerData[NUM_OF_CAM];

class FeatureTrackerNode : public rclcpp::Node
{
public:
  FeatureTrackerNode()
  : Node("feature_tracker")
  {
    // Set logger level to INFO
    
  
    // Read parameters (migrated from readParameters)
    readParameters(this);

    // Initialize intrinsic parameters for each camera
    for (int i = 0; i < NUM_OF_CAM; i++) {
      trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);
    }

    // If fisheye, load masks
    if (FISHEYE) {
      for (int i = 0; i < NUM_OF_CAM; i++) {
        trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
        if (!trackerData[i].fisheye_mask.data) {
          RCLCPP_INFO(this->get_logger(), "load mask fail");
          rclcpp::shutdown();
          return;
        } else {
          RCLCPP_INFO(this->get_logger(), "load mask success");
        }
      }
    }

    // Publishers
    pub_pc2_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("feature_tracker/feature", 1000);
    pub_match_ = this->create_publisher<sensor_msgs::msg::Image>("feature_tracker/feature_img", 1000);
    pub_restart_ = this->create_publisher<std_msgs::msg::Bool>("feature_tracker/restart", 1000);

    // Subscriber
    sub_img_ = this->create_subscription<sensor_msgs::msg::Image>(
      IMAGE_TOPIC, 1000, std::bind(&FeatureTrackerNode::img_callback, this, _1));

    RCLCPP_INFO(this->get_logger(), "FeatureTrackerNode initialized and subscribed to %s", IMAGE_TOPIC.c_str());
  }

private:
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pc2_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_match_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_restart_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
  bool init_pub = false;
  void img_callback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
  {
    // Handle first image logic
    if (first_image_flag) {
      first_image_flag = false;
      first_image_time = img_msg->header.stamp.sec + img_msg->header.stamp.nanosec * 1e-9;
      last_image_time = first_image_time;
      return;
    }

    double current_time = img_msg->header.stamp.sec + img_msg->header.stamp.nanosec * 1e-9;
    // Detect unstable camera stream
    if (current_time - last_image_time > 1.0 || current_time < last_image_time) {
      RCLCPP_WARN(this->get_logger(), "image discontinue! reset the feature tracker!");
      first_image_flag = true;
      last_image_time = 0.0;
      pub_count = 1;
      auto restart_flag = std::make_unique<std_msgs::msg::Bool>();
      restart_flag->data = true;
      pub_restart_->publish(std::move(restart_flag));
      return;
    }
    last_image_time = current_time;

    // Frequency control
    if (std::round(1.0 * pub_count / (current_time - first_image_time)) <= FREQ) {
      PUB_THIS_FRAME = true;
      if (std::fabs(1.0 * pub_count / (current_time - first_image_time) - FREQ) < 0.01 * FREQ) {
        first_image_time = current_time;
        pub_count = 0;
      }
    } else {
      PUB_THIS_FRAME = false;
    }

    // Convert to OpenCV Mono8
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1") {
      sensor_msgs::msg::Image img;
      img.header = img_msg->header;
      img.height = img_msg->height;
      img.width = img_msg->width;
      img.is_bigendian = img_msg->is_bigendian;
      img.step = img_msg->step;
      img.data = img_msg->data;
      img.encoding = "mono8";
      ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    } else {
      ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    }

    cv::Mat show_img = ptr->image;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++) {
      RCLCPP_DEBUG(this->get_logger(), "processing camera %d", i);
      if (i != 1 || !STEREO_TRACK) {
        RCLCPP_INFO(rclcpp::get_logger("FeatureTracker"),
                     "readimage begins");
        trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), current_time);
      } else {
        if (EQUALIZE) {
          cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
          clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
        } else {
          trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }
      }
#if SHOW_UNDISTORTION
      trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    for (unsigned int i = 0;; i++) {
      bool completed = false;
      for (int j = 0; j < NUM_OF_CAM; j++) {
        if (j != 1 || !STEREO_TRACK) {
          completed |= trackerData[j].updateID(i);
        }
      }
      if (!completed) {
        break;
      }
    }

    if (PUB_THIS_FRAME) {
      pub_count++;

      // 1) 收集所有特征点
      struct Pt { float x,y,z,id,u,v,vx,vy; };
      std::vector<Pt> all;
      for (int cam = 0; cam < NUM_OF_CAM; cam++) 
      {
        auto &un_pts  = trackerData[cam].cur_un_pts;
        auto &cur_pts = trackerData[cam].cur_pts;
        auto &ids     = trackerData[cam].ids;
        auto &vel     = trackerData[cam].pts_velocity;
        for (size_t j = 0; j < ids.size(); j++) 
        {
          if (trackerData[cam].track_cnt[j] > 1) 
          {
            Pt p;
            p.x  = un_pts[j].x;
            p.y  = un_pts[j].y;
            p.z  = 1.0f;
            p.id = ids[j] * NUM_OF_CAM + cam;
            p.u  = cur_pts[j].x;
            p.v  = cur_pts[j].y;
            p.vx = vel[j].x;
            p.vy = vel[j].y;
            all.push_back(p);
          }
        }
      }

      // 2) 构造 PointCloud2 消息
      sensor_msgs::msg::PointCloud2 pc2;
      pc2.header         = img_msg->header;
      pc2.header.frame_id= "world";
      pc2.height         = 1;
      pc2.width          = all.size();
      pc2.is_dense       = false;
      pc2.is_bigendian   = false;
      pc2.point_step     = 32;  // 8 个 float32，每个 4 字节
      pc2.row_step       = pc2.point_step * pc2.width;
      pc2.data.resize(pc2.row_step);
      pc2.fields.clear();
      {
        sensor_msgs::msg::PointField pf;
        pf.datatype = sensor_msgs::msg::PointField::FLOAT32;
        pf.count = 1;
        // 按顺序 push_back
        pf.name = "x";  pf.offset =  0; pc2.fields.push_back(pf);
        pf.name = "y";  pf.offset =  4; pc2.fields.push_back(pf);
        pf.name = "z";  pf.offset =  8; pc2.fields.push_back(pf);
        pf.name = "id"; pf.offset = 12; pc2.fields.push_back(pf);
        pf.name = "u";  pf.offset = 16; pc2.fields.push_back(pf);
        pf.name = "v";  pf.offset = 20; pc2.fields.push_back(pf);
        pf.name = "vx"; pf.offset = 24; pc2.fields.push_back(pf);
        pf.name = "vy"; pf.offset = 28; pc2.fields.push_back(pf);
      }

      // 3) 填充数据
      sensor_msgs::PointCloud2Iterator<float> it_x(pc2, "x");
      sensor_msgs::PointCloud2Iterator<float> it_y(pc2, "y");
      sensor_msgs::PointCloud2Iterator<float> it_z(pc2, "z");
      sensor_msgs::PointCloud2Iterator<float> it_id(pc2, "id");
      sensor_msgs::PointCloud2Iterator<float> it_u(pc2, "u");
      sensor_msgs::PointCloud2Iterator<float> it_v(pc2, "v");
      sensor_msgs::PointCloud2Iterator<float> it_vx(pc2, "vx");
      sensor_msgs::PointCloud2Iterator<float> it_vy(pc2, "vy");
      for (auto &p : all) {
        *it_x   = p.x;  ++it_x;
        *it_y   = p.y;  ++it_y;
        *it_z   = p.z;  ++it_z;
        *it_id  = p.id; ++it_id;
        *it_u   = p.u;  ++it_u;
        *it_v   = p.v;  ++it_v;
        *it_vx  = p.vx; ++it_vx;
        *it_vy  = p.vy; ++it_vy;
      }

      // 4) 发布
      if (!init_pub) 
      {
        init_pub = true;
      } else {
        pub_pc2_->publish(pc2);
      }
   
      if (SHOW_TRACK) 
      {
        ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
        cv::Mat stereo_img = ptr->image;

        for (int i = 0; i < NUM_OF_CAM; i++) 
        {
          cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
          cv::cvtColor(show_img, tmp_img, cv::COLOR_GRAY2RGB);

          for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++) 
          {
            double len = std::min(1.0, trackerData[i].track_cnt[j] * 1.0 / WINDOW_SIZE);
            cv::circle(tmp_img,
                        trackerData[i].cur_pts[j],
                        2,
                        cv::Scalar(255 * (1 - len), 0, 255 * len),
                        2);
          }
        }
        auto msg = ptr->toImageMsg();   // msg 是 sensor_msgs::msg::Image
        pub_match_->publish(*msg);
      }
    }  
  }
};

}  // namespace feature_tracker_ros2

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<feature_tracker_ros2::FeatureTrackerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

/*
#include "feature_tracker.h"
#include "parameters.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>

#define SHOW_UNDISTORTION 0

using std::placeholders::_1;

// 全局状态变量
static bool first_image_flag = true;
static double first_image_time = 0.0;
static double last_image_time = 0.0;
static int pub_count = 1;
static bool init_pub = false;

// 特征跟踪器实例
FeatureTracker trackerData[NUM_OF_CAM];

// ROS 2 发布者
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pc2;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_match;
rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_restart;

void img_callback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
    // 首帧处理
    if (first_image_flag) {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.sec + img_msg->header.stamp.nanosec * 1e-9;
        last_image_time = first_image_time;
        return;
    }
    double current_time = img_msg->header.stamp.sec + img_msg->header.stamp.nanosec * 1e-9;
    // 检测图像中断
    if (current_time - last_image_time > 1.0 || current_time < last_image_time) {
        RCLCPP_WARN(rclcpp::get_logger("feature_tracker"), "image discontinue! reset the feature tracker!");
        first_image_flag = true;
        last_image_time = 0.0;
        pub_count = 1;
        auto restart_flag = std::make_shared<std_msgs::msg::Bool>();
        restart_flag->data = true;
        pub_restart->publish(*restart_flag);
        return;
    }
    last_image_time = current_time;

    // 频率控制
    bool PUB_THIS_FRAME = false;
    if (std::round(1.0 * pub_count / (current_time - first_image_time)) <= FREQ) {
        PUB_THIS_FRAME = true;
        if (std::fabs(1.0 * pub_count / (current_time - first_image_time) - FREQ) < 0.01 * FREQ) {
            first_image_time = current_time;
            pub_count = 0;
        }
    }

    // 转为 Mono8
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1") {
        sensor_msgs::msg::Image mono_msg;
        mono_msg.header = img_msg->header;
        mono_msg.height = img_msg->height;
        mono_msg.width = img_msg->width;
        mono_msg.is_bigendian = img_msg->is_bigendian;
        mono_msg.step = img_msg->step;
        mono_msg.data = img_msg->data;
        mono_msg.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(mono_msg, sensor_msgs::image_encodings::MONO8);
    } else {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    }

    cv::Mat show_img = ptr->image;
    TicToc t_r;
    // 多相机处理
    for (int i = 0; i < NUM_OF_CAM; ++i) {
        RCLCPP_DEBUG(rclcpp::get_logger("feature_tracker"), "processing camera %d", i);
        if (i != 1 || !STEREO_TRACK) {
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), current_time);
        } else {
            if (EQUALIZE) {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            } else {
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
            }
        }
#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistortion_" + std::to_string(i));
#endif
    }
    // 更新 ID
    for (unsigned int iter = 0;; ++iter) {
        bool completed = false;
        for (int cam = 0; cam < NUM_OF_CAM; ++cam) {
            if (cam != 1 || !STEREO_TRACK)
                completed |= trackerData[cam].updateID(iter);
        }
        if (!completed) break;
    }

    if (PUB_THIS_FRAME) {
        ++pub_count;
        // 收集特征点
        struct Pt { float x,y,z,id,u,v,vx,vy; };
        std::vector<Pt> all;
        for (int cam = 0; cam < NUM_OF_CAM; ++cam) {
            auto &un_pts  = trackerData[cam].cur_un_pts;
            auto &cur_pts = trackerData[cam].cur_pts;
            auto &ids     = trackerData[cam].ids;
            auto &vel     = trackerData[cam].pts_velocity;
            for (size_t j = 0; j < ids.size(); ++j) {
                if (trackerData[cam].track_cnt[j] > 1) {
                    Pt p;
                    p.x  = un_pts[j].x;
                    p.y  = un_pts[j].y;
                    p.z  = 1.0f;
                    p.id = ids[j] * NUM_OF_CAM + cam;
                    p.u  = cur_pts[j].x;
                    p.v  = cur_pts[j].y;
                    p.vx = vel[j].x;
                    p.vy = vel[j].y;
                    all.push_back(p);
                }
            }
        }
        // 构造 PointCloud2
        sensor_msgs::msg::PointCloud2 pc2;
        pc2.header         = img_msg->header;
        pc2.header.frame_id= "world";
        pc2.height         = 1;
        pc2.width          = all.size();
        pc2.is_dense       = false;
        pc2.is_bigendian   = false;
        pc2.point_step     = 32;
        pc2.row_step       = pc2.point_step * pc2.width;
        pc2.data.resize(pc2.row_step);

        // 定义字段
        pc2.fields = std::vector<sensor_msgs::msg::PointField>();
        for (int idx = 0; idx < 8; ++idx) {
            sensor_msgs::msg::PointField pf;
            pf.datatype = sensor_msgs::msg::PointField::FLOAT32;
            pf.count    = 1;
            switch (idx) {
                case 0: pf.name = "x";  pf.offset =  0; break;
                case 1: pf.name = "y";  pf.offset =  4; break;
                case 2: pf.name = "z";  pf.offset =  8; break;
                case 3: pf.name = "id"; pf.offset = 12; break;
                case 4: pf.name = "u";  pf.offset = 16; break;
                case 5: pf.name = "v";  pf.offset = 20; break;
                case 6: pf.name = "vx"; pf.offset = 24; break;
                case 7: pf.name = "vy"; pf.offset = 28; break;
            }
            
          }
        // 填充数据
        sensor_msgs::PointCloud2Iterator<float> it_x(pc2, "x");
        sensor_msgs::PointCloud2Iterator<float> it_y(pc2, "y");
        sensor_msgs::PointCloud2Iterator<float> it_z(pc2, "z");
        sensor_msgs::PointCloud2Iterator<float> it_id(pc2, "id");
        sensor_msgs::PointCloud2Iterator<float> it_u(pc2, "u");
        sensor_msgs::PointCloud2Iterator<float> it_v(pc2, "v");
        sensor_msgs::PointCloud2Iterator<float> it_vx(pc2, "vx");
        sensor_msgs::PointCloud2Iterator<float> it_vy(pc2, "vy");
        for (auto &p : all) {
            *it_x = p.x;  ++it_x;
            *it_y = p.y;  ++it_y;
            *it_z = p.z;  ++it_z;
            *it_id = p.id; ++it_id;
            *it_u = p.u;  ++it_u;
            *it_v = p.v;  ++it_v;
            *it_vx = p.vx; ++it_vx;
            *it_vy = p.vy; ++it_vy;
        }
        pub_pc2->publish(pc2);
      }

    if (SHOW_TRACK) {
        auto color_ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
        cv::Mat stereo_img = color_ptr->image;
        for (int i = 0; i < NUM_OF_CAM; ++i) {
            cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
            cv::cvtColor(show_img, tmp_img, cv::COLOR_GRAY2RGB);
            for (size_t j = 0; j < trackerData[i].cur_pts.size(); ++j) {
                double len = std::min(1.0, trackerData[i].track_cnt[j] / (double)WINDOW_SIZE);
                cv::circle(tmp_img, trackerData[i].cur_pts[j], 2,
                           cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
            }
        }
        pub_match->publish(*color_ptr->toImageMsg());
    }
}


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("feature_tracker");
    // 读取参数并初始化
    readParameters(node.get());
    for (int i = 0; i < NUM_OF_CAM; ++i)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);
    if (FISHEYE) {
        for (int i = 0; i < NUM_OF_CAM; ++i) {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if (!trackerData[i].fisheye_mask.data) {
                RCLCPP_ERROR(node->get_logger(), "load mask fail");
                return 1;
            }
        }
    }
    // 创建发布者和订阅者
    pub_pc2    = node->create_publisher<sensor_msgs::msg::PointCloud2>("feature_tracker/feature", 1000);
    pub_match  = node->create_publisher<sensor_msgs::msg::Image>(     "feature_tracker/feature_img",1000);
    pub_restart= node->create_publisher<std_msgs::msg::Bool>(         "feature_tracker/restart", 1000);
    node->create_subscription<sensor_msgs::msg::Image>(IMAGE_TOPIC,1000, img_callback);
    RCLCPP_INFO(node->get_logger(), "Feature tracker node started, subscribed to %s", IMAGE_TOPIC.c_str());
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
*/