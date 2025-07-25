// feature_tracker.cpp (ROS 2 version, Camodocal 功能已注释)

#include "feature_tracker.h"
#include <rclcpp/rclcpp.hpp>   // ROS 2 logging and utilities
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <queue>
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>
int FeatureTracker::n_id = 0;

// Helper: check if a point lies inside the valid image border
static bool inBorder(const cv::Point2f &pt)
{
    constexpr int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return (BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE &&
            BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE);
}

// Helper: reduce a vector<cv::Point2f> by status mask
static void reduceVector(std::vector<cv::Point2f> &v, const std::vector<uchar> &status)
{
    int j = 0;
    for (int i = 0; i < static_cast<int>(v.size()); i++) {
        if (status[i]) {
            v[j++] = v[i];
        }
    }
    v.resize(j);
}

// Helper: reduce a vector<int> by status mask
static void reduceVector(std::vector<int> &v, const std::vector<uchar> &status)
{
    int j = 0;
    for (int i = 0; i < static_cast<int>(v.size()); i++) {
        if (status[i]) {
            v[j++] = v[i];
        }
    }
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if (FISHEYE) {
        mask = fisheye_mask.clone();
    } else {
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    }

    // Prefer to keep features that are tracked for a long time
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;
    for (unsigned int i = 0; i < forw_pts.size(); i++) {
        cnt_pts_id.push_back(std::make_pair(track_cnt[i], std::make_pair(forw_pts[i], ids[i])));                                           
    }

    std::sort(cnt_pts_id.begin(), cnt_pts_id.end(),
              [](const std::pair<int, std::pair<cv::Point2f, int>> &a,
                 const std::pair<int, std::pair<cv::Point2f, int>> &b) {
                  return a.first > b.first;
              });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id) {
        if (mask.at<uchar>(static_cast<int>(it.second.first.y),
                           static_cast<int>(it.second.first.x)) == 255) {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts) {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        RCLCPP_DEBUG(rclcpp::get_logger("FeatureTracker"),
                     "CLAHE costs: %fms", t_c.toc());
    } else {
        img = _img;
    }

    if (forw_img.empty()) {
        prev_img = cur_img = forw_img = img;
    } else {
        forw_img = img;
    }

    forw_pts.clear();

    if (!cur_pts.empty()) {
        TicToc t_o;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts,
                                 status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < static_cast<int>(forw_pts.size()); i++) {
            if (status[i] && !inBorder(forw_pts[i])) {
                status[i] = 0;
            }
        }
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        RCLCPP_DEBUG(rclcpp::get_logger("FeatureTracker"),
                     "temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt) {
        n++;
    }

    if (PUB_THIS_FRAME) {
        RCLCPP_INFO(rclcpp::get_logger("FeatureTracker"),
                     "PUB_THIS_FRAME=true rejectWithF begin");
        rejectWithF();
        RCLCPP_DEBUG(rclcpp::get_logger("FeatureTracker"),
                     "set mask begins");
        TicToc t_m;
        setMask();
        RCLCPP_DEBUG(rclcpp::get_logger("FeatureTracker"),
                     "set mask costs %fms", t_m.toc());

        RCLCPP_DEBUG(rclcpp::get_logger("FeatureTracker"),
                     "detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0) {
            if (mask.empty()) {
                RCLCPP_ERROR(rclcpp::get_logger("FeatureTracker"),
                             "mask is empty");
            }
            if (mask.type() != CV_8UC1) {
                RCLCPP_ERROR(rclcpp::get_logger("FeatureTracker"),
                             "mask type wrong");
            }
            if (mask.size() != forw_img.size()) {
                RCLCPP_ERROR(rclcpp::get_logger("FeatureTracker"),
                             "wrong size");
            }
            cv::goodFeaturesToTrack(forw_img, n_pts,
                                    static_cast<int>(n_max_cnt),
                                    0.01, MIN_DIST, mask);
        } else {
            n_pts.clear();
        }
        RCLCPP_DEBUG(rclcpp::get_logger("FeatureTracker"),
                     "detect feature costs: %fms", t_t.toc());

        RCLCPP_DEBUG(rclcpp::get_logger("FeatureTracker"),
                     "add feature begins");
        TicToc t_a;
        addPoints();
        RCLCPP_DEBUG(rclcpp::get_logger("FeatureTracker"),
                     "selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    RCLCPP_INFO(rclcpp::get_logger("FeatureTracker"), "rejectWithF: cur_pts=%zu, forw_pts=%zu", cur_pts.size(), forw_pts.size());

    if (forw_pts.size() >= 8) {
        RCLCPP_DEBUG(rclcpp::get_logger("FeatureTracker"),
                     "FM ransac begins");
        TicToc t_f;

        // Camodocal 功能：
         std::vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
         for (unsigned int i = 0; i < cur_pts.size(); i++) {
             Eigen::Vector3d tmp_p;
             m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
             un_cur_pts[i] = cv::Point2f(static_cast<float>(tmp_p.x() / tmp_p.z() * FOCAL_LENGTH + COL / 2.0),
                                         static_cast<float>(tmp_p.y() / tmp_p.z() * FOCAL_LENGTH + ROW / 2.0));
        
             m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
             un_forw_pts[i] = cv::Point2f(static_cast<float>(tmp_p.x() / tmp_p.z() * FOCAL_LENGTH + COL / 2.0),
                                          static_cast<float>(tmp_p.y() / tmp_p.z() * FOCAL_LENGTH + ROW / 2.0));
         }
        RCLCPP_INFO(rclcpp::get_logger("FeatureTracker"),
                     "camera_model begins");
        // Camodocal ：做 RANSAC
        std::vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = static_cast<int>(cur_pts.size());
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        RCLCPP_INFO(rclcpp::get_logger("FeatureTracker"),
                     "FM ransac: %d -> %lu: %f", size_a, forw_pts.size(),
                     1.0 * forw_pts.size() / size_a);
        RCLCPP_DEBUG(rclcpp::get_logger("FeatureTracker"),
                     "FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size()) {
        if (ids[i] == -1) {
            ids[i] = n_id++;
        }
        return true;
    } else {
        return false;
    }
}

void FeatureTracker::readIntrinsicParameter(const std::string &calib_file)
{
   
    // 原 Camodocal 调用已注释：
    
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    
    if (!m_camera) 
    {
        RCLCPP_ERROR(rclcpp::get_logger("FeatureTracker"),
        "FeatureTracker: Failed to load camera model from \"%s\"",
        calib_file.c_str());
        throw std::runtime_error("Failed to load camera model");
    }
}
/*
void FeatureTracker::showUndistortion(const std::string &name)
{
    

    // 若需恢复，请解注以下代码：
    
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    std::vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++) {
        for (int j = 0; j < ROW; j++) {
            Eigen::Vector3d b;
            m_camera->liftProjective(Eigen::Vector2d(i, j), b);
            distortedp.push_back(Eigen::Vector2d(i, j));
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
        }
    }
    for (int i = 0; i < static_cast<int>(undistortedp.size()); i++) {
        cv::Point2f pp;
        pp.x = static_cast<float>(undistortedp[i].x() * FOCAL_LENGTH + COL / 2);
        pp.y = static_cast<float>(undistortedp[i].y() * FOCAL_LENGTH + ROW / 2);
        int px = static_cast<int>(pp.x) + 300;
        int py = static_cast<int>(pp.y) + 300;
        if (px >= 0 && px < COL + 600 && py >= 0 && py < ROW + 600) {
            undistortedImg.at<uchar>(py, px) = cur_img.at<uchar>(
              static_cast<int>(distortedp[i].y()), static_cast<int>(distortedp[i].x()));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
    
}
*/
void FeatureTracker::showUndistortion(const std::string &name)//把去畸变前后像素点的对应关系一一对应
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    std::vector<Eigen::Vector2d> distortedp, undistortedp;//局部容器
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg); //利用opencv gui可视化工具展示 展示之后函数结束就释放掉
    cv::waitKey(0);
}
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();

    // Camodocal 功能跳过：直接使用原始像素坐标作为“去畸点”
    /*for (unsigned int i = 0; i < cur_pts.size(); i++) {
        cur_un_pts.push_back(cur_pts[i]);
        cur_un_pts_map[ids[i]] = cur_pts[i];
        // 若需恢复 Camodocal，请解注以下代码：
        
        Eigen::Vector3d b;
        m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), b);
        cv::Point2f up(static_cast<float>(b.x() / b.z()), static_cast<float>(b.y() / b.z()));
        cur_un_pts.push_back(up);
        cur_un_pts_map[ids[i]] = up;
        
    }
    */
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(std::make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }

    /*
    pts_velocity.clear();
    if (!prev_un_pts_map.empty()) {
        double dt = cur_time - prev_time;
        for (unsigned int i = 0; i < cur_un_pts.size(); i++) {
            if (ids[i] != -1) {
                auto it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end()) {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(static_cast<float>(v_x),
                                                       static_cast<float>(v_y)));
                } else {
                    pts_velocity.push_back(cv::Point2f(0, 0));
                }
            } else {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    } else {
        for (unsigned int i = 0; i < cur_pts.size(); i++) {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    
    prev_un_pts_map = cur_un_pts_map;*/
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
