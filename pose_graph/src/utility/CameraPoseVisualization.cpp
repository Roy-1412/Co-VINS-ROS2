#include "CameraPoseVisualization.h"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>

// 静态常量的定义
const Eigen::Vector3d CameraPoseVisualization::imlt = Eigen::Vector3d(-1.0, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::imrt = Eigen::Vector3d( 1.0, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::imlb = Eigen::Vector3d(-1.0,  0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::imrb = Eigen::Vector3d( 1.0,  0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::lt0  = Eigen::Vector3d(-0.7, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::lt1  = Eigen::Vector3d(-0.7, -0.2, 1.0);
const Eigen::Vector3d CameraPoseVisualization::lt2  = Eigen::Vector3d(-1.0, -0.2, 1.0);
const Eigen::Vector3d CameraPoseVisualization::oc   = Eigen::Vector3d( 0.0,  0.0, 0.0);

// 辅助函数：Eigen→geometry_msgs::msg::Point
static void Eigen2Point(const Eigen::Vector3d &v, geometry_msgs::msg::Point &p) {
  p.x = v.x();
  p.y = v.y();
  p.z = v.z();
}

CameraPoseVisualization::CameraPoseVisualization(float r, float g, float b, float a)
  : m_marker_ns("CameraPoseVisualization"),
    m_scale(0.2),
    m_line_width(0.01),
    m_loop_edge_count(20)
{
  m_image_boundary_color.r = r;
  m_image_boundary_color.g = g;
  m_image_boundary_color.b = b;
  m_image_boundary_color.a = a;

  m_optical_center_connector_color = m_image_boundary_color;
}

void CameraPoseVisualization::setImageBoundaryColor(float r, float g, float b, float a) {
  m_image_boundary_color.r = r;
  m_image_boundary_color.g = g;
  m_image_boundary_color.b = b;
  m_image_boundary_color.a = a;
}

void CameraPoseVisualization::setOpticalCenterConnectorColor(float r, float g, float b, float a) {
  m_optical_center_connector_color.r = r;
  m_optical_center_connector_color.g = g;
  m_optical_center_connector_color.b = b;
  m_optical_center_connector_color.a = a;
}

void CameraPoseVisualization::setScale(double s) {
  m_scale = s;
}

void CameraPoseVisualization::setLineWidth(double width) {
  m_line_width = width;
}

void CameraPoseVisualization::addEdge(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1) {
  visualization_msgs::msg::Marker marker;
  marker.ns        = m_marker_ns;
  marker.id        = static_cast<int>(m_markers.size()) + 1;
  marker.type      = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action    = visualization_msgs::msg::Marker::ADD;
  marker.scale.x   = 0.01;
  marker.color.b   = 1.0f;
  marker.color.a   = 1.0f;
  // 将 lifetime 清零
  marker.lifetime.sec = 0;
  marker.lifetime.nanosec = 0;

  geometry_msgs::msg::Point pt0, pt1;
  Eigen2Point(p0, pt0);
  Eigen2Point(p1, pt1);

  marker.points.push_back(pt0);
  marker.points.push_back(pt1);

  m_markers.push_back(marker);
}

void CameraPoseVisualization::addLoopEdge(const Eigen::Vector3d& p0,
                                          const Eigen::Vector3d& p1,
                                          int color_mode)
{
  visualization_msgs::msg::Marker marker;
  marker.ns       = m_marker_ns;
  marker.id       = static_cast<int>(m_markers.size()) + 1;
  marker.type     = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action   = visualization_msgs::msg::Marker::ADD;
  // 将 lifetime 清零
  marker.lifetime.sec = 0;
  marker.lifetime.nanosec = 0;
  marker.scale.x  = 0.02;

  // 不同 color_mode 可自定义色彩
  if (color_mode == 0) {
    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.a = 1.0f;
  } else {
    marker.color.r = 1.0f;
    marker.color.a = 1.0f;
  }

  geometry_msgs::msg::Point pt0, pt1;
  Eigen2Point(p0, pt0);
  Eigen2Point(p1, pt1);
  marker.points.push_back(pt0);
  marker.points.push_back(pt1);

  m_markers.push_back(marker);
}

void CameraPoseVisualization::addPose(const Eigen::Vector3d& p,
                                      const Eigen::Quaterniond& q)
{
  visualization_msgs::msg::Marker marker;
  marker.ns       = m_marker_ns;
  marker.id       = 0;
  marker.type     = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action   = visualization_msgs::msg::Marker::ADD;
  marker.scale.x  = m_line_width;
  // 将 lifetime 清零
  marker.lifetime.sec = 0;
  marker.lifetime.nanosec = 0;

  // 无需额外 pose.transform，直接在点上累积 p, q
  geometry_msgs::msg::Point lt, lb, rt, rb, oc_pt, t0, t1, t2;
  Eigen2Point(q * (m_scale * imlt) + p, lt);
  Eigen2Point(q * (m_scale * imlb) + p, lb);
  Eigen2Point(q * (m_scale * imrt) + p, rt);
  Eigen2Point(q * (m_scale * imrb) + p, rb);
  Eigen2Point(q * (m_scale * lt0 ) + p, t0);
  Eigen2Point(q * (m_scale * lt1 ) + p, t1);
  Eigen2Point(q * (m_scale * lt2 ) + p, t2);
  Eigen2Point(q * (m_scale * oc  ) + p, oc_pt);

  // 图像边界
  auto push_edge = [&](const geometry_msgs::msg::Point& a,
                       const geometry_msgs::msg::Point& b,
                       const std_msgs::msg::ColorRGBA& c) {
    marker.points.push_back(a);
    marker.points.push_back(b);
    marker.colors.push_back(c);
    marker.colors.push_back(c);
  };

  push_edge(lt, lb, m_image_boundary_color);
  push_edge(lb, rb, m_image_boundary_color);
  push_edge(rb, rt, m_image_boundary_color);
  push_edge(rt, lt, m_image_boundary_color);
  // 左上角指示
  push_edge(t0, t1, m_image_boundary_color);
  push_edge(t1, t2, m_image_boundary_color);
  // 光心连线
  push_edge(lt, oc_pt, m_optical_center_connector_color);
  push_edge(lb, oc_pt, m_optical_center_connector_color);
  push_edge(rt, oc_pt, m_optical_center_connector_color);
  push_edge(rb, oc_pt, m_optical_center_connector_color);

  m_markers.push_back(marker);
}

void CameraPoseVisualization::reset() {
  m_markers.clear();
}

void CameraPoseVisualization::publishBy(
  const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& pub,
  const std_msgs::msg::Header& header)
{
  visualization_msgs::msg::MarkerArray arr;
  for (auto &marker : m_markers) {
    marker.header = header;
    arr.markers.push_back(marker);
  }
  pub->publish(arr);
}
