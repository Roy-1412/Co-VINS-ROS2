#include "utility/CameraPoseVisualization.h"
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/header.hpp>

using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;
using geometry_msgs::msg::Point;
using std_msgs::msg::Header;

// Static camera frustum corners
const Eigen::Vector3d CameraPoseVisualization::imlt = {-1.0, -0.5, 1.0};
const Eigen::Vector3d CameraPoseVisualization::imrt = { 1.0, -0.5, 1.0};
const Eigen::Vector3d CameraPoseVisualization::imlb = {-1.0,  0.5, 1.0};
const Eigen::Vector3d CameraPoseVisualization::imrb = { 1.0,  0.5, 1.0};
const Eigen::Vector3d CameraPoseVisualization::lt0  = {-0.7, -0.5, 1.0};
const Eigen::Vector3d CameraPoseVisualization::lt1  = {-0.7, -0.2, 1.0};
const Eigen::Vector3d CameraPoseVisualization::lt2  = {-1.0, -0.2, 1.0};
const Eigen::Vector3d CameraPoseVisualization::oc   = { 0.0,  0.0, 0.0};

static void Eigen2Point(const Eigen::Vector3d& v, Point& p) {
  p.x = v.x();
  p.y = v.y();
  p.z = v.z();
}

CameraPoseVisualization::CameraPoseVisualization(
  float r, float g, float b, float a)
  : m_marker_ns("CameraPoseVisualization"),
    m_scale(0.2),
    m_line_width(0.01f)
{
  m_image_boundary_color.r = r;
  m_image_boundary_color.g = g;
  m_image_boundary_color.b = b;
  m_image_boundary_color.a = a;
  m_optical_center_connector_color = m_image_boundary_color;
}

void CameraPoseVisualization::setImageBoundaryColor(
  float r, float g, float b, float a)
{
  m_image_boundary_color.r = r;
  m_image_boundary_color.g = g;
  m_image_boundary_color.b = b;
  m_image_boundary_color.a = a;
}

void CameraPoseVisualization::setOpticalCenterConnectorColor(
  float r, float g, float b, float a)
{
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

void CameraPoseVisualization::add_edge(
  const Eigen::Vector3d& p0,
  const Eigen::Vector3d& p1)
{
  Marker marker;
  marker.ns = m_marker_ns;
  marker.id = static_cast<int>(m_markers.size());
  marker.type = Marker::LINE_LIST;
  marker.action = Marker::ADD;
  marker.scale.x = m_line_width;
  marker.color.r = 0.0f;
  marker.color.g = 1.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0f;

  Point pt0, pt1;
  Eigen2Point(p0, pt0);
  Eigen2Point(p1, pt1);
  marker.points = {pt0, pt1};

  m_markers.push_back(marker);
}

void CameraPoseVisualization::add_loopedge(
  const Eigen::Vector3d& p0,
  const Eigen::Vector3d& p1)
{
  Marker marker;
  marker.ns = m_marker_ns;
  marker.id = static_cast<int>(m_markers.size());
  marker.type = Marker::LINE_LIST;
  marker.action = Marker::ADD;
  marker.scale.x = 0.04f;
  marker.color.r = 1.0f;
  marker.color.g = 0.0f;
  marker.color.b = 1.0f;
  marker.color.a = 1.0f;

  Point pt0, pt1;
  Eigen2Point(p0, pt0);
  Eigen2Point(p1, pt1);
  marker.points = {pt0, pt1};

  m_markers.push_back(marker);
}

void CameraPoseVisualization::add_pose(
  const Eigen::Vector3d& p,
  const Eigen::Quaterniond& q)
{
  Marker marker;
  marker.ns = m_marker_ns;
  marker.id = static_cast<int>(m_markers.size());
  marker.type = Marker::LINE_STRIP;
  marker.action = Marker::ADD;
  marker.scale.x = m_line_width;

  // compute frustum corners in world frame
  std::array<Eigen::Vector3d, 8> corners = {
    q * (m_scale * imlt) + p,
    q * (m_scale * imlb) + p,
    q * (m_scale * imrb) + p,
    q * (m_scale * imrt) + p,
    q * (m_scale * lt0)  + p,
    q * (m_scale * lt1)  + p,
    q * (m_scale * lt2)  + p,
    q * (m_scale * oc)   + p
  };

  std::vector<Point> pts(8);
  for (size_t i = 0; i < 8; ++i) Eigen2Point(corners[i], pts[i]);

  // boundaries
  std::vector<std::pair<int,int>> edges = {
    {0,1},{1,2},{2,3},{3,0}, // boundary
    {4,5},{5,6},             // top-left indicator
    {0,7},{1,7},{2,7},{3,7}  // connections to optical center
  };

  for (auto &e : edges) {
    marker.points.push_back(pts[e.first]);
    marker.points.push_back(pts[e.second]);
    marker.colors.push_back(m_image_boundary_color);
    marker.colors.push_back(m_image_boundary_color);
  }

  m_markers.push_back(marker);
}

void CameraPoseVisualization::reset() {
  m_markers.clear();
}

void CameraPoseVisualization::publish_by(
  rclcpp::Publisher<MarkerArray>::SharedPtr pub,
  const Header &header)
{
  MarkerArray array;
  for (auto &m : m_markers) {
    m.header = header;
    array.markers.push_back(m);
  }
  pub->publish(array);
}
