#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include "parameters.h"

class CameraPoseVisualization
{
public:
  explicit CameraPoseVisualization(float r, float g, float b, float a);

  void setImageBoundaryColor(float r, float g, float b, float a = 1.0f);
  void setOpticalCenterConnectorColor(float r, float g, float b, float a = 1.0f);
  void setScale(double s);
  void setLineWidth(double width);

  void addPose(const Eigen::Vector3d &p, const Eigen::Quaterniond &q);
  void addEdge(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1);
  void addLoopEdge(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1, int color_mode);
  void reset();

  /**
   * @brief Publish the accumulated markers as a MarkerArray on ROS2.
   * @param pub SharedPtr to the ROS2 publisher for MarkerArray.
   * @param header Header to apply to all markers (frame_id, stamp, etc.).
   */
  void publishBy(
    const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr &pub,
    const std_msgs::msg::Header &header);

private:
  std::string                             m_marker_ns;
  std::vector<visualization_msgs::msg::Marker> m_markers;
  std_msgs::msg::ColorRGBA                m_image_boundary_color;
  std_msgs::msg::ColorRGBA                m_optical_center_connector_color;
  double                                  m_scale{1.0};
  double                                  m_line_width{0.01};
  int                                     m_loop_edge_count{0};

  // These static constants define the normalized corners/points of the camera "frustum"
  static const Eigen::Vector3d imlt;  // image top-left
  static const Eigen::Vector3d imlb;  // image bottom-left
  static const Eigen::Vector3d imrt;  // image top-right
  static const Eigen::Vector3d imrb;  // image bottom-right
  static const Eigen::Vector3d oc;    // optical center (origin)
  static const Eigen::Vector3d lt0;   // auxiliary line point 0
  static const Eigen::Vector3d lt1;   // auxiliary line point 1
  static const Eigen::Vector3d lt2;   // auxiliary line point 2
};
