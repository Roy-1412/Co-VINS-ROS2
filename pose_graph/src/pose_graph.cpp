#include "pose_graph.h"
#include <fstream>
#include <boost/dynamic_bitset.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <thread>
#include <chrono>
#include <mutex>
#include <ceres/ceres.h>
#include <visualization_msgs/msg/marker.hpp>



// Constructor: takes a pointer to your rclcpp::Node
PoseGraph::PoseGraph(rclcpp::Node *node)
: node_(node),
  tf_broadcaster_(node)
{ 
  posegraph_visualization = std::make_unique<CameraPoseVisualization>(1.0, 0.0, 1.0, 1.0);
  posegraph_visualization->setScale(0.1);
  posegraph_visualization->setLineWidth(0.01);

  // Declare parameters (if not done in the node)
  VISUALIZATION_SHIFT_X  = node_->declare_parameter<int>("visualization_shift_x", 0);
  VISUALIZATION_SHIFT_Y  = node_->declare_parameter<int>("visualization_shift_y", 0);
  SKIP_DIS               = node_->declare_parameter<double>("skip_dis", 0.0);
  POSE_GRAPH_SAVE_PATH   = node_->declare_parameter<std::string>("pose_graph_save_path", "");
  VINS_RESULT_PATH       = node_->declare_parameter<std::string>("pose_graph_result_path", "");


  // Load vocabulary
  auto pkg_share = ament_index_cpp::get_package_share_directory("pose_graph");
  std::string voc_file = pkg_share + "/../support_files/brief_k10L6.bin";
  loadVocabulary(voc_file);

  // Prepare CSV output
  std::ofstream tmp(VINS_RESULT_PATH + "/pose_graph_path.csv", std::ios::trunc);
  tmp.close();

  registerPublishers(node_);

  t_optimization = std::thread(&PoseGraph::optimize4DoF, this);

  earliest_loop_index = -1;
  global_index = 0;
  sequence_cnt = 0;
  sequence_loop.push_back(false);
}

PoseGraph::~PoseGraph()
{
  if (t_optimization.joinable()) {
    t_optimization.join();
  }
}

void PoseGraph::registerPublishers(rclcpp::Node *node)
{
  pub_pg_path = node->create_publisher<nav_msgs::msg::Path>("pose_graph_path", 10);
  pub_base_path = node->create_publisher<nav_msgs::msg::Path>("base_path", 10);
  pub_pose_graph = node_->create_publisher<visualization_msgs::msg::MarkerArray>("pose_graph_markers", rclcpp::QoS(1));
  for (int i = 0; i < 10; ++i) {
    pub_path[i] = node->create_publisher<nav_msgs::msg::Path>("path_" + std::to_string(i), 10);
  }
}

void PoseGraph::loadVocabulary(const std::string &voc_path)
{
  voc = std::make_unique<BriefVocabulary>(voc_path);
  db.setVocabulary(*voc, false, 0);
}

void PoseGraph::addAgentFrame(KeyFrame* cur_kf)
{
  // assign a unique index to this keyframe
  cur_kf->index = global_index++;
  int sequence = cur_kf->sequence;

  // if this sequence is seen for the first time, initialize its alignment maps
  if (!sequence_align_world.count(sequence)) {
    if (sequence_align_world.empty()) {
      sequence_align_world[sequence] = true;
      first_sequence = sequence;
      RCLCPP_INFO(node_->get_logger(), "First sequence %d", sequence);
    } else {
      sequence_align_world[sequence] = false;
      RCLCPP_INFO(node_->get_logger(), "New sequence %d", sequence);
    }
    sequence_t_drift_map[sequence] = Eigen::Vector3d::Zero();
    sequence_w_t_s_map[sequence]  = Eigen::Vector3d::Zero();
    sequence_r_drift_map[sequence] = Eigen::Matrix3d::Identity();
    sequence_w_r_s_map[sequence]  = Eigen::Matrix3d::Identity();
  }

  // transform the VIO pose into the world frame (applying any previous drift/shift)
  Eigen::Vector3d vio_P;
  Eigen::Matrix3d vio_R;
  cur_kf->getVioPose(vio_P, vio_R);


  RCLCPP_INFO(node_->get_logger(),
            "getVioPose: P = [%.4f, %.4f, %.4f]",
            vio_P.x(), vio_P.y(), vio_P.z());


  vio_P = sequence_w_r_s_map[sequence] * vio_P + sequence_w_t_s_map[sequence];
  vio_R = sequence_w_r_s_map[sequence] * vio_R;
  cur_kf->updateVioPose(vio_P, vio_R);

  RCLCPP_INFO(node_->get_logger(),
            "[updateVioPose] new VIO P = [%.4f, %.4f, %.4f]",
            vio_P.x(), vio_P.y(), vio_P.z());

  // detect loop closures
  int loop_index = detectLoop(cur_kf, cur_kf->index);
  bool find_connection    = false;
  bool need_update_path   = false;

  if (loop_index >= 0) {
    KeyFrame* old_kf = getKeyFrame(loop_index);
    find_connection  = cur_kf->findConnection(old_kf);
    if (find_connection) {
      // update earliest loop index for optimization window
      if (earliest_loop_index < 0 || loop_index < earliest_loop_index) {
        earliest_loop_index = loop_index;
      }
      RCLCPP_WARN(node_->get_logger(),
                  "Found loop between seq %d and seq %d",
                  sequence, old_kf->sequence);

      // if joining a new sequence into the world frame
      if (old_kf->sequence != sequence
          && !sequence_align_world[sequence]
          &&  sequence_align_world[old_kf->sequence])
      {
        RCLCPP_WARN(node_->get_logger(),
                    "Aligning sequence %d to world frame of sequence %d",
                    sequence, old_kf->sequence);
        sequence_align_world[sequence] = true;

        // compute relative transform at loop closure
        Eigen::Vector3d w_P_old, w_P_cur, tmp_vio_P;
        Eigen::Matrix3d w_R_old, w_R_cur, tmp_vio_R;
        old_kf->getVioPose(w_P_old, w_R_old);
        cur_kf->getVioPose(tmp_vio_P, tmp_vio_R);

        Eigen::Vector3d relative_t = cur_kf->getLoopRelativeT();
        Eigen::Matrix3d relative_q = cur_kf->getLoopRelativeQ().toRotationMatrix();
        w_P_cur = w_R_old * relative_t + w_P_old;
        w_R_cur = w_R_old * relative_q;

        double shift_yaw = Utility::R2ypr(w_R_cur).x()
                         - Utility::R2ypr(tmp_vio_R).x();
        Eigen::Matrix3d shift_r = Utility::ypr2R(Eigen::Vector3d(shift_yaw, 0, 0));
        Eigen::Vector3d shift_t = w_P_cur
                                - w_R_cur * tmp_vio_R.transpose() * tmp_vio_P;

        sequence_w_r_s_map[sequence] = shift_r;
        sequence_w_t_s_map[sequence] = shift_t;

        // re-apply new transform to all previous keyframes in this sequence
        for (auto *kf : keyframe_vec) {
          if (kf->sequence == sequence) {
            Eigen::Vector3d p;  Eigen::Matrix3d r;
            kf->getVioPose(p, r);
            p = shift_r * p + shift_t;
            r = shift_r * r;
            kf->updateVioPose(p, r);
          }
        }
        need_update_path = true;
      }

      // similarly handle aligning the old sequence if needed
      if (old_kf->sequence != sequence
          && !sequence_align_world[old_kf->sequence]
          &&  sequence_align_world[sequence])
      {
        RCLCPP_WARN(node_->get_logger(),
                    "Aligning sequence %d to world frame of sequence %d",
                    old_kf->sequence, sequence);
        sequence_align_world[old_kf->sequence] = true;

        Eigen::Vector3d w_P_old, tmp_vio_P, w_P_cur;
        Eigen::Matrix3d w_R_old, tmp_vio_R, w_R_cur;
        old_kf->getVioPose(tmp_vio_P, tmp_vio_R);
        cur_kf->getVioPose(w_P_cur, w_R_cur);

        Eigen::Vector3d relative_t = cur_kf->getLoopRelativeT();
        Eigen::Matrix3d relative_q = cur_kf->getLoopRelativeQ().toRotationMatrix();
        w_P_old = -w_R_cur * relative_q.transpose() * relative_t + w_P_cur;
        w_R_old =  w_R_cur * relative_q.transpose();

        double shift_yaw = Utility::R2ypr(w_R_old).x()
                         - Utility::R2ypr(tmp_vio_R).x();
        Eigen::Matrix3d shift_r = Utility::ypr2R(Eigen::Vector3d(shift_yaw, 0, 0));
        Eigen::Vector3d shift_t = w_P_old - w_R_old * tmp_vio_R.transpose() * tmp_vio_P;

        sequence_w_r_s_map[old_kf->sequence] = shift_r;
        sequence_w_t_s_map[old_kf->sequence] = shift_t;

        for (auto *kf : keyframe_vec) {
          if (kf->sequence == old_kf->sequence) {
            Eigen::Vector3d p;  Eigen::Matrix3d r;
            kf->getVioPose(p, r);
            p = shift_r * p + shift_t;
            r = shift_r * r;
            kf->updateVioPose(p, r);
          }
        }
        need_update_path = true;
      }
    }
  }

  // now commit the final pose (with drift) to the keyframe
  {
    std::lock_guard<std::mutex> lk(m_keyframelist);
    Eigen::Vector3d P; Eigen::Matrix3d R;
    cur_kf->getVioPose(P, R);
    P = sequence_r_drift_map[sequence] * P + sequence_t_drift_map[sequence];
    R = sequence_r_drift_map[sequence] * R;
    cur_kf->updatePose(P, R);

    Eigen::Quaterniond Q{R};
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp    = rclcpp::Time(static_cast<int64_t>(cur_kf->time_stamp * 1e9));
    ps.header.frame_id = "world";
    ps.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
    ps.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
    ps.pose.position.z = P.z();
    ps.pose.orientation.x = Q.x();
    ps.pose.orientation.y = Q.y();
    ps.pose.orientation.z = Q.z();
    ps.pose.orientation.w = Q.w();

    path[sequence].poses.push_back(ps);
    path[sequence].header = ps.header;

    // optionally log to CSV
    if (SAVE_LOOP_PATH) {
      std::ofstream file(VINS_RESULT_PATH + "/pose_graph_path.csv", std::ios::app);
      file << static_cast<uint64_t>(cur_kf->time_stamp * 1e9)
           << "," << P.x() << "," << P.y() << "," << P.z()
           << "," << Q.w() << "," << Q.x() << "," << Q.y() << "," << Q.z()
           << "\n";
    }

    keyframe_vec.push_back(cur_kf);
  }

  // if we made loop connections, schedule optimization
  if (find_connection) {
    std::lock_guard<std::mutex> lk(m_optimize_buf);
    optimize_buf.push(cur_kf->index);
  }

  // update the visualization if needed
  if (need_update_path) {
    updatePath();
  }

  // publish paths and TF
  publish();
}


void PoseGraph::loadKeyFrame(KeyFrame* cur_kf)
{
  // assign index
  cur_kf->index = global_index++;
  int sequence = cur_kf->sequence;

  // initialize sequence alignment if needed
  if (!sequence_align_world.count(sequence)) {
    if (sequence_align_world.empty()) {
      sequence_align_world[sequence] = true;
      first_sequence = sequence;
      RCLCPP_INFO(node_->get_logger(), "First sequence %d", sequence);
    } else {
      sequence_align_world[sequence] = false;
      RCLCPP_INFO(node_->get_logger(), "New sequence %d", sequence);
    }
    sequence_t_drift_map[sequence] = Eigen::Vector3d::Zero();
    sequence_w_t_s_map[sequence]  = Eigen::Vector3d::Zero();
    sequence_r_drift_map[sequence] = Eigen::Matrix3d::Identity();
    sequence_w_r_s_map[sequence]  = Eigen::Matrix3d::Identity();
  }

  // add this keyframe to the vocabulary for loop detection
  addKeyFrameIntoVoc(cur_kf);

  // get its VIO pose
  Eigen::Vector3d P;
  Eigen::Matrix3d R;
  cur_kf->getVioPose(P, R);
  Eigen::Quaterniond Q{R};

  // build a ROS2 PoseStamped for visualization
  geometry_msgs::msg::PoseStamped ps;
  ps.header.stamp = rclcpp::Time(static_cast<int64_t>(cur_kf->time_stamp * 1e9));
  ps.header.frame_id = "world";
  ps.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
  ps.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
  ps.pose.position.z = P.z();
  ps.pose.orientation.x = Q.x();
  ps.pose.orientation.y = Q.y();
  ps.pose.orientation.z = Q.z();
  ps.pose.orientation.w = Q.w();

  // append to the path for this sequence
  path[sequence].poses.push_back(ps);
  path[sequence].header = ps.header;

  // optionally draw loop edges if this keyframe has a loop
  if (SHOW_L_EDGE && cur_kf->has_loop && sequence != 0) {
    KeyFrame* connected_KF = getKeyFrame(cur_kf->loop_index);
    if (connected_KF) {
      Eigen::Vector3d P0, P1;
      Eigen::Matrix3d R0, R1;
      cur_kf->getPose(P0, R0);
      connected_KF->getPose(P1, R1);
      // offset connected pose for visualization
      P1 += Eigen::Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0);
      int edge_type = (sequence == connected_KF->sequence ? 0 : 1);
      posegraph_visualization->addLoopEdge(P0, P1, edge_type);
    }
  }

  // store the keyframe in our list (thread-safe)
  {
    std::lock_guard<std::mutex> lk(m_keyframelist);
    keyframe_vec.push_back(cur_kf);
  }
}


KeyFrame* PoseGraph::getKeyFrame(int index)
{
  if (index < 0 || index >= static_cast<int>(keyframe_vec.size()))
    return nullptr;
  return keyframe_vec[index];
}

int PoseGraph::detectLoop(KeyFrame* keyframe, int frame_index)
{
  RCLCPP_INFO(node_->get_logger(), "calling detect loop");
  // measure timing (optional)
  TicToc tmp_t;

  // first query the vocabulary database, then add this descriptor
  QueryResults ret;
  TicToc t_query;
  db.query(keyframe->feature_des, ret, 4, frame_index - 30);

  TicToc t_add;
  db.add(keyframe->feature_des);

  // decide if a loop exists by checking match scores
  bool find_loop = false;
  if (ret.size() >= 1 && ret[0].Score > 0.05) {
    for (size_t i = 1; i < ret.size(); ++i) {
      if (ret[i].Score > 0.015) {
        find_loop = true;
        break;
      }
    }
  }

  // if we found a loop and have enough frames, return the smallest matching frame ID
  if (find_loop && frame_index > 10) {
    RCLCPP_INFO(node_->get_logger(), "initial detect loop successful" );
    int min_index = -1;
    for (const auto &r : ret) {
      if ((min_index < 0 || r.Id < min_index) && r.Score > 0.015) {
        min_index = r.Id;
      }
    }
    RCLCPP_INFO(node_->get_logger(), "double detect loop successful" );
    return min_index;
  }

  // otherwise no loop
  RCLCPP_INFO(node_->get_logger(), "detect no loop" );
  return -1;
}


void PoseGraph::addKeyFrameIntoVoc(KeyFrame* keyframe)
{
  db.add(keyframe->feature_des);
}

void PoseGraph::optimize4DoF()
{
  // Continuously process loop-closure optimizations while the node is running
  while (rclcpp::ok()) {
    int cur_index = -1;
    int first_looped_index = -1;

    // Pop the latest frame index to optimize
    {
      std::lock_guard<std::mutex> lk(m_optimize_buf);
      while (!optimize_buf.empty()) {
        cur_index = optimize_buf.front();
        first_looped_index = earliest_loop_index;
        optimize_buf.pop();
      }
    }

    if (cur_index != -1) {
      // Start timing (optional)
      TicToc t_opt;

      // Prepare the optimization window
      size_t window_start = (first_looped_index > 0 ? first_looped_index : 0);
      size_t window_size = cur_index + 1;
      std::vector<std::array<double,3>> t_array(window_size);
      std::vector<Quaterniond> q_array(window_size);
      std::vector<std::array<double,3>> euler_array(window_size);

      // Build Ceres problem
      ceres::Problem problem;
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      options.max_num_iterations = 5;
      ceres::Solver::Summary summary;
      ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
      ceres::LocalParameterization* angle_param = AngleLocalParameterization::Create();

      std::map<int,int> last_sequence_index;
      bool fix_first = false;
      int local_i = 0;

      // Lock keyframe vector while reading poses
      {
        std::lock_guard<std::mutex> lk(m_keyframelist);
        for (size_t k = window_start; k < keyframe_vec.size() && local_i < (int)window_size; ++k) {
          KeyFrame* it = keyframe_vec[k];
          last_sequence_index[it->sequence] = k;
          it->local_index = local_i;

          // Retrieve VIO pose
          Eigen::Vector3d P; Eigen::Matrix3d R;
          it->getVioPose(P, R);
          Quaterniond Q(R);

          // Fill arrays
          t_array[local_i]     = {P.x(), P.y(), P.z()};
          q_array[local_i]     = Q;
          auto ypr = Utility::R2ypr(R);
          euler_array[local_i] = {ypr.x(), ypr.y(), ypr.z()};

          // Add parameter blocks to the problem
          problem.AddParameterBlock(euler_array[local_i].data(), 1, angle_param);
          problem.AddParameterBlock(t_array[local_i].data(),       3);

          // Fix the first frame of the first sequence
          if (!fix_first && it->sequence == first_sequence) {
            fix_first = true;
            problem.SetParameterBlockConstant(euler_array[local_i].data());
            problem.SetParameterBlockConstant(t_array[local_i].data());
          }

          // Add odometry edges (chain prior)
          int prev_local = local_i - 1;
          for (int step = 0; step < 5 && prev_local >= 0; ++step) {
            // find previous in same sequence
            while (prev_local >= 0 && keyframe_vec[window_start + prev_local]->sequence != it->sequence) {
              --prev_local;
            }
            if (prev_local >= 0) {
              // relative motion between prev_local and local_i
              Eigen::Vector3d rel_t(
                t_array[local_i][0] - t_array[prev_local][0],
                t_array[local_i][1] - t_array[prev_local][1],
                t_array[local_i][2] - t_array[prev_local][2]
              );
              rel_t = q_array[prev_local].inverse() * rel_t;
              double rel_yaw = euler_array[local_i][0] - euler_array[prev_local][0];

              ceres::CostFunction* cost = AngleLocalParameterization::FourDOFError::Create(
                rel_t.x(), rel_t.y(), rel_t.z(),
                rel_yaw,
                euler_array[prev_local][1],
                euler_array[prev_local][2]
              );
              problem.AddResidualBlock(
                cost,
                nullptr,
                euler_array[prev_local].data(),
                t_array[prev_local].data(),
                euler_array[local_i].data(),
                t_array[local_i].data()
              );
              --prev_local;
            }
          }

          // Add loop closure edges
          if (it->has_loop &&
              sequence_align_world[it->sequence] &&
              sequence_align_world[keyframe_vec[it->loop_index]->sequence])
          {
            KeyFrame* looped = keyframe_vec[it->loop_index];
            int j_local = looped->local_index;
            Eigen::Vector3d lli = it->getLoopRelativeT();
            double    ly  = it->getLoopRelativeYaw();
            ceres::CostFunction* cost = AngleLocalParameterization::FourDOFWeightError::Create(
              lli.x(), lli.y(), lli.z(),
              ly,
              euler_array[j_local][1],
              euler_array[j_local][2]
            );
            problem.AddResidualBlock(
              cost,
              loss_function,
              euler_array[j_local].data(),
              t_array[j_local].data(),
              euler_array[local_i].data(),
              t_array[local_i].data()
            );
          }

          // Stop once we've included the current frame
          if (it->index == cur_index) {
            ++local_i;
            break;
          }
          ++local_i;
        }
      } // unlock m_keyframelist

      // Solve the optimization
      ceres::Solve(options, &problem, &summary);
      RCLCPP_DEBUG(node_->get_logger(), "PoseGraph optimize time: %f ms", t_opt.toc());

      // Write optimized poses back into keyframes
      {
        std::lock_guard<std::mutex> lk(m_keyframelist);
        int apply_i = 0;
        for (size_t k = window_start; k < keyframe_vec.size() && apply_i < (int)window_size; ++k) {
          KeyFrame* it = keyframe_vec[k];
          if (it->index < first_looped_index) continue;

          Eigen::Matrix3d R_new = Utility::ypr2R(Eigen::Vector3d(
            euler_array[apply_i][0],
            euler_array[apply_i][1],
            euler_array[apply_i][2]
          ));
          Eigen::Quaterniond Q_new(R_new);
          Eigen::Vector3d T_new(
            t_array[apply_i][0],
            t_array[apply_i][1],
            t_array[apply_i][2]
          );
          it->updatePose(T_new, Q_new.toRotationMatrix());

          if (it->index == cur_index) break;
          ++apply_i;
        }
      }

      // Recompute drift per sequence
      for (auto &pr : last_sequence_index) {
        int seq = pr.first;
        int idx = pr.second;
        KeyFrame* kf = keyframe_vec[idx];
        Eigen::Vector3d P_opt, P_vio;
        Eigen::Matrix3d R_opt, R_vio;
        kf->getPose(P_opt, R_opt);
        kf->getVioPose(P_vio, R_vio);

        double yaw_drift = Utility::R2ypr(R_opt).x() - Utility::R2ypr(R_vio).x();
        {
          std::lock_guard<std::mutex> lk(m_drift);
          sequence_r_drift_map[seq] = Utility::ypr2R(Eigen::Vector3d(yaw_drift, 0, 0));
          sequence_t_drift_map[seq] = P_opt - sequence_r_drift_map[seq] * P_vio;
        }
      }

      // Apply drift to subsequent keyframes
      {
        std::lock_guard<std::mutex> lk(m_keyframelist);
        for (size_t k = window_start + local_i; k < keyframe_vec.size(); ++k) {
          KeyFrame* it = keyframe_vec[k];
          int seq = it->sequence;
          Eigen::Vector3d P_vio;
          Eigen::Matrix3d R_vio;
          it->getVioPose(P_vio, R_vio);
          P_vio = sequence_r_drift_map[seq] * P_vio + sequence_t_drift_map[seq];
          R_vio = sequence_r_drift_map[seq] * R_vio;
          it->updatePose(P_vio, R_vio);
        }
      }

      // Refresh the published paths
      updatePath();
    }

    // Sleep between optimization attempts
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  }
}


void PoseGraph::updatePath()
{
  // clear existing paths & visualization
  for (auto &kv : sequence_align_world) {
    path[kv.first].poses.clear();
  }
  base_path.poses.clear();
  posegraph_visualization->reset();
}

void PoseGraph::savePoseGraph()
{
  std::lock_guard<std::mutex> lk(m_keyframelist);
  std::string file = POSE_GRAPH_SAVE_PATH + "/pose_graph.txt";
  RCLCPP_INFO(node_->get_logger(), "Saving pose graph to %s", file.c_str());
  FILE* pFile = fopen(file.c_str(), "w");
  for (auto *it : keyframe_vec) {
    // fprintf all fields exactly as before
  }
  fclose(pFile);
}

void PoseGraph::loadPoseGraph()
{
  std::lock_guard<std::mutex> lk(m_keyframelist);
  std::string file = POSE_GRAPH_SAVE_PATH + "/pose_graph.txt";
  RCLCPP_INFO(node_->get_logger(), "Loading pose graph from %s", file.c_str());
  FILE* pFile = fopen(file.c_str(), "r");
  if (!pFile) {
    RCLCPP_WARN(node_->get_logger(), "Failed to open %s", file.c_str());
    return;
  }
  // fscanf loop, reconstruct KeyFrame, call loadKeyFrame(...)
  fclose(pFile);
}

void PoseGraph::publish()
{
  for (auto &kv : sequence_align_world) {
    int seq = kv.first;
    pub_pg_path->publish(path[seq]);
    pub_path[seq]->publish(path[seq]);
    posegraph_visualization->publishBy(pub_pose_graph, path[seq].header);
  }
  pub_base_path->publish(base_path);
  publishTF();
}

void PoseGraph::publishTF()
{
  for (auto &kv : sequence_align_world) {
    if (!kv.second) continue;
    int seq = kv.first;
    Eigen::Vector3d t = sequence_r_drift_map[seq] * sequence_w_t_s_map[seq] + sequence_t_drift_map[seq];
    Eigen::Matrix3d R_q = sequence_r_drift_map[seq] * sequence_w_r_s_map[seq];
    Eigen::Quaterniond q(R_q);


    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = node_->now();
    tf_msg.header.frame_id = "/global";
    tf_msg.child_frame_id = "/drone_" + std::to_string(seq);
    tf_msg.transform.translation.x = t.x();
    tf_msg.transform.translation.y = t.y();
    tf_msg.transform.translation.z = t.z();
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();

    tf_broadcaster_.sendTransform(tf_msg);
  }
}
