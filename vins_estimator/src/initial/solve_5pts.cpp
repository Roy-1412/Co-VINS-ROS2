#include "solve_5pts.h"
#include "estimator.h"


rclcpp::Logger logger = rclcpp::get_logger("vins_estimator");

namespace cv {
    void decomposeEssentialMat( InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t )
    {

        Mat E = _E.getMat().reshape(1, 3);
        CV_Assert(E.cols == 3 && E.rows == 3);

        Mat D, U, Vt;
        SVD::compute(E, D, U, Vt);

        if (determinant(U) < 0) U *= -1.;
        if (determinant(Vt) < 0) Vt *= -1.;

        Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
        W.convertTo(W, E.type());

        Mat R1, R2, t;
        R1 = U * W * Vt;
        R2 = U * W.t() * Vt;
        t = U.col(2) * 1.0;

        R1.copyTo(_R1);
        R2.copyTo(_R2);
        t.copyTo(_t);
    }

    int recoverPose( InputArray E, InputArray _points1, InputArray _points2, InputArray _cameraMatrix,
                         OutputArray _R, OutputArray _t, InputOutputArray _mask)
    {

        Mat points1, points2, cameraMatrix;
        _points1.getMat().convertTo(points1, CV_64F);
        _points2.getMat().convertTo(points2, CV_64F);
        _cameraMatrix.getMat().convertTo(cameraMatrix, CV_64F);

        int npoints = points1.checkVector(2);

        CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
                                  points1.type() == points2.type());

        CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3 && cameraMatrix.channels() == 1);

        if (points1.channels() > 1)
        {
            points1 = points1.reshape(1, npoints);
            points2 = points2.reshape(1, npoints);
        }

        double fx = cameraMatrix.at<double>(0,0);
        double fy = cameraMatrix.at<double>(1,1);
        double cx = cameraMatrix.at<double>(0,2);
        double cy = cameraMatrix.at<double>(1,2);

        points1.col(0) = (points1.col(0) - cx) / fx;
        points2.col(0) = (points2.col(0) - cx) / fx;
        points1.col(1) = (points1.col(1) - cy) / fy;
        points2.col(1) = (points2.col(1) - cy) / fy;

        points1 = points1.t();
        points2 = points2.t();

        Mat R1, R2, t;
        decomposeEssentialMat(E, R1, R2, t);
        Mat P0 = Mat::eye(3, 4, R1.type());
        Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
        P1(Range::all(), Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
        P2(Range::all(), Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
        P3(Range::all(), Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
        P4(Range::all(), Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;

        // Do the cheirality check.
        // Notice here a threshold dist is used to filter
        // out far away points (i.e. infinite points) since
        // there depth may vary between postive and negtive.
        double dist = 50.0;
        Mat Q;
        triangulatePoints(P0, P1, points1, points2, Q);
        Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;

        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask1 = (Q.row(2) < dist) & mask1;
        Q = P1 * Q;
        mask1 = (Q.row(2) > 0) & mask1;
        mask1 = (Q.row(2) < dist) & mask1;
   

        triangulatePoints(P0, P2, points1, points2, Q);
        Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask2 = (Q.row(2) < dist) & mask2;
        Q = P2 * Q;
        mask2 = (Q.row(2) > 0) & mask2;
        mask2 = (Q.row(2) < dist) & mask2;

        triangulatePoints(P0, P3, points1, points2, Q);
        Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask3 = (Q.row(2) < dist) & mask3;
        Q = P3 * Q;
        mask3 = (Q.row(2) > 0) & mask3;
        mask3 = (Q.row(2) < dist) & mask3;

        triangulatePoints(P0, P4, points1, points2, Q);
        Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask4 = (Q.row(2) < dist) & mask4;
        Q = P4 * Q;
        mask4 = (Q.row(2) > 0) & mask4;
        mask4 = (Q.row(2) < dist) & mask4;

        mask1 = mask1.t();
        mask2 = mask2.t();
        mask3 = mask3.t();
        mask4 = mask4.t();

        // If _mask is given, then use it to filter outliers.
        if (!_mask.empty())
        {
            Mat mask = _mask.getMat();
            CV_Assert(mask.size() == mask1.size());
            bitwise_and(mask, mask1, mask1);
            bitwise_and(mask, mask2, mask2);
            bitwise_and(mask, mask3, mask3);
            bitwise_and(mask, mask4, mask4);
        }
        if (_mask.empty() && _mask.needed())
        {
            _mask.create(mask1.size(), CV_8U);
        }

        CV_Assert(_R.needed() && _t.needed());
        _R.create(3, 3, R1.type());
        _t.create(3, 1, t.type());

        int good1 = countNonZero(mask1);
        int good2 = countNonZero(mask2);
        int good3 = countNonZero(mask3);
        int good4 = countNonZero(mask4);

        if (good1 >= good2 && good1 >= good3 && good1 >= good4)
        {
            R1.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask1.copyTo(_mask);
            return good1;
        }
        else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
        {
            R2.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask2.copyTo(_mask);
            return good2;
        }
        else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
        {
            t = -t;
            R1.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask3.copyTo(_mask);
            return good3;
        }
        else
        {
            t = -t;
            R2.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask4.copyTo(_mask);
            return good4;
        }
    }

    int recoverPose( InputArray E, InputArray _points1, InputArray _points2, OutputArray _R,
                         OutputArray _t, double focal, Point2d pp, InputOutputArray _mask)
    {
        Mat cameraMatrix = (Mat_<double>(3,3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
        return cv::recoverPose(E, _points1, _points2, cameraMatrix, _R, _t, _mask);
    }
}



/*
bool MotionEstimator::solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &Rotation, Vector3d &Translation)
{
    if (corres.size() >= 15)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat mask;
        cv::Mat F = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 3, 0.99, mask);

        int raw_inliers = cv::countNonZero(mask);
        RCLCPP_WARN(
            rclcpp::get_logger("vins_estimator"),
            "after findFundamentalMat: raw_mask inliers = %d / %d",
            raw_inliers, (int)ll.size()
            );

        
           
        //cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        //cv::Mat rot, trans;
        double fx = 2.1387619122017772e+03;
        double fy = 2.1315886210259278e+03;
        double cx = 3.6119856633263799e+02;
        double cy = 2.4827644773395667e+02;

        // 构造内参矩阵 K
        cv::Mat K = (cv::Mat_<double>(3,3) << 
            fx, 0,  cx,
            0,  fy, cy,
            0,  0,  1
        );

        cv::Mat pose_mask;
        cv::Mat E = cv::findEssentialMat(
            ll, rr,
            K,     // 相机内参
            cv::RANSAC,
            0.99,
            3.0,   // 重投影阈值
            pose_mask
        );
        
        cv::SVD svd(E, cv::SVD::MODIFY_A);
        svd.w.at<double>(0) = 1;
        svd.w.at<double>(1) = 1;
        svd.w.at<double>(2) = 0;
        E = svd.u * cv::Mat::diag(svd.w) * svd.vt;

        cv::Mat rot, trans;
        //int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        cv::Mat e_mask;
        int inlier_cnt = cv::recoverPose(E, ll, rr, K, rot, trans, e_mask);
        int final_inliers = cv::countNonZero(e_mask);
        RCLCPP_WARN(
            rclcpp::get_logger("vins_estimator"), 
            "[E-stage] inliers = %d / %d", 
            final_inliers, 
            (int)ll.size()
        );
        //cout << "inlier_cnt " << inlier_cnt << endl;
        std::cout << "[solveRelativeRT] inlier_cnt = "
                  << inlier_cnt
                  << " / "
                  << corres.size()
                  << std::endl;

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++)
        {   
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }

        Rotation = R.transpose();
        Translation = -R.transpose() * T;
        if(inlier_cnt > 10)
            return true;
        else
            return false;
    }
    return false;
}
*/
bool MotionEstimator::solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &Rotation, Vector3d &Translation)
{
    RCLCPP_INFO(logger, "calling solveRelativeRT ");
    if (corres.size() >= 15)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {

            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat mask;
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        int raw_inliers = cv::countNonZero(mask);
        
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        //cout << "inlier_cnt " << inlier_cnt << endl;
        int final_inliers = cv::countNonZero(mask);
   
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++)
        {   
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }

        Rotation = R.transpose();
        Translation = -R.transpose() * T;
        
 /*Eigen::Matrix3d R_eig;  Eigen::Vector3d T_eig;
    for (int i = 0; i < 3; ++i) {
        T_eig(i) = trans.at<double>(i,0);
        for (int j = 0; j < 3; ++j)
            R_eig(i,j) = rot.at<double>(i,j);
    }
    // “相机→世界” 形式
    Rotation    = R_eig.transpose();
    Translation = -R_eig.transpose() * T_eig;

    // 4. 基于 R,T 构造基础矩阵 F = K⁻ᵀ [T]_× R K⁻¹（K = I 时简化）
    cv::Mat t_x = (cv::Mat_<double>(3,3) <<
         0, -T_eig.z(),  T_eig.y(),
         T_eig.z(), 0, -T_eig.x(),
        -T_eig.y(), T_eig.x(),  0);
    cv::Mat F = t_x * rot;  // 因为 K = I

    // 5. 计算极线并测距离
    vector<cv::Vec3f> lines1, lines2;
    cv::computeCorrespondEpilines(ll, 1, F, lines1);
    cv::computeCorrespondEpilines(rr, 2, F, lines2);

    auto ptLineDist = [&](const cv::Vec3f &l, const cv::Point2f &p) {
        return std::abs(l[0]*p.x + l[1]*p.y + l[2]) /
               std::sqrt(l[0]*l[0] + l[1]*l[1]);
    };

    double sumErr = 0, maxErr = 0;
    int goodInliers = 0;
    for (int i = 0; i < int(corres.size()); ++i) {
        if (mask.at<uchar>(i) == 0) continue;
        double e1 = ptLineDist(lines1[i], rr[i]);
        double e2 = ptLineDist(lines2[i], ll[i]);
        double e  = 0.5*(e1 + e2);
        sumErr += e; 
        maxErr = std::max(maxErr, e);
        ++goodInliers;
    }
    double meanErr = sumErr / std::max(1, goodInliers);

    RCLCPP_INFO(logger, 
      "[SFM] inliers=%d/%d, epipolar mean=%.4fpx, max=%.4fpx",
      goodInliers, int(corres.size()), meanErr, maxErr);
*/



        if(inlier_cnt > 12)
            return true;
        else
             //RCLCPP_WARN(logger, "[SFM] Relative R/T not reliable: meanErr=%.4f, inliers=%d", meanErr, inlier_cnt);
          return false;
    }
    return false;
}



