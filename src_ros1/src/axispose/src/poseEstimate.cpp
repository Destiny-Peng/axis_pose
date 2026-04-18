#include "axispose/poseEstimate.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <limits>
#include <tuple>

#include <boost/bind.hpp>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

#ifdef AXISPOSE_HAS_CERES
#include "axispose/ceres_joint_optimizer.hpp"
#endif

namespace axispose
{

    namespace
    {
        std::string toLowerCopy(std::string s)
        {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });
            return s;
        }
    }

    PoseEstimate::PoseEstimate(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh)
    {
        std::string depth_topic = "/camera/depth/image_raw";
        std::string mask_topic = "/yolo/mask";
        std::string color_camera_info_topic = "/camera/color/camera_info";
        std::string depth_camera_info_topic = "/camera/depth/camera_info";
        std::string pose_topic = "/shaft/pose";
        std::string debug_cloud_topic = "/shaft/debug_cloud";

        pnh_.param<std::string>("depth_topic", depth_topic, depth_topic);
        pnh_.param<std::string>("mask_topic", mask_topic, mask_topic);
        pnh_.param<std::string>("color_camera_info_topic", color_camera_info_topic, color_camera_info_topic);
        pnh_.param<std::string>("depth_camera_info_topic", depth_camera_info_topic, depth_camera_info_topic);
        pnh_.param<std::string>("pose_topic", pose_topic, pose_topic);
        pnh_.param<std::string>("debug_cloud_topic", debug_cloud_topic, debug_cloud_topic);

        pnh_.param<std::string>("algorithm_type", algorithm_type_, algorithm_type_);
        algorithm_type_ = toLowerCopy(algorithm_type_);

        pnh_.param<std::string>("statistics_directory_path", statistics_directory_path_, std::string("statistics/runtime"));
        pnh_.param<bool>("statistics_enabled", statistics_enabled_, statistics_enabled_);

        pnh_.param<double>("voxel_leaf_size", voxel_leaf_size_, voxel_leaf_size_);
        pnh_.param<double>("sor_mean_k", sor_mean_k_, sor_mean_k_);
        pnh_.param<double>("sor_std_mul", sor_std_mul_, sor_std_mul_);
        pnh_.param<bool>("use_sor", use_sor_, use_sor_);
        pnh_.param<bool>("use_euclidean_cluster", use_euclidean_cluster_, use_euclidean_cluster_);
        pnh_.param<int>("cluster_mode", cluster_mode_, cluster_mode_);
        pnh_.param<double>("sacline_distance_threshold", sacline_distance_threshold_, sacline_distance_threshold_);
        pnh_.param<bool>("align_depth_to_color", align_depth_to_color_, align_depth_to_color_);
        pnh_.param<bool>("runtime_log_enabled", runtime_log_enabled_, runtime_log_enabled_);
        pnh_.param<double>("runtime_log_interval_sec", runtime_log_interval_sec_, runtime_log_interval_sec_);

        ROS_INFO("PoseEstimate starting. alg=%s depth=%s mask=%s stats=%s runtime_log=%s interval=%.1fs align_depth_to_color=%s",
                 algorithm_type_.c_str(),
                 depth_topic.c_str(),
                 mask_topic.c_str(),
                 statistics_directory_path_.c_str(),
                 runtime_log_enabled_ ? "true" : "false",
                 runtime_log_interval_sec_,
                 align_depth_to_color_ ? "true" : "false");

        depth_sub_.subscribe(nh_, depth_topic, 5);
        mask_sub_.subscribe(nh_, mask_topic, 5);
        sync_.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(10), depth_sub_, mask_sub_));
        sync_->registerCallback(boost::bind(&PoseEstimate::syncCallback, this, _1, _2));

        camera_info_color_sub_ = nh_.subscribe<sensor_msgs::CameraInfo>(
            color_camera_info_topic, 5, &PoseEstimate::cameraInfoColorCallback, this);
        camera_info_depth_sub_ = nh_.subscribe<sensor_msgs::CameraInfo>(
            depth_camera_info_topic, 5, &PoseEstimate::cameraInfoDepthCallback, this);

        pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(pose_topic, 10);
        debug_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(debug_cloud_topic, 1);

        pc_processor_ = std::make_unique<PointCloudProcessor>();
        depth_aligner_ = std::make_unique<DepthAligner>();

        try
        {
            benchmark_ = std::make_unique<AlgorithmBenchmark>(
                statistics_directory_path_,
                "metrics.csv",
                statistics_enabled_,
                std::vector<std::string>{"valid_points", "pose_x", "pose_y", "pose_z", "qx", "qy", "qz", "qw"});
        }
        catch (const std::exception &e)
        {
            ROS_WARN("Benchmark init failed: %s", e.what());
            benchmark_.reset();
        }

        param_timer_ = nh_.createTimer(ros::Duration(1.0), &PoseEstimate::updateParameters, this);
        health_timer_ = nh_.createTimer(ros::Duration(10.0), &PoseEstimate::healthCheck, this);
    }

    void PoseEstimate::updateParameters(const ros::TimerEvent &)
    {
        std::string new_alg = algorithm_type_;
        pnh_.param<std::string>("algorithm_type", new_alg, algorithm_type_);
        new_alg = toLowerCopy(new_alg);
        if (new_alg != algorithm_type_)
        {
            ROS_INFO("algorithm_type changed: %s -> %s", algorithm_type_.c_str(), new_alg.c_str());
            algorithm_type_ = new_alg;
        }

        pnh_.param<double>("voxel_leaf_size", voxel_leaf_size_, voxel_leaf_size_);
        pnh_.param<double>("sor_mean_k", sor_mean_k_, sor_mean_k_);
        pnh_.param<double>("sor_std_mul", sor_std_mul_, sor_std_mul_);
        pnh_.param<bool>("use_sor", use_sor_, use_sor_);
        pnh_.param<bool>("use_euclidean_cluster", use_euclidean_cluster_, use_euclidean_cluster_);
        pnh_.param<int>("cluster_mode", cluster_mode_, cluster_mode_);
        pnh_.param<double>("sacline_distance_threshold", sacline_distance_threshold_, sacline_distance_threshold_);
        pnh_.param<bool>("align_depth_to_color", align_depth_to_color_, align_depth_to_color_);
        pnh_.param<bool>("runtime_log_enabled", runtime_log_enabled_, runtime_log_enabled_);
        pnh_.param<double>("runtime_log_interval_sec", runtime_log_interval_sec_, runtime_log_interval_sec_);
    }

    void PoseEstimate::healthCheck(const ros::TimerEvent &)
    {
        if (!got_depth_frame_)
        {
            ROS_WARN("No depth frames received yet. Check depth_topic remap and camera driver.");
        }
        if (!got_mask_frame_)
        {
            ROS_WARN("No mask frames received yet. Check /yolo/mask producer and image topic.");
        }
    }

    void PoseEstimate::resetFrameDiagnostics()
    {
        frame_fallback_used_ = false;
        frame_fallback_reason_.clear();
        frame_error_code_ = 0;
    }

    void PoseEstimate::markFrameFallback(const std::string &reason)
    {
        frame_fallback_used_ = true;
        frame_fallback_reason_ = reason;
    }

    void PoseEstimate::logRuntimeSummary(const ros::Time &stamp)
    {
        if (!runtime_log_enabled_)
        {
            return;
        }

        if (last_runtime_log_stamp_.isZero())
        {
            last_runtime_log_stamp_ = stamp;
            return;
        }

        if ((stamp - last_runtime_log_stamp_).toSec() < runtime_log_interval_sec_)
        {
            return;
        }
        last_runtime_log_stamp_ = stamp;

        const double success_rate = total_frames_ > 0 ? static_cast<double>(pose_published_frames_) / static_cast<double>(total_frames_) : 0.0;
        const double fallback_rate = total_frames_ > 0 ? static_cast<double>(fallback_frames_) / static_cast<double>(total_frames_) : 0.0;
        const double avg_total_ms = total_frames_ > 0 ? cumulative_total_latency_ms_ / static_cast<double>(total_frames_) : 0.0;
        const double avg_denoise_ms = total_frames_ > 0 ? cumulative_denoise_latency_ms_ / static_cast<double>(total_frames_) : 0.0;
        const double avg_pose_ms = total_frames_ > 0 ? cumulative_pose_latency_ms_ / static_cast<double>(total_frames_) : 0.0;

        ROS_INFO("Runtime summary: frames=%llu pose_ok=%llu success_rate=%.3f fallback=%llu fallback_rate=%.3f avg_total_ms=%.2f avg_denoise_ms=%.2f avg_pose_ms=%.2f empty_cloud=%llu depth_convert_fail=%llu mask_convert_fail=%llu",
                 static_cast<unsigned long long>(total_frames_),
                 static_cast<unsigned long long>(pose_published_frames_),
                 success_rate,
                 static_cast<unsigned long long>(fallback_frames_),
                 fallback_rate,
                 avg_total_ms,
                 avg_denoise_ms,
                 avg_pose_ms,
                 static_cast<unsigned long long>(empty_cloud_frames_),
                 static_cast<unsigned long long>(depth_convert_failures_),
                 static_cast<unsigned long long>(mask_convert_failures_));
    }

    void PoseEstimate::cameraInfoDepthCallback(const sensor_msgs::CameraInfoConstPtr &msg)
    {
        if (!have_intrinsics_depth_)
        {
            if (!(msg->K[0] > 0.0 && msg->K[4] > 0.0))
            {
                ROS_ERROR_THROTTLE(5.0, "Invalid depth intrinsics fx/fy from camera_info.");
                return;
            }
            depth_camera_matrix_.at<double>(0, 0) = msg->K[0];
            depth_camera_matrix_.at<double>(1, 1) = msg->K[4];
            depth_camera_matrix_.at<double>(0, 2) = msg->K[2];
            depth_camera_matrix_.at<double>(1, 2) = msg->K[5];
            frame_id_ = msg->header.frame_id;
            have_intrinsics_depth_ = true;
            ROS_INFO("Depth intrinsics ready. fx=%.2f fy=%.2f cx=%.2f cy=%.2f frame=%s",
                     depth_camera_matrix_.at<double>(0, 0),
                     depth_camera_matrix_.at<double>(1, 1),
                     depth_camera_matrix_.at<double>(0, 2),
                     depth_camera_matrix_.at<double>(1, 2),
                     frame_id_.c_str());
        }
    }

    void PoseEstimate::cameraInfoColorCallback(const sensor_msgs::CameraInfoConstPtr &msg)
    {
        if (!have_intrinsics_color_)
        {
            if (!(msg->K[0] > 0.0 && msg->K[4] > 0.0))
            {
                ROS_ERROR_THROTTLE(5.0, "Invalid color intrinsics fx/fy from camera_info.");
                return;
            }
            color_camera_matrix_.at<double>(0, 0) = msg->K[0];
            color_camera_matrix_.at<double>(1, 1) = msg->K[4];
            color_camera_matrix_.at<double>(0, 2) = msg->K[2];
            color_camera_matrix_.at<double>(1, 2) = msg->K[5];
            color_frame_id_ = msg->header.frame_id;
            have_intrinsics_color_ = true;
            ROS_INFO("Color intrinsics ready. fx=%.2f fy=%.2f cx=%.2f cy=%.2f frame=%s",
                     color_camera_matrix_.at<double>(0, 0),
                     color_camera_matrix_.at<double>(1, 1),
                     color_camera_matrix_.at<double>(0, 2),
                     color_camera_matrix_.at<double>(1, 2),
                     color_frame_id_.c_str());
        }
    }

    cv::Mat PoseEstimate::alignDepthToColor(const cv::Mat &depth, int color_width, int color_height)
    {
        if (!depth_aligner_)
        {
            return cv::Mat();
        }
        return depth_aligner_->align(depth, depth_camera_matrix_, color_camera_matrix_, color_width, color_height);
    }

    void PoseEstimate::denoisePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        PointCloudDenoiseOptions options;
        options.voxel_leaf_size = voxel_leaf_size_;
        options.use_sor = use_sor_;
        options.sor_mean_k = static_cast<int>(sor_mean_k_);
        options.sor_std_mul = sor_std_mul_;
        options.use_euclidean_cluster = use_euclidean_cluster_;
        options.cluster_mode = cluster_mode_;
        options.sacline_distance_threshold = sacline_distance_threshold_;
        pc_processor_->denoisePointCloud(cloud, options);
    }

    void PoseEstimate::syncCallback(const sensor_msgs::ImageConstPtr &depth_msg, const sensor_msgs::ImageConstPtr &mask_msg)
    {
        ++total_frames_;
        resetFrameDiagnostics();

        const auto frame_t0 = std::chrono::steady_clock::now();
        got_depth_frame_ = true;
        got_mask_frame_ = true;
        if (!have_intrinsics_depth_)
        {
            frame_error_code_ = 1;
            ROS_WARN_THROTTLE(5.0, "No depth intrinsics yet, skipping frame");
            return;
        }

        ROS_DEBUG_THROTTLE(1.0, "Frame input: depth_encoding=%s depth=%dx%d mask=%dx%d",
                           depth_msg->encoding.c_str(),
                           depth_msg->width,
                           depth_msg->height,
                           mask_msg->width,
                           mask_msg->height);

        cv::Mat depth_cv;
        try
        {
            if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
                depth_msg->encoding == sensor_msgs::image_encodings::MONO16)
            {
                depth_cv = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image.clone();
            }
            else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
            {
                depth_cv = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1)->image.clone();
            }
            else
            {
                ROS_WARN_THROTTLE(5.0, "Unsupported depth encoding: %s", depth_msg->encoding.c_str());
                return;
            }
        }
        catch (const std::exception &e)
        {
            ++depth_convert_failures_;
            frame_error_code_ = 2;
            ROS_ERROR("Depth conversion failed: %s", e.what());
            return;
        }

        cv::Mat mask_cv;
        try
        {
            mask_cv = cv_bridge::toCvShare(mask_msg, sensor_msgs::image_encodings::MONO8)->image.clone();
        }
        catch (const std::exception &e)
        {
            ++mask_convert_failures_;
            frame_error_code_ = 3;
            ROS_ERROR("Mask conversion failed: %s", e.what());
            return;
        }

        const int mask_pixels = cv::countNonZero(mask_cv);
        ROS_DEBUG_THROTTLE(1.0, "Mask pixels: %d / %d", mask_pixels, mask_cv.rows * mask_cv.cols);

        cv::Mat depth_filtered;
        cv::Mat point_cloud_camera_matrix = depth_camera_matrix_;
        if (align_depth_to_color_ && have_intrinsics_color_)
        {
            cv::Mat aligned_depth = alignDepthToColor(depth_cv, mask_cv.cols, mask_cv.rows);
            depth_filtered = cv::Mat::zeros(aligned_depth.size(), aligned_depth.type());
            aligned_depth.copyTo(depth_filtered, mask_cv);
            point_cloud_camera_matrix = color_camera_matrix_;
        }
        else
        {
            if (align_depth_to_color_ && !have_intrinsics_color_)
            {
                ROS_WARN_THROTTLE(5.0, "align_depth_to_color=true but color intrinsics unavailable, fallback to depth grid.");
            }
            if (depth_cv.size() != mask_cv.size())
            {
                cv::resize(mask_cv, mask_cv, depth_cv.size(), 0, 0, cv::INTER_NEAREST);
            }
            depth_filtered = cv::Mat::zeros(depth_cv.size(), depth_cv.type());
            depth_cv.copyTo(depth_filtered, mask_cv);
        }

        auto organized = pc_processor_->depthMaskToPointCloud(depth_filtered, point_cloud_camera_matrix);
        if (!organized || organized->empty())
        {
            ++empty_cloud_frames_;
            frame_error_code_ = 4;
            ROS_WARN_THROTTLE(2.0, "Empty point cloud after mask filtering");
            return;
        }

        ROS_DEBUG_THROTTLE(1.0, "Organized cloud points=%zu", organized->points.size());

        pcl::PointCloud<pcl::PointXYZ>::Ptr valid_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        valid_cloud->header = organized->header;
        for (const auto &pt : organized->points)
        {
            if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
            {
                valid_cloud->points.push_back(pt);
            }
        }
        valid_cloud->width = static_cast<uint32_t>(valid_cloud->points.size());
        valid_cloud->height = 1;

        if (valid_cloud->empty())
        {
            ++empty_cloud_frames_;
            frame_error_code_ = 5;
            ROS_WARN_THROTTLE(2.0, "No valid points after conversion");
            return;
        }

        ROS_DEBUG_THROTTLE(1.0, "Valid cloud points before denoise=%zu", valid_cloud->points.size());

        const auto denoise_t0 = std::chrono::steady_clock::now();

        if (benchmark_)
        {
            benchmark_->run("denoise", [this, &valid_cloud]() -> size_t
                            {
                                denoisePointCloud(valid_cloud);
                                return valid_cloud->size(); }, [](size_t n)
                            { return std::vector<std::string>{std::to_string(n)}; });
        }
        else
        {
            denoisePointCloud(valid_cloud);
        }

        const auto denoise_t1 = std::chrono::steady_clock::now();
        const double denoise_ms = std::chrono::duration<double, std::milli>(denoise_t1 - denoise_t0).count();
        cumulative_denoise_latency_ms_ += denoise_ms;
        ROS_DEBUG_THROTTLE(1.0, "Valid cloud points after denoise=%zu denoise_ms=%.3f", valid_cloud->points.size(), denoise_ms);

        geometry_msgs::PoseStamped pose_msg;
        const std::string bench_name = std::string("pose_") + algorithm_type_;
        const auto pose_t0 = std::chrono::steady_clock::now();
        if (benchmark_)
        {
            auto tup = benchmark_->run(bench_name, [this, &valid_cloud, &mask_cv, &depth_msg]()
                                       {
                                           auto p = computePoseByAlgorithm(valid_cloud, mask_cv, depth_msg->header.stamp);
                                           return std::make_tuple(p, static_cast<size_t>(valid_cloud->size())); }, [](const std::tuple<geometry_msgs::PoseStamped, size_t> &t)
                                       {
                                           const auto &pose = std::get<0>(t).pose;
                                           size_t n = std::get<1>(t);
                                           return std::vector<std::string>{
                                               std::to_string(n),
                                               std::to_string(pose.position.x),
                                               std::to_string(pose.position.y),
                                               std::to_string(pose.position.z),
                                               std::to_string(pose.orientation.x),
                                               std::to_string(pose.orientation.y),
                                               std::to_string(pose.orientation.z),
                                               std::to_string(pose.orientation.w)}; });
            pose_msg = std::get<0>(tup);
        }
        else
        {
            pose_msg = computePoseByAlgorithm(valid_cloud, mask_cv, depth_msg->header.stamp);
        }
        const auto pose_t1 = std::chrono::steady_clock::now();
        const double pose_ms = std::chrono::duration<double, std::milli>(pose_t1 - pose_t0).count();
        cumulative_pose_latency_ms_ += pose_ms;

        pose_msg.header.stamp = depth_msg->header.stamp;
        pose_msg.header.frame_id = have_intrinsics_color_ ? color_frame_id_ : frame_id_;
        pose_pub_.publish(pose_msg);
        ++pose_published_frames_;

        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*organized, cloud_msg);
        cloud_msg.header.stamp = depth_msg->header.stamp;
        cloud_msg.header.frame_id = pose_msg.header.frame_id;
        debug_cloud_pub_.publish(cloud_msg);

        const auto frame_t1 = std::chrono::steady_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(frame_t1 - frame_t0).count();
        cumulative_total_latency_ms_ += total_ms;

        if (frame_fallback_used_)
        {
            ++fallback_frames_;
            ROS_WARN_THROTTLE(1.0, "Pose fallback occurred: %s", frame_fallback_reason_.c_str());
        }

        ROS_DEBUG_THROTTLE(1.0,
                           "Frame done: total_ms=%.3f denoise_ms=%.3f pose_ms=%.3f valid_points=%zu error_code=%d",
                           total_ms,
                           denoise_ms,
                           pose_ms,
                           valid_cloud->points.size(),
                           frame_error_code_);
        logRuntimeSummary(depth_msg->header.stamp);
    }

    Eigen::Vector3d PoseEstimate::stabilizeAxisDirection(const Eigen::Vector3d &axis, const Eigen::Vector3d *reference_axis)
    {
        Eigen::Vector3d out = axis.normalized();
        if (!std::isfinite(out.x()) || !std::isfinite(out.y()) || !std::isfinite(out.z()))
        {
            return Eigen::Vector3d::UnitX();
        }

        if (reference_axis)
        {
            Eigen::Vector3d ref = reference_axis->normalized();
            if (std::isfinite(ref.x()) && std::isfinite(ref.y()) && std::isfinite(ref.z()) && out.dot(ref) < 0.0)
            {
                out = -out;
            }
        }

        if (axis_history_valid_)
        {
            if (out.dot(axis_history_) < 0.0)
            {
                out = -out;
            }
            axis_history_ = (0.8 * axis_history_ + 0.2 * out).normalized();
        }
        else
        {
            if (out.x() < 0.0)
            {
                out = -out;
            }
            axis_history_ = out;
            axis_history_valid_ = true;
        }
        return out;
    }

    geometry_msgs::PoseStamped PoseEstimate::computePoseByAlgorithm(
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
        const cv::Mat &mask_cv,
        const ros::Time &stamp)
    {
        if (algorithm_type_ == "pca")
            return computePoseFromCloud(cloud, stamp);
        if (algorithm_type_ == "ransac")
            return computePoseFromSACLine(cloud, stamp);
        if (algorithm_type_ == "gaussian")
            return computePoseGaussian(cloud, mask_cv, stamp);
        if (algorithm_type_ == "ceres")
            return computePoseCeres(cloud, mask_cv, stamp);

        markFrameFallback("unknown_algorithm->pca:" + algorithm_type_);
        ROS_WARN_THROTTLE(5.0, "Unknown algorithm_type=%s, fallback pca", algorithm_type_.c_str());
        return computePoseFromCloud(cloud, stamp);
    }

    geometry_msgs::PoseStamped PoseEstimate::computePoseFromCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const ros::Time &stamp)
    {
        (void)stamp;
        geometry_msgs::PoseStamped pose;
        if (!cloud || cloud->empty())
            return pose;

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud, centroid);

        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*cloud, centroid, covariance);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        Eigen::Matrix3f eig_vecs = solver.eigenvectors();
        Eigen::Vector3f eig_vals = solver.eigenvalues();

        const double denom = std::max(1e-9f, eig_vals[0] + eig_vals[1] + eig_vals[2]);
        const double eta = eig_vals[2] / denom;
        const double lambda = std::sqrt(std::max(0.0f, eig_vals[1] + eig_vals[0]));
        ROS_INFO_THROTTLE(1.0, "PCA eig: %.6f %.6f %.6f eta=%.6f lambda=%.6f", eig_vals[0], eig_vals[1], eig_vals[2], eta, lambda);

        Eigen::Vector3f axis_x = eig_vecs.col(2).normalized();
        Eigen::Vector3f axis_y = eig_vecs.col(1).normalized();
        axis_x = stabilizeAxisDirection(axis_x.cast<double>(), nullptr).cast<float>();

        Eigen::Vector3f corrected_z = axis_x.cross(axis_y).normalized();
        Eigen::Matrix3f R;
        R.col(0) = axis_x;
        R.col(1) = axis_y;
        R.col(2) = corrected_z;

        Eigen::Quaternionf q(R);
        q.normalize();

        pose.pose.position.x = centroid[0];
        pose.pose.position.y = centroid[1];
        pose.pose.position.z = centroid[2];
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        return pose;
    }

    geometry_msgs::PoseStamped PoseEstimate::computePoseFromSACLine(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const ros::Time &stamp)
    {
        (void)stamp;
        geometry_msgs::PoseStamped pose;
        if (!cloud || cloud->empty())
            return pose;

        pcl::ModelCoefficients coefficients;
        pcl::PointIndices inliers;
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_LINE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(300);
        seg.setDistanceThreshold(sacline_distance_threshold_);
        seg.setInputCloud(cloud);
        seg.segment(inliers, coefficients);

        if (coefficients.values.size() < 6)
        {
            markFrameFallback("ransac_fit_failed->pca");
            ROS_WARN_THROTTLE(2.0, "RANSAC line fit failed, fallback PCA");
            return computePoseFromCloud(cloud, stamp);
        }

        Eigen::Vector3f line_point(coefficients.values[0], coefficients.values[1], coefficients.values[2]);
        Eigen::Vector3f axis_x(coefficients.values[3], coefficients.values[4], coefficients.values[5]);
        axis_x.normalize();
        axis_x = stabilizeAxisDirection(axis_x.cast<double>(), nullptr).cast<float>();

        Eigen::Vector3f axis_y = axis_x.unitOrthogonal();
        Eigen::Vector3f axis_z = axis_x.cross(axis_y).normalized();

        Eigen::Matrix3f R;
        R.col(0) = axis_x;
        R.col(1) = axis_y;
        R.col(2) = axis_z;

        Eigen::Quaternionf q(R);
        q.normalize();

        pose.pose.position.x = line_point[0];
        pose.pose.position.y = line_point[1];
        pose.pose.position.z = line_point[2];
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        return pose;
    }

    geometry_msgs::PoseStamped PoseEstimate::computePoseGaussian(
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
        const cv::Mat &mask_cv,
        const ros::Time &stamp)
    {
        (void)mask_cv;
        (void)stamp;
        geometry_msgs::PoseStamped pose;
        if (!cloud || cloud->empty())
            return pose;

        GaussianMapSolver solver;
        Eigen::Vector3f out_axis, out_point;
        float out_radius = 0.05f;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>(*cloud));
        if (!solver.estimateAxis(cloud_ptr, out_axis, out_point, out_radius))
        {
            markFrameFallback("gaussian_failed->pca");
            ROS_WARN_THROTTLE(2.0, "GaussianMapSolver failed, fallback PCA");
            return computePoseFromCloud(cloud, stamp);
        }

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud_ptr, centroid);
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*cloud_ptr, centroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(covariance);
        Eigen::Vector3f pca_axis = eig.eigenvectors().col(2).normalized();

        if (std::abs(out_axis.dot(pca_axis)) < std::cos(8.0 * M_PI / 180.0))
        {
            out_axis = pca_axis;
        }
        Eigen::Vector3d pca_axis_d = pca_axis.cast<double>();
        out_axis = stabilizeAxisDirection(out_axis.cast<double>(), &pca_axis_d).cast<float>();

        pose.pose.position.x = out_point.x();
        pose.pose.position.y = out_point.y();
        pose.pose.position.z = out_point.z();

        Eigen::Vector3d axis_x = out_axis.cast<double>().normalized();
        Eigen::Vector3d up = Eigen::Vector3d::UnitY();
        if (std::abs(axis_x.dot(up)) > 0.99)
            up = Eigen::Vector3d::UnitZ();
        Eigen::Vector3d axis_y = (up - up.dot(axis_x) * axis_x).normalized();
        Eigen::Vector3d axis_z = axis_x.cross(axis_y).normalized();

        Eigen::Matrix3d R;
        R.col(0) = axis_x;
        R.col(1) = axis_y;
        R.col(2) = axis_z;
        Eigen::Quaterniond q(R);
        q.normalize();

        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        return pose;
    }

    geometry_msgs::PoseStamped PoseEstimate::computePoseCeres(
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
        const cv::Mat &mask_cv,
        const ros::Time &stamp)
    {
#ifndef AXISPOSE_HAS_CERES
        markFrameFallback("ceres_unavailable->pca");
        ROS_WARN_THROTTLE(5.0, "Ceres not compiled in, fallback pca");
        return computePoseFromCloud(cloud, stamp);
#else
        geometry_msgs::PoseStamped pose;
        if (!cloud || cloud->empty())
            return pose;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>(*cloud));

        // Ceres initialization: use PCA axis + centroid as line point for robustness.
        Eigen::Vector4f centroid4;
        pcl::compute3DCentroid(*cloud_ptr, centroid4);
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*cloud_ptr, centroid4, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(covariance);
        Eigen::Vector3d pca_axis = eig.eigenvectors().col(2).cast<double>().normalized();
        if (!std::isfinite(pca_axis.x()) || !std::isfinite(pca_axis.y()) || !std::isfinite(pca_axis.z()))
        {
            markFrameFallback("ceres_init_pca_invalid->pca");
            ROS_WARN_THROTTLE(2.0, "Ceres skipped: invalid PCA init, fallback pca");
            return computePoseFromCloud(cloud, stamp);
        }
        Eigen::Vector3d d = stabilizeAxisDirection(pca_axis, nullptr).normalized();
        Eigen::Vector3d p = centroid4.head<3>().cast<double>();
        Eigen::Vector3d m = p.cross(d);

        std::vector<cv::Point> nz;
        cv::findNonZero(mask_cv, nz);
        if (nz.size() < 20)
        {
            markFrameFallback("ceres_mask_points_insufficient->pca");
            ROS_WARN_THROTTLE(2.0, "Ceres skipped: not enough mask points");
            return computePoseFromCloud(cloud, stamp);
        }

        cv::Vec4f line;
        cv::fitLine(nz, line, cv::DIST_L2, 0, 0.01, 0.01);
        const double vx = static_cast<double>(line[0]);
        const double vy = static_cast<double>(line[1]);
        const double x0 = static_cast<double>(line[2]);
        const double y0 = static_cast<double>(line[3]);
        if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(x0) || !std::isfinite(y0))
        {
            markFrameFallback("ceres_line_fit_invalid->pca");
            ROS_WARN_THROTTLE(2.0, "Ceres skipped: invalid line fit values");
            return computePoseFromCloud(cloud, stamp);
        }
        const double norm_v = std::hypot(vx, vy);
        if (norm_v < 1e-8)
        {
            markFrameFallback("ceres_line_norm_invalid->pca");
            ROS_WARN_THROTTLE(2.0, "Ceres skipped: invalid fitted line");
            return computePoseFromCloud(cloud, stamp);
        }

        Eigen::Vector3d line2d_abc;
        line2d_abc << -vy, vx, (vy * x0 - vx * y0);
        line2d_abc /= std::sqrt(line2d_abc.x() * line2d_abc.x() + line2d_abc.y() * line2d_abc.y());

        const cv::Mat &Kcv = have_intrinsics_color_ ? color_camera_matrix_ : depth_camera_matrix_;
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0, 0) = Kcv.at<double>(0, 0);
        K(1, 1) = Kcv.at<double>(1, 1);
        K(0, 2) = Kcv.at<double>(0, 2);
        K(1, 2) = Kcv.at<double>(1, 2);

        CeresJointOptimizer optimizer;
        const bool ok = optimizer.optimizePose(d, m, cloud_ptr, K, line2d_abc, p, 50, 90, 3.0, 20.0);
        if (!ok)
        {
            markFrameFallback("ceres_optimize_failed->pca");
            ROS_WARN_THROTTLE(2.0, "Ceres optimize failed, fallback pca");
            return computePoseFromCloud(cloud, stamp);
        }

        d = stabilizeAxisDirection(d, &pca_axis).normalized();

        const double dn2 = d.squaredNorm();
        Eigen::Vector3d optimized_point = p;
        if (dn2 > 1e-12)
        {
            optimized_point = d.cross(m) / dn2;
            const Eigen::Vector3d centroid = centroid4.head<3>().cast<double>();
            optimized_point = optimized_point + d * d.dot(centroid - optimized_point);
            const double max_shift = 0.20;
            Eigen::Vector3d delta = optimized_point - p;
            const double norm_delta = delta.norm();
            if (norm_delta > max_shift)
            {
                optimized_point = p + delta * (max_shift / norm_delta);
            }
        }

        pose.pose.position.x = optimized_point.x();
        pose.pose.position.y = optimized_point.y();
        pose.pose.position.z = optimized_point.z();

        Eigen::Vector3d axis_x = d.normalized();
        Eigen::Vector3d up = Eigen::Vector3d::UnitY();
        if (std::abs(axis_x.dot(up)) > 0.99)
            up = Eigen::Vector3d::UnitZ();
        Eigen::Vector3d axis_y = (up - up.dot(axis_x) * axis_x).normalized();
        Eigen::Vector3d axis_z = axis_x.cross(axis_y).normalized();

        Eigen::Matrix3d R;
        R.col(0) = axis_x;
        R.col(1) = axis_y;
        R.col(2) = axis_z;
        Eigen::Quaterniond q(R);
        q.normalize();

        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        return pose;
#endif
    }

} // namespace axispose

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pose_estimate_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    axispose::PoseEstimate node(nh, pnh);

    ros::spin();
    return 0;
}