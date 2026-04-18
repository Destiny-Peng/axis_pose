#include "axispose/poseEstimate.hpp"
#include "axispose/debug_manager.hpp"
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <Eigen/Dense>
#include <cmath>
#include <sstream>
#include <cstdint>
#include <tuple>
#include <algorithm>

namespace axispose
{

    PoseEstimateBase::PoseEstimateBase(const std::string &node_name, const rclcpp::NodeOptions &options) : rclcpp::Node(node_name, options)
    {
        // parameters
        this->declare_parameter("depth_image_topic", std::string("/camera/depth/image_raw"));
        this->declare_parameter("mask_topic", std::string("/yolo/mask"));
        this->declare_parameter("color_camera_info_topic", std::string("/camera/color/camera_info"));
        this->declare_parameter("depth_camera_info_topic", std::string("/camera/depth/camera_info"));
        this->declare_parameter("voxel_leaf_size", voxel_leaf_size_);
        this->declare_parameter("sor_mean_k", sor_mean_k_);
        this->declare_parameter("sor_std_mul", sor_std_mul_);
        this->declare_parameter("use_sor", use_sor_);
        this->declare_parameter("use_euclidean_cluster", use_euclidean_cluster_);
        this->declare_parameter("cluster_mode", cluster_mode_); // 0: closest to origin, 1: largest cluster, 2: RANSAC line inliers
        this->declare_parameter("sacline_distance_threshold", sacline_distance_threshold_);
        this->declare_parameter("ceres_max_iterations", ceres_max_iterations_);
        this->declare_parameter("ceres_max_points", ceres_max_points_);
        this->declare_parameter("ceres_weight_2d", ceres_weight_2d_);
        this->declare_parameter("ceres_weight_pos", ceres_weight_pos_);

        // statistics parameters
        this->declare_parameter("statistics_directory_path", statistics_directory_path_);
        this->declare_parameter("statistics_enabled", statistics_enabled_);
        this->declare_parameter("kalman_enabled", kalman_enabled_);
        this->declare_parameter("kalman_position_process_noise", kalman_position_process_noise_);
        this->declare_parameter("kalman_position_measurement_noise", kalman_position_measurement_noise_);
        this->declare_parameter("kalman_axis_process_noise", kalman_axis_process_noise_);
        this->declare_parameter("kalman_axis_measurement_noise", kalman_axis_measurement_noise_);
        this->declare_parameter("kalman_initial_covariance", kalman_initial_covariance_);
        this->declare_parameter("kalman_min_dt", kalman_min_dt_);
        this->declare_parameter("kalman_max_dt", kalman_max_dt_);

        std::string depth_image_topic = this->get_parameter("depth_image_topic").as_string();
        std::string mask_topic = this->get_parameter("mask_topic").as_string();
        std::string color_camera_info_topic = this->get_parameter("color_camera_info_topic").as_string();
        std::string depth_camera_info_topic = this->get_parameter("depth_camera_info_topic").as_string();
        voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
        sor_mean_k_ = this->get_parameter("sor_mean_k").as_double();
        sor_std_mul_ = this->get_parameter("sor_std_mul").as_double();
        statistics_directory_path_ = this->get_parameter("statistics_directory_path").as_string();
        statistics_enabled_ = this->get_parameter("statistics_enabled").as_bool();
        use_sor_ = this->get_parameter("use_sor").as_bool();
        use_euclidean_cluster_ = this->get_parameter("use_euclidean_cluster").as_bool();
        cluster_mode_ = this->get_parameter("cluster_mode").as_int();
        sacline_distance_threshold_ = this->get_parameter("sacline_distance_threshold").as_double();
        ceres_max_iterations_ = this->get_parameter("ceres_max_iterations").as_int();
        ceres_max_points_ = this->get_parameter("ceres_max_points").as_int();
        ceres_weight_2d_ = this->get_parameter("ceres_weight_2d").as_double();
        ceres_weight_pos_ = this->get_parameter("ceres_weight_pos").as_double();
        kalman_enabled_ = this->get_parameter("kalman_enabled").as_bool();
        kalman_position_process_noise_ = this->get_parameter("kalman_position_process_noise").as_double();
        kalman_position_measurement_noise_ = this->get_parameter("kalman_position_measurement_noise").as_double();
        kalman_axis_process_noise_ = this->get_parameter("kalman_axis_process_noise").as_double();
        kalman_axis_measurement_noise_ = this->get_parameter("kalman_axis_measurement_noise").as_double();
        kalman_initial_covariance_ = this->get_parameter("kalman_initial_covariance").as_double();
        kalman_min_dt_ = this->get_parameter("kalman_min_dt").as_double();
        kalman_max_dt_ = this->get_parameter("kalman_max_dt").as_double();

        RCLCPP_INFO(this->get_logger(), "PoseEstimate node starting. Depth: %s Mask: %s ", depth_image_topic.c_str(), mask_topic.c_str());
        if (kalman_enabled_)
        {
            RCLCPP_INFO(this->get_logger(), "Pose Kalman enabled: q_pos=%.6f r_pos=%.6f q_axis=%.6f r_axis=%.6f dt=[%.4f, %.4f]",
                        kalman_position_process_noise_,
                        kalman_position_measurement_noise_,
                        kalman_axis_process_noise_,
                        kalman_axis_measurement_noise_,
                        kalman_min_dt_,
                        kalman_max_dt_);
        }
        RCLCPP_INFO(this->get_logger(), "Ceres params: iter=%d points=%d w2d=%.3f wpos=%.3f",
                    ceres_max_iterations_, ceres_max_points_, ceres_weight_2d_, ceres_weight_pos_);

        rclcpp::QoS qos(rclcpp::KeepLast(5));
        // message_filters subscribers
        depth_sub_.subscribe(this, depth_image_topic, qos.get_rmw_qos_profile());
        mask_sub_.subscribe(this, mask_topic, qos.get_rmw_qos_profile());

        sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(ApproxSyncPolicy(10), depth_sub_, mask_sub_);
        sync_->registerCallback(&PoseEstimateBase::syncCallback, this);

        // subscribe to color and depth camera_info topics separately
        camera_info_color_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            color_camera_info_topic, qos,
            std::bind(&PoseEstimateBase::cameraInfoColorCallback, this, std::placeholders::_1));
        camera_info_depth_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            depth_camera_info_topic, qos,
            std::bind(&PoseEstimateBase::cameraInfoDepthCallback, this, std::placeholders::_1));

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/shaft/pose", 10);
        debug_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/shaft/debug_cloud", 1);

        // initialize debug manager (controls runtime debug flags via parameter `debug_flags`)
        debug_ = std::make_shared<DebugManager>(this);

        // initialize helpers
        pc_processor_ = std::make_unique<PointCloudProcessor>();
        depth_aligner_ = std::make_unique<DepthAligner>();

        // initialize benchmark/metrics logger (records per-frame metrics to CSV)
        try
        {
            benchmark_ = std::make_unique<AlgorithmBenchmark>(statistics_directory_path_, "metrics.csv", statistics_enabled_, std::vector<std::string>{"valid_points", "pose_x", "pose_y", "pose_z", "qx", "qy", "qz", "qw"});
            if (statistics_enabled_)
                RCLCPP_INFO(this->get_logger(), "Benchmark metrics enabled, output: %s/metrics.csv", statistics_directory_path_.c_str());
        }
        catch (const std::exception &e)
        {
            RCLCPP_WARN(this->get_logger(), "Failed to initialize benchmark logger: %s", e.what());
            // continue without metrics
            benchmark_.reset();
        }
    }

    void PoseEstimateBase::cameraInfoDepthCallback(const CameraInfo::SharedPtr msg)
    {
        if (!have_intrinsics_depth_)
        {
            depth_camera_matrix_.at<double>(0, 0) = msg->k[0];
            depth_camera_matrix_.at<double>(1, 1) = msg->k[4];
            depth_camera_matrix_.at<double>(0, 2) = msg->k[2];
            depth_camera_matrix_.at<double>(1, 2) = msg->k[5];
            frame_id_ = msg->header.frame_id;
            have_intrinsics_depth_ = true;
            RCLCPP_INFO(this->get_logger(), "Got depth camera intrinsics fx=%.2f fy=%.2f cx=%.2f cy=%.2f frame=%s",
                        depth_camera_matrix_.at<double>(0, 0),
                        depth_camera_matrix_.at<double>(1, 1),
                        depth_camera_matrix_.at<double>(0, 2),
                        depth_camera_matrix_.at<double>(1, 2),
                        frame_id_.c_str());
        }
    }

    void PoseEstimateBase::cameraInfoColorCallback(const CameraInfo::SharedPtr msg)
    {
        if (!have_intrinsics_color_)
        {
            color_camera_matrix_.at<double>(0, 0) = msg->k[0];
            color_camera_matrix_.at<double>(1, 1) = msg->k[4];
            color_camera_matrix_.at<double>(0, 2) = msg->k[2];
            color_camera_matrix_.at<double>(1, 2) = msg->k[5];
            color_frame_id_ = msg->header.frame_id;
            have_intrinsics_color_ = true;
            RCLCPP_INFO(this->get_logger(), "Got color camera intrinsics fx=%.2f fy=%.2f cx=%.2f cy=%.2f frame=%s",
                        color_camera_matrix_.at<double>(0, 0),
                        color_camera_matrix_.at<double>(1, 1),
                        color_camera_matrix_.at<double>(0, 2),
                        color_camera_matrix_.at<double>(1, 2),
                        msg->header.frame_id.c_str());
        }
    }

    void PoseEstimateBase::syncCallback(const Image::ConstSharedPtr depth_msg, const Image::ConstSharedPtr mask_msg)
    {
        if (!have_intrinsics_depth_)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "No depth camera intrinsics yet, skipping frame");
            return;
        }

        // convert to cv::Mat
        cv::Mat depth_cv;
        try
        {
            depth_cv = cv::Mat(depth_msg->height, depth_msg->width, CV_16U, const_cast<unsigned char *>(depth_msg->data.data())).clone();
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge depth conversion failed: %s", e.what());
            return;
        }

        cv::Mat mask_cv;
        try
        {
            mask_cv = cv::Mat(mask_msg->height, mask_msg->width, CV_8U, const_cast<unsigned char *>(mask_msg->data.data())).clone();
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge mask conversion failed: %s", e.what());
            return;
        }
        // std::cout << "organized cloud size: " << organized->width << " x " << organized->height << std::endl;

        // If we have color intrinsics, align depth to color image size and mask directly.
        cv::Mat depth_filtered;
        cv::Mat point_cloud_camera_matrix = depth_camera_matrix_;
        if (have_intrinsics_color_)
        {
            // align depth to mask (color) resolution
            cv::Mat aligned_depth = alignDepthToColor(depth_cv, mask_cv.cols, mask_cv.rows);
            depth_filtered = cv::Mat::zeros(aligned_depth.size(), aligned_depth.type());
            aligned_depth.copyTo(depth_filtered, mask_cv);
            // Depth has been remapped to color image grid.
            point_cloud_camera_matrix = color_camera_matrix_;
        }
        else
        {
            // fallback: resize mask to depth size and apply
            if (depth_cv.size() != mask_cv.size())
            {
                cv::resize(mask_cv, mask_cv, depth_cv.size(), 0, 0, cv::INTER_NEAREST);
            }
            depth_filtered = cv::Mat::zeros(depth_cv.size(), depth_cv.type());
            depth_cv.copyTo(depth_filtered, mask_cv);
        }

        // convert to organized point cloud (one point per pixel, NaN for invalid) - now only depth is passed
        auto organized = pc_processor_->depthMaskToPointCloud(depth_filtered, point_cloud_camera_matrix);
        if (!organized || organized->empty())
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Empty organized point cloud after mask filtering");
            return;
        }

        // build an unorganized valid cloud for processing (no NaNs), ensuring one point per pixel at most
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
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "No valid points after organized->valid extraction");
            return;
        }

        // denoise (record time and remaining valid points)
        if (benchmark_)
        {
            benchmark_->run("denoise", [this, &valid_cloud]() -> size_t
                            {
                this->denoisePointCloud(valid_cloud);
                return valid_cloud->size(); }, [](size_t n) -> std::vector<std::string>
                            { return {std::to_string(n)}; });
        }
        else
        {
            denoisePointCloud(valid_cloud);
        }

        // Compute pose using a polymorphic algorithm implementation.
        geometry_msgs::msg::PoseStamped pose_msg;
        if (benchmark_)
        {
            auto tup = benchmark_->run(benchmarkLabel(), [this, &valid_cloud, &mask_cv, &depth_msg]()
                                       {
                auto p = this->computePoseByAlgorithm(valid_cloud, mask_cv, depth_msg->header.stamp);
                p = this->applyKalmanFilterToPose(p, depth_msg->header.stamp);
                return std::make_tuple(p, static_cast<size_t>(valid_cloud->size())); }, [](const std::tuple<geometry_msgs::msg::PoseStamped, size_t> &t) -> std::vector<std::string>
                                       {
                const auto &pose = std::get<0>(t).pose;
                size_t n = std::get<1>(t);
                std::vector<std::string> out;
                out.reserve(8);
                out.push_back(std::to_string(n));
                out.push_back(std::to_string(pose.position.x));
                out.push_back(std::to_string(pose.position.y));
                out.push_back(std::to_string(pose.position.z));
                out.push_back(std::to_string(pose.orientation.x));
                out.push_back(std::to_string(pose.orientation.y));
                out.push_back(std::to_string(pose.orientation.z));
                out.push_back(std::to_string(pose.orientation.w));
                return out; });
            pose_msg = std::get<0>(tup);
        }
        else
        {
            pose_msg = computePoseByAlgorithm(valid_cloud, mask_cv, depth_msg->header.stamp);
            pose_msg = applyKalmanFilterToPose(pose_msg, depth_msg->header.stamp);
        }
        pose_msg.header.stamp = depth_msg->header.stamp;
        pose_msg.header.frame_id = have_intrinsics_color_ ? color_frame_id_ : frame_id_;
        pose_pub_->publish(pose_msg);

        // publish debug organized cloud
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*organized, cloud_msg);
        // pcl::toROSMsg(*valid_cloud, cloud_msg);
        cloud_msg.header.stamp = depth_msg->header.stamp;
        cloud_msg.header.frame_id = pose_msg.header.frame_id;
        if (!debug_ || debug_->enabled("debug_cloud"))
        {
            debug_cloud_pub_->publish(cloud_msg);
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr PoseEstimateBase::depthMaskToPointCloud(const cv::Mat &depth)
    {
        return pc_processor_->depthMaskToPointCloud(depth, depth_camera_matrix_);
    }

    cv::Mat PoseEstimateBase::alignDepthToColor(const cv::Mat &depth, int color_width, int color_height)
    {
        if (depth_aligner_)
        {
            return depth_aligner_->align(depth, depth_camera_matrix_, color_camera_matrix_, color_width, color_height);
        }
        return cv::Mat();
    }

    void PoseEstimateBase::denoisePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
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

    Eigen::Vector3d PoseEstimateBase::stabilizeAxisDirection(const Eigen::Vector3d &axis, const Eigen::Vector3d *reference_axis)
    {
        Eigen::Vector3d out = axis.normalized();
        if (!std::isfinite(out.x()) || !std::isfinite(out.y()) || !std::isfinite(out.z()))
        {
            return Eigen::Vector3d::UnitX();
        }

        // Prefer external reference when available (e.g., PCA axis from current frame).
        if (reference_axis)
        {
            Eigen::Vector3d ref = reference_axis->normalized();
            if (std::isfinite(ref.x()) && std::isfinite(ref.y()) && std::isfinite(ref.z()) && out.dot(ref) < 0.0)
            {
                out = -out;
            }
        }

        // Keep temporal consistency to avoid yaw sign flicker between adjacent frames.
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
            // First frame: keep forward-facing preference in camera frame.
            if (out.x() < 0.0)
            {
                out = -out;
            }
            axis_history_ = out;
            axis_history_valid_ = true;
        }

        return out;
    }

    void PoseEstimateBase::kalmanPredictUpdate1D(double measurement,
                                                 double dt,
                                                 double process_noise,
                                                 double measurement_noise,
                                                 double &x,
                                                 double &v,
                                                 Eigen::Matrix2d &P) const
    {
        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;

        const double x_pred = x + dt * v;
        const double v_pred = v;

        Eigen::Matrix2d F;
        F << 1.0, dt,
            0.0, 1.0;

        Eigen::Matrix2d Q;
        Q << 0.25 * dt4 * process_noise, 0.5 * dt3 * process_noise,
            0.5 * dt3 * process_noise, dt2 * process_noise;

        Eigen::Matrix2d P_pred = F * P * F.transpose() + Q;

        const double innovation = measurement - x_pred;
        const double S = P_pred(0, 0) + measurement_noise;
        if (S <= 1e-12)
        {
            x = x_pred;
            v = v_pred;
            P = P_pred;
            return;
        }

        const double k0 = P_pred(0, 0) / S;
        const double k1 = P_pred(1, 0) / S;

        x = x_pred + k0 * innovation;
        v = v_pred + k1 * innovation;

        Eigen::Matrix2d KH;
        KH << k0, 0.0,
            k1, 0.0;
        P = (Eigen::Matrix2d::Identity() - KH) * P_pred;
    }

    void PoseEstimateBase::resetKalmanState(const Eigen::Vector3d &position, const Eigen::Vector3d &axis, double stamp_s)
    {
        kalman_initialized_ = true;
        kalman_last_stamp_s_ = stamp_s;
        kalman_pos_x_ = position;
        kalman_pos_v_.setZero();
        kalman_axis_x_ = axis;
        kalman_axis_v_.setZero();

        for (int i = 0; i < 3; ++i)
        {
            kalman_pos_P_[i] = Eigen::Matrix2d::Identity() * kalman_initial_covariance_;
            kalman_axis_P_[i] = Eigen::Matrix2d::Identity() * kalman_initial_covariance_;
        }
    }

    geometry_msgs::msg::PoseStamped PoseEstimateBase::applyKalmanFilterToPose(const geometry_msgs::msg::PoseStamped &raw_pose, const rclcpp::Time &stamp)
    {
        if (!kalman_enabled_)
        {
            return raw_pose;
        }

        const auto &rp = raw_pose.pose;
        const Eigen::Vector3d raw_pos(rp.position.x, rp.position.y, rp.position.z);
        if (!std::isfinite(raw_pos.x()) || !std::isfinite(raw_pos.y()) || !std::isfinite(raw_pos.z()))
        {
            return raw_pose;
        }

        Eigen::Quaterniond q_raw(rp.orientation.w, rp.orientation.x, rp.orientation.y, rp.orientation.z);
        if (q_raw.norm() <= 1e-9)
        {
            return raw_pose;
        }
        q_raw.normalize();
        Eigen::Vector3d raw_axis = q_raw.toRotationMatrix().col(0).normalized();
        if (!std::isfinite(raw_axis.x()) || !std::isfinite(raw_axis.y()) || !std::isfinite(raw_axis.z()))
        {
            return raw_pose;
        }

        const double stamp_s = stamp.seconds();
        if (!kalman_initialized_ || stamp_s <= 0.0)
        {
            Eigen::Vector3d axis0 = stabilizeAxisDirection(raw_axis, &raw_axis);
            resetKalmanState(raw_pos, axis0, stamp_s);
            geometry_msgs::msg::PoseStamped out = raw_pose;
            out.pose.orientation.x = q_raw.x();
            out.pose.orientation.y = q_raw.y();
            out.pose.orientation.z = q_raw.z();
            out.pose.orientation.w = q_raw.w();
            return out;
        }

        double dt = stamp_s - kalman_last_stamp_s_;
        if (!std::isfinite(dt) || dt <= 0.0)
        {
            dt = kalman_min_dt_;
        }
        if (dt < kalman_min_dt_)
            dt = kalman_min_dt_;
        if (dt > kalman_max_dt_)
            dt = kalman_max_dt_;

        Eigen::Vector3d axis_meas = raw_axis;
        if (axis_meas.dot(kalman_axis_x_) < 0.0)
        {
            axis_meas = -axis_meas;
        }

        for (int i = 0; i < 3; ++i)
        {
            kalmanPredictUpdate1D(raw_pos[i], dt, kalman_position_process_noise_, kalman_position_measurement_noise_,
                                  kalman_pos_x_[i], kalman_pos_v_[i], kalman_pos_P_[i]);
            kalmanPredictUpdate1D(axis_meas[i], dt, kalman_axis_process_noise_, kalman_axis_measurement_noise_,
                                  kalman_axis_x_[i], kalman_axis_v_[i], kalman_axis_P_[i]);
        }
        kalman_last_stamp_s_ = stamp_s;

        Eigen::Vector3d filtered_axis = kalman_axis_x_;
        const double axis_norm = filtered_axis.norm();
        if (axis_norm > 1e-9)
        {
            filtered_axis /= axis_norm;
        }
        else
        {
            filtered_axis = axis_meas;
        }
        filtered_axis = stabilizeAxisDirection(filtered_axis, &axis_meas);

        Eigen::Vector3d up = Eigen::Vector3d::UnitY();
        if (std::abs(filtered_axis.dot(up)) > 0.99)
        {
            up = Eigen::Vector3d::UnitZ();
        }
        Eigen::Vector3d axis_y = (up - up.dot(filtered_axis) * filtered_axis).normalized();
        Eigen::Vector3d axis_z = filtered_axis.cross(axis_y).normalized();

        Eigen::Matrix3d R;
        R.col(0) = filtered_axis;
        R.col(1) = axis_y;
        R.col(2) = axis_z;
        Eigen::Quaterniond q_filtered(R);
        q_filtered.normalize();

        geometry_msgs::msg::PoseStamped out = raw_pose;
        out.pose.position.x = kalman_pos_x_.x();
        out.pose.position.y = kalman_pos_x_.y();
        out.pose.position.z = kalman_pos_x_.z();
        out.pose.orientation.x = q_filtered.x();
        out.pose.orientation.y = q_filtered.y();
        out.pose.orientation.z = q_filtered.z();
        out.pose.orientation.w = q_filtered.w();
        return out;
    }

    geometry_msgs::msg::PoseStamped PoseEstimateBase::computePoseFromCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const rclcpp::Time &stamp)
    {
        geometry_msgs::msg::PoseStamped pose;
        (void)stamp;
        if (!cloud || cloud->empty())
        {
            return pose;
        }

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud, centroid);

        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*cloud, centroid, covariance);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        Eigen::Matrix3f eig_vecs = solver.eigenvectors();
        Eigen::Vector3f eig_vals = solver.eigenvalues();
        double eta = eig_vals[2] / (eig_vals[0] + eig_vals[1] + eig_vals[2]);
        double lambda = sqrt(eig_vals[1] + eig_vals[0]);
        RCLCPP_INFO(this->get_logger(), "Eigenvalues: %.6f %.6f %.6f eta=%.6f lambda=%.6f", eig_vals[0], eig_vals[1], eig_vals[2], eta, lambda);

        // eigenvectors are in ascending order; principal axis is last column
        Eigen::Vector3f axis_x = eig_vecs.col(2).normalized();
        Eigen::Vector3f axis_y = eig_vecs.col(1).normalized();
        Eigen::Vector3f axis_z = eig_vecs.col(0).normalized();
        (void)axis_z;
        axis_x = stabilizeAxisDirection(axis_x.cast<double>(), nullptr).cast<float>();
        // Eigen::Vector3f axis_x_ = coefficients->values[3] * Eigen::Vector3f::UnitX() +
        //                           coefficients->values[4] * Eigen::Vector3f::UnitY() +
        //                           coefficients->values[5] * Eigen::Vector3f::UnitZ();
        // axis_x_.normalize();
        // double angle_diff = acos(axis_x.dot(axis_x_)) * 180.0 / M_PI;
        // RCLCPP_INFO(this->get_logger(), "Angle between covariance axis and RANSAC axis: %.4f degrees", angle_diff);
        // RCLCPP_INFO(this->get_logger(), "Axis from covariance: [%.4f %.4f %.4f], Axis from RANSAC: [%.4f %.4f %.4f]", axis_x[0], axis_x[1], axis_x[2], axis_x_[0], axis_x_[1], axis_x_[2]);
        // ensure right-handedness
        Eigen::Vector3f corrected_z = axis_x.cross(axis_y).normalized();
        Eigen::Matrix3f R;
        R.col(0) = axis_x;
        R.col(1) = axis_y;
        R.col(2) = corrected_z;

        Eigen::Quaternionf q(R);
        q.normalize();

        // --- 新增：使用 PointDistanceLogger 实例追加点到拟合直线的距离 ---
        // if (statistics_enabled_ && distance_recorder_)
        // {
        //     Eigen::Vector3f line_point(centroid[0], centroid[1], centroid[2]);
        //     Eigen::Vector3f line_dir = axis_x.normalized();
        //     if (!distance_recorder_->appendDistancesToLine(cloud, line_point, line_dir))
        //     {
        //         RCLCPP_WARN(this->get_logger(), "PointDistanceLogger failed to append distances");
        //     }
        // }

        pose.pose.position.x = centroid[0];
        pose.pose.position.y = centroid[1];
        pose.pose.position.z = centroid[2];

        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();

        return pose;
    }

    geometry_msgs::msg::PoseStamped PoseEstimateBase::computePoseFromSACLine(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const rclcpp::Time &stamp)
    {
        geometry_msgs::msg::PoseStamped pose;
        (void)stamp;
        if (!cloud || cloud->empty())
        {
            return pose;
        }

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
            RCLCPP_WARN(this->get_logger(), "RANSAC line fitting failed, fallback to PCA pose");
            return computePoseFromCloud(cloud, stamp);
        }

        Eigen::Vector3f line_point(coefficients.values[0], coefficients.values[1], coefficients.values[2]);
        Eigen::Vector3f axis_x = coefficients.values[3] * Eigen::Vector3f::UnitX() +
                                 coefficients.values[4] * Eigen::Vector3f::UnitY() +
                                 coefficients.values[5] * Eigen::Vector3f::UnitZ();
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

        // --- 新增：使用 PointDistanceLogger 实例追加点到拟合直线的距离 ---
        // if (statistics_enabled_ && distance_recorder_1)
        // {
        //     Eigen::Vector3f line_dir = axis_x;
        //     if (!distance_recorder_1->appendDistancesToLine(cloud, line_point, line_dir))
        //     {
        //         RCLCPP_WARN(this->get_logger(), "PointDistanceLogger failed to append distances");
        //     }
        // }

        pose.pose.position.x = line_point[0];
        pose.pose.position.y = line_point[1];
        pose.pose.position.z = line_point[2];

        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();

        return pose;
    }

    geometry_msgs::msg::PoseStamped PoseEstimateBase::computePoseGaussian(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const cv::Mat &mask_cv, const rclcpp::Time &stamp)
    {
        geometry_msgs::msg::PoseStamped pose;
        (void)mask_cv;
        (void)stamp;

        axispose::GaussianMapSolver solver;
        Eigen::Vector3f out_axis, out_point;
        float out_radius = 0.05f;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>(*cloud));
        if (!solver.estimateAxis(cloud_ptr, out_axis, out_point, out_radius))
        {
            RCLCPP_WARN(this->get_logger(), "GaussianMapSolver failed");
            return pose;
        }

        // Sanity-check direction against PCA axis to avoid occasional normal-space outliers.
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

        // Keep convention consistent with PCA path: line direction is local X axis.
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

    geometry_msgs::msg::PoseStamped PoseEstimateBase::computePoseCeres(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const cv::Mat &mask_cv, const rclcpp::Time &stamp)
    {
        geometry_msgs::msg::PoseStamped pose;
        (void)stamp;

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
            RCLCPP_WARN(this->get_logger(), "Ceres skipped: invalid PCA init, fallback PCA pose");
            return computePoseFromCloud(cloud, stamp);
        }
        Eigen::Vector3d d = stabilizeAxisDirection(pca_axis, nullptr).normalized();
        Eigen::Vector3d p = centroid4.head<3>().cast<double>();
        Eigen::Vector3d m = p.cross(d);

        // Fit a 2D line from YOLO mask as supervision: A*u + B*v + C = 0.
        std::vector<cv::Point> nz;
        cv::findNonZero(mask_cv, nz);
        if (nz.size() < 20)
        {
            RCLCPP_WARN(this->get_logger(), "Ceres skipped: not enough mask points (%zu)", nz.size());
            return computePoseFromCloud(cloud, stamp);
        }

        cv::Vec4f line;
        cv::fitLine(nz, line, cv::DIST_L2, 0, 0.01, 0.01);
        const double vx = static_cast<double>(line[0]);
        const double vy = static_cast<double>(line[1]);
        const double x0 = static_cast<double>(line[2]);
        const double y0 = static_cast<double>(line[3]);
        const double norm_v = std::hypot(vx, vy);
        if (norm_v < 1e-8)
        {
            RCLCPP_WARN(this->get_logger(), "Ceres skipped: invalid fitted 2D line");
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

        axispose::CeresJointOptimizer optimizer;
        const bool ok = optimizer.optimizePose(
            d,
            m,
            cloud_ptr,
            K,
            line2d_abc,
            p,
            std::max(1, ceres_max_iterations_),
            std::max(8, ceres_max_points_),
            std::max(0.0, ceres_weight_2d_),
            std::max(0.0, ceres_weight_pos_));
        if (!ok)
        {
            RCLCPP_WARN(this->get_logger(), "Ceres optimizer failed, fallback to PCA pose");
            return computePoseFromCloud(cloud, stamp);
        }

        // Disambiguate line direction sign using current-frame PCA + temporal history.
        d = stabilizeAxisDirection(d, &pca_axis).normalized();

        // Recover line point from optimized Plucker (d, m), then anchor near cloud centroid.
        const double dn2 = d.squaredNorm();
        Eigen::Vector3d optimized_point = p;
        if (dn2 > 1e-12)
        {
            optimized_point = d.cross(m) / dn2;
            const Eigen::Vector3d centroid = centroid4.head<3>().cast<double>();
            optimized_point = optimized_point + d * d.dot(centroid - optimized_point);

            // Keep motion bounded against initial point to avoid occasional jumps.
            const double max_shift = 0.20; // meters
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

        // Keep convention consistent with PCA path: line direction is local X axis.
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
    }

    PoseEstimatePCA::PoseEstimatePCA(const rclcpp::NodeOptions &options)
        : PoseEstimateBase("pose_estimate_pca", options)
    {
    }

    geometry_msgs::msg::PoseStamped PoseEstimatePCA::computePoseByAlgorithm(
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
        const cv::Mat &mask_cv,
        const rclcpp::Time &stamp)
    {
        (void)mask_cv;
        return computePoseFromCloud(cloud, stamp);
    }

    std::string PoseEstimatePCA::benchmarkLabel() const
    {
        return "pose_pca";
    }

    PoseEstimateRANSAC::PoseEstimateRANSAC(const rclcpp::NodeOptions &options)
        : PoseEstimateBase("pose_estimate_ransac", options)
    {
    }

    geometry_msgs::msg::PoseStamped PoseEstimateRANSAC::computePoseByAlgorithm(
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
        const cv::Mat &mask_cv,
        const rclcpp::Time &stamp)
    {
        (void)mask_cv;
        return computePoseFromSACLine(cloud, stamp);
    }

    std::string PoseEstimateRANSAC::benchmarkLabel() const
    {
        return "pose_ransac";
    }

    PoseEstimateGaussian::PoseEstimateGaussian(const rclcpp::NodeOptions &options)
        : PoseEstimateBase("pose_estimate_gaussian", options)
    {
    }

    geometry_msgs::msg::PoseStamped PoseEstimateGaussian::computePoseByAlgorithm(
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
        const cv::Mat &mask_cv,
        const rclcpp::Time &stamp)
    {
        return computePoseGaussian(cloud, mask_cv, stamp);
    }

    std::string PoseEstimateGaussian::benchmarkLabel() const
    {
        return "pose_gaussian";
    }

    PoseEstimateCeres::PoseEstimateCeres(const rclcpp::NodeOptions &options)
        : PoseEstimateBase("pose_estimate_ceres", options)
    {
    }

    geometry_msgs::msg::PoseStamped PoseEstimateCeres::computePoseByAlgorithm(
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
        const cv::Mat &mask_cv,
        const rclcpp::Time &stamp)
    {
        return computePoseCeres(cloud, mask_cv, stamp);
    }

    std::string PoseEstimateCeres::benchmarkLabel() const
    {
        return "pose_ceres";
    }

} // namespace axispose
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(axispose::PoseEstimatePCA)
RCLCPP_COMPONENTS_REGISTER_NODE(axispose::PoseEstimateRANSAC)
RCLCPP_COMPONENTS_REGISTER_NODE(axispose::PoseEstimateGaussian)
RCLCPP_COMPONENTS_REGISTER_NODE(axispose::PoseEstimateCeres)
