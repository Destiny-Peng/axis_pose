#include "axispose/poseEstimate.hpp"

#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <Eigen/Dense>
#include "axispose/logger.hpp"
#include <sstream>
#include <cstdint>

namespace axispose
{

    PoseEstimate::PoseEstimate(const rclcpp::NodeOptions &options) : rclcpp::Node("pose_estimate", options)
    {
        // parameters
        this->declare_parameter("depth_image_topic", std::string("/camera/depth/image_raw"));
        this->declare_parameter("mask_topic", std::string("/yolo/mask"));
        this->declare_parameter("camera_info_topic", std::string("/camera/camera_info"));
        this->declare_parameter("voxel_leaf_size", voxel_leaf_size_);
        this->declare_parameter("sor_mean_k", sor_mean_k_);
        this->declare_parameter("sor_std_mul", sor_std_mul_);
        this->declare_parameter("use_sor", use_sor_);
        this->declare_parameter("use_sacline", use_sacline_);
        this->declare_parameter("use_euclidean_cluster", use_euclidean_cluster_);
        this->declare_parameter("cluster_mode", cluster_mode_); // 0: closest to origin, 1: largest cluster, 2: RANSAC line inliers
        this->declare_parameter("sacline_distance_threshold", sacline_distance_threshold_);

        // statistics parameters
        this->declare_parameter("statistics_directory_path", statistics_directory_path_);
        this->declare_parameter("statistics_enabled", statistics_enabled_);

        std::string depth_image_topic = this->get_parameter("depth_image_topic").as_string();
        std::string mask_topic = this->get_parameter("mask_topic").as_string();
        std::string camera_info_topic = this->get_parameter("camera_info_topic").as_string();
        voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
        sor_mean_k_ = this->get_parameter("sor_mean_k").as_double();
        sor_std_mul_ = this->get_parameter("sor_std_mul").as_double();
        statistics_directory_path_ = this->get_parameter("statistics_directory_path").as_string();
        statistics_enabled_ = this->get_parameter("statistics_enabled").as_bool();
        use_sor_ = this->get_parameter("use_sor").as_bool();
        use_sacline_ = this->get_parameter("use_sacline").as_bool();
        use_euclidean_cluster_ = this->get_parameter("use_euclidean_cluster").as_bool();
        cluster_mode_ = this->get_parameter("cluster_mode").as_int();
        sacline_distance_threshold_ = this->get_parameter("sacline_distance_threshold").as_double();

        RCLCPP_INFO(this->get_logger(), "PoseEstimate node starting. Depth: %s Mask: %s CameraInfo: %s", depth_image_topic.c_str(), mask_topic.c_str(), camera_info_topic.c_str());

        rclcpp::QoS qos(rclcpp::KeepLast(5));
        // message_filters subscribers
        depth_sub_.subscribe(this, depth_image_topic, qos.get_rmw_qos_profile());
        mask_sub_.subscribe(this, mask_topic, qos.get_rmw_qos_profile());

        sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(ApproxSyncPolicy(10), depth_sub_, mask_sub_);
        sync_->registerCallback(&PoseEstimate::syncCallback, this);

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic, qos,
            std::bind(&PoseEstimate::cameraInfoCallback, this, std::placeholders::_1));

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/shaft/pose", 10);
        debug_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/shaft/debug_cloud", 1);

        // initialize statistics collector instance
        if (statistics_enabled_)
        {
            distance_recorder_ = std::make_shared<PointDistanceLogger>(statistics_directory_path_ + "/" + distance_file_name);
            distance_recorder_1 = std::make_shared<PointDistanceLogger>(statistics_directory_path_ + "/" + distance_file_name_1);
            point_number_logger_ = std::make_shared<PointNumberLogger>(statistics_directory_path_ + "/" + pointnumber_file_name);
            timing_logger_ = std::make_shared<SimpleTimingLogger>(statistics_directory_path_ + "/" + timing_file_name);
            RCLCPP_INFO(this->get_logger(), "PointDistanceLogger initialized, output path: %s", statistics_directory_path_.c_str());
        }
    }

    void PoseEstimate::cameraInfoCallback(const CameraInfo::SharedPtr msg)
    {
        if (!have_intrinsics_)
        {
            fx_ = msg->k[0];
            fy_ = msg->k[4];
            cx_ = msg->k[2];
            cy_ = msg->k[5];
            cx_ *= this->scale_x;
            cy_ *= this->scale_y;
            fx_ *= this->scale_x;
            fy_ *= this->scale_y;
            frame_id_ = msg->header.frame_id;
            have_intrinsics_ = true;
            RCLCPP_INFO(this->get_logger(), "Got camera intrinsics fx=%.2f fy=%.2f cx=%.2f cy=%.2f frame=%s", fx_, fy_, cx_, cy_, frame_id_.c_str());
        }
    }

    void PoseEstimate::syncCallback(const Image::ConstSharedPtr depth_msg, const Image::ConstSharedPtr mask_msg)
    {
        if (!have_intrinsics_)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "No camera intrinsics yet, skipping frame");
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

        // size不一样，对mask做resize,缩小到和depth一样大
        if (depth_cv.size() != mask_cv.size())
        {
            // adjust cx, cy, fx, fy according to resize
            if (this->scale_x != static_cast<double>(depth_cv.cols) / static_cast<double>(mask_cv.cols))
            {
                this->scale_x = static_cast<double>(depth_cv.cols) / static_cast<double>(mask_cv.cols);
                this->scale_y = static_cast<double>(depth_cv.rows) / static_cast<double>(mask_cv.rows);
                have_intrinsics_ = false;
                return;
            }
            cv::resize(mask_cv, mask_cv, depth_cv.size(), 0, 0, cv::INTER_NEAREST);
        }

        // filter depth with mask (produce depth_filtered but we will use only depth_filtered downstream)
        cv::Mat depth_filtered = cv::Mat::zeros(depth_cv.size(), depth_cv.type());
        depth_cv.copyTo(depth_filtered, mask_cv);

        // convert to organized point cloud (one point per pixel, NaN for invalid) - now only depth is passed
        auto organized = depthMaskToPointCloud(depth_filtered);
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

        // denoise
        if (statistics_enabled_)
        {
            timing_logger_->tik("denoise");
        }
        denoisePointCloud(valid_cloud);
        if (statistics_enabled_)
        {
            timing_logger_->tok("denoise");
        }

        // compute pose using valid_cloud
        geometry_msgs::msg::PoseStamped pose_msg;
        if (use_sacline_)
        {
            if (statistics_enabled_)
            {
                timing_logger_->tik("sacline");
            }
            pose_msg = computePoseFromSACLine(valid_cloud, depth_msg->header.stamp);
            if (statistics_enabled_)
            {
                timing_logger_->tok("sacline");
            }
        }
        else
        {
            if (statistics_enabled_)
            {
                timing_logger_->tik("covariance");
            }
            pose_msg = computePoseFromCloud(valid_cloud, depth_msg->header.stamp);
            if (statistics_enabled_)
            {
                timing_logger_->tok("covariance");
            }
        }
        pose_msg.header.stamp = depth_msg->header.stamp;
        pose_msg.header.frame_id = frame_id_;
        pose_pub_->publish(pose_msg);

        // publish debug organized cloud
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*valid_cloud, cloud_msg);
        cloud_msg.header.stamp = depth_msg->header.stamp;
        cloud_msg.header.frame_id = frame_id_;
        debug_cloud_pub_->publish(cloud_msg);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr PoseEstimate::depthMaskToPointCloud(const cv::Mat &depth)
    {
        // Create an organized cloud with width = cols, height = rows
        const int rows = depth.rows;
        const int cols = depth.cols;
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        cloud->is_dense = false;
        cloud->width = static_cast<uint32_t>(cols);
        cloud->height = static_cast<uint32_t>(rows);
        cloud->points.resize(cloud->width * cloud->height);
        cloud->header.frame_id = frame_id_;

        // compute constants similar to image_pipeline
        double unit_scaling = 0.001; // assume uint16 in mm -> meters
        bool depth_is_float = (depth.type() == CV_32F);
        if (depth_is_float)
            unit_scaling = 1.0; // already in meters

        float constant_x = static_cast<float>(unit_scaling / fx_);
        float constant_y = static_cast<float>(unit_scaling / fy_);
        float center_x = static_cast<float>(cx_);
        float center_y = static_cast<float>(cy_);
        float bad_point = std::numeric_limits<float>::quiet_NaN();

        for (int v = 0; v < rows; ++v)
        {
            for (int u = 0; u < cols; ++u)
            {
                size_t idx = static_cast<size_t>(v) * cols + u;
                // default bad
                pcl::PointXYZ &pt = cloud->points[idx];
                pt.x = bad_point;
                pt.y = bad_point;
                pt.z = bad_point;

                double depth_val = 0.0;
                if (!depth_is_float)
                {
                    uint16_t d = depth.at<uint16_t>(v, u);
                    if (d == 0 || d > 10000) // ignore zero
                        continue;
                    depth_val = static_cast<double>(d); // in mm
                }
                else
                {
                    float d = depth.at<float>(v, u);
                    if (!(d > 0.0f))
                        continue;
                    depth_val = static_cast<double>(d); // in meters
                }

                // compute XYZ using image_pipeline style: X = (u - cx) * depth * (unit_scaling / fx)
                float z = static_cast<float>(depth_val * unit_scaling);
                float x = (static_cast<float>(u) - center_x) * static_cast<float>(depth_val) * constant_x;
                float y = (static_cast<float>(v) - center_y) * static_cast<float>(depth_val) * constant_y;

                // adjust coordinate system to match previous convention (z forward -> keep as z)
                pt.x = z;
                pt.y = -x;
                pt.z = -y;
            }
        }

        return cloud;
    }

    void PoseEstimate::denoisePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        // voxel grid downsample and statistical outlier removal, then keep nearest cluster to origin
        // reuse a single temporary cloud buffer to avoid repeated allocations
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>());

        // voxel grid downsample
        if (voxel_leaf_size_ > 0)
        {
            pcl::VoxelGrid<pcl::PointXYZ> vg;
            vg.setInputCloud(cloud);
            vg.setLeafSize(static_cast<float>(voxel_leaf_size_), static_cast<float>(voxel_leaf_size_), static_cast<float>(voxel_leaf_size_));
            vg.filter(*tmp);
            if (statistics_enabled_)
            {
                point_number_logger_->logCounts(cloud->size(), tmp->size());
            }
            cloud.swap(tmp);
            tmp->clear();
        }
        if (use_sor_)
        {

            // statistical outlier removal
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(cloud);
            sor.setMeanK(static_cast<int>(sor_mean_k_));
            sor.setStddevMulThresh(sor_std_mul_);
            sor.filter(*tmp);
            cloud.swap(tmp);
            tmp->clear();
        }
        if (use_euclidean_cluster_)
        {

            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
            tree->setInputCloud(cloud);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(0.5f); // 50cm
            ec.setMinClusterSize(50);
            ec.setMaxClusterSize(25000);
            ec.setSearchMethod(tree);
            ec.setInputCloud(cloud);
            ec.extract(cluster_indices);
            pcl::PointIndices::Ptr best_cluster_indices(new pcl::PointIndices);
            std::cout << "Found " << cluster_indices.size() << " clusters." << std::endl;
            if (!cluster_indices.empty())
            {
                // 找到质心离原点 (0,0,0) 最近的 cluster
                size_t best_idx = 0;
                switch (cluster_mode_)
                {
                case 0:
                { // 策略1：找质心离原点最近的cluster
                    float min_dist = std::numeric_limits<float>::max();
                    for (size_t i = 0; i < cluster_indices.size(); ++i)
                    {
                        Eigen::Vector4f centroid;
                        pcl::compute3DCentroid(*cloud, cluster_indices[i].indices, centroid);

                        float dist = centroid.head<3>().norm();

                        if (dist < min_dist)
                        {
                            min_dist = dist;
                            best_idx = i;
                        }
                    }
                }
                break;
                case 1:
                { // 策略2：找点数最多的cluster
                    size_t max_size = 0;
                    for (const auto &indices : cluster_indices)
                    {
                        if (indices.indices.size() > max_size)
                        {
                            max_size = indices.indices.size();
                            best_idx = &indices - &cluster_indices[0];
                        }
                    }
                }
                break;
                case 2:
                default:
                { // 策略3：对每个cluster做ransac,找拟合直线inliers最多的cluster
                    if (statistics_enabled_)
                    {
                        timing_logger_->tik("cluster_ransac");
                    }
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    size_t inlier_cnt = 0;
                    // 使用extraction提取当前cluster的点云
                    for (const auto &cluster : cluster_indices)
                    {
                        cluster_cloud->points.clear();
                        for (const auto &indices : cluster.indices)
                        {
                            cluster_cloud->points.push_back(cloud->points[indices]);
                        }
                        cluster_cloud->width = cluster_cloud->points.size();
                        cluster_cloud->height = 1;

                        pcl::ModelCoefficients::Ptr coefficients_tmp(new pcl::ModelCoefficients);
                        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                        pcl::SACSegmentation<pcl::PointXYZ> seg;
                        seg.setOptimizeCoefficients(true);
                        seg.setModelType(pcl::SACMODEL_LINE);
                        seg.setMethodType(pcl::SAC_RANSAC);
                        seg.setMaxIterations(200);
                        seg.setDistanceThreshold(sacline_distance_threshold_);
                        seg.setInputCloud(cluster_cloud);
                        seg.segment(*inliers, *coefficients_tmp);

                        if (!inliers->indices.empty() && inliers->indices.size() > inlier_cnt)
                        {
                            inlier_cnt = inliers->indices.size();
                            best_idx = &cluster - &cluster_indices[0];
                            *coefficients = *coefficients_tmp;
                            *best_cluster_indices = *inliers;
                        }
                    }
                    if (statistics_enabled_)
                    {
                        timing_logger_->tok("cluster_ransac");
                    }
                }
                break;
                }

                pcl::ExtractIndices<pcl::PointXYZ> extract;
                extract.setInputCloud(cloud);
                // *best_cluster_indices = cluster_indices[best_idx];
                pcl::PointIndices::Ptr indices_ptr(new pcl::PointIndices(cluster_indices[best_idx]));
                extract.setIndices(indices_ptr);
                extract.setNegative(false);
                extract.filter(*tmp);
                cloud.swap(tmp);
                tmp->clear();
                if (use_sacline_)
                {
                    extract.setInputCloud(cloud);
                    extract.setIndices(best_cluster_indices);
                    extract.setNegative(false);
                    extract.filter(*tmp);
                    cloud.swap(tmp);
                }
            }
        }

        // if (use_sacline_)
        // {
        //     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        //     // Create the segmentation object
        //     pcl::SACSegmentation<pcl::PointXYZ> seg;
        //     // Optional
        //     seg.setOptimizeCoefficients(true);
        //     // Mandatory
        //     seg.setModelType(pcl::SACMODEL_LINE);
        //     seg.setMethodType(pcl::SAC_RANSAC);
        //     seg.setDistanceThreshold(sacline_distance_threshold_);

        //     seg.setInputCloud(cloud);
        //     seg.segment(*inliers, *coefficients);
        //     if (!inliers->indices.empty())
        //     {
        //         pcl::ExtractIndices<pcl::PointXYZ> extract;
        //         extract.setInputCloud(cloud);
        //         extract.setIndices(inliers);
        //         extract.setNegative(false);
        //         extract.filter(*tmp);
        //         cloud.swap(tmp);
        //         tmp->clear();
        //     }
        // }
    }

    geometry_msgs::msg::PoseStamped PoseEstimate::computePoseFromCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const rclcpp::Time &stamp)
    {
        geometry_msgs::msg::PoseStamped pose;
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

    geometry_msgs::msg::PoseStamped PoseEstimate::computePoseFromSACLine(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const rclcpp::Time &stamp)
    {
        geometry_msgs::msg::PoseStamped pose;
        if (!cloud || cloud->empty())
        {
            return pose;
        }

        Eigen::Vector3f line_point(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        Eigen::Vector3f axis_x = coefficients->values[3] * Eigen::Vector3f::UnitX() +
                                 coefficients->values[4] * Eigen::Vector3f::UnitY() +
                                 coefficients->values[5] * Eigen::Vector3f::UnitZ();
        axis_x.normalize();
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

} // namespace axispose

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(axispose::PoseEstimate)