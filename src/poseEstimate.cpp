#include "axispose/poseEstimate.hpp"

#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <Eigen/Dense>

namespace axispose
{

    PoseEstimate::PoseEstimate(const rclcpp::NodeOptions &options) : rclcpp::Node("pose_estimate", options)
    {
        // parameters
        this->declare_parameter("depth_topic", std::string("/camera/depth/image_raw"));
        this->declare_parameter("mask_topic", std::string("/yolo/mask"));
        this->declare_parameter("camera_info_topic", std::string("/camera/camera_info"));
        this->declare_parameter("voxel_leaf_size", voxel_leaf_size_);
        this->declare_parameter("sor_mean_k", sor_mean_k_);
        this->declare_parameter("sor_std_mul", sor_std_mul_);

        std::string depth_topic = this->get_parameter("depth_topic").as_string();
        std::string mask_topic = this->get_parameter("mask_topic").as_string();
        std::string camera_info_topic = this->get_parameter("camera_info_topic").as_string();
        voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
        sor_mean_k_ = this->get_parameter("sor_mean_k").as_double();
        sor_std_mul_ = this->get_parameter("sor_std_mul").as_double();

        RCLCPP_INFO(this->get_logger(), "PoseEstimate node starting. Depth: %s Mask: %s CameraInfo: %s", depth_topic.c_str(), mask_topic.c_str(), camera_info_topic.c_str());

        rclcpp::QoS qos(rclcpp::KeepLast(5));
        // message_filters subscribers
        depth_sub_.subscribe(this, "/camera/depth/image_raw", qos.get_rmw_qos_profile());
        mask_sub_.subscribe(this, "/yolo/mask", qos.get_rmw_qos_profile());

        sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(ApproxSyncPolicy(10), depth_sub_, mask_sub_);
        sync_->registerCallback(&PoseEstimate::syncCallback, this);

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic, qos,
            std::bind(&PoseEstimate::cameraInfoCallback, this, std::placeholders::_1));

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/shaft/pose", 10);
        debug_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/shaft/debug_cloud", 1);
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
        denoisePointCloud(valid_cloud);

        // compute pose using valid_cloud
        auto pose_msg = computePoseFromCloud(valid_cloud, depth_msg->header.stamp);
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
                    if (d == 0) // ignore zero
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
            cloud.swap(tmp);
            tmp->clear();
        }

        // statistical outlier removal
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(static_cast<int>(sor_mean_k_));
        sor.setStddevMulThresh(sor_std_mul_);
        sor.filter(*tmp);
        cloud.swap(tmp);
        tmp->clear();

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.5f); // 50cm
        ec.setMinClusterSize(10);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        // 找到质心离原点 (0,0,0) 最近的 cluster
        if (!cluster_indices.empty())
        {
            size_t best_idx = 0;
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

            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(cloud);
            auto indices_ptr = std::make_shared<pcl::PointIndices>(cluster_indices[best_idx]);
            extract.setIndices(indices_ptr);
            extract.setNegative(false);
            extract.filter(*tmp);
            cloud.swap(tmp);
        }
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

        // ensure right-handedness
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

} // namespace axispose

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(axispose::PoseEstimate)