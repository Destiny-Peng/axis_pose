#include "axispose/poseEstimate.hpp"

#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>

namespace axispose
{

    PoseEstimate::PoseEstimate(ros::NodeHandle &nh, ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh)
    {
        std::string depth_topic = "/camera/depth/image_raw";
        std::string mask_topic = "/yolo/mask";
        std::string camera_info_topic = "/camera/camera_info";
        pnh_.param<std::string>("depth_topic", depth_topic, depth_topic);
        pnh_.param<std::string>("mask_topic", mask_topic, mask_topic);
        pnh_.param<std::string>("camera_info_topic", camera_info_topic, camera_info_topic);

        pnh_.param<double>("voxel_leaf_size", voxel_leaf_size_, voxel_leaf_size_);
        pnh_.param<double>("sor_mean_k", sor_mean_k_, sor_mean_k_);
        pnh_.param<double>("sor_std_mul", sor_std_mul_, sor_std_mul_);

        // clustering parameters (exposed)
        pnh_.param<double>("cluster_tolerance", cluster_tolerance_, cluster_tolerance_);
        pnh_.param<int>("cluster_min_size", cluster_min_size_, cluster_min_size_);
        pnh_.param<int>("cluster_max_size", cluster_max_size_, cluster_max_size_);

        ROS_INFO("PoseEstimate node starting. depth_topic=%s mask_topic=%s camera_info=%s", depth_topic.c_str(), mask_topic.c_str(), camera_info_topic.c_str());

        // subscribe using configured topic names
        depth_sub_.subscribe(nh_, depth_topic, 5);
        mask_sub_.subscribe(nh_, mask_topic, 5);

        sync_.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(10), depth_sub_, mask_sub_));
        sync_->registerCallback(boost::bind(&PoseEstimate::syncCallback, this, _1, _2));

        camera_info_sub_ = nh_.subscribe<sensor_msgs::CameraInfo>(camera_info_topic, 5, &PoseEstimate::cameraInfoCallback, this);

        pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/shaft/pose", 10);
        debug_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/shaft/debug_cloud", 1);

        // start parameter polling timer for dynamic updates (1 Hz)
        param_timer_ = nh_.createTimer(ros::Duration(1.0), &PoseEstimate::updateParameters, this);
    }

    void PoseEstimate::updateParameters(const ros::TimerEvent &)
    {
        // read params from the private node handle and update members if changed
        double new_voxel = voxel_leaf_size_;
        double new_sor_mean = sor_mean_k_;
        double new_sor_std = sor_std_mul_;
        double new_cluster_tol = cluster_tolerance_;
        int new_cluster_min = cluster_min_size_;
        int new_cluster_max = cluster_max_size_;

        pnh_.param<double>("voxel_leaf_size", new_voxel, voxel_leaf_size_);
        pnh_.param<double>("sor_mean_k", new_sor_mean, sor_mean_k_);
        pnh_.param<double>("sor_std_mul", new_sor_std, sor_std_mul_);
        pnh_.param<double>("cluster_tolerance", new_cluster_tol, cluster_tolerance_);
        pnh_.param<int>("cluster_min_size", new_cluster_min, cluster_min_size_);
        pnh_.param<int>("cluster_max_size", new_cluster_max, cluster_max_size_);

        bool changed = false;
        if (new_voxel != voxel_leaf_size_)
        {
            ROS_INFO("voxel_leaf_size changed: %.6f -> %.6f", voxel_leaf_size_, new_voxel);
            voxel_leaf_size_ = new_voxel;
            changed = true;
        }
        if (new_sor_mean != sor_mean_k_)
        {
            ROS_INFO("sor_mean_k changed: %.1f -> %.1f", sor_mean_k_, new_sor_mean);
            sor_mean_k_ = new_sor_mean;
            changed = true;
        }
        if (new_sor_std != sor_std_mul_)
        {
            ROS_INFO("sor_std_mul changed: %.3f -> %.3f", sor_std_mul_, new_sor_std);
            sor_std_mul_ = new_sor_std;
            changed = true;
        }
        if (new_cluster_tol != cluster_tolerance_)
        {
            ROS_INFO("cluster_tolerance changed: %.3f -> %.3f", cluster_tolerance_, new_cluster_tol);
            cluster_tolerance_ = new_cluster_tol;
            changed = true;
        }
        if (new_cluster_min != cluster_min_size_)
        {
            ROS_INFO("cluster_min_size changed: %d -> %d", cluster_min_size_, new_cluster_min);
            cluster_min_size_ = new_cluster_min;
            changed = true;
        }
        if (new_cluster_max != cluster_max_size_)
        {
            ROS_INFO("cluster_max_size changed: %d -> %d", cluster_max_size_, new_cluster_max);
            cluster_max_size_ = new_cluster_max;
            changed = true;
        }
        if (changed)
        {
            // nothing else required immediately; new params will apply on next denoise invocation
        }
    }

    void PoseEstimate::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr &msg)
    {
        if (!have_intrinsics_)
        {
            fx_ = msg->K[0];
            fy_ = msg->K[4];
            cx_ = msg->K[2];
            cy_ = msg->K[5];
            cx_ *= this->scale_x;
            cy_ *= this->scale_y;
            fx_ *= this->scale_x;
            fy_ *= this->scale_y;
            frame_id_ = msg->header.frame_id;
            have_intrinsics_ = true;
            ROS_INFO("Got camera intrinsics fx=%.2f fy=%.2f cx=%.2f cy=%.2f frame=%s", fx_, fy_, cx_, cy_, frame_id_.c_str());
        }
    }

    void PoseEstimate::syncCallback(const sensor_msgs::ImageConstPtr depth_msg, const sensor_msgs::ImageConstPtr mask_msg)
    {
        if (!have_intrinsics_)
        {
            ROS_WARN_THROTTLE(5.0, "No camera intrinsics yet, skipping frame");
            return;
        }

        cv::Mat depth_cv;
        try
        {
            depth_cv = cv::Mat(depth_msg->height, depth_msg->width, CV_16U, const_cast<unsigned char *>(depth_msg->data.data())).clone();
        }
        catch (const std::exception &e)
        {
            ROS_ERROR("cv_bridge depth conversion failed: %s", e.what());
            return;
        }

        cv::Mat mask_cv;
        try
        {
            mask_cv = cv::Mat(mask_msg->height, mask_msg->width, CV_8U, const_cast<unsigned char *>(mask_msg->data.data())).clone();
        }
        catch (const std::exception &e)
        {
            ROS_ERROR("cv_bridge mask conversion failed: %s", e.what());
            return;
        }

        if (depth_cv.size() != mask_cv.size())
        {
            if (this->scale_x != static_cast<double>(depth_cv.cols) / static_cast<double>(mask_cv.cols))
            {
                this->scale_x = static_cast<double>(depth_cv.cols) / static_cast<double>(mask_cv.cols);
                this->scale_y = static_cast<double>(depth_cv.rows) / static_cast<double>(mask_cv.rows);
                have_intrinsics_ = false;
                return;
            }
            cv::resize(mask_cv, mask_cv, depth_cv.size(), 0, 0, cv::INTER_NEAREST);
        }

        cv::Mat depth_filtered = cv::Mat::zeros(depth_cv.size(), depth_cv.type());
        depth_cv.copyTo(depth_filtered, mask_cv);

        auto organized = depthMaskToPointCloud(depth_filtered);
        if (!organized || organized->empty())
        {
            ROS_WARN_THROTTLE(2.0, "Empty organized point cloud after mask filtering");
            return;
        }

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
            ROS_WARN_THROTTLE(2.0, "No valid points after organized->valid extraction");
            return;
        }

        denoisePointCloud(valid_cloud);

        auto pose_msg = computePoseFromCloud(valid_cloud, depth_msg->header.stamp);
        pose_msg.header.stamp = depth_msg->header.stamp;
        pose_msg.header.frame_id = frame_id_;
        pose_pub_.publish(pose_msg);

        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*valid_cloud, cloud_msg);
        cloud_msg.header.stamp = depth_msg->header.stamp;
        cloud_msg.header.frame_id = frame_id_;
        debug_cloud_pub_.publish(cloud_msg);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr PoseEstimate::depthMaskToPointCloud(const cv::Mat &depth)
    {
        const int rows = depth.rows;
        const int cols = depth.cols;
        auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr;

        cloud->is_dense = false;
        cloud->width = static_cast<uint32_t>(cols);
        cloud->height = static_cast<uint32_t>(rows);
        cloud->points.resize(cloud->width * cloud->height);
        cloud->header.frame_id = frame_id_;

        double unit_scaling = 0.001;
        bool depth_is_float = (depth.type() == CV_32F);
        if (depth_is_float)
            unit_scaling = 1.0;

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
                pcl::PointXYZ &pt = cloud->points[idx];
                pt.x = bad_point;
                pt.y = bad_point;
                pt.z = bad_point;

                double depth_val = 0.0;
                if (!depth_is_float)
                {
                    uint16_t d = depth.at<uint16_t>(v, u);
                    if (d == 0)
                        continue;
                    depth_val = static_cast<double>(d);
                }
                else
                {
                    float d = depth.at<float>(v, u);
                    if (!(d > 0.0f))
                        continue;
                    depth_val = static_cast<double>(d);
                }

                float z = static_cast<float>(depth_val * unit_scaling);
                float x = (static_cast<float>(u) - center_x) * static_cast<float>(depth_val) * constant_x;
                float y = (static_cast<float>(v) - center_y) * static_cast<float>(depth_val) * constant_y;

                pt.x = z;
                pt.y = -x;
                pt.z = -y;
            }
        }

        return cloud;
    }

    void PoseEstimate::denoisePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>());

        if (voxel_leaf_size_ > 0)
        {
            pcl::VoxelGrid<pcl::PointXYZ> vg;
            vg.setInputCloud(cloud);
            vg.setLeafSize(static_cast<float>(voxel_leaf_size_), static_cast<float>(voxel_leaf_size_), static_cast<float>(voxel_leaf_size_));
            vg.filter(*tmp);
            cloud.swap(tmp);
            tmp->clear();
        }

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
        ec.setClusterTolerance(cluster_tolerance_);
        ec.setMinClusterSize(cluster_min_size_);
        ec.setMaxClusterSize(cluster_max_size_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

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
            // Use PCL's Ptr (boost::shared_ptr) for indices to avoid ownership/allocator mismatch
            pcl::PointIndices::Ptr indices_ptr(new pcl::PointIndices(cluster_indices[best_idx]));
            extract.setIndices(indices_ptr);
            extract.setNegative(false);
            extract.filter(*tmp);
            cloud.swap(tmp);
        }
    }

    geometry_msgs::PoseStamped PoseEstimate::computePoseFromCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const ros::Time &)
    {
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

        Eigen::Vector3f axis_x = eig_vecs.col(2).normalized();
        Eigen::Vector3f axis_y = eig_vecs.col(1).normalized();
        Eigen::Vector3f axis_z = eig_vecs.col(0).normalized();
        ROS_INFO("Eigenvalues: %.6f %.6f %.6f eta=%.6f lambda=%.6f", eig_vals[0], eig_vals[1], eig_vals[2], eta, lambda);

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
