#ifndef AXISPOSE_POSEESTIMATE_HPP_
#define AXISPOSE_POSEESTIMATE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>

#include "axispose/logger.hpp"

#include <memory>

namespace axispose
{

    class PoseEstimate : public rclcpp::Node
    {
    public:
        explicit PoseEstimate(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

    private:
        using Image = sensor_msgs::msg::Image;
        using CameraInfo = sensor_msgs::msg::CameraInfo;
        using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;

        // Subscribers and synchronizer
        message_filters::Subscriber<Image> depth_sub_;
        message_filters::Subscriber<Image> mask_sub_;
        std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;

        rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_color_sub_;
        rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_depth_sub_;

        // Publishers
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_cloud_pub_;

        // Intrinsics
        std::atomic<bool> have_intrinsics_{false};
        // depth intrinsics (used for point cloud conversion)
        std::atomic<bool> have_intrinsics_depth_{false};
        double fx_{0.0}, fy_{0.0}, cx_{0.0}, cy_{0.0};
        // color intrinsics (used for alignment)
        std::atomic<bool> have_intrinsics_color_{false};
        double color_fx_{0.0}, color_fy_{0.0}, color_cx_{0.0}, color_cy_{0.0};
        double scale_x{1.0}, scale_y{1.0};
        std::string frame_id_ = "base_link";

        // Parameters
        double voxel_leaf_size_ = 0.05; // meters
        double sor_mean_k_ = 50;
        double sor_std_mul_ = 1.0;
        // Statistics collector (instance-based), created in constructor
        std::string statistics_directory_path_;
        std::shared_ptr<PointDistanceLogger> distance_recorder_;
        std::string distance_file_name = "pose_point_to_line.csv";
        std::shared_ptr<PointDistanceLogger> distance_recorder_1;
        std::string distance_file_name_1 = "pose_point_to_line_sac.csv";
        std::shared_ptr<PointNumberLogger> point_number_logger_;
        std::string pointnumber_file_name = "point_numbers.csv";
        std::shared_ptr<SimpleTimingLogger> timing_logger_;
        std::string timing_file_name = "timing.csv";
        pcl::ModelCoefficients::Ptr coefficients = std::make_shared<pcl::ModelCoefficients>();
        bool statistics_enabled_ = true;
        bool use_sor_ = true;
        bool use_sacline_ = true;
        bool use_euclidean_cluster_ = true;
        int cluster_mode_ = 0;                     // 0: closest to origin, 1: largest cluster, 2: RANSAC line inliers
        double sacline_distance_threshold_ = 0.05; // meters

        // Callbacks
        void cameraInfoDepthCallback(const CameraInfo::SharedPtr msg);
        void cameraInfoColorCallback(const CameraInfo::SharedPtr msg);
        void syncCallback(const Image::ConstSharedPtr depth_msg, const Image::ConstSharedPtr mask_msg);

        // Helpers
        pcl::PointCloud<pcl::PointXYZ>::Ptr depthMaskToPointCloud(const cv::Mat &depth);
        // align depth image to color image size using intrinsics
        cv::Mat alignDepthToColor(const cv::Mat &depth, int color_width, int color_height);
        void denoisePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
        geometry_msgs::msg::PoseStamped computePoseFromCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const rclcpp::Time &stamp);
        geometry_msgs::msg::PoseStamped computePoseFromSACLine(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const rclcpp::Time &stamp);
    };

} // namespace axispose

#endif // AXISPOSE_POSEESTIMATE_HPP_