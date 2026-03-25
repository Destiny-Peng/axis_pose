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

#include "axispose/benchmark.hpp"
#include "axispose/debug_manager.hpp"
#include "axispose/point_cloud_processor.hpp"
#include "axispose/depth_aligner.hpp"
#include "axispose/gaussian_map_solver.hpp"
#include "axispose/ceres_joint_optimizer.hpp"

#include <memory>
#include <atomic>

namespace axispose
{

    class PoseEstimateBase : public rclcpp::Node
    {
    public:
        explicit PoseEstimateBase(const std::string &node_name, const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
        virtual ~PoseEstimateBase() = default;

    protected:
        using Image = sensor_msgs::msg::Image;
        using CameraInfo = sensor_msgs::msg::CameraInfo;
        using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;

        virtual geometry_msgs::msg::PoseStamped computePoseByAlgorithm(
            pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
            const cv::Mat &mask_cv,
            const rclcpp::Time &stamp) = 0;
        virtual std::string benchmarkLabel() const = 0;

        geometry_msgs::msg::PoseStamped computePoseFromCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const rclcpp::Time &stamp);
        geometry_msgs::msg::PoseStamped computePoseFromSACLine(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const rclcpp::Time &stamp);
        geometry_msgs::msg::PoseStamped computePoseGaussian(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const cv::Mat &mask_cv, const rclcpp::Time &stamp);
        geometry_msgs::msg::PoseStamped computePoseCeres(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const cv::Mat &mask_cv, const rclcpp::Time &stamp);

    private:
        // Subscribers and synchronizer
        message_filters::Subscriber<Image> depth_sub_;
        message_filters::Subscriber<Image> mask_sub_;
        std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;

        rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_color_sub_;
        rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_depth_sub_;

        // Publishers
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_cloud_pub_;

        // Intrinsics (stored as OpenCV camera matrices for readability and consistency).
        std::atomic<bool> have_intrinsics_depth_{false};
        cv::Mat depth_camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
        std::atomic<bool> have_intrinsics_color_{false};
        cv::Mat color_camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
        std::string color_frame_id_;
        std::string frame_id_ = "base_link";

        // Parameters
        double voxel_leaf_size_ = 0.05; // meters
        double sor_mean_k_ = 50;
        double sor_std_mul_ = 1.0;
        // Statistics collector path
        std::string statistics_directory_path_;
        // Benchmark / metrics logger
        std::unique_ptr<AlgorithmBenchmark> benchmark_;
        bool statistics_enabled_ = true;
        bool use_sor_ = true;
        bool use_euclidean_cluster_ = true;
        int cluster_mode_ = 0;                     // 0: closest to origin, 1: largest cluster, 2: RANSAC line inliers
        double sacline_distance_threshold_ = 0.05; // meters

        // Debug manager (runtime flags via parameter `debug_flags`)
        std::shared_ptr<DebugManager> debug_;

        // Helpers (extracted)
        std::unique_ptr<PointCloudProcessor> pc_processor_;
        std::unique_ptr<DepthAligner> depth_aligner_;

        // Callbacks
        void cameraInfoDepthCallback(const CameraInfo::SharedPtr msg);
        void cameraInfoColorCallback(const CameraInfo::SharedPtr msg);
        void syncCallback(const Image::ConstSharedPtr depth_msg, const Image::ConstSharedPtr mask_msg);

        // Helpers
        pcl::PointCloud<pcl::PointXYZ>::Ptr depthMaskToPointCloud(const cv::Mat &depth);
        // align depth image to color image size using intrinsics
        cv::Mat alignDepthToColor(const cv::Mat &depth, int color_width, int color_height);
        void denoisePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    };

    class PoseEstimatePCA : public PoseEstimateBase
    {
    public:
        explicit PoseEstimatePCA(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

    protected:
        geometry_msgs::msg::PoseStamped computePoseByAlgorithm(
            pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
            const cv::Mat &mask_cv,
            const rclcpp::Time &stamp) override;
        std::string benchmarkLabel() const override;
    };

    class PoseEstimateRANSAC : public PoseEstimateBase
    {
    public:
        explicit PoseEstimateRANSAC(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

    protected:
        geometry_msgs::msg::PoseStamped computePoseByAlgorithm(
            pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
            const cv::Mat &mask_cv,
            const rclcpp::Time &stamp) override;
        std::string benchmarkLabel() const override;
    };

    class PoseEstimateGaussian : public PoseEstimateBase
    {
    public:
        explicit PoseEstimateGaussian(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

    protected:
        geometry_msgs::msg::PoseStamped computePoseByAlgorithm(
            pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
            const cv::Mat &mask_cv,
            const rclcpp::Time &stamp) override;
        std::string benchmarkLabel() const override;
    };

    class PoseEstimateCeres : public PoseEstimateBase
    {
    public:
        explicit PoseEstimateCeres(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

    protected:
        geometry_msgs::msg::PoseStamped computePoseByAlgorithm(
            pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
            const cv::Mat &mask_cv,
            const rclcpp::Time &stamp) override;
        std::string benchmarkLabel() const override;
    };

} // namespace axispose
#endif
