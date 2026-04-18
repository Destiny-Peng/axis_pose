#pragma once

#include <memory>
#include <string>
#include <cstdint>
#include <vector>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "axispose/benchmark.hpp"
#include "axispose/point_cloud_processor.hpp"
#include "axispose/depth_aligner.hpp"
#include "axispose/gaussian_map_solver.hpp"

namespace axispose
{

    class PoseEstimate
    {
    public:
        PoseEstimate(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    private:
        using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;

        void cameraInfoDepthCallback(const sensor_msgs::CameraInfoConstPtr &msg);
        void cameraInfoColorCallback(const sensor_msgs::CameraInfoConstPtr &msg);
        void syncCallback(const sensor_msgs::ImageConstPtr &depth_msg, const sensor_msgs::ImageConstPtr &mask_msg);
        void updateParameters(const ros::TimerEvent &);
        void healthCheck(const ros::TimerEvent &);
        void resetFrameDiagnostics();
        void markFrameFallback(const std::string &reason);
        void logRuntimeSummary(const ros::Time &stamp);

        cv::Mat alignDepthToColor(const cv::Mat &depth, int color_width, int color_height);
        void denoisePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

        geometry_msgs::PoseStamped computePoseByAlgorithm(
            pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
            const cv::Mat &mask_cv,
            const ros::Time &stamp);
        geometry_msgs::PoseStamped computePoseFromCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const ros::Time &stamp);
        geometry_msgs::PoseStamped computePoseFromSACLine(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const ros::Time &stamp);
        geometry_msgs::PoseStamped computePoseGaussian(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const cv::Mat &mask_cv, const ros::Time &stamp);
        geometry_msgs::PoseStamped computePoseCeres(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const cv::Mat &mask_cv, const ros::Time &stamp);

        Eigen::Vector3d stabilizeAxisDirection(const Eigen::Vector3d &axis, const Eigen::Vector3d *reference_axis = nullptr);

        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;

        message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
        message_filters::Subscriber<sensor_msgs::Image> mask_sub_;
        std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;

        ros::Subscriber camera_info_depth_sub_;
        ros::Subscriber camera_info_color_sub_;
        ros::Publisher pose_pub_;
        ros::Publisher debug_cloud_pub_;

        cv::Mat depth_camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat color_camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
        bool have_intrinsics_depth_ = false;
        bool have_intrinsics_color_ = false;
        std::string frame_id_ = "camera";
        std::string color_frame_id_ = "camera";

        std::string algorithm_type_ = "pca";
        std::string statistics_directory_path_;
        bool statistics_enabled_ = true;

        double voxel_leaf_size_ = 0.05;
        double sor_mean_k_ = 50;
        double sor_std_mul_ = 1.0;
        bool use_sor_ = true;
        bool use_euclidean_cluster_ = true;
        int cluster_mode_ = 0;
        double sacline_distance_threshold_ = 0.05;
        bool align_depth_to_color_ = false;

        std::unique_ptr<PointCloudProcessor> pc_processor_;
        std::unique_ptr<DepthAligner> depth_aligner_;
        std::unique_ptr<AlgorithmBenchmark> benchmark_;

        bool axis_history_valid_ = false;
        Eigen::Vector3d axis_history_{1.0, 0.0, 0.0};

        ros::Timer param_timer_;
        ros::Timer health_timer_;
        bool got_depth_frame_ = false;
        bool got_mask_frame_ = false;

        bool runtime_log_enabled_ = true;
        double runtime_log_interval_sec_ = 5.0;
        ros::Time last_runtime_log_stamp_;

        std::uint64_t total_frames_ = 0;
        std::uint64_t pose_published_frames_ = 0;
        std::uint64_t empty_cloud_frames_ = 0;
        std::uint64_t depth_convert_failures_ = 0;
        std::uint64_t mask_convert_failures_ = 0;
        std::uint64_t fallback_frames_ = 0;
        double cumulative_total_latency_ms_ = 0.0;
        double cumulative_denoise_latency_ms_ = 0.0;
        double cumulative_pose_latency_ms_ = 0.0;

        bool frame_fallback_used_ = false;
        std::string frame_fallback_reason_;
        int frame_error_code_ = 0;
    };

} // namespace axispose
