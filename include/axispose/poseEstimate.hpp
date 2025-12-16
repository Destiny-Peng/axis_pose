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

        rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_sub_;

        // Publishers
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_cloud_pub_;

        // Intrinsics
        std::atomic<bool> have_intrinsics_{false};
        double fx_{0.0}, fy_{0.0}, cx_{0.0}, cy_{0.0};
        double scale_x{1.0}, scale_y{1.0};
        std::string frame_id_ = "base_link";

        // Parameters
        double voxel_leaf_size_ = 0.05; // meters
        double sor_mean_k_ = 50;
        double sor_std_mul_ = 1.0;

        // Callbacks
        void cameraInfoCallback(const CameraInfo::SharedPtr msg);
        void syncCallback(const Image::ConstSharedPtr depth_msg, const Image::ConstSharedPtr mask_msg);

        // Helpers
        pcl::PointCloud<pcl::PointXYZ>::Ptr depthMaskToPointCloud(const cv::Mat &depth);
        void denoisePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
        geometry_msgs::msg::PoseStamped computePoseFromCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const rclcpp::Time &stamp);
    };

} // namespace axispose

#endif // AXISPOSE_POSEESTIMATE_HPP_