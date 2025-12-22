#pragma once

#include <memory>
#include <string>
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
#include <opencv2/opencv.hpp> // OpenCV required for cv::Mat in function signatures

namespace axispose
{

    class PoseEstimate
    {
    public:
        PoseEstimate(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    private:
        void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr &msg);
        void syncCallback(const sensor_msgs::ImageConstPtr &depth_msg, const sensor_msgs::ImageConstPtr &mask_msg);
        pcl::PointCloud<pcl::PointXYZ>::Ptr depthMaskToPointCloud(const cv::Mat &depth);
        void denoisePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
        geometry_msgs::PoseStamped computePoseFromCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const ros::Time &stamp);
        // dynamic parameter update
        void updateParameters(const ros::TimerEvent &);

        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;

        message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
        message_filters::Subscriber<sensor_msgs::Image> mask_sub_;
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ApproxSyncPolicy;
        std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;

        ros::Subscriber camera_info_sub_;
        ros::Publisher pose_pub_;
        ros::Publisher debug_cloud_pub_;

        bool have_intrinsics_ = false;
        double fx_, fy_, cx_, cy_;
        double scale_x = 1.0, scale_y = 1.0;
        std::string frame_id_;

        // denoise / filter parameters (configurable at runtime)
        double voxel_leaf_size_ = 0.01;
        double sor_mean_k_ = 50;
        double sor_std_mul_ = 1.0;
        // clustering parameters (exposed as ROS params)
        double cluster_tolerance_ = 0.5; // meters
        int cluster_min_size_ = 10;
        int cluster_max_size_ = 25000;

        // timer to poll parameters at runtime
        ros::Timer param_timer_;
    };

} // namespace axispose
