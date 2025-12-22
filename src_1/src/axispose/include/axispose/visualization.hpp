#pragma once

#include <memory>
#include <string>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

namespace axispose
{

    class Visualization
    {
    public:
        Visualization(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    private:
        void syncCallback(const sensor_msgs::ImageConstPtr &rgb_msg,
                          const geometry_msgs::PoseStampedConstPtr &pose_msg,
                          const sensor_msgs::CameraInfoConstPtr &cam_info_msg,
                          const sensor_msgs::ImageConstPtr &mask_msg);

        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;

        message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
        message_filters::Subscriber<geometry_msgs::PoseStamped> pose_sub_;
        message_filters::Subscriber<sensor_msgs::CameraInfo> caminfo_sub_;
        message_filters::Subscriber<sensor_msgs::Image> mask_sub_;
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped, sensor_msgs::CameraInfo, sensor_msgs::Image> ApproxSyncPolicy;
        std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;

        ros::Publisher vis_pub_;
        double axis_length_ = 0.1;
    };

} // namespace axispose
