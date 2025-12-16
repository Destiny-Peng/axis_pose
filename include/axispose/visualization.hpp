#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

namespace axispose
{

    class Visualization : public rclcpp::Node
    {
    public:
        explicit Visualization(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

    private:
        using Image = sensor_msgs::msg::Image;
        using Pose = geometry_msgs::msg::PoseStamped;
        using CameraInfo = sensor_msgs::msg::CameraInfo;

        // message_filters subscribers (wrapped in shared_ptr for lifetime control)
        message_filters::Subscriber<Image> rgb_sub_;
        message_filters::Subscriber<Image> mask_sub_;
        message_filters::Subscriber<Pose> pose_sub_;
        message_filters::Subscriber<CameraInfo> caminfo_sub_;

        // approximate time sync policy (RGB, Pose, CameraInfo, Mask)
        using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Pose, CameraInfo, Image>;
        std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;

        // publisher for visualization image
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr vis_pub_;

        // callback invoked with synchronized messages
        void syncCallback(const Image::ConstSharedPtr rgb_msg,
                          const Pose::ConstSharedPtr pose_msg,
                          const CameraInfo::ConstSharedPtr cam_info_msg,
                          const Image::ConstSharedPtr mask_msg);

        // parameters
        double axis_length_ = 0.25; // meters for visualized axis half-length
    };

} // namespace axispose
