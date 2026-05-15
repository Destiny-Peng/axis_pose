#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <axispose_msgs/msg/tracked_object_array.hpp>
#include <axispose_msgs/msg/tracked_pose_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <fstream>

namespace axispose
{

    class Visualization : public rclcpp::Node
    {
    public:
        explicit Visualization(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

    private:
        using Image = sensor_msgs::msg::Image;
        using CameraInfo = sensor_msgs::msg::CameraInfo;
        using TrackedObjectArray = axispose_msgs::msg::TrackedObjectArray;
        using TrackedPoseArray = axispose_msgs::msg::TrackedPoseArray;

        // message_filters subscribers (wrapped in shared_ptr for lifetime control)
        message_filters::Subscriber<Image> rgb_sub_;
        message_filters::Subscriber<TrackedPoseArray> pose_array_sub_;
        message_filters::Subscriber<TrackedObjectArray> object_array_sub_;
        // camera_info is static/latched; keep regular rclcpp subscriptions for
        // both color and depth caminfo and cache them.
        rclcpp::Subscription<CameraInfo>::SharedPtr caminfo_color_sub_;
        rclcpp::Subscription<CameraInfo>::SharedPtr caminfo_depth_sub_;
        CameraInfo::SharedPtr cached_caminfo_color_;
        CameraInfo::SharedPtr cached_caminfo_depth_;

        // approximate time sync policy (RGB, TrackedPoseArray, TrackedObjectArray)
        using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<Image, TrackedPoseArray, TrackedObjectArray>;
        std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;

        // publisher for visualization image
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr vis_pub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

        // image saving
        bool save_annotated_ = false;
        std::string save_dir_;
        int save_every_n_ = 1; // save every N frames (1 = every frame)
        uint64_t save_counter_ = 0;

        // 2D line evaluation logging
        bool line_eval_enabled_ = true;
        std::string line_eval_csv_path_;
        std::ofstream line_eval_ofs_;
        uint64_t line_eval_counter_ = 0;

        // callback invoked with synchronized messages
        void syncCallback(const Image::ConstSharedPtr rgb_msg,
                          const TrackedPoseArray::ConstSharedPtr pose_array_msg,
                          const TrackedObjectArray::ConstSharedPtr object_array_msg);

        // parameters
        double axis_length_ = 1.5; // meters for visualized axis half-length
        // multiplier applied to axis_length_ when drawing RViz markers
        double marker_length_scale_ = 1.0;
        std::string pose_array_topic_{"/shaft/tracked_poses"};
        std::string object_array_topic_{"/yolo/tracked_objects"};
        std::string marker_topic_{"/shaft/vis_markers"};
    };

} // namespace axispose
