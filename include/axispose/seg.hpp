#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>

#include "trtyolo.hpp"
namespace axispose
{
    class SegmentNode : public rclcpp::Node
    {
    public:
        explicit SegmentNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

    private:
        void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;

        std::unique_ptr<trtyolo::SegmentModel> model_;
        trtyolo::InferOption option_;
    };

} // namespace axispose
