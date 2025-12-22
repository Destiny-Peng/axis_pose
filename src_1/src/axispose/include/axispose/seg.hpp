#pragma once

#include <memory>
#include <string>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "trtyolo.hpp"

namespace axispose
{

    class SegmentNode
    {
    public:
        SegmentNode(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    private:
        void imageCallback(const sensor_msgs::ImageConstPtr &msg);

        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;
        ros::Subscriber sub_;
        ros::Publisher pub_;

        trtyolo::InferOption option_;
        std::unique_ptr<trtyolo::SegmentModel> model_;
    };

} // namespace axispose
