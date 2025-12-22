#pragma once

#include <string>
#include <vector>
#include <memory>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <camera_info_manager/camera_info_manager.h>

#include <opencv2/opencv.hpp>

namespace axispose
{

    class CameraDriver
    {
    public:
        CameraDriver(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    private:
        void preloadImages();
        void loadImageLists();
        bool loadCameraInfoFromFile(const std::string &yaml_file, sensor_msgs::CameraInfo &info);
        void publishTimerCallback(const ros::TimerEvent &ev);

        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;

        // publishers
        ros::Publisher rgb_pub_;
        ros::Publisher depth_pub_;
        ros::Publisher caminfo_pub_;

        // camera info manager
        std::shared_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;

        // parameters
        std::string rgb_dir_;
        std::string depth_dir_;
        std::string camera_info_file_;
        std::string frame_id_;
        double publish_rate_;
        bool loop_;

        // state
        std::vector<std::string> rgb_files_;
        std::vector<std::string> depth_files_;
        std::vector<sensor_msgs::Image> rgb_msgs_; // preloaded messages
        std::vector<sensor_msgs::Image> depth_msgs_;
        sensor_msgs::CameraInfo camera_info_msg_;
        size_t index_;
        ros::Timer timer_;
    };

} // namespace axispose
