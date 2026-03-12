#ifndef AXISPOSE_DRIVER_HPP_
#define AXISPOSE_DRIVER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <string>
#include <vector>
#include <memory>

namespace axispose
{

    class CameraDriver : public rclcpp::Node
    {
    public:
        explicit CameraDriver(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

    private:
        void load_image_lists();
        void preload_images();
        bool load_camera_info_from_file(const std::string &yaml_file, sensor_msgs::msg::CameraInfo &info);
        void publish_timer_callback();

        // publishers
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
        rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr color_caminfo_pub_;
        rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_caminfo_pub_;

        // parameters
        std::string rgb_dir_;
        std::string depth_dir_;
        std::string color_camera_info_file_;
        std::string depth_camera_info_file_;
        std::string frame_id_;
        std::string camera_name_;
        double publish_rate_;
        bool loop_;

        // file lists and index
        std::vector<std::string> rgb_files_;
        std::vector<std::string> depth_files_;
        size_t index_;

        // preloaded messages (to keep memory small but fast access; user said <=10 images)
        std::vector<sensor_msgs::msg::Image::SharedPtr> rgb_msgs_;
        std::vector<sensor_msgs::msg::Image::SharedPtr> depth_msgs_;

        // camera info template
        sensor_msgs::msg::CameraInfo color_camera_info_msg_;
        sensor_msgs::msg::CameraInfo depth_camera_info_msg_;

        // camera_info_manager
        std::shared_ptr<camera_info_manager::CameraInfoManager> color_camera_info_manager_;
        std::shared_ptr<camera_info_manager::CameraInfoManager> depth_camera_info_manager_;

        // timers
        rclcpp::TimerBase::SharedPtr image_timer_;

        // parameter callback handle for dynamic publish_rate updates
        std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
        std::shared_ptr<rclcpp::ParameterCallbackHandle> publish_rate_cb_handle_;
    };

} // namespace axispose

#endif // AXISPOSE_DRIVER_HPP_