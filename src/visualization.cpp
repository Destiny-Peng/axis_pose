#include "axispose/visualization.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
// Use Eigen instead of tf2 for rotations
#include <Eigen/Geometry>

namespace axispose
{

    Visualization::Visualization(const rclcpp::NodeOptions &options) : rclcpp::Node("visualization", options)
    {
        this->declare_parameter("mask_topic", std::string("/yolo/mask"));
        this->declare_parameter("color_image_topic", std::string("/camera/rgb/image_raw"));
        this->declare_parameter("pose_topic", std::string("/shaft/pose"));
        this->declare_parameter("camera_info_topic", std::string("/camera/color/camera_info"));
        this->declare_parameter("axis_length", axis_length_);

        std::string mask_topic = this->get_parameter("mask_topic").as_string();
        std::string color_image_topic = this->get_parameter("color_image_topic").as_string();
        std::string pose_topic = this->get_parameter("pose_topic").as_string();
        std::string caminfo_topic = this->get_parameter("camera_info_topic").as_string();
        axis_length_ = this->get_parameter("axis_length").as_double();

        rclcpp::QoS qos(rclcpp::KeepLast(5));

        // Use message_filters subscribers
        rgb_sub_.subscribe(this, color_image_topic, qos.get_rmw_qos_profile());
        pose_sub_.subscribe(this, pose_topic, qos.get_rmw_qos_profile());
        // subscribe to color camera_info (used for projecting into color image)
        caminfo_sub_.subscribe(this, caminfo_topic, qos.get_rmw_qos_profile());
        mask_sub_.subscribe(this, mask_topic, qos.get_rmw_qos_profile());

        sync_.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(10), rgb_sub_, pose_sub_, caminfo_sub_, mask_sub_));
        sync_->registerCallback(&Visualization::syncCallback, this);

        vis_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/shaft/vis_image", 1);

        RCLCPP_INFO(this->get_logger(), "Visualization node started. Subscribed to %s %s %s %s", color_image_topic.c_str(), pose_topic.c_str(), caminfo_topic.c_str(), mask_topic.c_str());
    }

    // project a 3D point (in camera frame) to pixel using intrinsics
    static bool projectPoint(const geometry_msgs::msg::Point &p, const sensor_msgs::msg::CameraInfo &cam, cv::Point2f &out)
    {
        // K: [k0 k1 k2; k3 k4 k5; k6 k7 k8]
        double fx = cam.k[0];
        double fy = cam.k[4];
        double cx = cam.k[2];
        double cy = cam.k[5];
        if (p.x <= 0.0)
            return false;
        out.x = static_cast<float>(-(p.y / p.x) * fx + cx);
        out.y = static_cast<float>(-(p.z / p.x) * fy + cy);
        return true;
    }

    void Visualization::syncCallback(const Image::ConstSharedPtr rgb_msg,
                                     const Pose::ConstSharedPtr pose_msg,
                                     const CameraInfo::ConstSharedPtr cam_info_msg,
                                     const Image::ConstSharedPtr mask_msg)
    {
        // convert rgb to cv::Mat (bgr8 assumed)
        cv::Mat rgb_cv;
        try
        {
            auto img_ptr = cv_bridge::toCvCopy(rgb_msg, "bgr8");
            rgb_cv = img_ptr->image.clone();
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge rgb conversion failed: %s", e.what());
            return;
        }

        // convert mask to cv::Mat (assume mono8)
        cv::Mat mask_cv;
        try
        {
            auto mask_ptr = cv_bridge::toCvCopy(mask_msg, "mono8");
            mask_cv = mask_ptr->image.clone();
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge mask conversion failed: %s", e.what());
            return;
        }

        // ensure same size
        if (rgb_cv.size() != mask_cv.size())
        {
            cv::resize(mask_cv, mask_cv, rgb_cv.size(), 0, 0, cv::INTER_NEAREST);
        }

        // get pose in camera frame (pose message frame must be camera frame)
        // pose_msg.pose.position is centroid, orientation encodes axis rotation. We'll draw center and one axis direction (x axis of pose)
        geometry_msgs::msg::Point center = pose_msg->pose.position;

        // compute axis endpoint in camera frame: transform local axis vector
        // convert geometry_msgs quaternion to Eigen quaternion (w, x, y, z)
        Eigen::Quaterniond q_e(pose_msg->pose.orientation.w,
                               pose_msg->pose.orientation.x,
                               pose_msg->pose.orientation.y,
                               pose_msg->pose.orientation.z);
        // rotation matrix and axis direction (local x axis)
        Eigen::Matrix3d R = q_e.toRotationMatrix();
        Eigen::Vector3d axis_dir_e = R * Eigen::Vector3d(1.0, 0.0, 0.0);
        geometry_msgs::msg::Point axis_end;
        axis_end.x = center.x + axis_dir_e.x() * axis_length_;
        axis_end.y = center.y + axis_dir_e.y() * axis_length_;
        axis_end.z = center.z + axis_dir_e.z() * axis_length_;

        // project center and axis_end
        cv::Point2f pc, pe;
        bool okc = projectPoint(center, *cam_info_msg, pc);
        bool oke = projectPoint(axis_end, *cam_info_msg, pe);

        // draw mask overlay (colored)
        cv::Mat vis = rgb_cv.clone();
        cv::Mat color_mask;
        cv::cvtColor(mask_cv, color_mask, cv::COLOR_GRAY2BGR);
        // colorize mask green
        color_mask.setTo(cv::Scalar(0, 255, 0), mask_cv);
        double alpha = 0.4;
        cv::addWeighted(color_mask, alpha, vis, 1.0 - alpha, 0.0, vis);

        // draw center and axis line if projection succeeded
        if (okc)
        {
            cv::circle(vis, pc, 6, cv::Scalar(0, 0, 255), -1);
        }
        if (okc && oke)
        {
            cv::line(vis, pc, pe, cv::Scalar(255, 0, 0), 2);
        }

        // convert back to sensor_msgs::Image
        std::shared_ptr<sensor_msgs::msg::Image> out_msg;
        try
        {
            auto out_ptr = cv_bridge::CvImage(rgb_msg->header, "bgr8", vis).toImageMsg();
            out_msg = std::make_shared<sensor_msgs::msg::Image>(*out_ptr);
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge toImageMsg failed: %s", e.what());
            return;
        }

        out_msg->header.stamp = rgb_msg->header.stamp;
        out_msg->header.frame_id = rgb_msg->header.frame_id;
        vis_pub_->publish(*out_msg);
    }

} // namespace axispose

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(axispose::Visualization)
