#include "axispose/visualization.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/bind.hpp>

namespace axispose
{

    Visualization::Visualization(ros::NodeHandle &nh, ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh)
    {
        std::string mask_topic = "/yolo/mask";
        std::string rgb_topic = "/camera/rgb/image_raw";
        std::string pose_topic = "/shaft/pose";
        std::string caminfo_topic = "/camera/camera_info";
        pnh_.param<std::string>("mask_topic", mask_topic, mask_topic);
        pnh_.param<std::string>("rgb_topic", rgb_topic, rgb_topic);
        pnh_.param<std::string>("pose_topic", pose_topic, pose_topic);
        pnh_.param<std::string>("camera_info_topic", caminfo_topic, caminfo_topic);
        pnh_.param<double>("axis_length", axis_length_, axis_length_);

        // Use message_filters subscribers
        rgb_sub_.subscribe(nh_, rgb_topic, 5);
        pose_sub_.subscribe(nh_, pose_topic, 5);
        caminfo_sub_.subscribe(nh_, caminfo_topic, 5);
        mask_sub_.subscribe(nh_, mask_topic, 5);

        sync_.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(10), rgb_sub_, pose_sub_, caminfo_sub_, mask_sub_));
        sync_->registerCallback(boost::bind(&Visualization::syncCallback, this, _1, _2, _3, _4));

        vis_pub_ = nh_.advertise<sensor_msgs::Image>("/shaft/vis_image", 1);

        ROS_INFO("Visualization node started. Subscribed to %s %s %s %s", rgb_topic.c_str(), pose_topic.c_str(), caminfo_topic.c_str(), mask_topic.c_str());
    }

    static bool projectPoint(const geometry_msgs::Point &p, const sensor_msgs::CameraInfo &cam, cv::Point2f &out)
    {
        double fx = cam.K[0];
        double fy = cam.K[4];
        double cx = cam.K[2];
        double cy = cam.K[5];
        if (p.x <= 0.0)
            return false;
        out.x = static_cast<float>(-(p.y / p.x) * fx + cx);
        out.y = static_cast<float>(-(p.z / p.x) * fy + cy);
        return true;
    }

    void Visualization::syncCallback(const sensor_msgs::ImageConstPtr &rgb_msg,
                                     const geometry_msgs::PoseStampedConstPtr &pose_msg,
                                     const sensor_msgs::CameraInfoConstPtr &cam_info_msg,
                                     const sensor_msgs::ImageConstPtr &mask_msg)
    {
        cv::Mat rgb_cv;
        try
        {
            auto img_ptr = cv_bridge::toCvCopy(rgb_msg, "bgr8");
            rgb_cv = img_ptr->image.clone();
        }
        catch (const cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge rgb conversion failed: %s", e.what());
            return;
        }

        cv::Mat mask_cv;
        try
        {
            auto mask_ptr = cv_bridge::toCvCopy(mask_msg, "mono8");
            mask_cv = mask_ptr->image.clone();
        }
        catch (const cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge mask conversion failed: %s", e.what());
            return;
        }

        if (rgb_cv.size() != mask_cv.size())
        {
            cv::resize(mask_cv, mask_cv, rgb_cv.size(), 0, 0, cv::INTER_NEAREST);
        }

        geometry_msgs::Point center = pose_msg->pose.position;

        Eigen::Quaterniond q_e(pose_msg->pose.orientation.w,
                               pose_msg->pose.orientation.x,
                               pose_msg->pose.orientation.y,
                               pose_msg->pose.orientation.z);
        Eigen::Matrix3d R = q_e.toRotationMatrix();
        Eigen::Vector3d axis_dir_e = R * Eigen::Vector3d(1.0, 0.0, 0.0);
        geometry_msgs::Point axis_end;
        axis_end.x = center.x + axis_dir_e.x() * axis_length_;
        axis_end.y = center.y + axis_dir_e.y() * axis_length_;
        axis_end.z = center.z + axis_dir_e.z() * axis_length_;

        cv::Point2f pc, pe;
        bool okc = projectPoint(center, *cam_info_msg, pc);
        bool oke = projectPoint(axis_end, *cam_info_msg, pe);

        cv::Mat vis = rgb_cv.clone();
        cv::Mat color_mask;
        cv::cvtColor(mask_cv, color_mask, cv::COLOR_GRAY2BGR);
        color_mask.setTo(cv::Scalar(0, 255, 0), mask_cv);
        double alpha = 0.4;
        cv::addWeighted(color_mask, alpha, vis, 1.0 - alpha, 0.0, vis);

        if (okc)
        {
            cv::circle(vis, pc, 6, cv::Scalar(0, 0, 255), -1);
        }
        if (okc && oke)
        {
            cv::line(vis, pc, pe, cv::Scalar(255, 0, 0), 2);
        }

        sensor_msgs::ImagePtr out_msg;
        try
        {
            out_msg = cv_bridge::CvImage(rgb_msg->header, "bgr8", vis).toImageMsg();
        }
        catch (const cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge toImageMsg failed: %s", e.what());
            return;
        }

        out_msg->header.stamp = rgb_msg->header.stamp;
        out_msg->header.frame_id = rgb_msg->header.frame_id;
        vis_pub_.publish(out_msg);
    }

} // namespace axispose
#include <ros/ros.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "visualization_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    axispose::Visualization node(nh, pnh);

    ros::spin();
    return 0;
}