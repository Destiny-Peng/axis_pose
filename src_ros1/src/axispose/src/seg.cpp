#include "axispose/seg.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

using std::placeholders::_1;
namespace axispose
{
    SegmentNode::SegmentNode(ros::NodeHandle &nh, ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh)
    {
        std::string engine_path;
        pnh_.param<std::string>("engine", engine_path, "");
        if (engine_path.empty())
        {
            ROS_ERROR("Parameter 'engine' is required.");
            throw std::runtime_error("Parameter 'engine' is required.");
        }
        if (!std::filesystem::exists(engine_path))
        {
            ROS_ERROR("Engine file does not exist: %s", engine_path.c_str());
            throw std::runtime_error("Engine file missing: " + engine_path);
        }

        const auto preload = std::getenv("LD_PRELOAD");
        if (!preload || std::string(preload).empty())
        {
            ROS_WARN("LD_PRELOAD is empty. CUDA plugin loading may fail on this platform.");
        }
        else
        {
            ROS_INFO("LD_PRELOAD=%s", preload);
        }

        std::string image_topic_;
        pnh_.param<std::string>("image_topic", image_topic_, "/camera/rgb/image_raw");

        option_.enableSwapRB();
        const auto load_t0 = std::chrono::steady_clock::now();
        model_ = std::make_unique<trtyolo::SegmentModel>(engine_path, option_);
        const auto load_t1 = std::chrono::steady_clock::now();
        const double load_ms = std::chrono::duration<double, std::milli>(load_t1 - load_t0).count();

        std::error_code ec;
        const auto engine_size = std::filesystem::file_size(engine_path, ec);

        sub_ = nh_.subscribe(image_topic_, 5, &SegmentNode::imageCallback, this);
        pub_ = nh_.advertise<sensor_msgs::Image>("/yolo/mask", 1);

        ROS_INFO("Segment node initialized. Engine=%s size_bytes=%llu load_ms=%.2f image_topic=%s",
                 engine_path.c_str(),
                 static_cast<unsigned long long>(ec ? 0ULL : engine_size),
                 load_ms,
                 image_topic_.c_str());
    }

    void SegmentNode::imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        try
        {
            cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
            trtyolo::Image input(image.data, image.cols, image.rows);

            const auto infer_t0 = std::chrono::steady_clock::now();
            auto result = model_->predict(input);
            const auto infer_t1 = std::chrono::steady_clock::now();
            const double infer_ms = std::chrono::duration<double, std::milli>(infer_t1 - infer_t0).count();

            cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(0));

            for (size_t i = 0; i < result.num; ++i)
            {
                auto &box = result.boxes[i];
                auto xyxy = box.xyxy();
                int w = std::max(xyxy[2] - xyxy[0] + 1, 1);
                int h = std::max(xyxy[3] - xyxy[1] + 1, 1);

                int x1 = std::max(0, xyxy[0]);
                int y1 = std::max(0, xyxy[1]);
                int x2 = std::min(image.cols, xyxy[2] + 1);
                int y2 = std::min(image.rows, xyxy[3] + 1);

                cv::Mat float_mask(result.masks[i].height, result.masks[i].width, CV_32FC1, result.masks[i].data.data());
                cv::resize(float_mask, float_mask, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
                cv::Mat bool_mask;
                cv::threshold(float_mask, bool_mask, 0.5, 255, cv::THRESH_BINARY);
                bool_mask.convertTo(bool_mask, CV_8UC1);

                int src_x_offset = std::max(0, -xyxy[0]);
                int src_y_offset = std::max(0, -xyxy[1]);

                int target_w = x2 - x1;
                int target_h = y2 - y1;
                if (target_w <= 0 || target_h <= 0)
                    continue;

                if (src_x_offset + target_w > bool_mask.cols)
                    target_w = bool_mask.cols - src_x_offset;
                if (src_y_offset + target_h > bool_mask.rows)
                    target_h = bool_mask.rows - src_y_offset;
                if (target_w <= 0 || target_h <= 0)
                    continue;

                cv::Rect source_rect(src_x_offset, src_y_offset, target_w, target_h);
                cv::Rect target_rect(x1, y1, target_w, target_h);

                cv::Mat roi = mask(target_rect);
                cv::Mat src = bool_mask(source_rect);
                cv::bitwise_or(roi, src, roi);
            }

            const int mask_pixels = cv::countNonZero(mask);
            ++frames_in_;
            if (first_frame_stamp_.isZero())
            {
                first_frame_stamp_ = ros::Time::now();
            }
            const double elapsed = std::max(1e-3, (ros::Time::now() - first_frame_stamp_).toSec());
            const double fps = static_cast<double>(frames_in_) / elapsed;

            if (!first_infer_logged_)
            {
                first_infer_logged_ = true;
                ROS_INFO("First inference done: infer_ms=%.3f detections=%zu mask_pixels=%d", infer_ms, result.num, mask_pixels);
            }
            ROS_INFO_THROTTLE(5.0, "Segment runtime: fps=%.2f infer_ms=%.3f detections=%zu mask_pixels=%d",
                              fps, infer_ms, result.num, mask_pixels);

            auto out_msg = cv_bridge::CvImage(msg->header, "mono8", mask).toImageMsg();
            pub_.publish(*out_msg);
        }
        catch (const std::exception &e)
        {
            ROS_ERROR("Exception in image callback: %s", e.what());
        }
    }
} // namespace axispose
#include <ros/ros.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "segment_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    axispose::SegmentNode node(nh, pnh);

    ros::spin();
    return 0;
}
