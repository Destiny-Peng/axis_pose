#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <opencv2/opencv.hpp>

#include "trtyolo.hpp"
namespace axispose
{
    class SegmentNode : public rclcpp::Node
    {
    public:
        explicit SegmentNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

    private:
        struct TrackCandidate
        {
            int index{-1};
            cv::Rect bbox;
            cv::Point2d center{0.0, 0.0};
            double area{0.0};
            float confidence{0.0f};
            cv::Mat mask;
        };

        void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
        static double computeIoU(const cv::Rect &lhs, const cv::Rect &rhs);
        static double computeNormalizedCenterDistance(const cv::Point2d &lhs, const cv::Point2d &rhs, const cv::Size &frame_size);
        static bool isSimilarCandidate(const TrackCandidate &lhs, const TrackCandidate &rhs, const cv::Size &frame_size, double center_dist_threshold_ratio);
        TrackCandidate selectInitialCandidate(const std::vector<TrackCandidate> &candidates, const cv::Size &frame_size) const;
        TrackCandidate selectTrackCandidate(const std::vector<TrackCandidate> &candidates, const cv::Size &frame_size, double *out_iou, double *out_center_dist_norm) const;
        std::string buildDebugString(const std::string &state,
                                     const std::vector<TrackCandidate> &candidates,
                                     const TrackCandidate *selected_candidate,
                                     int candidate_count,
                                     int selected_index,
                                     int lost_frames,
                                     int pending_frames,
                                     double matched_iou,
                                     double matched_center_dist_norm) const;

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_pub_;

        std::unique_ptr<trtyolo::SegmentModel> model_;
        trtyolo::InferOption option_;

        bool tracking_enabled_{true};
        bool track_active_{false};
        int tracked_index_{-1};
        cv::Rect tracked_bbox_;
        cv::Point2d tracked_center_{0.0, 0.0};
        double tracked_area_{0.0};
        float tracked_confidence_{0.0f};

        cv::Rect pending_bbox_;
        cv::Point2d pending_center_{0.0, 0.0};
        double pending_area_{0.0};
        float pending_confidence_{0.0f};
        int pending_confirm_count_{0};

        int lost_frames_{0};

        int lost_grace_frames_{12};
        int reacquire_confirm_frames_{2};
        double iou_keep_min_{0.15};
        double center_dist_max_ratio_{0.14};
        bool debug_enabled_{true};
        std::string debug_topic_{"/yolo/seg_debug"};
    };

} // namespace axispose
