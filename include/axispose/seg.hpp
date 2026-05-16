#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/region_of_interest.hpp>
#include <std_msgs/msg/string.hpp>
#include <opencv2/opencv.hpp>

#include <axispose_msgs/msg/tracked_object.hpp>
#include <axispose_msgs/msg/tracked_object_array.hpp>

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

        struct TrackState
        {
            uint32_t track_id{0};
            TrackCandidate candidate;
            uint32_t age{0};
            uint32_t confirm_count{0};
            uint32_t lost_frames{0};
            bool confirmed{false};
            bool matched_this_frame{false};
        };

        void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
        static double computeIoU(const cv::Rect &lhs, const cv::Rect &rhs);
        static double computeNormalizedCenterDistance(const cv::Point2d &lhs, const cv::Point2d &rhs, const cv::Size &frame_size);
        static bool isSimilarCandidate(const TrackCandidate &lhs, const TrackCandidate &rhs, const cv::Size &frame_size, double center_dist_threshold_ratio);
        TrackCandidate selectInitialCandidate(const std::vector<TrackCandidate> &candidates, const cv::Size &frame_size) const;
        TrackCandidate selectTrackCandidate(const std::vector<TrackCandidate> &candidates, const cv::Size &frame_size, double *out_iou, double *out_center_dist_norm) const;
        axispose_msgs::msg::TrackedObject buildTrackedObjectMessage(const TrackCandidate &candidate,
                                                                    const std_msgs::msg::Header &header,
                                                                    uint32_t track_id,
                                                                    uint8_t status,
                                                                    uint32_t age,
                                                                    uint32_t lost_frames) const;
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
        rclcpp::Publisher<axispose_msgs::msg::TrackedObjectArray>::SharedPtr tracked_objects_pub_;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_pub_;

        std::unique_ptr<trtyolo::SegmentModel> model_;
        trtyolo::InferOption option_;

        bool tracking_enabled_{true};
        bool track_active_{false};
        int tracked_index_{-1};
        uint32_t current_track_id_{0}; // Persistent track ID for the currently tracked object
        uint32_t next_track_id_{1};    // Next available track ID to assign
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
        std::unordered_map<uint32_t, TrackState> tracks_;

        int lost_grace_frames_{12};
        int reacquire_confirm_frames_{2};
        double iou_keep_min_{0.15};
        double center_dist_max_ratio_{0.14};
        bool debug_enabled_{true};
        std::string debug_topic_{"/yolo/seg_debug"};
    };

} // namespace axispose
