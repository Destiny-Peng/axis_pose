#include "axispose/seg.hpp"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

using std::placeholders::_1;

namespace axispose
{
    SegmentNode::SegmentNode(const rclcpp::NodeOptions &options)
        : rclcpp::Node("segment_node", options)
    {
        this->declare_parameter<std::string>("engine", "");
        this->declare_parameter<std::string>("color_image_topic", "/camera/color/image_raw");
        this->declare_parameter<std::string>("mask_topic", "/yolo/mask");
        this->declare_parameter<std::string>("tracked_object_topic", "/yolo/tracked_objects");
        this->declare_parameter<bool>("tracking_enabled", true);
        this->declare_parameter<int>("lost_grace_frames", 12);
        this->declare_parameter<int>("reacquire_confirm_frames", 2);
        this->declare_parameter<double>("iou_keep_min", 0.15);
        this->declare_parameter<double>("center_dist_max_ratio", 0.14);
        this->declare_parameter<std::string>("debug_topic", "/yolo/seg_debug");

        const std::string engine_path = this->get_parameter("engine").as_string();
        const std::string color_topic = this->get_parameter("color_image_topic").as_string();
        const std::string mask_topic = this->get_parameter("mask_topic").as_string();
        const std::string tracked_object_topic = this->get_parameter("tracked_object_topic").as_string();
        tracking_enabled_ = this->get_parameter("tracking_enabled").as_bool();
        lost_grace_frames_ = std::max(0, static_cast<int>(this->get_parameter("lost_grace_frames").as_int()));
        reacquire_confirm_frames_ = std::max(1, static_cast<int>(this->get_parameter("reacquire_confirm_frames").as_int()));
        iou_keep_min_ = std::clamp(this->get_parameter("iou_keep_min").as_double(), 0.0, 1.0);
        center_dist_max_ratio_ = std::max(0.0, this->get_parameter("center_dist_max_ratio").as_double());
        debug_topic_ = this->get_parameter("debug_topic").as_string();

        if (engine_path.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Parameter 'engine' is required.");
            throw std::runtime_error("Parameter 'engine' is required.");
        }

        option_.enableSwapRB();
        model_ = std::make_unique<trtyolo::SegmentModel>(engine_path, option_);

        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            color_topic, rclcpp::QoS(rclcpp::KeepLast(5)).reliable(), std::bind(&SegmentNode::image_callback, this, _1));

        pub_ = this->create_publisher<sensor_msgs::msg::Image>(mask_topic, 1);
        tracked_objects_pub_ = this->create_publisher<axispose_msgs::msg::TrackedObjectArray>(tracked_object_topic, 1);
        if (!debug_topic_.empty())
        {
            debug_pub_ = this->create_publisher<std_msgs::msg::String>(debug_topic_, 1);
        }

        RCLCPP_INFO(this->get_logger(), "Segment node initialized. Engine: %s", engine_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Segment tracking: enabled=%d grace=%d confirm=%d iou_min=%.3f center_dist_ratio=%.3f debug_topic=%s tracked_object_topic=%s",
                    tracking_enabled_ ? 1 : 0,
                    lost_grace_frames_,
                    reacquire_confirm_frames_,
                    iou_keep_min_,
                    center_dist_max_ratio_,
                    debug_topic_.c_str(),
                    tracked_object_topic.c_str());
    }

    double SegmentNode::computeIoU(const cv::Rect &lhs, const cv::Rect &rhs)
    {
        const cv::Rect intersection = lhs & rhs;
        if (intersection.empty())
            return 0.0;
        const double intersection_area = static_cast<double>(intersection.area());
        const double union_area = static_cast<double>(lhs.area() + rhs.area()) - intersection_area;
        if (union_area <= 0.0)
            return 0.0;
        return intersection_area / union_area;
    }

    double SegmentNode::computeNormalizedCenterDistance(const cv::Point2d &lhs, const cv::Point2d &rhs, const cv::Size &frame_size)
    {
        const double dx = lhs.x - rhs.x;
        const double dy = lhs.y - rhs.y;
        const double diagonal = std::hypot(static_cast<double>(frame_size.width), static_cast<double>(frame_size.height));
        if (diagonal <= 0.0)
            return std::numeric_limits<double>::infinity();
        return std::hypot(dx, dy) / diagonal;
    }

    bool SegmentNode::isSimilarCandidate(const TrackCandidate &lhs, const TrackCandidate &rhs, const cv::Size &frame_size, double center_dist_threshold_ratio)
    {
        const double iou = computeIoU(lhs.bbox, rhs.bbox);
        const double center_dist_norm = computeNormalizedCenterDistance(lhs.center, rhs.center, frame_size);
        return iou >= 0.75 || center_dist_norm <= center_dist_threshold_ratio;
    }

    SegmentNode::TrackCandidate SegmentNode::selectInitialCandidate(const std::vector<TrackCandidate> &candidates, const cv::Size &frame_size) const
    {
        TrackCandidate best_candidate;
        double best_score = -std::numeric_limits<double>::infinity();
        const cv::Point2d image_center(static_cast<double>(frame_size.width) * 0.5, static_cast<double>(frame_size.height) * 0.5);

        RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                     "[selectInitialCandidate] Finding initial candidate from %zu candidates", candidates.size());

        for (size_t i = 0; i < candidates.size(); ++i)
        {
            const auto &candidate = candidates[i];
            const double center_dist_norm = computeNormalizedCenterDistance(candidate.center, image_center, frame_size);
            const double score = candidate.area - center_dist_norm * candidate.area * 0.2 + static_cast<double>(candidate.confidence) * 10.0;

            RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                         "  [%zu] bbox:(%d,%d,%d,%d) center:(%.1f,%.1f) area:%.0f conf:%.3f center_dist_norm=%.4f score=%.2f",
                         i, candidate.bbox.x, candidate.bbox.y, candidate.bbox.width, candidate.bbox.height,
                         candidate.center.x, candidate.center.y, candidate.area, candidate.confidence, center_dist_norm, score);

            if (score > best_score)
            {
                RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                             "    → NEW BEST (prev_best_score=%.2f)", best_score);
                best_score = score;
                best_candidate = candidate;
            }
        }

        if (best_candidate.index >= 0)
        {
            RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                         "[selectInitialCandidate] SELECTED candidate[%d] with score=%.2f",
                         best_candidate.index, best_score);
        }
        else
        {
            RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                         "[selectInitialCandidate] NO CANDIDATE (best_score=%.2f)", best_score);
        }

        return best_candidate;
    }

    SegmentNode::TrackCandidate SegmentNode::selectTrackCandidate(const std::vector<TrackCandidate> &candidates, const cv::Size &frame_size, double *out_iou, double *out_center_dist_norm) const
    {
        TrackCandidate best_candidate;
        double best_score = -std::numeric_limits<double>::infinity();
        double best_iou = 0.0;
        double best_center_dist_norm = std::numeric_limits<double>::infinity();

        RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                     "[selectTrackCandidate] Tracked bbox: (%d,%d,%d,%d) center:(%.1f,%.1f) candidates=%zu",
                     tracked_bbox_.x, tracked_bbox_.y, tracked_bbox_.width, tracked_bbox_.height,
                     tracked_center_.x, tracked_center_.y, candidates.size());

        for (size_t i = 0; i < candidates.size(); ++i)
        {
            const auto &candidate = candidates[i];
            const double iou = computeIoU(candidate.bbox, tracked_bbox_);
            const double center_dist_norm = computeNormalizedCenterDistance(candidate.center, tracked_center_, frame_size);

            RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                         "  [%zu] bbox:(%d,%d,%d,%d) center:(%.1f,%.1f) area:%.0f conf:%.3f iou=%.4f center_dist=%.4f",
                         i, candidate.bbox.x, candidate.bbox.y, candidate.bbox.width, candidate.bbox.height,
                         candidate.center.x, candidate.center.y, candidate.area, candidate.confidence, iou, center_dist_norm);

            if (iou < iou_keep_min_ && center_dist_norm > center_dist_max_ratio_)
            {
                RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                             "    → FILTERED OUT (iou=%.4f<%f || center_dist=%.4f>%f)",
                             iou, iou_keep_min_, center_dist_norm, center_dist_max_ratio_);
                continue;
            }

            const double score = std::clamp(iou, 0.0, 1.0) * 0.7 + (1.0 - std::min(center_dist_norm / std::max(center_dist_max_ratio_, 1e-6), 1.0)) * 0.25 + static_cast<double>(candidate.confidence) * 0.05;
            RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                         "    → score=%.4f (iou*0.7=%.4f + center_dist*0.25=%.4f + conf*0.05=%.4f)",
                         score, std::clamp(iou, 0.0, 1.0) * 0.7,
                         (1.0 - std::min(center_dist_norm / std::max(center_dist_max_ratio_, 1e-6), 1.0)) * 0.25,
                         static_cast<double>(candidate.confidence) * 0.05);

            if (score > best_score)
            {
                RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                             "    → NEW BEST (prev_best_score=%.4f)", best_score);
                best_score = score;
                best_candidate = candidate;
                best_iou = iou;
                best_center_dist_norm = center_dist_norm;
            }
        }

        if (best_candidate.index >= 0)
        {
            RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                         "[selectTrackCandidate] SELECTED candidate[%d]: iou=%.4f center_dist=%.4f score=%.4f",
                         best_candidate.index, best_iou, best_center_dist_norm, best_score);
        }
        else
        {
            RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                         "[selectTrackCandidate] NO VALID CANDIDATE (best_score=%.4f)", best_score);
        }

        if (out_iou)
            *out_iou = best_iou;
        if (out_center_dist_norm)
            *out_center_dist_norm = best_center_dist_norm;
        return best_candidate;
    }

    std::string SegmentNode::buildDebugString(const std::string &state,
                                              const std::vector<TrackCandidate> &candidates,
                                              const TrackCandidate *selected_candidate,
                                              int candidate_count,
                                              int selected_index,
                                              int lost_frames,
                                              int pending_frames,
                                              double matched_iou,
                                              double matched_center_dist_norm) const
    {
        std::ostringstream oss;
        oss << "state=" << state
            << ";candidates=" << candidate_count
            << ";selected=" << selected_index
            << ";lost_frames=" << lost_frames
            << ";pending_frames=" << pending_frames
            << ";matched_iou=" << matched_iou
            << ";matched_center_dist_norm=" << matched_center_dist_norm;

        if (selected_candidate)
        {
            oss << ";selected_area=" << selected_candidate->area
                << ";selected_conf=" << selected_candidate->confidence
                << ";selected_bbox=" << selected_candidate->bbox.x << "," << selected_candidate->bbox.y << "," << selected_candidate->bbox.width << "," << selected_candidate->bbox.height;
        }

        if (!candidates.empty())
        {
            oss << ";top_candidates=";
            for (size_t i = 0; i < candidates.size(); ++i)
            {
                const auto &candidate = candidates[i];
                oss << "[" << candidate.index << ":a=" << candidate.area << ",c=" << candidate.confidence << "]";
                if (i + 1 < candidates.size())
                    oss << ",";
            }
        }

        return oss.str();
    }

    axispose_msgs::msg::TrackedObject SegmentNode::buildTrackedObjectMessage(const TrackCandidate &candidate,
                                                                             const std_msgs::msg::Header &header,
                                                                             uint32_t track_id,
                                                                             uint8_t status) const
    {
        axispose_msgs::msg::TrackedObject msg;
        msg.header = header;
        msg.track_id = track_id;
        msg.class_id = -1;
        msg.score = candidate.confidence;
        msg.status = status;
        msg.age = 1;
        msg.lost_frames = 0;
        msg.bbox.x_offset = static_cast<uint32_t>(std::max(0, candidate.bbox.x));
        msg.bbox.y_offset = static_cast<uint32_t>(std::max(0, candidate.bbox.y));
        msg.bbox.width = static_cast<uint32_t>(std::max(0, candidate.bbox.width));
        msg.bbox.height = static_cast<uint32_t>(std::max(0, candidate.bbox.height));
        msg.bbox.do_rectify = false;
        msg.mask = *cv_bridge::CvImage(header, "mono8", candidate.mask).toImageMsg();
        return msg;
    }

    void SegmentNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
            trtyolo::Image input(image.data, image.cols, image.rows);
            auto result = model_->predict(input);

            std::vector<TrackCandidate> candidates;
            size_t candidate_count = std::min(result.boxes.size(), result.masks.size());
            if (result.num >= 0)
            {
                candidate_count = std::min(candidate_count, static_cast<size_t>(result.num));
            }
            candidates.reserve(candidate_count);

            for (size_t i = 0; i < candidate_count; ++i)
            {
                auto &box = result.boxes[i];
                auto xyxy = box.xyxy();

                const int x1 = std::max(0, xyxy[0]);
                const int y1 = std::max(0, xyxy[1]);
                const int x2 = std::min(image.cols, xyxy[2] + 1);
                const int y2 = std::min(image.rows, xyxy[3] + 1);

                int w = std::max(x2 - x1, 1);
                int h = std::max(y2 - y1, 1);
                if (w <= 0 || h <= 0)
                    continue;

                const int src_x_offset = std::max(0, -xyxy[0]);
                const int src_y_offset = std::max(0, -xyxy[1]);

                cv::Mat float_mask(result.masks[i].height, result.masks[i].width, CV_32FC1, result.masks[i].data.data());
                cv::Mat resized_mask;
                cv::resize(float_mask, resized_mask, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

                cv::Mat bool_mask;
                cv::threshold(resized_mask, bool_mask, 0.5, 255, cv::THRESH_BINARY);
                bool_mask.convertTo(bool_mask, CV_8UC1);

                if (src_x_offset + w > bool_mask.cols)
                    w = bool_mask.cols - src_x_offset;
                if (src_y_offset + h > bool_mask.rows)
                    h = bool_mask.rows - src_y_offset;
                if (w <= 0 || h <= 0)
                    continue;

                const cv::Rect source_rect(src_x_offset, src_y_offset, w, h);
                const cv::Rect target_rect(x1, y1, w, h);
                cv::Mat candidate_mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
                bool_mask(source_rect).copyTo(candidate_mask(target_rect));

                const double area = static_cast<double>(cv::countNonZero(candidate_mask));
                if (area <= 0.0)
                    continue;

                TrackCandidate candidate;
                candidate.index = static_cast<int>(i);
                candidate.bbox = target_rect;
                candidate.center = cv::Point2d(target_rect.x + target_rect.width * 0.5, target_rect.y + target_rect.height * 0.5);
                candidate.area = area;
                candidate.confidence = (i < result.scores.size()) ? result.scores[i] : 0.0f;
                candidate.mask = std::move(candidate_mask);
                candidates.push_back(std::move(candidate));
            }

            // First, run the state machine to update tracked_index_ and current_track_id_
            cv::Mat output_mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
            std::string state = "empty";
            int selected_index = -1;
            double matched_iou = 0.0;
            double matched_center_dist_norm = std::numeric_limits<double>::infinity();
            const TrackCandidate *selected_candidate = nullptr;

            if (!tracking_enabled_)
            {
                for (size_t i = 0; i < candidate_count; ++i)
                {
                    auto &box = result.boxes[i];
                    auto xyxy = box.xyxy();
                    const int x1 = std::max(0, xyxy[0]);
                    const int y1 = std::max(0, xyxy[1]);
                    const int x2 = std::min(image.cols, xyxy[2] + 1);
                    const int y2 = std::min(image.rows, xyxy[3] + 1);
                    int w = std::max(x2 - x1, 1);
                    int h = std::max(y2 - y1, 1);
                    if (w <= 0 || h <= 0)
                        continue;

                    const int src_x_offset = std::max(0, -xyxy[0]);
                    const int src_y_offset = std::max(0, -xyxy[1]);

                    cv::Mat float_mask(result.masks[i].height, result.masks[i].width, CV_32FC1, result.masks[i].data.data());
                    cv::Mat resized_mask;
                    cv::resize(float_mask, resized_mask, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

                    cv::Mat bool_mask;
                    cv::threshold(resized_mask, bool_mask, 0.5, 255, cv::THRESH_BINARY);
                    bool_mask.convertTo(bool_mask, CV_8UC1);

                    if (src_x_offset + w > bool_mask.cols)
                        w = bool_mask.cols - src_x_offset;
                    if (src_y_offset + h > bool_mask.rows)
                        h = bool_mask.rows - src_y_offset;
                    if (w <= 0 || h <= 0)
                        continue;

                    const cv::Rect source_rect(src_x_offset, src_y_offset, w, h);
                    const cv::Rect target_rect(x1, y1, w, h);
                    bool_mask(source_rect).copyTo(output_mask(target_rect), bool_mask(source_rect));
                }
                state = "legacy_merge";
            }
            else if (candidates.empty())
            {
                if (track_active_)
                {
                    ++lost_frames_;
                    state = (lost_frames_ > lost_grace_frames_) ? "released" : "lost";
                    if (lost_frames_ > lost_grace_frames_)
                    {
                        RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                                     "→ RELEASED (no candidates): track_id=%d", current_track_id_);
                        track_active_ = false;
                        tracked_index_ = -1;
                        current_track_id_ = 0;
                        pending_confirm_count_ = 0;
                    }
                }
            }
            else if (track_active_)
            {
                RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                             "=== LOCKED STATE: permanent_track_id=%d candidates=%zu", current_track_id_, candidates.size());
                TrackCandidate best_candidate = selectTrackCandidate(candidates, image.size(), &matched_iou, &matched_center_dist_norm);
                if (best_candidate.index >= 0)
                {
                    lost_frames_ = 0;
                    tracked_index_ = best_candidate.index;
                    tracked_bbox_ = best_candidate.bbox;
                    tracked_center_ = best_candidate.center;
                    tracked_area_ = best_candidate.area;
                    tracked_confidence_ = best_candidate.confidence;
                    output_mask = best_candidate.mask;
                    selected_index = best_candidate.index;
                    selected_candidate = &best_candidate;
                    state = "locked";
                    RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                                 "→ STAY LOCKED: track_id=%d matched with detection[%d]", current_track_id_, best_candidate.index);
                }
                else
                {
                    ++lost_frames_;
                    state = (lost_frames_ > lost_grace_frames_) ? "released" : "lost";
                    RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                                 "→ NO MATCH: track_id=%d transition to %s (lost_frames=%d/%d)",
                                 current_track_id_, state.c_str(), lost_frames_, lost_grace_frames_);
                    if (lost_frames_ > lost_grace_frames_)
                    {
                        RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                                     "→ RELEASED: track_id=%d is now released", current_track_id_);
                        track_active_ = false;
                        tracked_index_ = -1;
                        current_track_id_ = 0;
                        pending_confirm_count_ = 0;
                    }
                }
            }
            else
            {
                TrackCandidate best_initial_candidate = selectInitialCandidate(candidates, image.size());
                if (best_initial_candidate.index >= 0)
                {
                    if (pending_confirm_count_ <= 0)
                    {
                        pending_bbox_ = best_initial_candidate.bbox;
                        pending_center_ = best_initial_candidate.center;
                        pending_area_ = best_initial_candidate.area;
                        pending_confidence_ = best_initial_candidate.confidence;
                        pending_confirm_count_ = 1;
                    }
                    else
                    {
                        TrackCandidate pending_candidate;
                        pending_candidate.bbox = pending_bbox_;
                        pending_candidate.center = pending_center_;
                        pending_candidate.area = pending_area_;
                        pending_candidate.confidence = pending_confidence_;

                        if (isSimilarCandidate(best_initial_candidate, pending_candidate, image.size(), center_dist_max_ratio_))
                        {
                            ++pending_confirm_count_;
                            pending_bbox_ = best_initial_candidate.bbox;
                            pending_center_ = best_initial_candidate.center;
                            pending_area_ = best_initial_candidate.area;
                            pending_confidence_ = best_initial_candidate.confidence;
                        }
                        else
                        {
                            pending_bbox_ = best_initial_candidate.bbox;
                            pending_center_ = best_initial_candidate.center;
                            pending_area_ = best_initial_candidate.area;
                            pending_confidence_ = best_initial_candidate.confidence;
                            pending_confirm_count_ = 1;
                        }
                    }

                    if (pending_confirm_count_ >= reacquire_confirm_frames_)
                    {
                        track_active_ = true;
                        current_track_id_ = next_track_id_++;
                        lost_frames_ = 0;
                        tracked_index_ = best_initial_candidate.index;
                        tracked_bbox_ = best_initial_candidate.bbox;
                        tracked_center_ = best_initial_candidate.center;
                        tracked_area_ = best_initial_candidate.area;
                        tracked_confidence_ = best_initial_candidate.confidence;
                        output_mask = best_initial_candidate.mask;
                        selected_index = best_initial_candidate.index;
                        selected_candidate = &best_initial_candidate;
                        pending_confirm_count_ = 0;
                        state = "locked";
                        RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                                     "→ CONFIRMED LOCK: assigned permanent track_id=%d", current_track_id_);
                    }
                    else
                    {
                        state = "pending";
                    }
                }
            }

            if (debug_pub_)
            {
                std_msgs::msg::String debug_msg;
                debug_msg.data = buildDebugString(state,
                                                  candidates,
                                                  selected_candidate,
                                                  static_cast<int>(candidates.size()),
                                                  selected_index,
                                                  lost_frames_,
                                                  pending_confirm_count_,
                                                  matched_iou,
                                                  matched_center_dist_norm);
                debug_pub_->publish(debug_msg);
            }

            // NOW build and publish tracked_objects_msg based on updated tracked_index_ and current_track_id_
            axispose_msgs::msg::TrackedObjectArray tracked_objects_msg;
            tracked_objects_msg.header = msg->header;
            tracked_objects_msg.objects.reserve(candidates.size());
            for (const auto &candidate : candidates)
            {
                const bool is_selected = track_active_ && candidate.index == tracked_index_;
                const uint8_t status = is_selected ? axispose_msgs::msg::TrackedObject::STATUS_CONFIRMED
                                                   : axispose_msgs::msg::TrackedObject::STATUS_TENTATIVE;
                // For confirmed tracks (LOCKED), use permanent track_id; for tentative detections, use 0
                uint32_t track_id = is_selected ? current_track_id_ : 0U;
                tracked_objects_msg.objects.push_back(
                    buildTrackedObjectMessage(candidate, msg->header, track_id, status));
            }

            // Print all track assignments for debugging (after state machine)
            RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                         "[MSG OUTPUT] Publishing %zu objects:", tracked_objects_msg.objects.size());
            for (size_t i = 0; i < tracked_objects_msg.objects.size(); ++i)
            {
                const auto &obj = tracked_objects_msg.objects[i];
                RCLCPP_DEBUG(rclcpp::get_logger("segment_node"),
                             "  [%zu] track_id=%d status=%s bbox:(%d,%d,%d,%d)",
                             i, obj.track_id,
                             (obj.status == axispose_msgs::msg::TrackedObject::STATUS_CONFIRMED) ? "CONFIRMED" : "TENTATIVE",
                             obj.bbox.x_offset, obj.bbox.y_offset, obj.bbox.width, obj.bbox.height);
            }

            auto out_msg = cv_bridge::CvImage(msg->header, "mono8", output_mask).toImageMsg();
            pub_->publish(*out_msg);
            if (tracked_objects_pub_)
            {
                tracked_objects_pub_->publish(tracked_objects_msg);
            }
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Exception in image callback: %s", e.what());
        }
    }
} // namespace axispose

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(axispose::SegmentNode)
