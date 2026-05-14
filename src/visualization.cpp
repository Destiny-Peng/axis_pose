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
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <limits>
#include <cmath>
#include <array>
#include <unordered_map>

namespace axispose
{

    static bool lineFromPointDir(const cv::Point2f &p, const cv::Point2f &v, double &A, double &B, double &C)
    {
        const double n = std::hypot(static_cast<double>(v.x), static_cast<double>(v.y));
        if (n < 1e-9)
            return false;
        const double vx = static_cast<double>(v.x) / n;
        const double vy = static_cast<double>(v.y) / n;
        A = -vy;
        B = vx;
        C = -(A * static_cast<double>(p.x) + B * static_cast<double>(p.y));
        return true;
    }

    static bool extractUpperLowerEdgeLines(const cv::Mat &mask, double &ua, double &ub, double &uc, double &la, double &lb, double &lc)
    {
        std::vector<cv::Point> mask_pts;
        cv::findNonZero(mask, mask_pts);
        if (mask_pts.size() < 40)
            return false;

        cv::Vec4f center_line;
        cv::fitLine(mask_pts, center_line, cv::DIST_L2, 0, 0.01, 0.01);
        cv::Point2f v(center_line[0], center_line[1]);
        const float nv = std::sqrt(v.x * v.x + v.y * v.y);
        if (nv < 1e-6f)
            return false;
        v.x /= nv;
        v.y /= nv;

        // Normal direction used to split upper/lower edge points.
        cv::Point2f n(-v.y, v.x);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        if (contours.empty())
            return false;

        size_t best_idx = 0;
        size_t best_size = 0;
        for (size_t i = 0; i < contours.size(); ++i)
        {
            if (contours[i].size() > best_size)
            {
                best_size = contours[i].size();
                best_idx = i;
            }
        }
        const auto &contour = contours[best_idx];
        if (contour.size() < 40)
            return false;

        double s_min = std::numeric_limits<double>::infinity();
        double s_max = -std::numeric_limits<double>::infinity();
        std::vector<double> svals;
        svals.reserve(contour.size());
        for (const auto &p : contour)
        {
            const double s = static_cast<double>(n.x) * static_cast<double>(p.x) + static_cast<double>(n.y) * static_cast<double>(p.y);
            svals.push_back(s);
            s_min = std::min(s_min, s);
            s_max = std::max(s_max, s);
        }

        const double span = s_max - s_min;
        if (span < 3.0)
            return false;

        const double band = std::max(2.0, 0.10 * span);
        std::vector<cv::Point> upper_pts;
        std::vector<cv::Point> lower_pts;
        upper_pts.reserve(contour.size() / 4);
        lower_pts.reserve(contour.size() / 4);

        for (size_t i = 0; i < contour.size(); ++i)
        {
            const double s = svals[i];
            if (s >= (s_max - band))
                upper_pts.push_back(contour[i]);
            if (s <= (s_min + band))
                lower_pts.push_back(contour[i]);
        }

        if (upper_pts.size() < 10 || lower_pts.size() < 10)
            return false;

        cv::Vec4f ul, ll;
        cv::fitLine(upper_pts, ul, cv::DIST_L2, 0, 0.01, 0.01);
        cv::fitLine(lower_pts, ll, cv::DIST_L2, 0, 0.01, 0.01);
        cv::Point2f uv(ul[0], ul[1]);
        cv::Point2f up(ul[2], ul[3]);
        cv::Point2f lv(ll[0], ll[1]);
        cv::Point2f lp(ll[2], ll[3]);

        if (!lineFromPointDir(up, uv, ua, ub, uc))
            return false;
        if (!lineFromPointDir(lp, lv, la, lb, lc))
            return false;
        return true;
    }

    Visualization::Visualization(const rclcpp::NodeOptions &options) : rclcpp::Node("visualization", options)
    {
        this->declare_parameter("mask_topic", std::string("/yolo/mask"));
        this->declare_parameter("color_image_topic", std::string("/camera/rgb/image_raw"));
        this->declare_parameter("pose_topic", std::string("/shaft/pose"));
        this->declare_parameter("tracked_pose_topic", std::string("/shaft/tracked_poses"));
        this->declare_parameter("tracked_object_topic", std::string("/yolo/tracked_objects"));
        this->declare_parameter("camera_info_topic", std::string("/camera/color/camera_info"));
        this->declare_parameter("depth_camera_info_topic", std::string("/camera/depth/camera_info"));
        this->declare_parameter("axis_length", axis_length_);
        this->declare_parameter("statistics_directory_path", std::string(""));
        // saving annotated images
        this->declare_parameter("save_visualization", false);
        this->declare_parameter("save_dir", std::string("statistics/vis"));
        this->declare_parameter("save_every_n", 1);
        this->declare_parameter("line_eval_enabled", true);

        std::string mask_topic = this->get_parameter("mask_topic").as_string();
        std::string color_image_topic = this->get_parameter("color_image_topic").as_string();
        std::string pose_topic = this->get_parameter("pose_topic").as_string();
        pose_array_topic_ = this->get_parameter("tracked_pose_topic").as_string();
        object_array_topic_ = this->get_parameter("tracked_object_topic").as_string();
        std::string caminfo_topic = this->get_parameter("camera_info_topic").as_string();
        std::string depth_caminfo_topic = this->get_parameter("depth_camera_info_topic").as_string();
        axis_length_ = this->get_parameter("axis_length").as_double();
        save_annotated_ = this->get_parameter("save_visualization").as_bool();
        save_dir_ = this->get_parameter("save_dir").as_string();
        std::string stats_dir = this->get_parameter("statistics_directory_path").as_string();
        line_eval_enabled_ = this->get_parameter("line_eval_enabled").as_bool();
        if (!stats_dir.empty())
        {
            try
            {
                std::filesystem::path p = std::filesystem::absolute(std::filesystem::path(stats_dir));
                p /= "vis";
                if (!std::filesystem::exists(p))
                    std::filesystem::create_directories(p);
                save_dir_ = p.string();
                RCLCPP_INFO(this->get_logger(), "Visualization: overridden save_dir -> %s (from statistics_directory_path)", save_dir_.c_str());

                if (line_eval_enabled_)
                {
                    line_eval_csv_path_ = (std::filesystem::path(stats_dir) / "line2d_metrics.csv").string();
                    line_eval_ofs_.open(line_eval_csv_path_, std::ios::out | std::ios::trunc);
                    if (line_eval_ofs_.is_open())
                    {
                        line_eval_ofs_ << "frame_idx,timestamp_sec,timestamp_nsec,angle_err_deg,offset_err_px,upper_a,upper_b,upper_c,lower_a,lower_b,lower_c,valid\n";
                        line_eval_ofs_.flush();
                    }
                    else
                    {
                        RCLCPP_WARN(this->get_logger(), "Visualization: failed to open line2d_metrics.csv at %s", line_eval_csv_path_.c_str());
                    }
                }
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN(this->get_logger(), "Visualization: failed to create statistics-based save_dir '%s': %s", stats_dir.c_str(), e.what());
            }
        }
        save_every_n_ = this->get_parameter("save_every_n").as_int();
        if (save_every_n_ <= 0)
            save_every_n_ = 1;

        if (save_annotated_)
        {
            try
            {
                std::filesystem::path p = std::filesystem::absolute(std::filesystem::path(save_dir_));
                if (!std::filesystem::exists(p))
                    std::filesystem::create_directories(p);
                save_dir_ = p.string();
                RCLCPP_INFO(this->get_logger(), "Visualization: saving annotated images to %s (every %d frames)", save_dir_.c_str(), save_every_n_);
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN(this->get_logger(), "Visualization: failed to create save_dir '%s': %s", save_dir_.c_str(), e.what());
                save_annotated_ = false;
            }

            if (line_eval_enabled_ && line_eval_csv_path_.empty())
            {
                // Fall back to save_dir parent when statistics_directory_path is not provided.
                std::filesystem::path fallback_dir = std::filesystem::path(save_dir_).parent_path();
                if (fallback_dir.empty())
                {
                    fallback_dir = std::filesystem::current_path();
                }
                line_eval_csv_path_ = (fallback_dir / "line2d_metrics.csv").string();
                line_eval_ofs_.open(line_eval_csv_path_, std::ios::out | std::ios::trunc);
                if (line_eval_ofs_.is_open())
                {
                    line_eval_ofs_ << "frame_idx,timestamp_sec,timestamp_nsec,angle_err_deg,offset_err_px,upper_a,upper_b,upper_c,lower_a,lower_b,lower_c,valid\n";
                    line_eval_ofs_.flush();
                }
            }
        }

        rclcpp::QoS qos(rclcpp::KeepLast(5));

        // Use message_filters subscribers
        rgb_sub_.subscribe(this, color_image_topic, qos.get_rmw_qos_profile());
        pose_array_sub_.subscribe(this, pose_array_topic_, qos.get_rmw_qos_profile());

        // camera_info is latched/transient_local and static — subscribe with
        // transient_local QoS and disable intra-process on these subscriptions
        // only so we can keep container-level intra-process enabled.
        rclcpp::QoS caminfo_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local();
        rclcpp::SubscriptionOptions caminfo_sub_options;
        caminfo_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
        caminfo_color_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            caminfo_topic, caminfo_qos,
            [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg)
            {
                this->cached_caminfo_color_ = msg;
                RCLCPP_DEBUG(this->get_logger(), "cached color camera_info received");
            },
            caminfo_sub_options);
        caminfo_depth_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            depth_caminfo_topic, caminfo_qos,
            [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg)
            {
                this->cached_caminfo_depth_ = msg;
                RCLCPP_DEBUG(this->get_logger(), "cached depth camera_info received");
            },
            caminfo_sub_options);

        object_array_sub_.subscribe(this, object_array_topic_, qos.get_rmw_qos_profile());

        // Synchronize RGB, tracked poses and tracked objects; CameraInfo is read from cache.
        sync_.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(10), rgb_sub_, pose_array_sub_, object_array_sub_));
        sync_->registerCallback(&Visualization::syncCallback, this);

        vis_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/shaft/vis_image", 1);

        RCLCPP_INFO(this->get_logger(), "Visualization node started. Subscribed to %s %s %s %s", color_image_topic.c_str(), pose_array_topic_.c_str(), caminfo_topic.c_str(), object_array_topic_.c_str());
    }

    static cv::Mat cameraMatrixFromInfo(const sensor_msgs::msg::CameraInfo &cam)
    {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = cam.k[0];
        K.at<double>(1, 1) = cam.k[4];
        K.at<double>(0, 2) = cam.k[2];
        K.at<double>(1, 2) = cam.k[5];
        return K;
    }

    // project a 3D point (in camera frame) to pixel using camera matrix
    static bool projectPoint(const geometry_msgs::msg::Point &p, const cv::Mat &camera_matrix, cv::Point2f &out)
    {
        const double fx = camera_matrix.at<double>(0, 0);
        const double fy = camera_matrix.at<double>(1, 1);
        const double cx = camera_matrix.at<double>(0, 2);
        const double cy = camera_matrix.at<double>(1, 2);
        if (p.x <= 0.0)
            return false;
        out.x = static_cast<float>(-(p.y / p.x) * fx + cx);
        out.y = static_cast<float>(-(p.z / p.x) * fy + cy);
        return true;
    }

    static std::string makeTimestampFilename(const builtin_interfaces::msg::Time &stamp)
    {
        std::time_t t = static_cast<std::time_t>(stamp.sec);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::ostringstream ss;
        // vis_YYYYMMDD_HHMMSS_mmm_nnnnnnnnn.png
        ss << "vis_" << std::put_time(&tm, "%Y%m%d_%H%M%S")
           << "_" << std::setw(3) << std::setfill('0') << (stamp.nanosec / 1000000U)
           << "_" << std::setw(9) << std::setfill('0') << stamp.nanosec
           << ".png";
        return ss.str();
    }

    void Visualization::syncCallback(const Image::ConstSharedPtr rgb_msg,
                                     const TrackedPoseArray::ConstSharedPtr pose_array_msg,
                                     const TrackedObjectArray::ConstSharedPtr object_array_msg)
    {
        RCLCPP_DEBUG(this->get_logger(), "Visualization::syncCallback invoked");
        if (!pose_array_msg || !object_array_msg)
        {
            return;
        }

        // choose camera_info matching the tracked pose frame if possible
        CameraInfo::SharedPtr caminfo_to_use;
        const std::string pose_frame = pose_array_msg->header.frame_id;
        if (cached_caminfo_color_ && pose_frame == cached_caminfo_color_->header.frame_id)
            caminfo_to_use = cached_caminfo_color_;
        else if (cached_caminfo_depth_ && pose_frame == cached_caminfo_depth_->header.frame_id)
            caminfo_to_use = cached_caminfo_depth_;
        else if (cached_caminfo_color_)
            caminfo_to_use = cached_caminfo_color_;
        else if (cached_caminfo_depth_)
            caminfo_to_use = cached_caminfo_depth_;

        if (!caminfo_to_use)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Visualization: waiting for any camera_info before processing");
            return;
        }
        const cv::Mat camera_matrix = cameraMatrixFromInfo(*caminfo_to_use);
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

        cv::Mat vis = rgb_cv.clone();
        struct PoseLookup
        {
            const axispose_msgs::msg::TrackedPose *pose{nullptr};
            const axispose_msgs::msg::TrackedObject *object{nullptr};
        };

        std::unordered_map<uint32_t, PoseLookup> lookup;
        lookup.reserve(std::min(pose_array_msg->poses.size(), object_array_msg->objects.size()));
        for (const auto &pose_item : pose_array_msg->poses)
        {
            lookup[pose_item.track_id].pose = &pose_item;
        }
        for (const auto &object_item : object_array_msg->objects)
        {
            lookup[object_item.track_id].object = &object_item;
        }

        auto trackColor = [](uint32_t track_id) -> cv::Scalar
        {
            static const std::array<cv::Scalar, 8> palette = {
                cv::Scalar(0, 255, 0),
                cv::Scalar(0, 165, 255),
                cv::Scalar(255, 0, 0),
                cv::Scalar(255, 0, 255),
                cv::Scalar(0, 255, 255),
                cv::Scalar(255, 255, 0),
                cv::Scalar(128, 255, 0),
                cv::Scalar(255, 128, 0)};
            return palette[track_id % palette.size()];
        };

        for (const auto &object_item : object_array_msg->objects)
        {
            cv::Mat mask_cv;
            try
            {
                auto mask_ptr = cv_bridge::toCvCopy(object_item.mask, "mono8");
                mask_cv = mask_ptr->image.clone();
            }
            catch (const cv_bridge::Exception &e)
            {
                RCLCPP_WARN(this->get_logger(), "Visualization: mask conversion failed for track %u: %s", object_item.track_id, e.what());
                continue;
            }

            if (mask_cv.size() != rgb_cv.size())
            {
                cv::resize(mask_cv, mask_cv, rgb_cv.size(), 0, 0, cv::INTER_NEAREST);
            }

            const cv::Scalar color = trackColor(object_item.track_id);
            cv::Mat color_mask = cv::Mat::zeros(rgb_cv.size(), CV_8UC3);
            color_mask.setTo(color, mask_cv);
            cv::addWeighted(color_mask, 0.35, vis, 0.65, 0.0, vis);

            cv::rectangle(vis,
                          cv::Rect(static_cast<int>(object_item.bbox.x_offset), static_cast<int>(object_item.bbox.y_offset), static_cast<int>(object_item.bbox.width), static_cast<int>(object_item.bbox.height)),
                          color, 2);

            auto it = lookup.find(object_item.track_id);
            if (it == lookup.end() || it->second.pose == nullptr)
            {
                continue;
            }

            const auto &pose_item = *it->second.pose;
            const auto &pose_msg = pose_item.pose.pose;
            geometry_msgs::msg::Point center;
            center.x = pose_msg.position.x;
            center.y = pose_msg.position.y;
            center.z = pose_msg.position.z;

            Eigen::Quaterniond q_e(pose_msg.orientation.w,
                                   pose_msg.orientation.x,
                                   pose_msg.orientation.y,
                                   pose_msg.orientation.z);
            Eigen::Vector3d axis_dir_e = q_e.toRotationMatrix() * Eigen::Vector3d(1.0, 0.0, 0.0);
            geometry_msgs::msg::Point axis_end;
            axis_end.x = center.x + axis_dir_e.x() * axis_length_;
            axis_end.y = center.y + axis_dir_e.y() * axis_length_;
            axis_end.z = center.z + axis_dir_e.z() * axis_length_;

            cv::Point2f pc, pe;
            const bool okc = projectPoint(center, camera_matrix, pc);
            const bool oke = projectPoint(axis_end, camera_matrix, pe);
            if (okc)
            {
                cv::circle(vis, pc, 5, color, -1);
            }

            if (okc)
            {
                cv::Point2f dir_pt;
                bool have_dir = false;
                if (oke)
                {
                    dir_pt = pe;
                    have_dir = true;
                }
                else
                {
                    geometry_msgs::msg::Point axis_far;
                    axis_far.x = center.x + axis_dir_e.x() * axis_length_ * 100.0;
                    axis_far.y = center.y + axis_dir_e.y() * axis_length_ * 100.0;
                    axis_far.z = center.z + axis_dir_e.z() * axis_length_ * 100.0;
                    cv::Point2f pf;
                    if (projectPoint(axis_far, camera_matrix, pf))
                    {
                        dir_pt = pf;
                        have_dir = true;
                    }
                }

                if (have_dir)
                {
                    cv::Point2f d = dir_pt - pc;
                    if (std::abs(d.x) < 1e-6 && std::abs(d.y) < 1e-6)
                    {
                        cv::line(vis, cv::Point2f(pc.x - 50.0f, pc.y), cv::Point2f(pc.x + 50.0f, pc.y), color, 2);
                    }
                    else
                    {
                        cv::line(vis, pc, pc + d, color, 2);
                    }
                }
            }

            if (line_eval_enabled_ && line_eval_ofs_.is_open())
            {
                double angle_err_deg = std::numeric_limits<double>::quiet_NaN();
                double offset_err_px = std::numeric_limits<double>::quiet_NaN();
                double upper_a = std::numeric_limits<double>::quiet_NaN();
                double upper_b = std::numeric_limits<double>::quiet_NaN();
                double upper_c = std::numeric_limits<double>::quiet_NaN();
                double lower_a = std::numeric_limits<double>::quiet_NaN();
                double lower_b = std::numeric_limits<double>::quiet_NaN();
                double lower_c = std::numeric_limits<double>::quiet_NaN();
                int valid_eval = 0;

                const bool edges_ok = extractUpperLowerEdgeLines(mask_cv, upper_a, upper_b, upper_c, lower_a, lower_b, lower_c);
                std::vector<cv::Point> mask_pts;
                cv::findNonZero(mask_cv, mask_pts);
                if (mask_pts.size() >= 20 && okc && edges_ok)
                {
                    cv::Vec4f mask_line;
                    cv::fitLine(mask_pts, mask_line, cv::DIST_L2, 0, 0.01, 0.01);
                    cv::Point2f pm(mask_line[2], mask_line[3]);
                    cv::Point2f vm(mask_line[0], mask_line[1]);

                    std::vector<cv::Point2f> proj_samples;
                    proj_samples.reserve(10);
                    const std::array<double, 10> scales = {-40.0, -20.0, -10.0, -5.0, -2.0, 2.0, 5.0, 10.0, 20.0, 40.0};
                    for (double s : scales)
                    {
                        geometry_msgs::msg::Point ps;
                        ps.x = center.x + axis_dir_e.x() * axis_length_ * s;
                        ps.y = center.y + axis_dir_e.y() * axis_length_ * s;
                        ps.z = center.z + axis_dir_e.z() * axis_length_ * s;
                        cv::Point2f pp;
                        if (projectPoint(ps, camera_matrix, pp))
                            proj_samples.push_back(pp);
                    }

                    cv::Point2f ve = proj_samples.size() >= 2 ? proj_samples.back() - proj_samples.front() : cv::Point2f(0.0f, 0.0f);
                    const float ne = std::sqrt(ve.x * ve.x + ve.y * ve.y);
                    const float nm = std::sqrt(vm.x * vm.x + vm.y * vm.y);
                    if (ne > 1e-6f && nm > 1e-6f)
                    {
                        cv::Point2f ve_n(ve.x / ne, ve.y / ne);
                        cv::Point2f vm_n(vm.x / nm, vm.y / nm);
                        float dot = std::abs(ve_n.x * vm_n.x + ve_n.y * vm_n.y);
                        dot = std::max(-1.0f, std::min(1.0f, dot));
                        angle_err_deg = std::acos(dot) * 180.0 / M_PI;
                        const double A = -static_cast<double>(ve_n.y);
                        const double B = static_cast<double>(ve_n.x);
                        const double C = -(A * pc.x + B * pc.y);
                        offset_err_px = std::abs(A * pm.x + B * pm.y + C);
                        valid_eval = 1;
                    }
                }

                line_eval_ofs_ << line_eval_counter_ << ","
                               << pose_array_msg->header.stamp.sec << ","
                               << pose_array_msg->header.stamp.nanosec << ","
                               << object_item.track_id << ","
                               << angle_err_deg << ","
                               << offset_err_px << ","
                               << upper_a << "," << upper_b << "," << upper_c << ","
                               << lower_a << "," << lower_b << "," << lower_c << ","
                               << valid_eval << "\n";
            }
        }

        if (line_eval_enabled_ && line_eval_ofs_.is_open())
        {
            line_eval_ofs_.flush();
            line_eval_counter_++;
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

        // optionally save annotated image to disk
        if (save_annotated_)
        {
            save_counter_++;
            if ((save_counter_ % static_cast<uint64_t>(save_every_n_)) == 0)
            {
                // filename: <save_dir>/vis_YYYYMMDD_HHMMSS_mmm_nnnnnnnnn.png
                std::filesystem::path out_path = std::filesystem::path(save_dir_) / makeTimestampFilename(rgb_msg->header.stamp);
                try
                {
                    cv::imwrite(out_path.string(), vis);
                    RCLCPP_DEBUG(this->get_logger(), "Saved annotated image %s", out_path.string().c_str());
                }
                catch (const std::exception &e)
                {
                    RCLCPP_WARN(this->get_logger(), "Failed to save annotated image %s: %s", out_path.string().c_str(), e.what());
                }
            }
        }
    }

} // namespace axispose

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(axispose::Visualization)
