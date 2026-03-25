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

namespace axispose
{

    Visualization::Visualization(const rclcpp::NodeOptions &options) : rclcpp::Node("visualization", options)
    {
        this->declare_parameter("mask_topic", std::string("/yolo/mask"));
        this->declare_parameter("color_image_topic", std::string("/camera/rgb/image_raw"));
        this->declare_parameter("pose_topic", std::string("/shaft/pose"));
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
                        line_eval_ofs_ << "frame_idx,timestamp_sec,timestamp_nsec,angle_err_deg,offset_err_px,valid\n";
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
                    line_eval_ofs_ << "frame_idx,timestamp_sec,timestamp_nsec,angle_err_deg,offset_err_px,valid\n";
                    line_eval_ofs_.flush();
                }
            }
        }

        rclcpp::QoS qos(rclcpp::KeepLast(5));

        // Use message_filters subscribers
        rgb_sub_.subscribe(this, color_image_topic, qos.get_rmw_qos_profile());
        pose_sub_.subscribe(this, pose_topic, qos.get_rmw_qos_profile());

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

        mask_sub_.subscribe(this, mask_topic, qos.get_rmw_qos_profile());

        // Synchronize RGB, Pose and Mask only; CameraInfo is read from cache.
        sync_.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(10), rgb_sub_, pose_sub_, mask_sub_));
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
                                     const Pose::ConstSharedPtr pose_msg,
                                     const Image::ConstSharedPtr mask_msg)
    {
        RCLCPP_DEBUG(this->get_logger(), "Visualization::syncCallback invoked");
        // choose camera_info matching the pose frame if possible
        CameraInfo::SharedPtr caminfo_to_use;
        if (cached_caminfo_color_ && pose_msg->header.frame_id == cached_caminfo_color_->header.frame_id)
            caminfo_to_use = cached_caminfo_color_;
        else if (cached_caminfo_depth_ && pose_msg->header.frame_id == cached_caminfo_depth_->header.frame_id)
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
        bool okc = projectPoint(center, *caminfo_to_use, pc);
        bool oke = projectPoint(axis_end, *caminfo_to_use, pe);

        // draw mask overlay (colored)
        cv::Mat vis = rgb_cv.clone();
        cv::Mat color_mask;
        cv::cvtColor(mask_cv, color_mask, cv::COLOR_GRAY2BGR);
        // colorize mask green
        color_mask.setTo(cv::Scalar(0, 255, 0), mask_cv);
        double alpha = 0.4;
        cv::addWeighted(color_mask, alpha, vis, 1.0 - alpha, 0.0, vis);

        // draw center and axis line if projection succeeded
        cv::Point2f eval_dir_pt;
        bool eval_have_dir = false;

        if (okc)
        {
            cv::circle(vis, pc, 6, cv::Scalar(0, 0, 255), -1);
        }

        if (okc)
        {
            // determine a projected axis direction point; if axis_end projection failed,
            // try projecting a far point along the axis to get a direction in image plane.
            cv::Point2f dir_pt;
            bool have_dir = false;
            if (oke)
            {
                dir_pt = pe;
                have_dir = true;
            }
            else
            {
                // try a far point along axis to estimate image direction
                geometry_msgs::msg::Point axis_far;
                axis_far.x = center.x + axis_dir_e.x() * axis_length_ * 100.0;
                axis_far.y = center.y + axis_dir_e.y() * axis_length_ * 100.0;
                axis_far.z = center.z + axis_dir_e.z() * axis_length_ * 100.0;
                cv::Point2f pf;
                if (projectPoint(axis_far, *caminfo_to_use, pf))
                {
                    dir_pt = pf;
                    have_dir = true;
                }
            }

            if (have_dir)
            {
                eval_dir_pt = dir_pt;
                eval_have_dir = true;
                // build a line through pc in direction (dir_pt - pc) and clip to image bounds
                cv::Point2f d = dir_pt - pc;
                const int W = vis.cols;
                const int H = vis.rows;
                std::vector<cv::Point2f> ints;
                // if direction is nearly zero, draw small short line
                if (std::abs(d.x) < 1e-6 && std::abs(d.y) < 1e-6)
                {
                    cv::Point2f p1(pc.x - 50.0f, pc.y);
                    cv::Point2f p2(pc.x + 50.0f, pc.y);
                    cv::line(vis, p1, p2, cv::Scalar(255, 0, 0), 2);
                }
                else
                {
                    // intersections with x=0 and x=W-1
                    if (std::abs(d.x) > 1e-9)
                    {
                        float t0 = (0.0f - pc.x) / d.x;
                        float y0 = pc.y + t0 * d.y;
                        if (y0 >= 0.0f && y0 <= static_cast<float>(H - 1))
                            ints.emplace_back(0.0f, y0);
                        float t1 = (static_cast<float>(W - 1) - pc.x) / d.x;
                        float y1 = pc.y + t1 * d.y;
                        if (y1 >= 0.0f && y1 <= static_cast<float>(H - 1))
                            ints.emplace_back(static_cast<float>(W - 1), y1);
                    }
                    // intersections with y=0 and y=H-1
                    if (std::abs(d.y) > 1e-9)
                    {
                        float t2 = (0.0f - pc.y) / d.y;
                        float x2 = pc.x + t2 * d.x;
                        if (x2 >= 0.0f && x2 <= static_cast<float>(W - 1))
                            ints.emplace_back(x2, 0.0f);
                        float t3 = (static_cast<float>(H - 1) - pc.y) / d.y;
                        float x3 = pc.x + t3 * d.x;
                        if (x3 >= 0.0f && x3 <= static_cast<float>(W - 1))
                            ints.emplace_back(x3, static_cast<float>(H - 1));
                    }

                    // dedupe almost-equal points
                    std::vector<cv::Point2f> uniq;
                    for (const auto &p : ints)
                    {
                        bool found = false;
                        for (const auto &q : uniq)
                        {
                            if (std::hypot(p.x - q.x, p.y - q.y) < 1e-2f)
                            {
                                found = true;
                                break;
                            }
                        }
                        if (!found)
                            uniq.push_back(p);
                    }

                    if (uniq.size() >= 2)
                    {
                        // pick two most distant points
                        size_t a = 0, b = 1;
                        double bestd = 0.0;
                        for (size_t i = 0; i < uniq.size(); ++i)
                            for (size_t j = i + 1; j < uniq.size(); ++j)
                            {
                                double dd = std::hypot(uniq[i].x - uniq[j].x, uniq[i].y - uniq[j].y);
                                if (dd > bestd)
                                {
                                    bestd = dd;
                                    a = i;
                                    b = j;
                                }
                            }
                        cv::line(vis, uniq[a], uniq[b], cv::Scalar(255, 0, 0), 2);
                    }
                    else
                    {
                        // fallback: draw short line from pc toward dir_pt
                        cv::Point2f p2 = pc + d * 50.0f;
                        cv::line(vis, pc, p2, cv::Scalar(255, 0, 0), 2);
                    }
                }
            }
        }

        // Optional quantitative 2D evaluation: compare projected 3D axis line vs fitted mask line.
        if (line_eval_enabled_ && line_eval_ofs_.is_open())
        {
            double angle_err_deg = std::numeric_limits<double>::quiet_NaN();
            double offset_err_px = std::numeric_limits<double>::quiet_NaN();
            int valid_eval = 0;

            std::vector<cv::Point> mask_pts;
            cv::findNonZero(mask_cv, mask_pts);
            if (mask_pts.size() >= 20)
            {
                cv::Vec4f mask_line;
                cv::fitLine(mask_pts, mask_line, cv::DIST_L2, 0, 0.01, 0.01);
                cv::Point2f pm(mask_line[2], mask_line[3]);
                cv::Point2f vm(mask_line[0], mask_line[1]);

                if (okc)
                {
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
                        if (projectPoint(ps, *caminfo_to_use, pp))
                            proj_samples.push_back(pp);
                    }

                    cv::Point2f ve;
                    if (proj_samples.size() >= 2)
                    {
                        ve = proj_samples.back() - proj_samples.front();
                    }
                    else if (eval_have_dir)
                    {
                        ve = eval_dir_pt - pc;
                    }
                    else
                    {
                        ve = cv::Point2f(0.0f, 0.0f);
                    }

                    const float ne = std::sqrt(ve.x * ve.x + ve.y * ve.y);
                    const float nm = std::sqrt(vm.x * vm.x + vm.y * vm.y);
                    if (ne > 1e-6f && nm > 1e-6f)
                    {
                        cv::Point2f ve_n(ve.x / ne, ve.y / ne);
                        cv::Point2f vm_n(vm.x / nm, vm.y / nm);
                        float dot = std::abs(ve_n.x * vm_n.x + ve_n.y * vm_n.y);
                        dot = std::max(-1.0f, std::min(1.0f, dot));
                        angle_err_deg = std::acos(dot) * 180.0 / M_PI;

                        // Signed normal form for projected estimated line through center with direction ve_n.
                        const double A = -static_cast<double>(ve_n.y);
                        const double B = static_cast<double>(ve_n.x);
                        const double C = -(A * pc.x + B * pc.y);
                        offset_err_px = std::abs(A * pm.x + B * pm.y + C);
                        valid_eval = 1;
                    }
                }
            }

            line_eval_ofs_ << line_eval_counter_ << ","
                           << pose_msg->header.stamp.sec << ","
                           << pose_msg->header.stamp.nanosec << ",";
            if (valid_eval)
            {
                line_eval_ofs_ << angle_err_deg << "," << offset_err_px;
            }
            else
            {
                line_eval_ofs_ << ",";
            }
            line_eval_ofs_ << "," << valid_eval << "\n";
            if ((save_counter_ % 10) == 0)
            {
                line_eval_ofs_.flush();
            }
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
