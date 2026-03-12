#include "axispose/driver.hpp"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <camera_info_manager/camera_info_manager.hpp>

#include "rclcpp_components/register_node_macro.hpp"

namespace fs = std::filesystem;

namespace axispose
{

    CameraDriver::CameraDriver(const rclcpp::NodeOptions &options)
        : Node("camera_driver", options), index_(0)
    {
        // parameters
        this->declare_parameter<std::string>("rgb_dir", "");
        this->declare_parameter<std::string>("depth_dir", "");
        this->declare_parameter<std::string>("color_camera_info_file", "");
        this->declare_parameter<std::string>("depth_camera_info_file", "");
        this->declare_parameter<std::string>("frame_id", "camera");
        this->declare_parameter<double>("publish_rate", 10.0);
        this->declare_parameter<bool>("loop", true);
        this->declare_parameter<std::string>("color_image_topic", "/camera/rgb/image_raw");
        this->declare_parameter<std::string>("depth_image_topic", "/camera/depth/image_raw");
        this->declare_parameter<std::string>("color_camera_info_topic", "/camera/color/camera_info");
        this->declare_parameter<std::string>("depth_camera_info_topic", "/camera/depth/camera_info");
        this->declare_parameter<std::string>("camera_name", "camera");

        this->get_parameter("rgb_dir", rgb_dir_);
        this->get_parameter("depth_dir", depth_dir_);
        this->get_parameter("color_camera_info_file", color_camera_info_file_);
        this->get_parameter("depth_camera_info_file", depth_camera_info_file_);
        this->get_parameter("frame_id", frame_id_);
        this->get_parameter("publish_rate", publish_rate_);
        this->get_parameter("loop", loop_);
        this->get_parameter("camera_name", camera_name_);
        std::string color_image_topic_ = this->get_parameter("color_image_topic").as_string();
        std::string depth_image_topic_ = this->get_parameter("depth_image_topic").as_string();
        std::string color_camera_info_topic_ = this->get_parameter("color_camera_info_topic").as_string();
        std::string depth_camera_info_topic_ = this->get_parameter("depth_camera_info_topic").as_string();

        RCLCPP_INFO(this->get_logger(), "CameraDriver: rgb_dir=%s depth_dir=%s color_camera_info_file=%s depth_camera_info_file=%s rate=%.2f frame_id=%s loop=%s",
                    rgb_dir_.c_str(), depth_dir_.c_str(), color_camera_info_file_.c_str(), depth_camera_info_file_.c_str(), publish_rate_, frame_id_.c_str(), loop_ ? "true" : "false");

        // publishers
        rclcpp::QoS qos(rclcpp::KeepLast(5));
        rgb_pub_ = this->create_publisher<sensor_msgs::msg::Image>(color_image_topic_, qos);
        depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(depth_image_topic_, qos);
        // camera_info should be latched for late subscribers: use transient_local QoS
        rclcpp::QoS caminfo_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local();
        color_caminfo_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(color_camera_info_topic_, caminfo_qos);
        depth_caminfo_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(depth_camera_info_topic_, caminfo_qos);

        // load camera info via camera_info_manager if available
        if (!color_camera_info_file_.empty())
        {
            try
            {
                color_camera_info_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(this, camera_name_ + "_color");
                std::string url = color_camera_info_file_;
                // if local path, convert to file:// URL if necessary
                if (url.rfind("file:", 0) != 0 && url.rfind("package:", 0) != 0)
                {
                    url = std::string("file://") + url;
                }
                if (color_camera_info_manager_->validateURL(url))
                {
                    color_camera_info_manager_->loadCameraInfo(url);
                    color_camera_info_msg_ = color_camera_info_manager_->getCameraInfo();
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "color_camera_info_manager could not validate URL %s, falling back to parser", url.c_str());
                    if (!load_camera_info_from_file(color_camera_info_file_, color_camera_info_msg_))
                    {
                        RCLCPP_WARN(this->get_logger(), "failed to load color_camera_info from %s", color_camera_info_file_.c_str());
                    }
                }
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN(this->get_logger(), "color_camera_info_manager exception: %s, falling back to parser", e.what());
                if (!load_camera_info_from_file(color_camera_info_file_, color_camera_info_msg_))
                {
                    RCLCPP_WARN(this->get_logger(), "failed to load color_camera_info from %s", color_camera_info_file_.c_str());
                }
            }
        }
        // set camera_info frame_ids for color/depth derived from base frame_id_
        color_camera_info_msg_.header.frame_id = frame_id_ + std::string("_color");

        // load depth camera info via camera_info_manager if available
        if (!depth_camera_info_file_.empty())
        {
            try
            {
                auto depth_camera_info_manager = std::make_shared<camera_info_manager::CameraInfoManager>(this, camera_name_ + "_depth");
                std::string url = depth_camera_info_file_;
                if (url.rfind("file:", 0) != 0 && url.rfind("package:", 0) != 0)
                {
                    url = std::string("file://") + url;
                }
                if (depth_camera_info_manager->validateURL(url))
                {
                    depth_camera_info_manager->loadCameraInfo(url);
                    depth_camera_info_msg_ = depth_camera_info_manager->getCameraInfo();
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "depth_camera_info_manager could not validate URL %s, falling back to parser", url.c_str());
                    if (!load_camera_info_from_file(depth_camera_info_file_, depth_camera_info_msg_))
                    {
                        RCLCPP_WARN(this->get_logger(), "failed to load depth_camera_info from %s", depth_camera_info_file_.c_str());
                    }
                }
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN(this->get_logger(), "depth_camera_info_manager exception: %s, falling back to parser", e.what());
                if (!load_camera_info_from_file(depth_camera_info_file_, depth_camera_info_msg_))
                {
                    RCLCPP_WARN(this->get_logger(), "failed to load depth_camera_info from %s", depth_camera_info_file_.c_str());
                }
            }
        }
        // ensure depth camera_info frame_id is set
        depth_camera_info_msg_.header.frame_id = frame_id_ + std::string("_depth");

        // Publish camera_info immediately (latched) so late subscribers receive them
        rclcpp::Time now = this->now();
        if (color_caminfo_pub_)
        {
            color_camera_info_msg_.header.stamp = now;
            color_caminfo_pub_->publish(color_camera_info_msg_);
        }
        if (depth_caminfo_pub_)
        {
            depth_camera_info_msg_.header.stamp = now;
            depth_caminfo_pub_->publish(depth_camera_info_msg_);
        }

        // load image lists
        load_image_lists();

        // preload images into messages
        preload_images();

        // timer for images
        auto period = std::chrono::duration<double>(1.0 / std::max(0.001, publish_rate_));
        image_timer_ = this->create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period), std::bind(&CameraDriver::publish_timer_callback, this));
        param_subscriber_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
        // parameter callback to allow runtime publish_rate change
        auto publish_rate_callback = [this](const rclcpp::Parameter &p)
        {
            if (p.get_name() == "publish_rate")
            {
                double new_rate = p.as_double();
                publish_rate_ = new_rate;
                // rebuild timer
                if (image_timer_)
                    image_timer_->cancel();
                auto period = std::chrono::duration<double>(1.0 / std::max(0.001, publish_rate_));
                image_timer_ = this->create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period), std::bind(&CameraDriver::publish_timer_callback, this));
                RCLCPP_INFO(this->get_logger(), "Timer updated to %.2f Hz", publish_rate_);
            }
        };
        this->publish_rate_cb_handle_ = param_subscriber_->add_parameter_callback("publish_rate", publish_rate_callback);
    }

    void CameraDriver::preload_images()
    {
        rgb_msgs_.clear();
        depth_msgs_.clear();

        // helper to convert cv::Mat to sensor_msgs::Image
        auto mat_to_msg = [this](const cv::Mat &mat, const std::string &encoding) -> sensor_msgs::msg::Image::SharedPtr
        {
            std_msgs::msg::Header h;
            // don't set frame_id here; set it explicitly depending on color/depth when pushing
            h.frame_id = std::string();
            h.stamp = this->now();
            auto msg = std::make_shared<sensor_msgs::msg::Image>();
            cv_bridge::CvImage cv_img(h, encoding, mat);
            *msg = *cv_img.toImageMsg();
            return msg;
        };

        // preload rgb
        for (const auto &f : rgb_files_)
        {
            cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
            if (img.empty())
            {
                RCLCPP_WARN(this->get_logger(), "failed to read rgb image for preload: %s", f.c_str());
                continue;
            }
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            rgb_msgs_.push_back(mat_to_msg(img, sensor_msgs::image_encodings::RGB8));
            // set color-specific frame_id
            rgb_msgs_.back()->header.frame_id = frame_id_ + std::string("_color");
        }

        // preload depth
        for (const auto &f : depth_files_)
        {
            cv::Mat dimg = cv::imread(f, cv::IMREAD_UNCHANGED);
            if (dimg.empty())
            {
                RCLCPP_WARN(this->get_logger(), "failed to read depth image for preload: %s", f.c_str());
                continue;
            }
            RCLCPP_INFO(this->get_logger(), "depth image %s type=%d", f.c_str(), dimg.type());
            // ensure depth is 16UC1
            if (dimg.type() != CV_16UC1)
            {
                RCLCPP_ERROR(this->get_logger(), "unsupported depth image type (not 16UC1): %s", f.c_str());
                std::abort();
            }

            depth_msgs_.push_back(mat_to_msg(dimg, sensor_msgs::image_encodings::TYPE_16UC1));
            // set depth-specific frame_id
            depth_msgs_.back()->header.frame_id = frame_id_ + std::string("_depth");
        }
    }
    void CameraDriver::load_image_lists()
    {
        rgb_files_.clear();
        depth_files_.clear();

        auto push_images = [](const std::string &dir, std::vector<std::string> &out)
        {
            if (dir.empty())
                return;
            try
            {
                for (auto &p : fs::directory_iterator(dir))
                {
                    if (!p.is_regular_file())
                        continue;
                    auto ext = p.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tiff" || ext == ".bmp")
                    {
                        out.push_back(p.path().string());
                    }
                }
                std::sort(out.begin(), out.end());
            }
            catch (const std::exception &e)
            {
                // ignore
            }
        };

        push_images(rgb_dir_, rgb_files_);
        push_images(depth_dir_, depth_files_);
        // cout the file names loaded
        // for (size_t i = 0; i < rgb_files_.size(); i++)
        // {
        //     RCLCPP_INFO(this->get_logger(), "Loaded rgb image: %s", rgb_files_[i].c_str());
        //     RCLCPP_INFO(this->get_logger(), "Loaded depth image: %s", depth_files_[i].c_str());
        // }

        RCLCPP_INFO(this->get_logger(), "Found %zu rgb images and %zu depth images", rgb_files_.size(), depth_files_.size());
    }

    bool CameraDriver::load_camera_info_from_file(const std::string &yaml_file, sensor_msgs::msg::CameraInfo &info)
    {
        // 使用 OpenCV FileStorage 读取 YAML，兼容多种 camera_info 格式
        cv::FileStorage fs(yaml_file, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            RCLCPP_ERROR(this->get_logger(), "cannot open camera_info file: %s", yaml_file.c_str());
            return false;
        }

        // image width/height
        int iw = 0, ih = 0;
        fs["image_width"] >> iw;
        fs["image_height"] >> ih;
        if (iw > 0)
            info.width = static_cast<uint32_t>(iw);
        if (ih > 0)
            info.height = static_cast<uint32_t>(ih);

        // distortion_model
        std::string dist_model;
        fs["distortion_model"] >> dist_model;
        if (!dist_model.empty())
            info.distortion_model = dist_model;
        else
            info.distortion_model = "plumb_bob";

        // helper to read a sequence of numbers into a std::vector<double>
        auto read_seq = [&](const std::string &key) -> std::vector<double>
        {
            std::vector<double> out;
            cv::FileNode node = fs[key];
            if (node.empty())
                return out;
            // case 1: node has subnode 'data' (typical camera_info format)
            cv::FileNode dataNode = node["data"];
            if (!dataNode.empty())
            {
                for (auto it = dataNode.begin(); it != dataNode.end(); ++it)
                {
                    double v = (double)(*it);
                    out.push_back(v);
                }
                return out;
            }
            // case 2: node itself is a sequence
            if (node.isSeq())
            {
                for (auto it = node.begin(); it != node.end(); ++it)
                {
                    double v = (double)(*it);
                    out.push_back(v);
                }
                return out;
            }
            // case 3: node is a map that can be read as cv::Mat
            cv::Mat m;
            try
            {
                fs[key] >> m;
            }
            catch (...)
            {
            }
            if (!m.empty())
            {
                out.reserve(m.rows * m.cols);
                for (int r = 0; r < m.rows; ++r)
                {
                    for (int c = 0; c < m.cols; ++c)
                    {
                        out.push_back(m.at<double>(r, c));
                    }
                }
            }
            return out;
        };

        // camera_matrix
        std::vector<double> cam = read_seq("camera_matrix");
        if (cam.size() == 9)
        {
            for (size_t i = 0; i < 9; ++i)
                info.k[i] = cam[i];
        }
        else if (cam.size() > 9)
        {
            for (size_t i = 0; i < 9; ++i)
                info.k[i] = cam[i];
        }

        // distortion_coefficients
        std::vector<double> d = read_seq("distortion_coefficients");
        if (!d.empty())
        {
            info.d = d;
        }

        // projection_matrix
        std::vector<double> p = read_seq("projection_matrix");
        if (p.size() >= 12)
        {
            for (size_t i = 0; i < 12; ++i)
                info.p[i] = p[i];
        }

        fs.release();
        return true;
    }

    void CameraDriver::publish_timer_callback()
    {
        // prefer preloaded messages
        if (rgb_msgs_.empty() && depth_msgs_.empty())
            return;

        // determine max index — if both exist, assume equal count; else use the non-empty size
        size_t max_idx = 0;
        if (!rgb_msgs_.empty() && !depth_msgs_.empty())
        {
            max_idx = std::min(rgb_msgs_.size(), depth_msgs_.size());
        }
        else if (!rgb_msgs_.empty())
        {
            max_idx = rgb_msgs_.size();
        }
        else
        {
            max_idx = depth_msgs_.size();
        }
        if (max_idx == 0)
            return;

        if (index_ >= max_idx)
        {
            if (loop_)
                index_ = 0;
            else
            {
                if (image_timer_)
                    image_timer_->cancel();
                RCLCPP_INFO(this->get_logger(), "Finished publishing images");
                return;
            }
        }

        rclcpp::Time now = this->now();

        // If preloaded messages exist, publish them directly (no copy)
        if (!rgb_msgs_.empty() && !depth_msgs_.empty())
        {
            size_t idx = index_ % max_idx;
            // update headers in-place and publish shared_ptr -> zero-copy
            rgb_msgs_[idx]->header.stamp = now;
            rgb_msgs_[idx]->header.frame_id = frame_id_ + std::string("_color");
            rgb_pub_->publish(*rgb_msgs_[idx]);

            depth_msgs_[idx]->header.stamp = now;
            depth_msgs_[idx]->header.frame_id = frame_id_ + std::string("_depth");
            depth_pub_->publish(*depth_msgs_[idx]);
            RCLCPP_INFO(this->get_logger(), "Published preloaded rgb/depth image %s,%s", rgb_files_[idx].c_str(), depth_files_[idx].c_str());
        }
        else
        {
            // Fallback: if one of the preloads is missing, try to read files; use same index for both
            size_t idx = index_ % std::max(rgb_files_.size(), depth_files_.size());

            if (!rgb_files_.empty())
            {
                const std::string &rgb_file = rgb_files_[idx % rgb_files_.size()];
                cv::Mat img = cv::imread(rgb_file, cv::IMREAD_COLOR);
                if (!img.empty())
                {
                    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
                    std_msgs::msg::Header h;
                    h.stamp = now;
                    h.frame_id = frame_id_ + std::string("_color");
                    auto cv_img = cv_bridge::CvImage(h, sensor_msgs::image_encodings::RGB8, img);
                    rgb_pub_->publish(*cv_img.toImageMsg());
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "failed to read rgb image: %s", rgb_file.c_str());
                }
            }

            if (!depth_files_.empty())
            {
                const std::string &depth_file = depth_files_[idx % depth_files_.size()];
                cv::Mat dimg = cv::imread(depth_file, cv::IMREAD_UNCHANGED);
                if (!dimg.empty())
                {
                    if (dimg.channels() == 3)
                        cv::cvtColor(dimg, dimg, cv::COLOR_BGR2GRAY);
                    cv::Mat depth_f;
                    dimg.convertTo(depth_f, CV_32F);
                    std_msgs::msg::Header h;
                    h.stamp = now;
                    h.frame_id = frame_id_ + std::string("_depth");
                    auto cv_img = cv_bridge::CvImage(h, sensor_msgs::image_encodings::TYPE_32FC1, depth_f);
                    depth_pub_->publish(*cv_img.toImageMsg());
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "failed to read depth image: %s", depth_file.c_str());
                }
            }
        }

        ++index_;
    }

} // namespace axispose

RCLCPP_COMPONENTS_REGISTER_NODE(axispose::CameraDriver)