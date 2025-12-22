#include "axispose/driver.hpp"

#include <filesystem>
#include <algorithm>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

namespace fs = std::filesystem;

namespace axispose
{

    CameraDriver::CameraDriver(ros::NodeHandle &nh, ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh), index_(0)
    {
        pnh_.param<std::string>("rgb_dir", rgb_dir_, "");
        pnh_.param<std::string>("depth_dir", depth_dir_, "");
        pnh_.param<std::string>("camera_info_file", camera_info_file_, "");
        pnh_.param<std::string>("frame_id", frame_id_, "camera");
        pnh_.param<double>("publish_rate", publish_rate_, 10.0);
        pnh_.param<bool>("loop", loop_, true);

        ROS_INFO("CameraDriver: rgb_dir=%s depth_dir=%s camera_info_file=%s rate=%.2f frame_id=%s loop=%s",
                 rgb_dir_.c_str(), depth_dir_.c_str(), camera_info_file_.c_str(), publish_rate_, frame_id_.c_str(), loop_ ? "true" : "false");

        rgb_pub_ = nh_.advertise<sensor_msgs::Image>("/camera/rgb/image_raw", 5);
        depth_pub_ = nh_.advertise<sensor_msgs::Image>("/camera/depth/image_raw", 5);
        caminfo_pub_ = nh_.advertise<sensor_msgs::CameraInfo>("/camera/camera_info", 5);

        if (!camera_info_file_.empty())
        {
            camera_info_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(nh_, "camera");
            std::string url = camera_info_file_;
            if (url.rfind("file:", 0) != 0 && url.rfind("package:", 0) != 0)
                url = std::string("file://") + url;
            if (camera_info_manager_->validateURL(url))
            {
                camera_info_manager_->loadCameraInfo(url);
                camera_info_msg_ = camera_info_manager_->getCameraInfo();
            }
            else
            {
                ROS_WARN("camera_info_manager could not validate URL %s, falling back to parser", url.c_str());
                if (!loadCameraInfoFromFile(camera_info_file_, camera_info_msg_))
                {
                    ROS_WARN("failed to load camera_info from %s", camera_info_file_.c_str());
                }
            }
        }
        camera_info_msg_.header.frame_id = frame_id_;

        loadImageLists();
        preloadImages();

        timer_ = nh_.createTimer(ros::Duration(1.0 / std::max(0.001, publish_rate_)), &CameraDriver::publishTimerCallback, this);
    }

    void CameraDriver::preloadImages()
    {
        rgb_msgs_.clear();
        depth_msgs_.clear();

        auto mat_to_msg = [this](const cv::Mat &mat, const std::string &encoding) -> sensor_msgs::Image
        {
            sensor_msgs::Image img;
            std_msgs::Header h;
            h.frame_id = frame_id_;
            h.stamp = ros::Time::now();
            cv_bridge::CvImage cv_img(h, encoding, mat);
            img = *cv_img.toImageMsg();
            return img;
        };

        for (const auto &f : rgb_files_)
        {
            cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
            if (img.empty())
            {
                ROS_WARN("failed to read rgb image for preload: %s", f.c_str());
                continue;
            }
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            rgb_msgs_.push_back(mat_to_msg(img, sensor_msgs::image_encodings::RGB8));
        }

        for (const auto &f : depth_files_)
        {
            cv::Mat dimg = cv::imread(f, cv::IMREAD_UNCHANGED);
            if (dimg.empty())
            {
                ROS_WARN("failed to read depth image for preload: %s", f.c_str());
                continue;
            }
            ROS_INFO("depth image %s type=%d", f.c_str(), dimg.type());
            if (dimg.type() != CV_16UC1)
            {
                ROS_ERROR("unsupported depth image type (not 16UC1): %s", f.c_str());
                std::abort();
            }
            depth_msgs_.push_back(mat_to_msg(dimg, sensor_msgs::image_encodings::TYPE_16UC1));
        }
    }

    void CameraDriver::loadImageLists()
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
            }
        };

        push_images(rgb_dir_, rgb_files_);
        push_images(depth_dir_, depth_files_);

        ROS_INFO("Found %zu rgb images and %zu depth images", rgb_files_.size(), depth_files_.size());
    }

    bool CameraDriver::loadCameraInfoFromFile(const std::string &yaml_file, sensor_msgs::CameraInfo &info)
    {
        cv::FileStorage fs(yaml_file, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            ROS_ERROR("cannot open camera_info file: %s", yaml_file.c_str());
            return false;
        }

        int iw = 0, ih = 0;
        fs["image_width"] >> iw;
        fs["image_height"] >> ih;
        if (iw > 0)
            info.width = static_cast<uint32_t>(iw);
        if (ih > 0)
            info.height = static_cast<uint32_t>(ih);

        std::string dist_model;
        fs["distortion_model"] >> dist_model;
        if (!dist_model.empty())
            info.distortion_model = dist_model;
        else
            info.distortion_model = "plumb_bob";

        auto read_seq = [&](const std::string &key) -> std::vector<double>
        {
            std::vector<double> out;
            cv::FileNode node = fs[key];
            if (node.empty())
                return out;
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
            if (node.isSeq())
            {
                for (auto it = node.begin(); it != node.end(); ++it)
                {
                    double v = (double)(*it);
                    out.push_back(v);
                }
                return out;
            }
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

        std::vector<double> cam = read_seq("camera_matrix");
        if (cam.size() >= 9)
        {
            for (size_t i = 0; i < 9; ++i)
                info.K[i] = cam[i];
        }

        std::vector<double> d = read_seq("distortion_coefficients");
        if (!d.empty())
            info.D = d;

        std::vector<double> p = read_seq("projection_matrix");
        if (p.size() >= 12)
        {
            for (size_t i = 0; i < 12; ++i)
                info.P[i] = p[i];
        }

        fs.release();
        return true;
    }

    void CameraDriver::publishTimerCallback(const ros::TimerEvent &)
    {
        if (rgb_msgs_.empty() && depth_msgs_.empty())
            return;

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
                timer_.stop();
                ROS_INFO("Finished publishing images");
                return;
            }
        }

        ros::Time now = ros::Time::now();

        if (!rgb_msgs_.empty() && !depth_msgs_.empty())
        {
            size_t idx = index_ % max_idx;
            auto rgb = rgb_msgs_[idx];
            rgb.header.stamp = now;
            rgb.header.frame_id = frame_id_;
            rgb_pub_.publish(rgb);

            auto depth = depth_msgs_[idx];
            depth.header.stamp = now;
            depth.header.frame_id = frame_id_;
            depth_pub_.publish(depth);

            ROS_INFO("Published preloaded rgb/depth image %s,%s", rgb_files_[idx].c_str(), depth_files_[idx].c_str());
        }
        else
        {
            size_t idx = index_ % std::max(rgb_files_.size(), depth_files_.size());
            if (!rgb_files_.empty())
            {
                const std::string &rgb_file = rgb_files_[idx % rgb_files_.size()];
                cv::Mat img = cv::imread(rgb_file, cv::IMREAD_COLOR);
                if (!img.empty())
                {
                    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
                    std_msgs::Header h;
                    h.stamp = now;
                    h.frame_id = frame_id_;
                    cv_bridge::CvImage cv_img(h, sensor_msgs::image_encodings::RGB8, img);
                    rgb_pub_.publish(*cv_img.toImageMsg());
                }
                else
                {
                    ROS_WARN("failed to read rgb image: %s", rgb_file.c_str());
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
                    std_msgs::Header h;
                    h.stamp = now;
                    h.frame_id = frame_id_;
                    cv_bridge::CvImage cv_img(h, sensor_msgs::image_encodings::TYPE_32FC1, depth_f);
                    depth_pub_.publish(*cv_img.toImageMsg());
                }
                else
                {
                    ROS_WARN("failed to read depth image: %s", depth_file.c_str());
                }
            }
        }

        if (caminfo_pub_)
        {
            camera_info_msg_.header.stamp = now;
            caminfo_pub_.publish(camera_info_msg_);
        }

        ++index_;
    }

} // namespace axispose
#include <ros/ros.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "camera_driver_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    axispose::CameraDriver driver(nh, pnh);

    ros::spin();
    return 0;
}
