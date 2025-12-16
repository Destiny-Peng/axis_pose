#include "axispose/seg.hpp"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

using std::placeholders::_1;
namespace axispose
{
    SegmentNode::SegmentNode(const rclcpp::NodeOptions &options)
        : rclcpp::Node("segment_node", options)
    {
        // 参数：engine 路径
        this->declare_parameter<std::string>("engine", "");

        std::string engine_path = this->get_parameter("engine").as_string();

        if (engine_path.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Parameter 'engine' is required.");
            throw std::runtime_error("Parameter 'engine' is required.");
        }

        option_.enableSwapRB();
        model_ = std::make_unique<trtyolo::SegmentModel>(engine_path, option_);

        // 订阅 RGB 图像
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/rgb/image_raw", rclcpp::QoS(rclcpp::KeepLast(5)).best_effort(), std::bind(&SegmentNode::image_callback, this, _1));

        pub_ = this->create_publisher<sensor_msgs::msg::Image>("/yolo/mask", 1);

        RCLCPP_INFO(this->get_logger(), "Segment node initialized. Engine: %s", engine_path.c_str());
    }

    void SegmentNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
            trtyolo::Image input(image.data, image.cols, image.rows);
            auto result = model_->predict(input);

            // 创建空的单通道掩码图像
            cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(0));

            // 如果模型返回多个目标，将所有目标掩码合并
            for (size_t i = 0; i < result.num; ++i)
            {
                auto &box = result.boxes[i];
                auto xyxy = box.xyxy();
                int w = std::max(xyxy[2] - xyxy[0] + 1, 1);
                int h = std::max(xyxy[3] - xyxy[1] + 1, 1);

                // 原始 bbox 可能有负值或超出边界，需要记录偏移量以便从 mask 中正确裁切
                int x1 = std::max(0, xyxy[0]);
                int y1 = std::max(0, xyxy[1]);
                int x2 = std::min(image.cols, xyxy[2] + 1);
                int y2 = std::min(image.rows, xyxy[3] + 1);

                cv::Mat float_mask(result.masks[i].height, result.masks[i].width, CV_32FC1, result.masks[i].data.data());
                cv::resize(float_mask, float_mask, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
                cv::Mat bool_mask;
                cv::threshold(float_mask, bool_mask, 0.5, 255, cv::THRESH_BINARY);
                bool_mask.convertTo(bool_mask, CV_8UC1);
                // 对bool_mask进行边界腐蚀
                int erosion_size = 1; // 根据需要调整腐蚀大小
                cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                                            cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                            cv::Point(erosion_size, erosion_size));
                // cv::erode(bool_mask, bool_mask, element, cv::Point(-1, -1), 2);

                // 计算从 bool_mask 中裁切的偏移（若 xyxy 左上角为负，则偏移为负值的绝对值）
                int src_x_offset = std::max(0, -xyxy[0]);
                int src_y_offset = std::max(0, -xyxy[1]);

                // 目标区域的宽高（在原图中的实际可用区域）
                int target_w = x2 - x1;
                int target_h = y2 - y1;
                if (target_w <= 0 || target_h <= 0)
                    continue;

                // 确保源裁切区域不会超出 bool_mask 的边界
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
                // 使用按位或将多个目标掩码合并
                cv::bitwise_or(roi, src, roi);
            }

            // 发布 mask (mono8)
            auto out_msg = cv_bridge::CvImage(msg->header, "mono8", mask).toImageMsg();
            pub_->publish(*out_msg);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Exception in image callback: %s", e.what());
        }
    }
} // namespace axispose
#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(axispose::SegmentNode)
