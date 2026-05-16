#include "axispose/joint_semantic_depth_preprocessor.hpp"

#include <opencv2/imgproc.hpp>

namespace axispose
{
    inline double JointSemanticDepthPreprocessor::readDepthMeters(const cv::Mat &depth, int v, int u)
    {
        if (depth.empty() || v < 0 || v >= depth.rows || u < 0 || u >= depth.cols)
            return -1.0;
        if (depth.type() == CV_16U)
        {
            uint16_t d = depth.at<uint16_t>(v, u);
            if (d == 0)
                return -1.0;
            return static_cast<double>(d) * 0.001; // mm -> m
        }
        else if (depth.type() == CV_32F)
        {
            float d = depth.at<float>(v, u);
            if (!(d > 0.0f))
                return -1.0;
            return static_cast<double>(d);
        }
        return -1.0;
    }

    cv::Mat JointSemanticDepthPreprocessor::applyAdaptiveThreshold(const cv::Mat &confidence_map, const cv::Mat &depth_map, float near_thresh, float far_thresh) const
    {
        CV_Assert(!confidence_map.empty());
        CV_Assert(confidence_map.type() == CV_32F);

        cv::Mat out = cv::Mat::zeros(confidence_map.size(), CV_8U);

        // compute depth min/max for valid pixels
        double zmin = std::numeric_limits<double>::infinity();
        double zmax = 0.0;
        for (int v = 0; v < depth_map.rows; ++v)
        {
            const uchar *dummy = nullptr;
            for (int u = 0; u < depth_map.cols; ++u)
            {
                double z = readDepthMeters(depth_map, v, u);
                if (z > 0.0 && std::isfinite(z))
                {
                    zmin = std::min(zmin, z);
                    zmax = std::max(zmax, z);
                }
            }
        }
        if (!std::isfinite(zmin) || zmax <= zmin)
        {
            // fallback to global threshold
            for (int v = 0; v < confidence_map.rows; ++v)
            {
                const float *cptr = confidence_map.ptr<float>(v);
                uchar *outptr = out.ptr<uchar>(v);
                for (int u = 0; u < confidence_map.cols; ++u)
                {
                    outptr[u] = (cptr[u] >= near_thresh) ? 255 : 0;
                }
            }
            return out;
        }

        // per-pixel threshold by linear interpolation on Z
        for (int v = 0; v < confidence_map.rows; ++v)
        {
            const float *cptr = confidence_map.ptr<float>(v);
            uchar *outptr = out.ptr<uchar>(v);
            for (int u = 0; u < confidence_map.cols; ++u)
            {
                double z = readDepthMeters(depth_map, v, u);
                float thresh = near_thresh;
                if (z > 0.0 && std::isfinite(z))
                {
                    double t = (z - zmin) / (zmax - zmin);
                    if (t < 0.0)
                        t = 0.0;
                    if (t > 1.0)
                        t = 1.0;
                    // 阈值线性插值：近处严格（near_thresh），远处放宽（far_thresh）
                    thresh = static_cast<float>(near_thresh * (1.0 - t) + far_thresh * t);
                }
                outptr[u] = (cptr[u] >= thresh) ? 255 : 0;
            }
        }
        return out;
    }

    cv::Mat JointSemanticDepthPreprocessor::dilateMaskAdaptive(const cv::Mat &boundary_mask, const cv::Mat &depth_map, float base_kernel_size, float alpha) const
    {
        CV_Assert(!boundary_mask.empty());
        CV_Assert(boundary_mask.type() == CV_8U);

        // Compute Zmin for mapping
        double zmin = std::numeric_limits<double>::infinity();
        double zmax = 0.0;
        for (int v = 0; v < depth_map.rows; ++v)
        {
            for (int u = 0; u < depth_map.cols; ++u)
            {
                double z = readDepthMeters(depth_map, v, u);
                if (z > 0.0 && std::isfinite(z))
                {
                    zmin = std::min(zmin, z);
                    zmax = std::max(zmax, z);
                }
            }
        }
        if (!std::isfinite(zmin))
            zmin = 0.0;

        // Quantize kernel sizes into small odd integers to avoid per-pixel morphology
        // Compute per-pixel desired kernel radius, then bucket into {0,1,2,3} -> kernel sizes {1,3,5,7}
        const std::vector<int> kernels = {1, 3, 5, 7};
        std::vector<cv::Mat> masks_per_kernel(kernels.size(), cv::Mat::zeros(boundary_mask.size(), CV_8U));

        for (int v = 0; v < boundary_mask.rows; ++v)
        {
            const uchar *bptr = boundary_mask.ptr<uchar>(v);
            for (int u = 0; u < boundary_mask.cols; ++u)
            {
                if (bptr[u] == 0)
                    continue;
                double z = readDepthMeters(depth_map, v, u);
                double dz = (z > 0.0 && std::isfinite(z)) ? (z - zmin) : 0.0;
                double kf = base_kernel_size + alpha * dz; // kernel radius in pixels
                if (kf < 0.0)
                    kf = 0.0;
                // map kernel radius to index
                int idx = 0;
                if (kf <= 1.0)
                    idx = 0;
                else if (kf <= 2.0)
                    idx = 1;
                else if (kf <= 3.0)
                    idx = 2;
                else
                    idx = 3;
                masks_per_kernel[idx].at<uchar>(v, u) = 255;
            }
        }

        // Apply dilation for each kernel and combine
        cv::Mat out = cv::Mat::zeros(boundary_mask.size(), CV_8U);
        for (size_t i = 0; i < kernels.size(); ++i)
        {
            int k = kernels[i];
            if (k <= 1)
            {
                cv::bitwise_or(out, masks_per_kernel[i], out);
                continue;
            }
            cv::Mat elem = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
            cv::Mat dilated;
            cv::dilate(masks_per_kernel[i], dilated, elem);
            cv::bitwise_or(out, dilated, out);
        }
        return out;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr JointSemanticDepthPreprocessor::cvMaskToPclCloud(const cv::Mat &depth, const cv::Mat &mask, double fx, double fy, double cx, double cy) const
    {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        if (depth.empty() || mask.empty())
            return cloud;

        CV_Assert(depth.size() == mask.size());

        bool depth_is_float = (depth.type() == CV_32F);
        double unit_scaling = depth_is_float ? 1.0 : 0.001; // if uint16 mm, convert to meters

        for (int v = 0; v < depth.rows; ++v)
        {
            for (int u = 0; u < depth.cols; ++u)
            {
                if (mask.at<uchar>(v, u) == 0)
                    continue;
                double depth_val = readDepthMeters(depth, v, u);
                if (!(depth_val > 0.0))
                    continue;
                double Z = depth_val;
                double X = (static_cast<double>(u) - cx) * Z / fx;
                double Y = (static_cast<double>(v) - cy) * Z / fy;
                pcl::PointXYZ p;
                // follow existing convention: pt.x=z, pt.y=-x, pt.z=-y
                p.x = static_cast<float>(Z);
                p.y = static_cast<float>(-X);
                p.z = static_cast<float>(-Y);
                cloud->points.push_back(p);
            }
        }
        cloud->width = static_cast<uint32_t>(cloud->points.size());
        cloud->height = 1;
        return cloud;
    }

} // namespace axispose
