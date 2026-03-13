#include "axispose/depth_aligner.hpp"

#include <cmath>

namespace axispose
{

    DepthAligner::DepthAligner() {}

    void DepthAligner::setDepthIntrinsics(double fx, double fy, double cx, double cy)
    {
        d_fx_ = fx;
        d_fy_ = fy;
        d_cx_ = cx;
        d_cy_ = cy;
    }

    void DepthAligner::setColorIntrinsics(double fx, double fy, double cx, double cy)
    {
        c_fx_ = fx;
        c_fy_ = fy;
        c_cx_ = cx;
        c_cy_ = cy;
    }

    cv::Mat DepthAligner::align(const cv::Mat &depth, int color_width, int color_height) const
    {
        cv::Mat aligned = cv::Mat::zeros(color_height, color_width, CV_16U);

        bool depth_is_float = (depth.type() == CV_32F);
        const int d_rows = depth.rows;
        const int d_cols = depth.cols;

        for (int v = 0; v < d_rows; ++v)
        {
            for (int u = 0; u < d_cols; ++u)
            {
                double depth_val_m = 0.0;
                if (!depth_is_float)
                {
                    uint16_t d = depth.at<uint16_t>(v, u);
                    if (d == 0 || d > 10000)
                        continue;
                    depth_val_m = static_cast<double>(d) * 0.001; // mm -> m
                }
                else
                {
                    float d = depth.at<float>(v, u);
                    if (!(d > 0.0f))
                        continue;
                    depth_val_m = static_cast<double>(d);
                }

                // Back-project to 3D in depth camera frame
                double X = (static_cast<double>(u) - d_cx_) * depth_val_m / d_fx_;
                double Y = (static_cast<double>(v) - d_cy_) * depth_val_m / d_fy_;
                double Z = depth_val_m;

                if (Z <= 0.0)
                    continue;
                int u_c = static_cast<int>(std::round((X * c_fx_ / Z) + c_cx_));
                int v_c = static_cast<int>(std::round((Y * c_fy_ / Z) + c_cy_));

                if (u_c < 0 || u_c >= color_width || v_c < 0 || v_c >= color_height)
                    continue;

                uint16_t d_mm = static_cast<uint16_t>(std::round(Z * 1000.0));
                uint16_t &cell = aligned.at<uint16_t>(v_c, u_c);
                if (cell == 0 || d_mm < cell)
                {
                    cell = d_mm;
                }
            }
        }

        return aligned;
    }

} // namespace axispose
