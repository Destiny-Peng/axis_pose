#include "axispose/depth_aligner.hpp"

#include <cmath>

namespace axispose
{

    DepthAligner::DepthAligner() {}

    cv::Mat DepthAligner::align(const cv::Mat &depth,
                                const cv::Mat &depth_camera_matrix,
                                const cv::Mat &color_camera_matrix,
                                int color_width,
                                int color_height) const
    {
        const double d_fx = depth_camera_matrix.at<double>(0, 0);
        const double d_fy = depth_camera_matrix.at<double>(1, 1);
        const double d_cx = depth_camera_matrix.at<double>(0, 2);
        const double d_cy = depth_camera_matrix.at<double>(1, 2);
        const double c_fx = color_camera_matrix.at<double>(0, 0);
        const double c_fy = color_camera_matrix.at<double>(1, 1);
        const double c_cx = color_camera_matrix.at<double>(0, 2);
        const double c_cy = color_camera_matrix.at<double>(1, 2);

        cv::Mat aligned = cv::Mat::zeros(color_height, color_width, CV_16U);

        const bool depth_is_float = (depth.type() == CV_32F);
        const int d_rows = depth.rows;
        const int d_cols = depth.cols;

        for (int v = 0; v < d_rows; ++v)
        {
            for (int u = 0; u < d_cols; ++u)
            {
                double depth_val_m = 0.0;
                if (!depth_is_float)
                {
                    const uint16_t d = depth.at<uint16_t>(v, u);
                    if (d == 0 || d > 10000)
                        continue;
                    depth_val_m = static_cast<double>(d) * 0.001;
                }
                else
                {
                    const float d = depth.at<float>(v, u);
                    if (!(d > 0.0f))
                        continue;
                    depth_val_m = static_cast<double>(d);
                }

                const double X = (static_cast<double>(u) - d_cx) * depth_val_m / d_fx;
                const double Y = (static_cast<double>(v) - d_cy) * depth_val_m / d_fy;
                const double Z = depth_val_m;
                if (Z <= 0.0)
                    continue;

                const int u_c = static_cast<int>(std::round((X * c_fx / Z) + c_cx));
                const int v_c = static_cast<int>(std::round((Y * c_fy / Z) + c_cy));

                if (u_c < 0 || u_c >= color_width || v_c < 0 || v_c >= color_height)
                    continue;

                const uint16_t d_mm = static_cast<uint16_t>(std::round(Z * 1000.0));
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
