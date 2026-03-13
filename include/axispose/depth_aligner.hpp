#ifndef AXISPOSE_DEPTH_ALIGNER_HPP_
#define AXISPOSE_DEPTH_ALIGNER_HPP_

#include <opencv2/opencv.hpp>

namespace axispose
{

    class DepthAligner
    {
    public:
        DepthAligner();

        void setDepthIntrinsics(double fx, double fy, double cx, double cy);
        void setColorIntrinsics(double fx, double fy, double cx, double cy);

        // align depth image (CV_16U mm or CV_32F meters) to color size; returns CV_16U in mm
        cv::Mat align(const cv::Mat &depth, int color_width, int color_height) const;

    private:
        double d_fx_{0.0}, d_fy_{0.0}, d_cx_{0.0}, d_cy_{0.0};
        double c_fx_{0.0}, c_fy_{0.0}, c_cx_{0.0}, c_cy_{0.0};
    };

} // namespace axispose

#endif // AXISPOSE_DEPTH_ALIGNER_HPP_
