#ifndef AXISPOSE_DEPTH_ALIGNER_HPP_
#define AXISPOSE_DEPTH_ALIGNER_HPP_

#include <opencv2/opencv.hpp>

namespace axispose
{

    class DepthAligner
    {
    public:
        DepthAligner();

        // align depth image (CV_16U mm or CV_32F meters) to color size; returns CV_16U in mm
        cv::Mat align(const cv::Mat &depth,
                      const cv::Mat &depth_camera_matrix,
                      const cv::Mat &color_camera_matrix,
                      int color_width,
                      int color_height) const;
    };

} // namespace axispose

#endif // AXISPOSE_DEPTH_ALIGNER_HPP_
