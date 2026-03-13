#ifndef AXISPOSE_POINT_CLOUD_PROCESSOR_HPP_
#define AXISPOSE_POINT_CLOUD_PROCESSOR_HPP_

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace axispose
{

    class PointCloudProcessor
    {
    public:
        PointCloudProcessor();

        // Convert organized depth (CV_16U mm or CV_32F meters) to organized pcl cloud using intrinsics
        pcl::PointCloud<pcl::PointXYZ>::Ptr depthMaskToPointCloud(const cv::Mat &depth, double fx, double fy, double cx, double cy) const;

        // Simple denoise: voxel grid + statistical outlier removal
        void denoisePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                               double voxel_leaf_size,
                               bool use_sor,
                               int sor_mean_k,
                               double sor_std_mul) const;
    };

} // namespace axispose

#endif // AXISPOSE_POINT_CLOUD_PROCESSOR_HPP_
