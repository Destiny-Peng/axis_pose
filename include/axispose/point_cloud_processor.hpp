#ifndef AXISPOSE_POINT_CLOUD_PROCESSOR_HPP_
#define AXISPOSE_POINT_CLOUD_PROCESSOR_HPP_

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace axispose
{

    struct PointCloudDenoiseOptions
    {
        double voxel_leaf_size{0.05};
        bool use_sor{true};
        int sor_mean_k{50};
        double sor_std_mul{1.0};
        bool use_euclidean_cluster{true};
        int cluster_mode{0}; // 0: closest to origin, 1: largest cluster, 2: max RANSAC inliers
        double sacline_distance_threshold{0.05};
    };

    class PointCloudProcessor
    {
    public:
        PointCloudProcessor();

        // Convert organized depth (CV_16U mm or CV_32F meters) to organized pcl cloud using K matrix.
        pcl::PointCloud<pcl::PointXYZ>::Ptr depthMaskToPointCloud(const cv::Mat &depth, const cv::Mat &camera_matrix) const;

        // Full denoise pipeline: voxel + SOR + optional cluster filtering.
        void denoisePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                               const PointCloudDenoiseOptions &options) const;
    };

} // namespace axispose

#endif // AXISPOSE_POINT_CLOUD_PROCESSOR_HPP_
