#include "axispose/point_cloud_processor.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace axispose
{

    PointCloudProcessor::PointCloudProcessor() {}

    pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudProcessor::depthMaskToPointCloud(const cv::Mat &depth, double fx, double fy, double cx, double cy) const
    {
        const int rows = depth.rows;
        const int cols = depth.cols;
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        cloud->is_dense = false;
        cloud->width = static_cast<uint32_t>(cols);
        cloud->height = static_cast<uint32_t>(rows);
        cloud->points.resize(cloud->width * cloud->height);

        double unit_scaling = 0.001; // assume uint16 in mm -> meters
        bool depth_is_float = (depth.type() == CV_32F);
        if (depth_is_float)
            unit_scaling = 1.0;

        float constant_x = static_cast<float>(unit_scaling / fx);
        float constant_y = static_cast<float>(unit_scaling / fy);
        float center_x = static_cast<float>(cx);
        float center_y = static_cast<float>(cy);
        float bad_point = std::numeric_limits<float>::quiet_NaN();

        for (int v = 0; v < rows; ++v)
        {
            for (int u = 0; u < cols; ++u)
            {
                size_t idx = static_cast<size_t>(v) * cols + u;
                pcl::PointXYZ &pt = cloud->points[idx];
                pt.x = bad_point;
                pt.y = bad_point;
                pt.z = bad_point;

                double depth_val = 0.0;
                if (!depth_is_float)
                {
                    uint16_t d = depth.at<uint16_t>(v, u);
                    if (d == 0)
                        continue;
                    depth_val = static_cast<double>(d);
                }
                else
                {
                    float d = depth.at<float>(v, u);
                    if (!(d > 0.0f))
                        continue;
                    depth_val = static_cast<double>(d);
                }

                float z = static_cast<float>(depth_val * unit_scaling);
                float x = (static_cast<float>(u) - center_x) * static_cast<float>(depth_val) * constant_x;
                float y = (static_cast<float>(v) - center_y) * static_cast<float>(depth_val) * constant_y;

                pt.x = z;
                pt.y = -x;
                pt.z = -y;
            }
        }

        return cloud;
    }

    void PointCloudProcessor::denoisePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                double voxel_leaf_size,
                                                bool use_sor,
                                                int sor_mean_k,
                                                double sor_std_mul) const
    {
        if (!cloud || cloud->empty())
            return;

        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>());

        if (voxel_leaf_size > 0)
        {
            pcl::VoxelGrid<pcl::PointXYZ> vg;
            vg.setInputCloud(cloud);
            vg.setLeafSize(static_cast<float>(voxel_leaf_size), static_cast<float>(voxel_leaf_size), static_cast<float>(voxel_leaf_size));
            vg.filter(*tmp);
            cloud.swap(tmp);
            tmp->clear();
        }

        if (use_sor)
        {
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(cloud);
            sor.setMeanK(sor_mean_k);
            sor.setStddevMulThresh(sor_std_mul);
            sor.filter(*tmp);
            cloud.swap(tmp);
            tmp->clear();
        }
    }

} // namespace axispose
