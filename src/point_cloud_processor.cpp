#include "axispose/point_cloud_processor.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <limits>

namespace axispose
{

    PointCloudProcessor::PointCloudProcessor() {}

    pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudProcessor::depthMaskToPointCloud(const cv::Mat &depth, const cv::Mat &camera_matrix) const
    {
        const double fx = camera_matrix.at<double>(0, 0);
        const double fy = camera_matrix.at<double>(1, 1);
        const double cx = camera_matrix.at<double>(0, 2);
        const double cy = camera_matrix.at<double>(1, 2);

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
                                                const PointCloudDenoiseOptions &options) const
    {
        if (!cloud || cloud->empty())
            return;

        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>());

        if (options.voxel_leaf_size > 0)
        {
            pcl::VoxelGrid<pcl::PointXYZ> vg;
            vg.setInputCloud(cloud);
            vg.setLeafSize(static_cast<float>(options.voxel_leaf_size),
                           static_cast<float>(options.voxel_leaf_size),
                           static_cast<float>(options.voxel_leaf_size));
            vg.filter(*tmp);
            cloud.swap(tmp);
            tmp->clear();
        }

        if (options.use_sor)
        {
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(cloud);
            sor.setMeanK(options.sor_mean_k);
            sor.setStddevMulThresh(options.sor_std_mul);
            sor.filter(*tmp);
            cloud.swap(tmp);
            tmp->clear();
        }

        if (!options.use_euclidean_cluster || cloud->empty())
        {
            return;
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.5f);
        ec.setMinClusterSize(50);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        if (cluster_indices.empty())
        {
            return;
        }

        size_t best_idx = 0;
        pcl::PointIndices::Ptr best_cluster_inliers(new pcl::PointIndices);
        bool handled_by_mode0 = false;

        switch (options.cluster_mode)
        {
        case 0:
        {
            // New behavior: select the largest cluster as the mixed candidate,
            // then perform Z-axis bin splitting and finally SACLine refinement.
            size_t max_size = 0;
            for (size_t i = 0; i < cluster_indices.size(); ++i)
            {
                const size_t sz = cluster_indices[i].indices.size();
                if (sz > max_size)
                {
                    max_size = sz;
                    best_idx = i;
                }
            }

            // Build a point cloud containing only the largest cluster
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto idx : cluster_indices[best_idx].indices)
                cluster_cloud->points.push_back(cloud->points[idx]);
            cluster_cloud->width = static_cast<uint32_t>(cluster_cloud->points.size());
            cluster_cloud->height = 1;

            // Determine which axis to bin on
            int axis = 2; // default z
            if (options.bin_axis == "x")
                axis = 0;
            else if (options.bin_axis == "y")
                axis = 1;

            // Build histogram
            double coord_min = std::numeric_limits<double>::infinity();
            double coord_max = -std::numeric_limits<double>::infinity();
            for (const auto &pt : cluster_cloud->points)
            {
                double v = (axis == 0) ? pt.x : ((axis == 1) ? pt.y : pt.z);
                if (std::isfinite(v))
                {
                    coord_min = std::min(coord_min, v);
                    coord_max = std::max(coord_max, v);
                }
            }
            if (!std::isfinite(coord_min) || !std::isfinite(coord_max) || coord_max <= coord_min)
            {
                // fallback to largest cluster as-is
                break;
            }

            const double bin_w = options.bin_width;
            int nbins = static_cast<int>(std::ceil((coord_max - coord_min) / bin_w));
            if (nbins < 1)
                nbins = 1;
            std::vector<int> hist(nbins, 0);
            std::vector<double> bin_centers(nbins, 0.0);
            for (int i = 0; i < nbins; ++i)
                bin_centers[i] = coord_min + (i + 0.5) * bin_w;

            for (const auto &pt : cluster_cloud->points)
            {
                double v = (axis == 0) ? pt.x : ((axis == 1) ? pt.y : pt.z);
                if (!std::isfinite(v))
                    continue;
                int idx = static_cast<int>(std::floor((v - coord_min) / bin_w));
                if (idx < 0)
                    idx = 0;
                if (idx >= nbins)
                    idx = nbins - 1;
                hist[idx]++;
            }

            // Smooth histogram with 3-bin moving average to reduce noise
            std::vector<int> hist_smooth(nbins, 0);
            for (int i = 0; i < nbins; ++i)
            {
                int c = 0;
                for (int k = -1; k <= 1; ++k)
                {
                    int j = i + k;
                    if (j >= 0 && j < nbins)
                    {
                        hist_smooth[i] += hist[j];
                        c++;
                    }
                }
                if (c > 0)
                    hist_smooth[i] = hist_smooth[i] / c;
            }

            // Find peak bin
            int peak_idx = 0;
            int peak_val = hist_smooth[0];
            for (int i = 1; i < nbins; ++i)
            {
                if (hist_smooth[i] > peak_val)
                {
                    peak_val = hist_smooth[i];
                    peak_idx = i;
                }
            }

            // Expand around peak while bins >= peak * bin_peak_ratio
            const double ratio = std::max(0.0, std::min(1.0, options.bin_peak_ratio));
            int left = peak_idx;
            int right = peak_idx;
            while (left - 1 >= 0 && hist_smooth[left - 1] >= static_cast<int>(peak_val * ratio))
                --left;
            while (right + 1 < nbins && hist_smooth[right + 1] >= static_cast<int>(peak_val * ratio))
                ++right;

            // Ensure minimum contiguous bins
            int span = right - left + 1;
            if (span < options.bin_min_bins)
            {
                int need = options.bin_min_bins - span;
                int expand_left = need / 2 + (need % 2);
                int expand_right = need / 2;
                left = std::max(0, left - expand_left);
                right = std::min(nbins - 1, right + expand_right);
            }

            // Collect points inside selected bin band
            pcl::PointCloud<pcl::PointXYZ>::Ptr band_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto &pt : cluster_cloud->points)
            {
                double v = (axis == 0) ? pt.x : ((axis == 1) ? pt.y : pt.z);
                if (!std::isfinite(v))
                    continue;
                int idx = static_cast<int>(std::floor((v - coord_min) / bin_w));
                if (idx < 0)
                    idx = 0;
                if (idx >= nbins)
                    idx = nbins - 1;
                if (idx >= left && idx <= right)
                    band_cloud->points.push_back(pt);
            }
            band_cloud->width = static_cast<uint32_t>(band_cloud->points.size());
            band_cloud->height = 1;

            if (band_cloud->empty())
            {
                // fallback to largest cluster
                break;
            }

            cloud.swap(band_cloud);
            handled_by_mode0 = true;
            break;

            // Final refinement: run SACLine on band_cloud
            pcl::PointIndices inliers;
            pcl::ModelCoefficients coeffs;
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_LINE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setMaxIterations(200);
            seg.setDistanceThreshold(options.sacline_distance_threshold);
            seg.setInputCloud(band_cloud);
            seg.segment(inliers, coeffs);

            if (!inliers.indices.empty())
            {
                pcl::ExtractIndices<pcl::PointXYZ> extract2;
                extract2.setInputCloud(band_cloud);
                pcl::PointIndices::Ptr inliers_ptr(new pcl::PointIndices(inliers));
                extract2.setIndices(inliers_ptr);
                extract2.setNegative(false);
                extract2.filter(*tmp);
                cloud.swap(tmp);
                tmp->clear();
                handled_by_mode0 = true;
            }
            else
            {
                // If SAC failed, use band_cloud as the filtered result
                cloud.swap(band_cloud);
                handled_by_mode0 = true;
            }
        }
        break;
        case 3:
        {
            // Raw mode: do not perform any clustering-based trimming; keep the cloud as-is
            handled_by_mode0 = true;
        }
        break;
        case 1:
        {
            size_t max_size = 0;
            for (size_t i = 0; i < cluster_indices.size(); ++i)
            {
                const size_t sz = cluster_indices[i].indices.size();
                if (sz > max_size)
                {
                    max_size = sz;
                    best_idx = i;
                }
            }
        }
        break;
        case 2:
        default:
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            size_t best_inlier_count = 0;

            for (size_t i = 0; i < cluster_indices.size(); ++i)
            {
                const auto &cluster = cluster_indices[i];
                cluster_cloud->points.clear();
                for (const auto idx : cluster.indices)
                {
                    cluster_cloud->points.push_back(cloud->points[idx]);
                }
                cluster_cloud->width = static_cast<uint32_t>(cluster_cloud->points.size());
                cluster_cloud->height = 1;

                pcl::ModelCoefficients coeffs;
                pcl::PointIndices inliers;
                pcl::SACSegmentation<pcl::PointXYZ> seg;
                seg.setOptimizeCoefficients(true);
                seg.setModelType(pcl::SACMODEL_LINE);
                seg.setMethodType(pcl::SAC_RANSAC);
                seg.setMaxIterations(200);
                seg.setDistanceThreshold(options.sacline_distance_threshold);
                seg.setInputCloud(cluster_cloud);
                seg.segment(inliers, coeffs);

                if (!inliers.indices.empty() && inliers.indices.size() > best_inlier_count)
                {
                    best_inlier_count = inliers.indices.size();
                    best_idx = i;
                    best_cluster_inliers->indices = inliers.indices;
                }
            }
        }
        break;
        }

        if (!handled_by_mode0)
        {
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(cloud);
            pcl::PointIndices::Ptr cluster_indices_ptr(new pcl::PointIndices(cluster_indices[best_idx]));
            extract.setIndices(cluster_indices_ptr);
            extract.setNegative(false);
            extract.filter(*tmp);
            cloud.swap(tmp);
            tmp->clear();

            if (!best_cluster_inliers->indices.empty())
            {
                extract.setInputCloud(cloud);
                extract.setIndices(best_cluster_inliers);
                extract.setNegative(false);
                extract.filter(*tmp);
                cloud.swap(tmp);
                tmp->clear();
            }
        }
    }

} // namespace axispose
