#ifndef AXISPOSE_GAUSSIAN_MAP_SOLVER_HPP
#define AXISPOSE_GAUSSIAN_MAP_SOLVER_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>

namespace axispose
{

    class GaussianMapSolver
    {
    public:
        GaussianMapSolver();
        ~GaussianMapSolver();

        /**
         * @brief Estimate the axis, center point, and radius of a cylinder given a point cloud.
         *
         * @param cloud_in Input point cloud.
         * @param out_axis Output axis direction vector.
         * @param out_point Output point on the axis.
         * @param out_radius Output radius of the cylinder.
         * @return true if successful, false otherwise.
         */
        bool estimateAxis(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in,
                          Eigen::Vector3f &out_axis,
                          Eigen::Vector3f &out_point,
                          float &out_radius);
    };

} // namespace axispose

#endif // AXISPOSE_GAUSSIAN_MAP_SOLVER_HPP
