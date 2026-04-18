#ifndef AXISPOSE_GAUSSIAN_MAP_SOLVER_HPP
#define AXISPOSE_GAUSSIAN_MAP_SOLVER_HPP

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace axispose
{

    class GaussianMapSolver
    {
    public:
        GaussianMapSolver();
        ~GaussianMapSolver();

        bool estimateAxis(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in,
                          Eigen::Vector3f &out_axis,
                          Eigen::Vector3f &out_point,
                          float &out_radius);
    };

} // namespace axispose

#endif // AXISPOSE_GAUSSIAN_MAP_SOLVER_HPP
