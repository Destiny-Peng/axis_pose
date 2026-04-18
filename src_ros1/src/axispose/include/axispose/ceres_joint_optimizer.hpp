#ifndef AXISPOSE_CERES_JOINT_OPTIMIZER_HPP
#define AXISPOSE_CERES_JOINT_OPTIMIZER_HPP

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace axispose
{

    class CeresJointOptimizer
    {
    public:
        CeresJointOptimizer();
        ~CeresJointOptimizer();

        bool optimizePose(Eigen::Vector3d &d,
                          Eigen::Vector3d &m,
                          const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                          const Eigen::Matrix3d &K,
                          const Eigen::Vector3d &line2d_abc,
                          const Eigen::Vector3d &point_prior,
                          int max_iterations = 15,
                          int max_points = 120,
                          double weight_2d = 4.0,
                          double weight_pos = 1.0);
    };

} // namespace axispose

#endif // AXISPOSE_CERES_JOINT_OPTIMIZER_HPP
