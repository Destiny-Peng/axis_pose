#include "axispose/gaussian_map_solver.hpp"

#include <algorithm>
#include <vector>

#include <Eigen/Dense>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>

namespace axispose
{

    GaussianMapSolver::GaussianMapSolver() {}
    GaussianMapSolver::~GaussianMapSolver() {}

    bool GaussianMapSolver::estimateAxis(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in,
                                         Eigen::Vector3f &out_axis,
                                         Eigen::Vector3f &out_point,
                                         float &out_radius)
    {
        if (!cloud_in || cloud_in->empty())
            return false;

        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

        if (cloud_in->isOrganized())
        {
            pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
            ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
            ne.setMaxDepthChangeFactor(0.02f);
            ne.setNormalSmoothingSize(10.0f);
            ne.setInputCloud(cloud_in);
            ne.compute(*normals);
        }
        else
        {
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
            ne.setInputCloud(cloud_in);
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
            ne.setSearchMethod(tree);
            ne.setKSearch(30);
            ne.compute(*normals);
        }

        std::vector<Eigen::Vector3f> valid_ns;
        valid_ns.reserve(normals->size());
        for (const auto &nm : normals->points)
        {
            if (!std::isfinite(nm.normal_x) || !std::isfinite(nm.normal_y) || !std::isfinite(nm.normal_z))
                continue;
            Eigen::Vector3f n(nm.normal_x, nm.normal_y, nm.normal_z);
            const float nn = n.norm();
            if (nn < 1e-6f)
                continue;
            valid_ns.push_back(n / nn);
        }
        if (valid_ns.size() < 6)
            return false;

        auto solve_axis = [](const std::vector<Eigen::Vector3f> &ns)
        {
            Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
            for (const auto &n : ns)
            {
                cov.noalias() += n * n.transpose();
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
            return eig.eigenvectors().col(0).normalized();
        };

        Eigen::Vector3f axis0 = solve_axis(valid_ns);

        constexpr float kOrthoDotMax = 0.25881905f;
        std::vector<Eigen::Vector3f> refined_ns;
        refined_ns.reserve(valid_ns.size());
        for (const auto &n : valid_ns)
        {
            if (std::abs(n.dot(axis0)) <= kOrthoDotMax)
                refined_ns.push_back(n);
        }

        if (refined_ns.size() >= 6)
            out_axis = solve_axis(refined_ns);
        else
            out_axis = axis0;

        Eigen::Vector3f center_of_mass(0, 0, 0);
        for (const auto &pt : cloud_in->points)
        {
            center_of_mass += Eigen::Vector3f(pt.x, pt.y, pt.z);
        }
        center_of_mass /= static_cast<float>(cloud_in->points.size());
        out_point = center_of_mass;

        std::vector<float> radial_dists;
        radial_dists.reserve(cloud_in->size());
        for (const auto &pt : cloud_in->points)
        {
            const Eigen::Vector3f p(pt.x, pt.y, pt.z);
            const Eigen::Vector3f v = p - out_point;
            const Eigen::Vector3f perp = v - v.dot(out_axis) * out_axis;
            const float r = perp.norm();
            if (std::isfinite(r))
                radial_dists.push_back(r);
        }
        if (radial_dists.empty())
            return false;

        std::nth_element(radial_dists.begin(), radial_dists.begin() + radial_dists.size() / 2, radial_dists.end());
        out_radius = radial_dists[radial_dists.size() / 2];
        if (!(out_radius > 1e-4f))
            out_radius = 0.05f;
        return true;
    }

} // namespace axispose
