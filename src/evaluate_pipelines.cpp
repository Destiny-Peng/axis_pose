#include "axispose/gaussian_map_solver.hpp"
#include "axispose/ceres_joint_optimizer.hpp"
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <Eigen/Eigenvalues>

using namespace axispose;

int main()
{
    std::cout << "--- Evaluating Pipelines ---" << std::endl;

    // Ground truth parameters
    Eigen::Vector3d gt_axis(0.0, 1.0, 0.0);
    gt_axis.normalize();
    Eigen::Vector3d gt_point(0.0, 0.0, 1.0);
    double gt_radius = 0.05;

    // Generate dummy point cloud for a cylinder with noise
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    std::mt19937 gen(42);
    std::uniform_real_distribution<> height_dist(-0.2, 0.2);
    std::uniform_real_distribution<> angle_dist(0.0, M_PI); // Visible half
    std::normal_distribution<> noise_dist(0.0, 0.005);      // 5mm noise

    for (int i = 0; i < 1000; ++i)
    {
        double h = height_dist(gen);
        double theta = angle_dist(gen);

        Eigen::Vector3d p;
        p.x() = gt_point.x() + gt_radius * std::cos(theta); // assuming axis is roughly Y
        p.y() = gt_point.y() + h;
        p.z() = gt_point.z() - gt_radius * std::sin(theta); // visible side

        // Add noise
        p.x() += noise_dist(gen);
        p.y() += noise_dist(gen);
        p.z() += noise_dist(gen);

        cloud->points.emplace_back(p.x(), p.y(), p.z());
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    std::cout << "Generated cloud with " << cloud->size() << " points." << std::endl;

    // Baseline: PCA
    Eigen::Vector4f pca_centroid;
    pcl::compute3DCentroid(*cloud, pca_centroid);
    Eigen::Matrix3f covariance_matrix;
    pcl::computeCovarianceMatrixNormalized(*cloud, pca_centroid, covariance_matrix);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
    Eigen::Vector3d pca_axis = eigen_solver.eigenvectors().col(2).cast<double>();
    if (pca_axis.z() > 0)
        pca_axis = -pca_axis; // Normalize direction
    Eigen::Vector3d pca_point = pca_centroid.head<3>().cast<double>();

    // New Pipeline: GaussianMapSolver + CeresJointOptimizer
    GaussianMapSolver gms;
    Eigen::Vector3f gms_axis, gms_point;
    float gms_radius = 0.0f;
    bool gms_success = gms.estimateAxis(cloud, gms_axis, gms_point, gms_radius);

    // Setup dummy 2D line data for Ceres
    Eigen::Matrix3d K;
    K << 600.0, 0.0, 320.0,
        0.0, 600.0, 240.0,
        0.0, 0.0, 1.0;

    // x = 320 as synthetic fitted centerline => 1*u + 0*v - 320 = 0
    Eigen::Vector3d line2d_abc(1.0, 0.0, -320.0);

    Eigen::Vector3d opt_axis = gms_axis.cast<double>().normalized();
    Eigen::Vector3d opt_point = gms_point.cast<double>();
    Eigen::Vector3d opt_m = opt_point.cross(opt_axis);

    CeresJointOptimizer cjo;
    bool cjo_success = false;
    if (gms_success)
    {
        cjo_success = cjo.optimizePose(opt_axis, opt_m, cloud, K, line2d_abc, opt_point, 12, 120, 3.0, 1.0);
        opt_point = opt_axis.cross(opt_m);
    }

    // Reports
    auto get_angle_error = [](const Eigen::Vector3d &a, const Eigen::Vector3d &b)
    {
        double dot = std::abs(a.dot(b));
        if (dot > 1.0)
            dot = 1.0;
        return std::acos(dot) * 180.0 / M_PI;
    };

    auto get_dist_error = [](const Eigen::Vector3d &point, const Eigen::Vector3d &axis, const Eigen::Vector3d &gt_point)
    {
        Eigen::Vector3d v = gt_point - point;
        return v.cross(axis).norm(); // Perpendicular distance
    };

    std::cout << "\n--- Ground Truth ---" << std::endl;
    std::cout << "Axis:   " << gt_axis.transpose() << std::endl;
    std::cout << "Point:  " << gt_point.transpose() << std::endl;
    std::cout << "Radius: " << gt_radius << std::endl;

    std::cout << "\n--- Baseline (PCA) ---" << std::endl;
    std::cout << "Axis:   " << pca_axis.transpose() << std::endl;
    std::cout << "Point:  " << pca_point.transpose() << std::endl;
    std::cout << "Axis Angular Error: " << get_angle_error(pca_axis, gt_axis) << " deg" << std::endl;
    std::cout << "Point Distance Error: " << get_dist_error(pca_point, pca_axis, gt_point) << " m" << std::endl;

    std::cout << "\n--- New Pipeline (GMS + Ceres) ---" << std::endl;
    if (!gms_success)
    {
        std::cout << "GMS Failed!" << std::endl;
    }
    else
    {
        std::cout << "Sub-Step: GMS Output" << std::endl;
        std::cout << "  Axis:   " << gms_axis.transpose() << std::endl;
        std::cout << "  Point:  " << gms_point.transpose() << std::endl;
        std::cout << "  Radius: " << gms_radius << std::endl;

        std::cout << "Final Output (Ceres Optimized)" << std::endl;
        std::cout << "  Axis:   " << opt_axis.transpose() << std::endl;
        std::cout << "  Point:  " << opt_point.transpose() << std::endl;
        std::cout << "Axis Angular Error: " << get_angle_error(opt_axis, gt_axis) << " deg" << std::endl;
        std::cout << "Point Distance Error: " << get_dist_error(opt_point, opt_axis, gt_point) << " m" << std::endl;
    }

    return 0;
}
