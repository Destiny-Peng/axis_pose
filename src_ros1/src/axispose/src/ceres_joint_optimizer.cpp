#include "axispose/ceres_joint_optimizer.hpp"

#ifdef AXISPOSE_HAS_CERES
#include <algorithm>
#include <thread>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace axispose
{

    struct PointToLineResidual
    {
        explicit PointToLineResidual(const Eigen::Vector3d &point) : pt(point) {}

        template <typename T>
        bool operator()(const T *const plucker, T *residual) const
        {
            const T *d = plucker;
            const T *m = plucker + 3;

            T p[3] = {T(pt.x()), T(pt.y()), T(pt.z())};
            T cross[3];
            ceres::CrossProduct(p, d, cross);

            residual[0] = cross[0] - m[0];
            residual[1] = cross[1] - m[1];
            residual[2] = cross[2] - m[2];
            return true;
        }

        Eigen::Vector3d pt;

        static ceres::CostFunction *Create(const Eigen::Vector3d &point)
        {
            return new ceres::AutoDiffCostFunction<PointToLineResidual, 3, 6>(
                new PointToLineResidual(point));
        }
    };

    struct ReprojectionLineResidual
    {
        ReprojectionLineResidual(const Eigen::Vector3d &pt,
                                 const Eigen::Matrix3d &K,
                                 const Eigen::Vector3d &line_abc,
                                 double weight)
            : pt3d_(pt), K_(K), line_abc_(line_abc), weight_(weight) {}

        template <typename T>
        bool operator()(const T *const plucker, T *residual) const
        {
            const T *d = plucker;
            const T *m = plucker + 3;

            T p[3] = {T(pt3d_.x()), T(pt3d_.y()), T(pt3d_.z())};

            T p0[3];
            ceres::CrossProduct(d, m, p0);

            T p_minus_p0[3] = {p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]};
            T t = ceres::DotProduct(p_minus_p0, d);

            T proj[3] = {
                p0[0] + t * d[0],
                p0[1] + t * d[1],
                p0[2] + t * d[2]};

            if (ceres::abs(proj[0]) < T(1e-6))
            {
                residual[0] = T(0);
                return true;
            }

            const T fx = T(K_(0, 0));
            const T fy = T(K_(1, 1));
            const T cx = T(K_(0, 2));
            const T cy = T(K_(1, 2));

            const T u = -fx * proj[1] / proj[0] + cx;
            const T v = -fy * proj[2] / proj[0] + cy;

            const T A = T(line_abc_.x());
            const T B = T(line_abc_.y());
            const T C = T(line_abc_.z());

            residual[0] = T(weight_) * (A * u + B * v + C);
            return true;
        }

        Eigen::Vector3d pt3d_;
        Eigen::Matrix3d K_;
        Eigen::Vector3d line_abc_;
        double weight_;

        static ceres::CostFunction *Create(const Eigen::Vector3d &point,
                                           const Eigen::Matrix3d &K,
                                           const Eigen::Vector3d &line_abc,
                                           double weight)
        {
            return new ceres::AutoDiffCostFunction<ReprojectionLineResidual, 1, 6>(
                new ReprojectionLineResidual(point, K, line_abc, weight));
        }
    };

    struct PointPriorResidual
    {
        PointPriorResidual(const Eigen::Vector3d &p_prior, double w)
            : prior_(p_prior), weight_(w) {}

        template <typename T>
        bool operator()(const T *const plucker, T *residual) const
        {
            const T *d = plucker;
            const T *m = plucker + 3;
            T p0[3];
            ceres::CrossProduct(d, m, p0);

            residual[0] = T(weight_) * (p0[0] - T(prior_.x()));
            residual[1] = T(weight_) * (p0[1] - T(prior_.y()));
            residual[2] = T(weight_) * (p0[2] - T(prior_.z()));
            return true;
        }

        Eigen::Vector3d prior_;
        double weight_;

        static ceres::CostFunction *Create(const Eigen::Vector3d &p_prior, double w)
        {
            return new ceres::AutoDiffCostFunction<PointPriorResidual, 3, 6>(
                new PointPriorResidual(p_prior, w));
        }
    };

    struct PluckerPlus
    {
        template <typename T>
        bool operator()(const T *x, const T *delta, T *x_plus_delta) const
        {
            for (int i = 0; i < 6; ++i)
            {
                x_plus_delta[i] = x[i] + delta[i];
            }

            T d_norm = ceres::sqrt(x_plus_delta[0] * x_plus_delta[0] +
                                   x_plus_delta[1] * x_plus_delta[1] +
                                   x_plus_delta[2] * x_plus_delta[2]);
            if (d_norm > T(0))
            {
                x_plus_delta[0] /= d_norm;
                x_plus_delta[1] /= d_norm;
                x_plus_delta[2] /= d_norm;
            }

            const T dot = ceres::DotProduct(x_plus_delta, x_plus_delta + 3);
            x_plus_delta[3] -= dot * x_plus_delta[0];
            x_plus_delta[4] -= dot * x_plus_delta[1];
            x_plus_delta[5] -= dot * x_plus_delta[2];

            return true;
        }
    };

    CeresJointOptimizer::CeresJointOptimizer() {}
    CeresJointOptimizer::~CeresJointOptimizer() {}

    bool CeresJointOptimizer::optimizePose(Eigen::Vector3d &d,
                                           Eigen::Vector3d &m,
                                           const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                           const Eigen::Matrix3d &K,
                                           const Eigen::Vector3d &line2d_abc,
                                           const Eigen::Vector3d &point_prior,
                                           int max_iterations,
                                           int max_points,
                                           double weight_2d,
                                           double weight_pos)
    {
        if (!cloud || cloud->empty())
            return false;

        std::vector<int> indices;
        indices.reserve(static_cast<size_t>(std::min<int>(cloud->size(), std::max(8, max_points))));
        const int n_all = static_cast<int>(cloud->size());
        const int target = std::max(8, std::min(max_points, n_all));
        const int stride = std::max(1, n_all / target);
        for (int i = 0; i < n_all && static_cast<int>(indices.size()) < target; i += stride)
        {
            indices.push_back(i);
        }

        ceres::Problem problem;
        ceres::LocalParameterization *plucker_param = new ceres::AutoDiffLocalParameterization<PluckerPlus, 6, 6>();

        double plucker_arr[6] = {d.x(), d.y(), d.z(), m.x(), m.y(), m.z()};
        problem.AddParameterBlock(plucker_arr, 6, plucker_param);

        for (const int idx : indices)
        {
            const auto &point = cloud->points[static_cast<size_t>(idx)];
            const Eigen::Vector3d pt(point.x, point.y, point.z);

            ceres::CostFunction *cost_3d = PointToLineResidual::Create(pt);
            problem.AddResidualBlock(cost_3d, new ceres::HuberLoss(0.03), plucker_arr);

            ceres::CostFunction *cost_2d = ReprojectionLineResidual::Create(pt, K, line2d_abc, weight_2d);
            problem.AddResidualBlock(cost_2d, nullptr, plucker_arr);
        }

        ceres::CostFunction *cost_pos = PointPriorResidual::Create(point_prior, weight_pos);
        problem.AddResidualBlock(cost_pos, new ceres::HuberLoss(0.2), plucker_arr);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = std::max(5, max_iterations);
        options.minimizer_progress_to_stdout = false;
        options.num_threads = std::max(1u, std::thread::hardware_concurrency());
        options.function_tolerance = 1e-5;
        options.gradient_tolerance = 1e-7;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        d = Eigen::Vector3d(plucker_arr[0], plucker_arr[1], plucker_arr[2]).normalized();
        m = Eigen::Vector3d(plucker_arr[3], plucker_arr[4], plucker_arr[5]);

        return summary.IsSolutionUsable();
    }

} // namespace axispose

#else
namespace axispose
{
    CeresJointOptimizer::CeresJointOptimizer() {}
    CeresJointOptimizer::~CeresJointOptimizer() {}
    bool CeresJointOptimizer::optimizePose(Eigen::Vector3d &, Eigen::Vector3d &, const pcl::PointCloud<pcl::PointXYZ>::Ptr &, const Eigen::Matrix3d &, const Eigen::Vector3d &, const Eigen::Vector3d &, int, int, double, double)
    {
        return false;
    }
} // namespace axispose
#endif
