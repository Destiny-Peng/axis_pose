#include "axispose/benchmark.hpp"
#include <Eigen/Dense>
#include <iostream>

using namespace axispose;

Eigen::Vector3d compute_pose()
{
    // simulate work
    Eigen::Vector3d v(1.2345, 2.3456, 3.4567);
    return v;
}

int main()
{
    // initialize benchmark: write to examples/benchmark.csv, record two extra columns x,y
    AlgorithmBenchmark bm("examples", "benchmark.csv", true, {"x", "y"});

    // run and record only x,y of the returned Eigen::Vector3d
    auto res = bm.run("compute_pose", []() -> Eigen::Vector3d
                      { return compute_pose(); }, [](const Eigen::Vector3d &v) -> std::vector<std::string>
                      { return {std::to_string(v.x()), std::to_string(v.y())}; });

    std::cout << "Result: " << res.transpose() << std::endl;
    return 0;
}
