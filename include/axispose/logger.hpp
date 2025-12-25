#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <chrono>

// PCL / Eigen for statistics collector
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/ModelCoefficients.h>
#include <Eigen/Dense>

namespace axispose
{

    // Simple CSV logger base class. Derived classes can call appendRow() to add rows.
    class CsvLogger
    {
    public:
        explicit CsvLogger(const std::string &path, const std::string &header = "")
            : path_(path), header_written_(false)
        {
            std::ifstream ifs(path_, std::ios::in);
            if (ifs.good())
            {
                ifs.seekg(0, std::ios::end);
                if (ifs.tellg() > 0)
                    header_written_ = true;
            }
            ifs.close();

            ofs_.open(path_, std::ios::app);
            if (ofs_.is_open() && !header.empty() && !header_written_)
            {
                ofs_ << header << '\n';
                header_written_ = true;
                ofs_.flush();
            }
        }

        virtual ~CsvLogger()
        {
            if (ofs_.is_open())
                ofs_.close();
        }

    protected:
        bool appendRow(const std::vector<std::string> &cols)
        {
            if (!ofs_.is_open())
            {
                ofs_.open(path_, std::ios::app);
                if (!ofs_.is_open())
                    return false;
            }

            for (size_t i = 0; i < cols.size(); ++i)
            {
                if (i)
                    ofs_ << ',';
                ofs_ << cols[i];
            }
            ofs_ << '\n';
            ofs_.flush();
            return true;
        }

        template <typename T>
        static std::string toString(T v, int precision = 6)
        {
            std::ostringstream ss;
            ss.setf(std::ios::fixed);
            ss.precision(precision);
            ss << v;
            return ss.str();
        }

    private:
        std::string path_;
        std::ofstream ofs_;
        bool header_written_;
    };

    // PointDistanceLogger: append per-point distances to line into CSV
    class PointDistanceLogger : public CsvLogger
    {
    public:
        explicit PointDistanceLogger(const std::string &path)
            : CsvLogger(path, "x,y,z,distance")
        {
        }

        bool appendDistancesToLine(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
                                   const Eigen::Vector3f &line_point,
                                   const Eigen::Vector3f &line_dir)
        {
            if (!cloud || cloud->empty())
                return false;

            std::vector<double> distances;
            if (!computeDistancesToLine(cloud, line_point, line_dir, distances))
                return false;

            for (size_t i = 0; i < cloud->points.size(); ++i)
            {
                const auto &p = cloud->points[i];
                std::vector<std::string> cols = {toString(p.x), toString(p.y), toString(p.z), toString(distances[i])};
                if (!appendRow(cols))
                    return false;
            }
            return true;
        }
        // 使用直线的coefficients计算点到直线的距离
        bool computeDistancesToLine(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
                                    const pcl::ModelCoefficients::Ptr &coefficients,
                                    std::vector<double> &out_distances)
        {
            if (!cloud || cloud->empty() || !coefficients)
                return false;

            // 计算每个点到直线的距离
            for (const auto &point : cloud->points)
            {
                double distance = std::abs(coefficients->values[0] * point.x +
                                           coefficients->values[1] * point.y +
                                           coefficients->values[2] * point.z +
                                           coefficients->values[3]) /
                                  std::sqrt(coefficients->values[0] * coefficients->values[0] +
                                            coefficients->values[1] * coefficients->values[1] +
                                            coefficients->values[2] * coefficients->values[2]);
                out_distances.push_back(distance);
            }
            return !out_distances.empty();
        }

    private:
        static bool computeDistancesToLine(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
                                           const Eigen::Vector3f &line_point,
                                           const Eigen::Vector3f &line_dir,
                                           std::vector<double> &out_distances)
        {
            if (!cloud || cloud->empty())
                return false;

            Eigen::VectorXf coeffs(6);
            coeffs[0] = line_point.x();
            coeffs[1] = line_point.y();
            coeffs[2] = line_point.z();
            Eigen::Vector3f dir = line_dir.normalized();
            coeffs[3] = dir.x();
            coeffs[4] = dir.y();
            coeffs[5] = dir.z();

            pcl::SampleConsensusModelLine<pcl::PointXYZ> model(cloud);
            model.getDistancesToModel(coeffs, out_distances);
            return !out_distances.empty();
        }
    };

    // SimpleTimingLogger: timing utility that logs label and elapsed milliseconds
    class SimpleTimingLogger : public CsvLogger
    {
    public:
        explicit SimpleTimingLogger(const std::string &path)
            : CsvLogger(path, "label,microseconds")
        {
        }

        void tik(const std::string &label)
        {
            starts_[label] = std::chrono::steady_clock::now();
        }

        void tok(const std::string &label)
        {
            auto it = starts_.find(label);
            if (it == starts_.end())
                return;

            auto end = std::chrono::steady_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - it->second).count();
            appendRow({label, toString(us, 0)});
            starts_.erase(it);
        }

    private:
        std::unordered_map<std::string, std::chrono::steady_clock::time_point> starts_;
    };

    class PointNumberLogger : public CsvLogger
    {
    private:
    public:
        explicit PointNumberLogger(const std::string &path) : CsvLogger(path, "point_count1,point_count2")
        {
        }
        void logCounts(size_t count1, size_t count2)
        {
            appendRow({toString(count1, 0), toString(count2, 0)});
        }
    };

} // namespace axispose
