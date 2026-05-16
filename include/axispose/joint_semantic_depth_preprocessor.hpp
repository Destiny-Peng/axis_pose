#pragma once

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace axispose
{
    // JointSemanticDepthPreprocessor
    // 提供基于深度的 mask 修复与投影辅助函数。
    class JointSemanticDepthPreprocessor
    {
    public:
        JointSemanticDepthPreprocessor() = default;

        // 深度引导的自适应膨胀
        // boundary_mask: 输入二值边界掩码 (CV_8U)
        // depth_map: 对齐到 color 的深度图 (CV_16U 毫米或 CV_32F 米)
        // base_kernel_size: 基础核半径（像素级）
        // alpha: 深度对核尺寸的线性放大系数（以米为单位）
        cv::Mat dilateMaskAdaptive(const cv::Mat &boundary_mask, const cv::Mat &depth_map, float base_kernel_size, float alpha) const;

        // 深度相关的置信度阈值放宽
        // confidence_map: YOLO 输出的置信度图 (CV_32F, 0..1)
        // depth_map: 对齐到 color 的深度图 (CV_16U 毫米或 CV_32F 米)
        // 返回二值掩码 CV_8U
        cv::Mat applyAdaptiveThreshold(const cv::Mat &confidence_map, const cv::Mat &depth_map, float near_thresh = 0.5f, float far_thresh = 0.2f) const;

        // 根据二值 mask 裁深度图并投影到 PCL 点云
        // depth: 对齐后的深度图 (CV_16U 毫米 或 CV_32F 米)
        // mask: 二值掩码 CV_8U (非零为有效)
        // 相机内参: fx,fy,cx,cy
        pcl::PointCloud<pcl::PointXYZ>::Ptr cvMaskToPclCloud(const cv::Mat &depth, const cv::Mat &mask, double fx, double fy, double cx, double cy) const;

    private:
        // 读深度为米（返回负值表示无效）
        static inline double readDepthMeters(const cv::Mat &depth, int v, int u);
    };

} // namespace axispose
