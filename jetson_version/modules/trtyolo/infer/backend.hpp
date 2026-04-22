/**
 * @file backend.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief TensorRT 推理后端定义
 * @date 2025-01-15
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#pragma once

#include <NvInferRuntime.h>

#include <memory>
#include <string>
#include <vector>

#include "core/buffer.hpp"
#include "core/core.hpp"
#include "letterbox.hpp"
#include "trtyolo.hpp"
#include "utils/common.hpp"

namespace trtyolo {
    /**
 * @brief 推理选项配置结构体
 *
 */
struct InferConfig {
    int                 device_id                 = 0;      // < GPU ID
    bool                cuda_mem                  = false;  // < 推理数据是否已经在 CUDA 显存中
    bool                enable_managed_memory     = false;  // < 是否启用统一内存
    bool                enable_performance_report = false;  // < 是否启用性能报告
    std::optional<int2> input_shape;                        // < 输入数据的高、宽，未设置时表示宽度可变（用于输入数据宽高确定的任务场景：监控视频分析，AI外挂等）
    ProcessConfig       config;                             // < 图像预处理配置
};

/**
 * @brief TensorRT 后端类，用于执行推理操作。
 */
class TrtBackend {
public:
    /**
     * @brief 构造函数，用于初始化 TrtBackend 对象。
     *
     * @param trt_engine_file TensorRT 引擎文件路径。
     * @param infer_config 推理配置指针。
     */
    TrtBackend(const std::string& trt_engine_file, const InferConfig& infer_config);

    /**
     * @brief 默认构造函数。
     */
    TrtBackend() = default;

    /**
     * @brief 析构函数。
     */
    ~TrtBackend();

    /**
     * @brief 克隆 TrtBackend 对象。
     *
     * @return 克隆后的 TrtBackend 对象的智能指针。
     */
    std::unique_ptr<TrtBackend> clone();

    /**
     * @brief 执行推理操作。
     *
     * @param inputs 输入图像向量。
     */
    void infer(const std::vector<Image>& inputs);

    cudaStream_t            stream;        // < CUDA 流
    InferConfig             infer_config;  // < 推理选项
    std::vector<TensorInfo> tensor_infos;  // < 张量信息向量
    std::vector<Transform>  transforms;    // < 仿射变换向量
    int4                    min_shape;     // < 最小形状
    int4                    max_shape;     // < 最大形状
    bool                    dynamic;       // < 是否为动态形状

private:
    void getTensorInfo();
    void initialize();
    void captureCudaGraph();
    void dynamicInfer(const std::vector<Image>& inputs);
    void staticInfer(const std::vector<Image>& inputs);

    std::unique_ptr<TRTManager> manager_;        // < TensorRT 管理器对象的智能指针
    CudaGraph                   cuda_graph_;     // < CUDA 图
    std::unique_ptr<BaseBuffer> inputs_buffer_;  // < 输入缓冲区智能指针
    BufferType                  buffer_type_;    // < 缓冲区类型

    bool zero_copy_;                             // < 是否为零拷贝

    int input_size_;                             // < 输入大小
    int infer_size_;                             // < 推理大小
};

}  // namespace trtyolo
