#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "sc_sift_affine.h"

namespace StudyCorr_GPU {

// SIFT批量匹配（输出配对后数组，直接对齐批量仿射/ICGN流接口）
class SiftFeatureBatchGpu {
public:
    SiftFeatureBatchGpu();
    ~SiftFeatureBatchGpu();

    void release();

    // 输入两幅图像，准备CUDA内存
    void prepare_cuda(const float* ref_img, const float* tar_img, int height, int width, cudaStream_t stream = 0);

    // 批量检测+匹配，直接输出配对后数组
    void compute_match_batch_cuda(cudaStream_t stream = 0);

    // 匹配结果数组（每个元素顺序一一配对）
    std::vector<SiftFeature2D> match_kp_ref;
    std::vector<SiftFeature2D> match_kp_tar;
    int num_match = 0;

private:
    int width_ = 0, height_ = 0;
    float* d_ref_img_ = nullptr;
    float* d_tar_img_ = nullptr;
    int max_feat_ = 25000;
};

} // namespace StudyCorr