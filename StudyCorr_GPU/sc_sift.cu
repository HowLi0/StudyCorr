#include "sc_sift.h"
#include <cudaSift.h>
#include <cudaImage.h>
#include <cstring>

namespace StudyCorr_GPU {

SiftFeatureBatchGpu::SiftFeatureBatchGpu() {}
SiftFeatureBatchGpu::~SiftFeatureBatchGpu() { release(); }

void SiftFeatureBatchGpu::release() {
    if (d_ref_img_) { cudaFree(d_ref_img_); d_ref_img_ = nullptr; }
    if (d_tar_img_) { cudaFree(d_tar_img_); d_tar_img_ = nullptr; }
    match_kp_ref.clear();
    match_kp_tar.clear();
    num_match = 0;
    width_ = height_ = 0;
}

void SiftFeatureBatchGpu::prepare_cuda(const float* ref_img, const float* tar_img, int width, int height, cudaStream_t stream) {
    release();
    width_ = width;
    height_ = height;
    size_t img_bytes = width_ * height_ * sizeof(float);

    cudaMalloc(&d_ref_img_, img_bytes);
    cudaMemcpyAsync(d_ref_img_, ref_img, img_bytes, cudaMemcpyHostToDevice, stream);

    cudaMalloc(&d_tar_img_, img_bytes);
    cudaMemcpyAsync(d_tar_img_, tar_img, img_bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
}

void SiftFeatureBatchGpu::compute_match_batch_cuda(cudaStream_t stream) {
    match_kp_ref.clear();
    match_kp_tar.clear();
    num_match = 0;

    // --- 检测参考图特征点 ---
    SiftData ref_data;
    InitSiftData(ref_data, max_feat_, true, true);
    CudaImage ref_img;
    ref_img.Allocate(width_, height_, iAlignUp(width_, 128), false, d_ref_img_);
    InitCuda(0);
    ExtractSift(ref_data, ref_img, 5, 1.0, 3.0f, 0.0f, false, nullptr);
    cudaStreamSynchronize(stream);

    // --- 检测目标图特征点 ---
    SiftData tar_data;
    InitSiftData(tar_data, max_feat_, true, true);
    CudaImage tar_img;
    tar_img.Allocate(width_, height_, iAlignUp(width_, 128), false, d_tar_img_);
    ExtractSift(tar_data, tar_img, 5, 1.0, 3.0f, 0.0f, false, nullptr);
    cudaStreamSynchronize(stream);

    // --- CUDA SIFT批量匹配 ---
    MatchSiftData(ref_data, tar_data);

    // 默认参数
    float minScore = 0.85f;
    float maxAmbiguity = 0.95f;

    // 只保留配对点
    for (int i = 0; i < ref_data.numPts; ++i) {
        const SiftPoint& ref_pt = ref_data.h_data[i];
        if (ref_pt.score >= minScore && ref_pt.ambiguity <= maxAmbiguity && ref_pt.match >= 0 && ref_pt.match < tar_data.numPts) {
            const SiftPoint& tar_pt = tar_data.h_data[ref_pt.match];
            SiftFeature2D fref, ftar;
            // 拷贝参考点
            fref.x = ref_pt.xpos;
            fref.y = ref_pt.ypos;
            fref.scale = ref_pt.scale;
            fref.orientation = ref_pt.orientation;
            fref.score = ref_pt.score;
            fref.ambiguity = ref_pt.ambiguity;
            fref.match = ref_pt.match;
            fref.match_x = ref_pt.match_xpos;
            fref.match_y = ref_pt.match_ypos;
            memcpy(fref.data, ref_pt.data, sizeof(float)*128);
            // 拷贝目标点
            ftar.x = tar_pt.xpos;
            ftar.y = tar_pt.ypos;
            ftar.scale = tar_pt.scale;
            ftar.orientation = tar_pt.orientation;
            ftar.score = tar_pt.score;
            ftar.ambiguity = tar_pt.ambiguity;
            ftar.match = tar_pt.match;
            ftar.match_x = tar_pt.match_xpos;
            ftar.match_y = tar_pt.match_ypos;
            memcpy(ftar.data, tar_pt.data, sizeof(float)*128);

            match_kp_ref.push_back(fref);
            match_kp_tar.push_back(ftar);
        }
    }
    num_match = int(match_kp_ref.size());

    FreeSiftData(ref_data);
    FreeSiftData(tar_data);
}

} // namespace StudyCorr