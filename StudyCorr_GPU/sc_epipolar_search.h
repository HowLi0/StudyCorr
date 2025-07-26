#pragma once
#include <cuda_runtime.h>
#include "sc_poi.h"
#include"sc_icgn_device_function.cuh"

namespace StudyCorr_GPU {

class EpipolarSearchGpu {
public:
    EpipolarSearchGpu(int subsetRadius, int search_step, cv::Mat F);
    ~EpipolarSearchGpu();

    void prepare_cuda(const float* ref_image, const float* tar_image, int height, int width, cudaStream_t stream = 0);
    void compute_batch_cuda(CudaPOI2D* pois, int N, cudaStream_t stream = 0);
    void release_cuda();

private:
    // host variables
    int subset_radius; // 子集半径
    int search_step; // 搜索步长
    float h_F[9]; // 正确：基础矩阵

    // device variables
    float* d_ref_image = nullptr;
    float* d_tar_image = nullptr;
    int height = 0, width = 0;
    float* d_F = nullptr; // Fundamental matrix
};

} // namespace StudyCorr_GPU