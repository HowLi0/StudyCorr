#pragma once
#include <cuda_runtime.h>
#include <cassert>
#include "sc_poi.h"
#include"sc_icgn_device_function.cuh"

namespace StudyCorr_GPU {

    struct ICGNParam {
        int subsetRadius = 21;
        double convergenceThreshold = 0.001f;
        int maxIterations = 10;
    };

    // 2D ICGN1批量CUDA接口
    class ICGN2D1BatchGpu {
    public:
        ICGN2D1BatchGpu();
        ~ICGN2D1BatchGpu();

        void prepare_cuda(const float* ref_image, const float* tar_image, int height, int width, const ICGNParam& param = ICGNParam(), cudaStream_t stream = 0);
        void compute_batch_cuda(CudaPOI2D* pois, int N, cudaStream_t stream = 0);
        void release_cuda();

    private:
        float* d_ref_image = nullptr;
        float* d_tar_image = nullptr;
        int height, width;
        ICGNParam param;
    };

    // 2D ICGN2批量CUDA接口
    class ICGN2D2BatchGpu {
    public:
        ICGN2D2BatchGpu();
        ~ICGN2D2BatchGpu();

        void prepare_cuda(const float* ref_image, const float* tar_image, int height, int width, const ICGNParam& param, cudaStream_t stream = 0);
        void compute_batch_cuda(CudaPOI2D* pois, int N, cudaStream_t stream = 0);
        void release_cuda();

    private:
        float* d_ref_image = nullptr;
        float* d_tar_image = nullptr;
        int height, width;
        ICGNParam param;
    };

    // 3D ICGN1批量CUDA接口
    class ICGN3D1BatchGpu {
    public:
        ICGN3D1BatchGpu();
        ~ICGN3D1BatchGpu();

        void prepare_cuda(const float* ref_image, const float* tar_image, int dim_x, int dim_y, int dim_z, const ICGNParam& param);
        void compute_batch_cuda(CudaPOI3D* pois, int N, cudaStream_t stream = 0);
        void release_cuda();

    private:
        float* d_ref_image = nullptr;
        float* d_tar_image = nullptr;
        int dim_x, dim_y, dim_z;
        ICGNParam param;
    };

}