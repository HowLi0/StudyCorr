/*
 * Data-Oriented Design (DOD) version of ICGN batch processing
 * Uses Structure of Arrays (SoA) layout for better performance
 */

#pragma once
#include <cuda_runtime.h>
#include "sc_poi_dod.h"

namespace StudyCorr_GPU {

    struct ICGNParam; // Forward declaration from sc_icgn.h

    // DOD version of 2D ICGN1 batch processing
    class ICGN2D1BatchGpuDOD {
    public:
        ICGN2D1BatchGpuDOD();
        ~ICGN2D1BatchGpuDOD();

        // Prepare images and parameters for computation
        void prepare_cuda(const float* ref_image, const float* tar_image, 
                         int height, int width, const ICGNParam& param, cudaStream_t stream = 0);
        
        // Compute batch using DOD SoA layout
        void compute_batch_cuda(POI2D_SoA& pois_soa, cudaStream_t stream = 0);
        
        // Release CUDA resources
        void release_cuda();

    private:
        float* d_ref_image = nullptr;
        float* d_tar_image = nullptr;
        int height, width;
        ICGNParam param;
    };

    // DOD version of 2D ICGN2 batch processing
    class ICGN2D2BatchGpuDOD {
    public:
        ICGN2D2BatchGpuDOD();
        ~ICGN2D2BatchGpuDOD();

        void prepare_cuda(const float* ref_image, const float* tar_image, 
                         int height, int width, const ICGNParam& param, cudaStream_t stream = 0);
        void compute_batch_cuda(POI2D_SoA& pois_soa, cudaStream_t stream = 0);
        void release_cuda();

    private:
        float* d_ref_image = nullptr;
        float* d_tar_image = nullptr;
        int height, width;
        ICGNParam param;
    };

    // DOD version of 3D ICGN1 batch processing
    // NOTE: This is currently experimental - full 3D ICGN iteration not yet implemented
    class ICGN3D1BatchGpuDOD {
    public:
        ICGN3D1BatchGpuDOD();
        ~ICGN3D1BatchGpuDOD();

        void prepare_cuda(const float* ref_image, const float* tar_image, 
                         int dim_x, int dim_y, int dim_z, const ICGNParam& param);
        void compute_batch_cuda(POI3D_SoA& pois_soa, cudaStream_t stream = 0);
        void release_cuda();

    private:
        float* d_ref_image = nullptr;
        float* d_tar_image = nullptr;
        int dim_x, dim_y, dim_z;
        ICGNParam param;
    };

} // namespace StudyCorr_GPU
