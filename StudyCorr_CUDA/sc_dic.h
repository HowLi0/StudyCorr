#pragma once

#include"sc_poi.h"
#include"sc_image.h"
#include"sc_array.h"
#include"sc_subset.h"

namespace StudyCorr
{
    struct CudaKeypointIndex
    {
        int kp_idx;
        float distance;
        CUDA_HD CudaKeypointIndex() : kp_idx(-1), distance(0.0f) {}
        CUDA_HD CudaKeypointIndex(int idx, float dist) : kp_idx(idx), distance(dist) {}

    };

    class CudaDIC
    {
        public:
        CudaImage2D* ref_img = nullptr;
        CudaImage2D* tar_img = nullptr;

        int subset_radius_x = 0, subset_radius_y = 0;

        CUDA_HD CudaDIC() {}
        CUDA_HD ~CudaDIC()= default;

        CUDA_HD void setImages(CudaImage2D* ref, CudaImage2D* tar) {
            ref_img = ref; tar_img = tar;
        }
        CUDA_HD void setSubsetRadius(int rx, int ry) {
            subset_radius_x = rx; subset_radius_y = ry;
        }

        virtual CUDA_HD void prepare();
        virtual CUDA_HD void compute(CudaPOI2D* poi) = 0;
        virtual CUDA_HD void compute(Array2D<CudaPOI2D>& poi_queue) = 0;
    };

    class CudaDVC
    {
        CudaImage3D* ref_img = nullptr;
        CudaImage3D* tar_img = nullptr;

        int subset_radius_x = 0, subset_radius_y = 0, subset_radius_z = 0;

        CUDA_HD CudaDVC() {}
        CUDA_HD ~CudaDVC()= default;

        CUDA_HD void setImages(CudaImage3D* ref, CudaImage3D* tar) {
            ref_img = ref; tar_img = tar;
        }
        CUDA_HD void setSubsetRadius(int rx, int ry, int rz) {
            subset_radius_x = rx; subset_radius_y = ry; subset_radius_z = rz;
        }

        virtual CUDA_HD void prepare();
        virtual CUDA_HD void compute(CudaPOI3D* poi) = 0;
        virtual CUDA_HD void compute(Array3D<CudaPOI3D>& poi_queue) = 0;
    };

    CUDA_HD inline
    bool sortByZNCC(const CudaPOI2D& p1, const CudaPOI2D& p2) {
        return p1.result.zncc > p2.result.zncc;
    }

    CUDA_HD inline
    bool sortByDistance(const CudaKeypointIndex& kp1, const CudaKeypointIndex& kp2)
    {
        return kp1.distance < kp2.distance;
    }
}