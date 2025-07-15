#pragma once

#include "sc_image.h"
#include "sc_point.h"
#include <opencv2/features2d.hpp>

namespace StudyCorr {

class CudaFeature2D
{
protected:
    CudaImage2D* ref_img = nullptr;
    CudaImage2D* tar_img = nullptr;

public:
    virtual ~CudaFeature2D() = default;

    // 设置参考和目标图像
    void setImages(CudaImage2D& ref, CudaImage2D& tar) {
        ref_img = &ref;
        tar_img = &tar;
    }

    // 数据预处理（如金字塔/特征预提取等）
    virtual void prepare() = 0;

    // 计算特征点/描述子（CUDA并行）
    virtual void compute() = 0;
};

class CudaFeature3D
{
protected:
    CudaImage3D* ref_img = nullptr;
    CudaImage3D* tar_img = nullptr;

public:
    virtual ~CudaFeature3D() = default;

    void setImages(CudaImage3D& ref, CudaImage3D& tar) {
        ref_img = &ref;
        tar_img = &tar;
    }

    virtual void prepare() = 0;
    virtual void compute() = 0;
};

} // namespace StudyCorr