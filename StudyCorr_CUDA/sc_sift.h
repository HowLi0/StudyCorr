#pragma once
#include "sc_feature.h"
#include "sc_array.h"
#include "cudaImage.h"
#include "cudaSift.h"

namespace StudyCorr {

struct Sift2dConfig
{
    int n_features = 0;     // 最大特征点数
    int n_octaves = 5;      // 金字塔octaves
    float init_blur = 1.6f; // 初始高斯
    float thresh = 3.0f;    // DoG阈值
    float matching_ratio = 0.8f;
};

class CudaSIFT2D : public CudaFeature2D
{
public:
    Sift2dConfig config;
    Array1D<Point2D> ref_matched_kp;
    Array1D<Point2D> tar_matched_kp;

    CudaSIFT2D();
    ~CudaSIFT2D();

    void setConfig(const Sift2dConfig& cfg) { config = cfg; }
    void prepare() override;
    void compute() override;
    void clear();

private:
    // cuSIFT数据
    CudaImage cu_ref_img, cu_tar_img;
    SiftData sift_ref, sift_tar;
    float *ref_data = nullptr, *tar_data = nullptr;
    int w = 0, h = 0;
    bool prepared = false;
};

} // namespace StudyCorr