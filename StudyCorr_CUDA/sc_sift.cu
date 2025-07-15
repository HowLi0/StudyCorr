#include "sc_sift.h"
#include <cstring>
#include <cassert>

namespace StudyCorr {

CudaSIFT2D::CudaSIFT2D()
    : ref_data(nullptr), tar_data(nullptr), prepared(false) {}

CudaSIFT2D::~CudaSIFT2D() {
    clear();
}

void CudaSIFT2D::clear() {
    if (ref_data) { delete[] ref_data; ref_data = nullptr; }
    if (tar_data) { delete[] tar_data; tar_data = nullptr; }
    FreeSiftData(sift_ref);
    FreeSiftData(sift_tar);
    prepared = false;
}

void CudaSIFT2D::prepare() {
    clear();

    // --- 1. 拷贝sc风格CudaImage2D数据到float*（0~255，float）
    w = ref_img->width;  h = ref_img->height;
    ref_data = new float[w * h];
    tar_data = new float[w * h];
    #pragma omp parallel for
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            ref_data[i * w + j] = ref_img->data(i, j);
            tar_data[i * w + j] = tar_img->data(i, j);
        }
    }
    // --- 2. cuSIFT: 分配CudaImage和SiftData
    cu_ref_img.Allocate(w, h, iAlignUp(w, 128), false, nullptr, ref_data);
    cu_tar_img.Allocate(w, h, iAlignUp(w, 128), false, nullptr, tar_data);

    cu_ref_img.Download();
    cu_tar_img.Download();

    int max_features = config.n_features > 0 ? config.n_features : 32768;
    InitSiftData(sift_ref, max_features, true, true);
    InitSiftData(sift_tar, max_features, true, true);

    prepared = true;
}

void CudaSIFT2D::compute() {
    assert(prepared);
    // --- 1. cuSIFT特征提取
    float *memoryTmp = AllocSiftTempMemory(w, h, config.n_octaves, false);
    ExtractSift(sift_ref, cu_ref_img, config.n_octaves, config.init_blur, config.thresh, 0.0f, false, memoryTmp);
    ExtractSift(sift_tar, cu_tar_img, config.n_octaves, config.init_blur, config.thresh, 0.0f, false, memoryTmp);
    FreeSiftTempMemory(memoryTmp);

    // --- 2. cuSIFT特征匹配
    MatchSiftData(sift_ref, sift_tar);

    // --- 3. 拷贝匹配点到sc风格
    ref_matched_kp = Array1D<Point2D>(sift_ref.numPts);
    tar_matched_kp = Array1D<Point2D>(sift_ref.numPts);
    #pragma omp parallel for
    for (int i = 0; i < sift_ref.numPts; ++i) {
        int match_idx = sift_ref.h_data[i].match;
        if (match_idx < 0 || match_idx >= sift_tar.numPts) continue;
        // ratio test
        if (sift_ref.h_data[i].ambiguity < config.matching_ratio) {
            ref_matched_kp[i] = Point2D(sift_ref.h_data[i].xpos, sift_ref.h_data[i].ypos);
            tar_matched_kp[i] = Point2D(sift_tar.h_data[match_idx].xpos, sift_tar.h_data[match_idx].ypos);
        }
    }
}

} // namespace StudyCorr