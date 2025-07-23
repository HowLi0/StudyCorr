#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "sc_poi.h"

namespace StudyCorr_GPU {

// SIFT特征结构（与cusift一致，可扩展）
struct SiftFeature2D {
    float x, y;
    float scale, orientation, score, ambiguity;
    int match;
    float match_x, match_y;
    float data[128];
};

// SIFT仿射参数结构，默认对齐OpenCorr
struct SiftAffineParam {
    int knn;                // 邻域点数，OpenCorr默认7
    int radius_x;           // 子集半径x
    int radius_y;           // 子集半径y
    float kd_radius;        // KDTree搜索半径，OpenCorr默认sqrt(radius_x^2 + radius_y^2)
    int trial_number;       // RANSAC最大次数，OpenCorr默认20
    int sample_number;      // RANSAC每次采样点数，OpenCorr默认3
    float error_threshold;  // RANSAC判据阈值，OpenCorr默认1.5
    int min_inlier;         // RANSAC最小内点数，OpenCorr默认7

    SiftAffineParam(int rx = 15, int ry = 15)
        : knn(7), radius_x(rx), radius_y(ry),
          kd_radius(std::sqrt(float(rx*rx + ry*ry))),
          trial_number(20), sample_number(3),
          error_threshold(1.5f), min_inlier(7) {}
};

// SIFT仿射初始化Batch CUDA接口（OpenCorr风格，支持host侧邻居选取）
class SiftAffineBatchGpu {
public:
    SiftAffineBatchGpu(const SiftAffineParam& param = SiftAffineParam());
    ~SiftAffineBatchGpu();

    // 设置待处理POI列表
    void set_poi_list(const std::vector<CudaPOI2D>& poi_list) { poi_list_ = poi_list; }
    const std::vector<CudaPOI2D>& get_poi_list() const { return poi_list_; }

    // SIFT特征输入，预处理KDTree/knn/暴力补齐选邻居，并分配/传输CUDA内存
    // 逻辑完全对齐OpenCorr: 先radiusSearch，再knn补齐，最后暴力补齐
    void prepare_cuda(const SiftFeature2D* ref_kp, const SiftFeature2D* tar_kp, int num_kp, cudaStream_t stream = 0);

    // CUDA并行仿射初始化，自动RANSAC，结果写入POI deformation，与sc_icgn一致
    void compute_batch_cuda(CudaPOI2D* pois, int N, cudaStream_t stream = 0);

    void release_cuda();

private:
    SiftAffineParam param_;
    SiftFeature2D* d_ref_kp = nullptr;
    SiftFeature2D* d_tar_kp = nullptr;
    int num_kp_ = 0;

    // host侧POI列表
    std::vector<CudaPOI2D> poi_list_;

    // host侧邻居索引与距离
    std::vector<int> neighbor_idx_;
    std::vector<float> neighbor_dist_;

    // device侧邻居索引与距离
    int* d_neighbor_idx_ = nullptr;
    float* d_neighbor_dist_ = nullptr;
};

} // namespace StudyCorr_GPU