#include "sc_sift_affine.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>
#include <nanoflann.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <nanoflann.hpp>

namespace StudyCorr_GPU {

// KDTree adaptor for nanoflann
struct SiftKeypointCloud {
    const SiftFeature2D* pts;
    size_t num;
    inline size_t kdtree_get_point_count() const { return num; }
    inline float kdtree_get_pt(const size_t idx, int dim) const {
        return (dim == 0) ? pts[idx].x : pts[idx].y;
    }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

// nanoflann KDTree: use explicit size_t for index type
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, SiftKeypointCloud>,
    SiftKeypointCloud, 2, uint32_t> SiftKDTree;

// CUDA RANSAC仿射核，OpenCorr风格
struct SiftAffineRansacParam {
    int trial_number;
    int sample_number;
    float error_threshold;
    int min_inlier;
};

__device__ float norm2(float x, float y) { return sqrtf(x*x + y*y); }

__global__ void estimate_affine_opencorr_kernel(
    const SiftFeature2D* ref_kp,
    const SiftFeature2D* tar_kp,
    const int* poi_neighbor_idx, // [N*max_neighbor]
    const float* poi_neighbor_dist, // [N*max_neighbor]
    int max_neighbor,
    CudaPOI2D* pois,
    int N,
    SiftAffineRansacParam param,
    unsigned long long seed
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    CudaPOI2D& poi = pois[idx];

    // 1. 邻居数据
    int neighbor_num = 0;
    int neighbor_idx[30];
    float neighbor_dist[30];
    for (int i = 0; i < max_neighbor; ++i) {
        int nidx = poi_neighbor_idx[idx*max_neighbor + i];
        if (nidx < 0) break;
        neighbor_idx[i] = nidx;
        neighbor_dist[i] = poi_neighbor_dist[idx*max_neighbor + i];
        neighbor_num++;
    }

    if (neighbor_num < param.sample_number) {
        poi.result.zncc = -1.f;
        poi.result.feature = 0;
        return;
    }

    // 2. 局部坐标转换
    float ref_X[30], ref_Y[30], tar_X[30], tar_Y[30];
    for (int i = 0; i < neighbor_num; ++i) {
        int kp_idx = neighbor_idx[i];
        ref_X[i] = ref_kp[kp_idx].x - poi.x;
        ref_Y[i] = ref_kp[kp_idx].y - poi.y;
        tar_X[i] = tar_kp[kp_idx].x - poi.x;
        tar_Y[i] = tar_kp[kp_idx].y - poi.y;
    }

    // 3. RANSAC洗牌采样
    int best_inlier = 0, best_trial_counter = 0;
    float best_affine[6] = {0};
    int best_inlier_indices[30];
    float best_mean_error = 1e10f;

    curandState state;
    curand_init(seed + idx, 0, 0, &state);

    int candidate_idx[30];
    for (int i = 0; i < neighbor_num; ++i) candidate_idx[i] = i;

    int trial_counter = 0;
    do {
        // 洗牌采样
        for (int i = neighbor_num-1; i > 0; --i) {
            int j = curand(&state) % (i+1);
            int tmp = candidate_idx[i]; candidate_idx[i] = candidate_idx[j]; candidate_idx[j] = tmp;
        }
        // 构造最小二乘
        float A[9] = {0}, Bx[3] = {0}, By[3] = {0};
        for (int s = 0; s < param.sample_number; ++s) {
            int k = candidate_idx[s];
            float x = ref_X[k], y = ref_Y[k], tx = tar_X[k], ty = tar_Y[k];
            float v[3] = {x, y, 1.f};
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
                A[r*3+c] += v[r]*v[c];
            for (int r = 0; r < 3; ++r) {
                Bx[r] += v[r]*tx;
                By[r] += v[r]*ty;
            }
        }
        float invA[9];
        float det = A[0]*A[4]*A[8] + A[1]*A[5]*A[6] + A[2]*A[3]*A[7]
                  - A[0]*A[5]*A[7] - A[1]*A[3]*A[8] - A[2]*A[4]*A[6];
        bool ok = fabs(det) >= 1e-6f;
        if (!ok) { trial_counter++; continue; }
        invA[0]=(A[4]*A[8]-A[5]*A[7])/det; invA[1]=(A[2]*A[7]-A[1]*A[8])/det; invA[2]=(A[1]*A[5]-A[2]*A[4])/det;
        invA[3]=(A[5]*A[6]-A[3]*A[8])/det; invA[4]=(A[0]*A[8]-A[2]*A[6])/det; invA[5]=(A[2]*A[3]-A[0]*A[5])/det;
        invA[6]=(A[3]*A[7]-A[4]*A[6])/det; invA[7]=(A[1]*A[6]-A[0]*A[7])/det; invA[8]=(A[0]*A[4]-A[1]*A[3])/det;
        float affine_x[3]={0}, affine_y[3]={0};
        for (int r=0;r<3;++r) for(int j=0;j<3;++j) {
            affine_x[r]+=invA[r*3+j]*Bx[j]; affine_y[r]+=invA[r*3+j]*By[j];
        }
        // 一致集判据
        int inlier=0;
        int inlier_indices[30];
        float mean_error = 0.0f;
        for(int i=0;i<neighbor_num;++i) {
            float x=ref_X[i], y=ref_Y[i];
            float tx=affine_x[0]*x + affine_x[1]*y + affine_x[2];
            float ty=affine_y[0]*x + affine_y[1]*y + affine_y[2];
            float err=norm2(tx-tar_X[i],ty-tar_Y[i]);
            if(err<param.error_threshold) {
                inlier_indices[inlier] = i;
                mean_error += err;
                inlier++;
            }
        }
        mean_error = inlier > 0 ? mean_error/inlier : 1e10f;
        if (inlier > best_inlier ||
            (inlier == best_inlier && mean_error < best_mean_error))
        {
            best_inlier = inlier;
            best_mean_error = mean_error;
            for (int k = 0; k < inlier; ++k) best_inlier_indices[k] = inlier_indices[k];
            for (int i=0;i<3;++i) { best_affine[i]=affine_x[i]; best_affine[i+3]=affine_y[i]; }
            best_trial_counter = trial_counter;
        }
        trial_counter++;
    } while (trial_counter < param.trial_number &&
        (best_inlier < param.min_inlier || best_mean_error > param.error_threshold/param.min_inlier));

    // 4. 最终仿射重算
    if (best_inlier < 3) {
        poi.deformation = DeformationVector2D();
        poi.result.zncc = -2.f;
        poi.result.feature = 0;
        poi.result.iteration = trial_counter;
        return;
    }
    float A[9]={0}, Bx[3]={0}, By[3]={0};
    for (int k = 0; k < best_inlier; ++k) {
        int i = best_inlier_indices[k];
        float x = ref_X[i], y = ref_Y[i];
        float v[3] = {x, y, 1.f};
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
            A[r*3+c] += v[r]*v[c];
        for (int r = 0; r < 3; ++r) {
            Bx[r] += v[r]*tar_X[i];
            By[r] += v[r]*tar_Y[i];
        }
    }
    float invA[9];
    float det = A[0]*A[4]*A[8] + A[1]*A[5]*A[6] + A[2]*A[3]*A[7]
              - A[0]*A[5]*A[7] - A[1]*A[3]*A[8] - A[2]*A[4]*A[6];
    bool ok = fabs(det)>=1e-6f;
    float affine_x[3]={0}, affine_y[3]={0};
    if (ok) {
        invA[0]=(A[4]*A[8]-A[5]*A[7])/det; invA[1]=(A[2]*A[7]-A[1]*A[8])/det; invA[2]=(A[1]*A[5]-A[2]*A[4])/det;
        invA[3]=(A[5]*A[6]-A[3]*A[8])/det; invA[4]=(A[0]*A[8]-A[2]*A[6])/det; invA[5]=(A[2]*A[3]-A[0]*A[5])/det;
        invA[6]=(A[3]*A[7]-A[4]*A[6])/det; invA[7]=(A[1]*A[6]-A[0]*A[7])/det; invA[8]=(A[0]*A[4]-A[1]*A[3])/det;
        for (int r=0;r<3;++r) for(int j=0;j<3;++j) {
            affine_x[r]+=invA[r*3+j]*Bx[j]; affine_y[r]+=invA[r*3+j]*By[j];
        }
    }
    // 仿射参数写回（OpenCorr风格）
    poi.deformation.u   = affine_x[2];
    poi.deformation.ux  = affine_x[0] - 1.f;
    poi.deformation.uy  = affine_x[1];
    poi.deformation.v   = affine_y[2];
    poi.deformation.vx  = affine_y[0];
    poi.deformation.vy  = affine_y[1] - 1.f;
    poi.deformation.uxx = poi.deformation.uxy = poi.deformation.uyy = 0.f;
    poi.deformation.vxx = poi.deformation.vxy = poi.deformation.vyy = 0.f;

    // debug/特征统计
    float dist_sum = 0.0f, dist_max = 0.0f, dist_min = 1e10f;
    for (int i = 0; i < neighbor_num; ++i) {
        dist_sum += neighbor_dist[i];
        if (neighbor_dist[i] > dist_max) dist_max = neighbor_dist[i];
        if (neighbor_dist[i] < dist_min) dist_min = neighbor_dist[i];
    }
    float dist_mean = neighbor_num > 0 ? dist_sum / neighbor_num : 0.0f;
    poi.result.feature = best_inlier;
    poi.result.zncc = 0.f;
    poi.result.u0 = dist_mean;
    poi.result.v0 = dist_max;
    poi.result.iteration = best_trial_counter;
    poi.result.convergence = best_mean_error;
}

// --- SiftAffineBatchGpu实现 ---
SiftAffineBatchGpu::SiftAffineBatchGpu(const SiftAffineParam& param)
    : param_(param)
{
}

SiftAffineBatchGpu::~SiftAffineBatchGpu() { release_cuda(); }


// KDTree/knnSearch/暴力补齐（OpenCorr式）+分配CUDA内存
void SiftAffineBatchGpu::prepare_cuda(const SiftFeature2D* ref_kp, const SiftFeature2D* tar_kp, int num_kp, cudaStream_t stream) {
    release_cuda();
    num_kp_ = num_kp;
    cudaMalloc(&d_ref_kp, num_kp_ * sizeof(SiftFeature2D));
    cudaMalloc(&d_tar_kp, num_kp_ * sizeof(SiftFeature2D));
    cudaMemcpy(d_ref_kp, ref_kp, num_kp_*sizeof(SiftFeature2D), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tar_kp, tar_kp, num_kp_*sizeof(SiftFeature2D), cudaMemcpyHostToDevice);

    neighbor_idx_.clear();
    neighbor_dist_.clear();

    SiftKeypointCloud cloud{ref_kp, size_t(num_kp)};
    SiftKDTree kdtree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();

    int max_neighbor = 30;
    neighbor_idx_.resize(poi_list_.size() * max_neighbor, -1);
    neighbor_dist_.resize(poi_list_.size() * max_neighbor, 1e10f);

    for (size_t pi = 0; pi < poi_list_.size(); ++pi) {
        float query_pt[2] = {poi_list_[pi].x, poi_list_[pi].y};
        std::vector<nanoflann::ResultItem<uint32_t, float>> ret_matches;

        // 1. radiusSearch
        nanoflann::SearchParameters params;
        params.sorted = false;
        kdtree.radiusSearch(query_pt, param_.kd_radius * param_.kd_radius, ret_matches, params);

        int got = 0;
        for (auto& m : ret_matches) {
            if (got >= max_neighbor) break;
            neighbor_idx_[pi*max_neighbor + got] = int(m.first);
            neighbor_dist_[pi*max_neighbor + got] = std::sqrt(m.second);
            got++;
        }

        // 2. knnSearch if not enough
        if (got < param_.min_inlier) {
            std::vector<uint32_t> knn_idx(max_neighbor);
            std::vector<float> knn_dist(max_neighbor);
            size_t found = kdtree.knnSearch(query_pt, max_neighbor, knn_idx.data(), knn_dist.data());
            for (size_t k = got; k < std::min(found, size_t(max_neighbor)); ++k) {
                neighbor_idx_[pi*max_neighbor + k] = int(knn_idx[k]);
                neighbor_dist_[pi*max_neighbor + k] = std::sqrt(knn_dist[k]);
            }
            got = int(std::max(got, int(found)));
        }

        // 3. 暴力补齐
        if (got < param_.min_inlier) {
            std::vector<std::pair<int, float>> dist_vec;
            for (int ki = 0; ki < num_kp_; ++ki) {
                float dx = ref_kp[ki].x - poi_list_[pi].x;
                float dy = ref_kp[ki].y - poi_list_[pi].y;
                float dist = std::sqrt(dx*dx + dy*dy);
                dist_vec.emplace_back(ki, dist);
            }
            std::sort(dist_vec.begin(), dist_vec.end(), [](auto& a, auto& b){ return a.second < b.second; });
            int fill = got;
            for (size_t k = 0; fill < param_.min_inlier && k < dist_vec.size() && fill < max_neighbor; ++k) {
                neighbor_idx_[pi*max_neighbor + fill] = dist_vec[k].first;
                neighbor_dist_[pi*max_neighbor + fill] = dist_vec[k].second;
                fill++;
            }
        }
        for (int k = 0; k < max_neighbor; ++k) {
            if (neighbor_idx_[pi*max_neighbor + k] < 0) {
                neighbor_dist_[pi*max_neighbor + k] = 1e10f;
            }
        }
    }

    // 分配并拷贝到device
    cudaMalloc(&d_neighbor_idx_, neighbor_idx_.size() * sizeof(int));
    cudaMalloc(&d_neighbor_dist_, neighbor_dist_.size() * sizeof(float));
    cudaMemcpy(d_neighbor_idx_, neighbor_idx_.data(), neighbor_idx_.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbor_dist_, neighbor_dist_.data(), neighbor_dist_.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(stream);
}

void SiftAffineBatchGpu::compute_batch_cuda(CudaPOI2D* pois, int N, cudaStream_t stream) {
    CudaPOI2D* d_pois;
    cudaMalloc(&d_pois, N*sizeof(CudaPOI2D));
    cudaMemcpyAsync(d_pois, pois, N*sizeof(CudaPOI2D), cudaMemcpyHostToDevice, stream);

    SiftAffineRansacParam d_param;
    d_param.trial_number = param_.trial_number;
    d_param.sample_number = param_.sample_number;
    d_param.error_threshold = param_.error_threshold;
    d_param.min_inlier = param_.min_inlier;

    int max_neighbor = 30;
    int block = 128, grid = (N + block - 1) / block;
    estimate_affine_opencorr_kernel<<<grid, block, 0, stream>>>(
        d_ref_kp, d_tar_kp,
        d_neighbor_idx_, d_neighbor_dist_, max_neighbor,
        d_pois, N, d_param, 123456789ULL
    );

    cudaMemcpyAsync(pois, d_pois, N*sizeof(CudaPOI2D), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_pois);
}

void SiftAffineBatchGpu::release_cuda() {
    if (d_ref_kp) cudaFree(d_ref_kp);
    if (d_tar_kp) cudaFree(d_tar_kp);
    if (d_neighbor_idx_) cudaFree(d_neighbor_idx_);
    if (d_neighbor_dist_) cudaFree(d_neighbor_dist_);
    d_ref_kp = d_tar_kp = nullptr;
    d_neighbor_idx_ = nullptr;
    d_neighbor_dist_ = nullptr;
}

} // namespace StudyCorr