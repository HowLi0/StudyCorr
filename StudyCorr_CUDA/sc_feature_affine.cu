#include "sc_feature_affine.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <vector>

namespace StudyCorr
{

// CUDA实现Householder QR分解，返回与Eigen等价的仿射最小二乘解
__device__ bool householder_affine_fit_qr(const Point2D* src, const Point2D* dst, int n, float aff[6])
{
    if (n < 3) return false;
    float A[64][3]; // 支持每POI最多64邻居，max_neighbors建议不超过64
    float B[64][2];
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        A[i][0] = src[i].x;
        A[i][1] = src[i].y;
        A[i][2] = 1.f;
        B[i][0] = dst[i].x;
        B[i][1] = dst[i].y;
    }
    float R[3][3] = {0};
    float QtB[3][2] = {0};
    for (int k = 0; k < 3; ++k) {
        float norm_x = 0.f;
        for (int i = k; i < n; ++i)
            norm_x += A[i][k] * A[i][k];
        norm_x = sqrtf(norm_x);
        if (norm_x < 1e-7f) return false;
        float sign = (A[k][k] >= 0) ? 1.f : -1.f;
        float u1 = A[k][k] + sign * norm_x;
        float beta = 1.f / (norm_x * (norm_x + fabsf(A[k][k])));
        float v[64] = {0};
        v[k] = u1;
        for (int i = k + 1; i < n; ++i)
            v[i] = A[i][k];
        for (int j = k; j < 3; ++j) {
            float dot = 0.f;
            for (int i = k; i < n; ++i) dot += v[i] * A[i][j];
            dot *= beta;
            for (int i = k; i < n; ++i) A[i][j] -= dot * v[i];
        }
        for (int j = 0; j < 2; ++j) {
            float dot = 0.f;
            for (int i = k; i < n; ++i) dot += v[i] * B[i][j];
            dot *= beta;
            for (int i = k; i < n; ++i) B[i][j] -= dot * v[i];
        }
        R[k][k] = A[k][k];
        for (int j = k + 1; j < 3; ++j)
            R[k][j] = A[k][j];
        QtB[k][0] = B[k][0];
        QtB[k][1] = B[k][1];
    }
    for (int j = 0; j < 2; ++j) {
        float x[3];
        for (int k = 2; k >= 0; --k) {
            x[k] = QtB[k][j];
            for (int t = k + 1; t < 3; ++t)
                x[k] -= R[k][t] * x[t];
            x[k] /= R[k][k];
        }
        aff[j * 3 + 0] = x[0];
        aff[j * 3 + 1] = x[1];
        aff[j * 3 + 2] = x[2];
    }
    return true;
}

// CUDA主核：每POI做RANSAC+仿射估计，完全CUDA并行
__global__ void cuda_affine_ransac_kernel(
    const Point2D* ref_pts, const Point2D* tar_pts,
    const int* nb_counts, const int* nb_indices,
    CudaPOI2D* pois, int n_poi, int max_neighbors,
    int ransac_trials, float inlier_thresh, int min_neighbor_num)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_poi) return;
    int nb = nb_counts[i];
    if (nb < min_neighbor_num) {
        pois[i].result.zncc = -1;
        return;
    }
    Point2D src[64], dst[64];
    Point2D poi = pois[i].location;
    for (int j = 0; j < nb; ++j) {
        int idx = nb_indices[i * max_neighbors + j];
        src[j] = ref_pts[idx] - poi;
        dst[j] = tar_pts[idx] - poi;
    }
    int best_inlier = 0;
    float best_aff[6] = {0};
    float best_mean_err = 1e9;
    curandState state;
    curand_init(1234ULL + i, 0, 0, &state);
    int idxs[64];
    for (int j = 0; j < nb; ++j) idxs[j] = j;
    int trial_counter = 0;
    do {
        for (int j = nb - 1; j > 0; --j) {
            int k = curand(&state) % (j + 1);
            int tmp = idxs[j]; idxs[j] = idxs[k]; idxs[k] = tmp;
        }
        Point2D s_src[3], s_dst[3];
        for (int k = 0; k < 3; ++k) {
            int idx = idxs[k];
            s_src[k] = src[idx];
            s_dst[k] = dst[idx];
        }
        float aff[6];
        if (!householder_affine_fit_qr(s_src, s_dst, 3, aff)) continue;
        int inlier = 0;
        float err_sum = 0.f;
        for (int j = 0; j < nb; ++j) {
            float xp = aff[0] * src[j].x + aff[1] * src[j].y + aff[2];
            float yp = aff[3] * src[j].x + aff[4] * src[j].y + aff[5];
            float err = hypotf(xp - dst[j].x, yp - dst[j].y);
            if (err < inlier_thresh) { ++inlier; err_sum += err; }
        }
        float mean_err = (inlier > 0) ? (err_sum / inlier) : 1e9;
        if ((inlier > best_inlier) || (inlier == best_inlier && mean_err < best_mean_err)) {
            best_inlier = inlier;
            best_mean_err = mean_err;
            for (int k = 0; k < 6; ++k) best_aff[k] = aff[k];
        }
        ++trial_counter;
    } while (trial_counter < ransac_trials &&
             (best_inlier < min_neighbor_num || best_mean_err > inlier_thresh / min_neighbor_num));
    Point2D inl_src[64], inl_dst[64];
    int n_inl = 0;
    for (int j = 0; j < nb; ++j) {
        float xp = best_aff[0] * src[j].x + best_aff[1] * src[j].y + best_aff[2];
        float yp = best_aff[3] * src[j].x + best_aff[4] * src[j].y + best_aff[5];
        float err = hypotf(xp - dst[j].x, yp - dst[j].y);
        if (err < inlier_thresh) {
            inl_src[n_inl] = src[j];
            inl_dst[n_inl] = dst[j];
            ++n_inl;
        }
    }
    float final_aff[6];
    if (n_inl < 3 || !householder_affine_fit_qr(inl_src, inl_dst, n_inl, final_aff)) {
        pois[i].result.zncc = -2;
        return;
    }
    pois[i].deformation.u  = final_aff[2];
    pois[i].deformation.ux = final_aff[0] - 1.f;
    pois[i].deformation.uy = final_aff[1];
    pois[i].deformation.v  = final_aff[5];
    pois[i].deformation.vx = final_aff[3];
    pois[i].deformation.vy = final_aff[4] - 1.f;
    pois[i].result.feature = float(n_inl);
    pois[i].result.zncc = 0.f;
    pois[i].result.iteration = float(trial_counter);
}

void FeatureAffine2D::compute(Array2D<CudaPOI2D>& poi_queue)
{
    int nrows = int(poi_queue.dim1());
    int ncols = int(poi_queue.dim2());
    int n_poi = nrows * ncols;
    Array1D<CudaPOI2D> flat_pois(n_poi);
    for (int i = 0; i < nrows; ++i)
        for (int j = 0; j < ncols; ++j)
            flat_pois[i * ncols + j] = poi_queue(i, j);

    Array1D<Point2D> poi_locs(n_poi);
    for (int i = 0; i < n_poi; ++i) poi_locs[i] = flat_pois[i].location;

    Array1D<int> neighbor_counts, neighbor_indices;
    // 1. GPU半径查找
    NearestNeighbor::radiusSearch2D(
        ref_pts_, poi_locs, cfg_.search_radius,
        neighbor_counts, neighbor_indices, cfg_.max_neighbors);

    // 2. 用KNN查找补齐不足的POI（也在GPU上）
    for (int i = 0; i < n_poi; ++i) {
        if (neighbor_counts[i] < cfg_.min_neighbor_num) {
            Array1D<int> knn_indices;
            NearestNeighbor::knnSearch2D(ref_pts_, Array1D<Point2D>(1, poi_locs[i]), cfg_.min_neighbor_num, knn_indices);
            for (int k = 0; k < cfg_.min_neighbor_num; ++k)
                neighbor_indices[i * cfg_.max_neighbors + k] = knn_indices[k];
            neighbor_counts[i] = cfg_.min_neighbor_num;
        }
    }

    // 3. CUDA并行RANSAC主流程
    cuda_affine_ransac(
        ref_pts_, tar_pts_,
        neighbor_counts, neighbor_indices,
        flat_pois.data, n_poi, cfg_.max_neighbors,
        cfg_.ransac_trials, cfg_.inlier_thresh, cfg_.min_neighbor_num);

    for (int i = 0; i < nrows; ++i)
        for (int j = 0; j < ncols; ++j)
            poi_queue(i, j) = flat_pois[i * ncols + j];
}

void FeatureAffine2D::cuda_affine_ransac(
        const Array1D<Point2D>& ref_pts,
        const Array1D<Point2D>& tar_pts,
        const Array1D<int>& nb_counts,
        const Array1D<int>& nb_indices,
        CudaPOI2D* pois, int n_poi, int max_neighbors,
        int ransac_trials, float inlier_thresh, int min_neighbor_num)
{
    int block = 256, grid = (n_poi + block - 1) / block;
    cuda_affine_ransac_kernel<<<grid, block>>>(
        ref_pts.data, tar_pts.data,
        nb_counts.data, nb_indices.data,
        pois, n_poi, max_neighbors,
        ransac_trials, inlier_thresh, min_neighbor_num);
    cudaDeviceSynchronize();
}

void FeatureAffine2D::setKeypoints(const Array1D<Point2D>& ref, const Array1D<Point2D>& tar) {
    ref_pts_ = ref;
    tar_pts_ = tar;
    n_ref_ = int(ref.dim());
}
void FeatureAffine2D::setAffineConfig(const AffineRansacConfig& cfg) { cfg_ = cfg; }
void FeatureAffine2D::setSearchParameters(float radius, int min_neighbor_num) {
    cfg_.search_radius = radius;
    cfg_.min_neighbor_num = min_neighbor_num;
}

} // namespace StudyCorr