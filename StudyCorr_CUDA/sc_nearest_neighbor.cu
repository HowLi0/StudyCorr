#include "sc_nearest_neighbor.h"
#include <cassert>

namespace StudyCorr
{

// 2D半径搜索
__global__ void radius_search_2d_kernel(
    const Point2D* ref_pts, int n_ref,
    const Point2D* queries, int n_query,
    float radius,
    int* out_counts,
    int* out_indices,
    int max_neighbors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_query) return;
    Point2D q = queries[idx];
    int count = 0;
    for (int j = 0; j < n_ref; ++j) {
        float dx = ref_pts[j].x - q.x;
        float dy = ref_pts[j].y - q.y;
        if (dx*dx + dy*dy <= radius*radius) {
            if (count < max_neighbors)
                out_indices[idx*max_neighbors+count] = j;
            ++count;
        }
    }
    out_counts[idx] = count > max_neighbors ? max_neighbors : count;
}

void NearestNeighbor::radiusSearch2D(
        const Array1D<Point2D>& ref_pts,
        const Array1D<Point2D>& queries,
        float radius,
        Array1D<int>& out_counts,
        Array1D<int>& out_indices,
        int max_neighbors)
{
    int n_ref = int(ref_pts.dim());
    int n_query = int(queries.dim());
    out_counts = Array1D<int>(n_query);
    out_indices = Array1D<int>(n_query*max_neighbors);

    int block = 256, grid = (n_query+block-1)/block;
    radius_search_2d_kernel<<<grid, block>>>(
        ref_pts.data, n_ref, queries.data, n_query, radius,
        out_counts.data, out_indices.data, max_neighbors);
    cudaDeviceSynchronize();
}

// 3D半径搜索
__global__ void radius_search_3d_kernel(
    const Point3D* ref_pts, int n_ref,
    const Point3D* queries, int n_query,
    float radius,
    int* out_counts,
    int* out_indices,
    int max_neighbors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_query) return;
    Point3D q = queries[idx];
    int count = 0;
    for (int j = 0; j < n_ref; ++j) {
        float dx = ref_pts[j].x - q.x;
        float dy = ref_pts[j].y - q.y;
        float dz = ref_pts[j].z - q.z;
        if (dx*dx + dy*dy + dz*dz <= radius*radius) {
            if (count < max_neighbors)
                out_indices[idx*max_neighbors+count] = j;
            ++count;
        }
    }
    out_counts[idx] = count > max_neighbors ? max_neighbors : count;
}

void NearestNeighbor::radiusSearch3D(
        const Array1D<Point3D>& ref_pts,
        const Array1D<Point3D>& queries,
        float radius,
        Array1D<int>& out_counts,
        Array1D<int>& out_indices,
        int max_neighbors)
{
    int n_ref = int(ref_pts.dim());
    int n_query = int(queries.dim());
    out_counts = Array1D<int>(n_query);
    out_indices = Array1D<int>(n_query*max_neighbors);

    int block = 256, grid = (n_query+block-1)/block;
    radius_search_3d_kernel<<<grid, block>>>(
        ref_pts.data, n_ref, queries.data, n_query, radius,
        out_counts.data, out_indices.data, max_neighbors);
    cudaDeviceSynchronize();
}

// 2D KNN搜索
__global__ void knn_search_2d_kernel(
    const Point2D* ref_pts, int n_ref,
    const Point2D* queries, int n_query,
    int K,
    int* out_indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_query) return;
    Point2D q = queries[idx];
    float best_dist[16];
    int best_idx[16];
    for (int k = 0; k < K; ++k) best_dist[k] = 1e20f, best_idx[k] = -1;
    for (int j = 0; j < n_ref; ++j) {
        float dx = ref_pts[j].x - q.x;
        float dy = ref_pts[j].y - q.y;
        float d2 = dx*dx + dy*dy;
        for (int k = 0; k < K; ++k) {
            if (d2 < best_dist[k]) {
                for (int m = K-1; m > k; --m) {
                    best_dist[m] = best_dist[m-1];
                    best_idx[m]  = best_idx[m-1];
                }
                best_dist[k] = d2;
                best_idx[k]  = j;
                break;
            }
        }
    }
    for (int k = 0; k < K; ++k)
        out_indices[idx*K + k] = best_idx[k];
}

void NearestNeighbor::knnSearch2D(
        const Array1D<Point2D>& ref_pts,
        const Array1D<Point2D>& queries,
        int K,
        Array1D<int>& out_indices)
{
    int n_ref = int(ref_pts.dim());
    int n_query = int(queries.dim());
    out_indices = Array1D<int>(n_query*K);

    int block = 256, grid = (n_query+block-1)/block;
    knn_search_2d_kernel<<<grid, block>>>(
        ref_pts.data, n_ref, queries.data, n_query, K, out_indices.data);
    cudaDeviceSynchronize();
}

// 3D KNN搜索
__global__ void knn_search_3d_kernel(
    const Point3D* ref_pts, int n_ref,
    const Point3D* queries, int n_query,
    int K,
    int* out_indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_query) return;
    Point3D q = queries[idx];
    float best_dist[16];
    int best_idx[16];
    for (int k = 0; k < K; ++k) best_dist[k] = 1e20f, best_idx[k] = -1;
    for (int j = 0; j < n_ref; ++j) {
        float dx = ref_pts[j].x - q.x;
        float dy = ref_pts[j].y - q.y;
        float dz = ref_pts[j].z - q.z;
        float d2 = dx*dx + dy*dy + dz*dz;
        for (int k = 0; k < K; ++k) {
            if (d2 < best_dist[k]) {
                for (int m = K-1; m > k; --m) {
                    best_dist[m] = best_dist[m-1];
                    best_idx[m]  = best_idx[m-1];
                }
                best_dist[k] = d2;
                best_idx[k]  = j;
                break;
            }
        }
    }
    for (int k = 0; k < K; ++k)
        out_indices[idx*K + k] = best_idx[k];
}

void NearestNeighbor::knnSearch3D(
        const Array1D<Point3D>& ref_pts,
        const Array1D<Point3D>& queries,
        int K,
        Array1D<int>& out_indices)
{
    int n_ref = int(ref_pts.dim());
    int n_query = int(queries.dim());
    out_indices = Array1D<int>(n_query*K);

    int block = 256, grid = (n_query+block-1)/block;
    knn_search_3d_kernel<<<grid, block>>>(
        ref_pts.data, n_ref, queries.data, n_query, K, out_indices.data);
    cudaDeviceSynchronize();
}

} // namespace StudyCorr