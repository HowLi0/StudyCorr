#pragma once
#include"sc_subset.h"

namespace StudyCorr 
{

    // CUDA kernel for 2D fill
    __global__ void subset2d_fill_kernel(
        float* subset_data, int s_height, int s_width,
        const float* img_data, int img_height, int img_width,
        int x0, int y0, int img_x)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < s_height && j < s_width) {
            int y = y0 + i, x = x0 + j;
            if(y >= 0 && y < img_height && x >= 0 && x < img_width)
                subset_data[i * s_width + j] = img_data[y * img_x + x];
            else
                subset_data[i * s_width + j] = 0.0f;
        }
    }

    // CUDA kernel for 2D zero mean norm (reduction)
    __global__ void subset2d_reduce_sum(const float* data, int n, float* sum_out) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        float mySum = 0;
        if(idx < n) mySum = data[idx];
        sdata[tid] = mySum;
        __syncthreads();
        // reduction
        for(int s = blockDim.x / 2; s > 0; s >>= 1) {
            if(tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if(tid == 0) atomicAdd(sum_out, sdata[0]);
    }

    __global__ void subset2d_sub_mean_and_sq(float* data, float mean, int n, float* sumsq_out) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < n) {
            data[idx] -= mean;
            float v = data[idx];
            atomicAdd(sumsq_out, v * v);
        }
    }

    void CudaSubset2D::fill(const CudaImage2D& image) {
            int x0 = int(center.x) - radius_x;
            int y0 = int(center.y) - radius_y;
            dim3 block(16, 16);
            dim3 grid((width + 15) / 16, (height + 15) / 16);
            subset2d_fill_kernel<<<grid, block>>>(
                data.get(), height, width,
                image.data.get(), image.height, image.width,
                x0, y0, image.data.dim2());
            cudaDeviceSynchronize();
        }

    float CudaSubset2D::zeroMeanNorm() {
        int n = width * height;
        float* d_sum = nullptr;
        cudaMallocManaged(&d_sum, sizeof(float));
        *d_sum = 0.0f;
        // 1. 求和
        int block = 256, grid = (n + block - 1) / block;
        subset2d_reduce_sum<<<grid, block, block * sizeof(float)>>>(data.get(), n, d_sum);
        cudaDeviceSynchronize();
        float mean = *d_sum / n;
        *d_sum = 0.0f;
        // 2. 减均值并平方和
        subset2d_sub_mean_and_sq<<<grid, block>>>(data.get(), mean, n, d_sum);
        cudaDeviceSynchronize();
        float result = std::sqrt(*d_sum);
        cudaFree(d_sum);
        return result;
    }

    __global__ void subset3d_fill_kernel(
        float* subset_data, int dim_x, int dim_y, int dim_z,
        const float* img_data, int img_height, int img_width, int img_depth,
        int x0, int y0, int z0, int img_y)
    {
        int i = blockIdx.z * blockDim.z + threadIdx.z;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < dim_z && j < dim_y && k < dim_x) {
            int x=x0 + k, y=y0 + j, z=z0 + i;
            if(y >= 0 && y < img_height && x >= 0 && x < img_width && z >= 0 && z < img_depth)
                subset_data[(i * dim_y + j) * dim_x + k] = img_data[(z * img_y + y) * img_width + x];
            else
                subset_data[(i * dim_y + j) * dim_x + k] = 0.0f;
        }
    }

    __global__ void subset3d_reduce_sum(const float* data, int n, float* sum_out) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        float mySum = 0;
        if(idx < n) mySum = data[idx];
        sdata[tid] = mySum;
        __syncthreads();
        // reduction
        for(int s = blockDim.x / 2; s > 0; s >>= 1) {
            if(tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if(tid == 0) atomicAdd(sum_out, sdata[0]);
    }

    __global__ void subset3d_sub_mean_and_sq(float* data, float mean, int n, float* sumsq_out) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < n) {
            data[idx] -= mean;
            float v = data[idx];
            atomicAdd(sumsq_out, v * v);
        }
    }

    void CudaSubset3D::fill(const CudaImage3D& image) {
        int x0 = center.x - radius_x;
        int y0 = center.y - radius_y;
        int z0 = center.z - radius_z;
        dim3 block(8, 8, 8);
        dim3 grid((dim_x + 7) / 8, (dim_y + 7) / 8, (dim_z + 7) / 8);
        subset3d_fill_kernel<<<grid, block>>>(
            data.get(), dim_x, dim_y, dim_z,
            image.data.get(), image.dim_x, image.dim_y, image.dim_z,
            x0, y0, z0, image.data.dim2());
        cudaDeviceSynchronize();
    }

    float CudaSubset3D::zeroMeanNorm() {
        int n = dim_x * dim_y * dim_z;
        float* d_sum = nullptr;
        cudaMallocManaged(&d_sum, sizeof(float));
        *d_sum = 0.0f;
        // 1. 求和
        int block = 256, grid = (n + block - 1) / block;
        subset3d_reduce_sum<<<grid, block, block * sizeof(float)>>>(data.get(), n, d_sum);
        cudaDeviceSynchronize();
        float mean = *d_sum / n;
        *d_sum = 0.0f;
        // 2. 减均值并平方和
        subset3d_sub_mean_and_sq<<<grid, block>>>(data.get(), mean, n, d_sum);
        cudaDeviceSynchronize();
        float result = std::sqrt(*d_sum);
        cudaFree(d_sum);
        return result;
    }
    

} // namespace StudyCorr