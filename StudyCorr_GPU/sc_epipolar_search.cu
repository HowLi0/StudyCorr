#include "sc_epipolar_search.h"

namespace StudyCorr_GPU
{ 

    __device__ float ComputeZNCC(const float* refImg, const float* tarImg, int width, int height,
                    float u0, float v0, float ur, float vr, int subsetRadius)
    {
        float sumRef = 0, sumTar = 0;
        float sumRef2 = 0, sumTar2 = 0;
        float sumCross = 0;
        int count = 0;

        for (int dy = -subsetRadius; dy <= subsetRadius; ++dy) {
            for (int dx = -subsetRadius; dx <= subsetRadius; ++dx) {
                float xl = u0 + dx;
                float yl = v0 + dy;
                float xr = ur + dx;
                float yr = vr + dy;

                float ref = BilinearInterpolation(xl, yl, refImg, width, height);
                float tar = BilinearInterpolation(xr, yr, tarImg, width, height);

                sumRef += ref;
                sumTar += tar;
                sumRef2 += ref * ref;
                sumTar2 += tar * tar;
                sumCross += ref * tar;
                count++;
            }
        }

        float meanRef = sumRef / count;
        float meanTar = sumTar / count;
        float denom = sqrtf((sumRef2 - count * meanRef * meanRef) *
                            (sumTar2 - count * meanTar * meanTar));
        if (denom < 1e-5f) return -2.0f;

        return (sumCross - count * meanRef * meanTar) / denom;
    }


    __global__ void zncc_epipolar_search_kernel(
        const float* refImg, const float* tarImg,
        int width, int height,
        const float* F, // 基础矩阵
        CudaPOI2D* pois, int subsetRadius,
        int searchLength,int N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;

        CudaPOI2D& poi = pois[idx];
        float u0 = poi.x, v0 = poi.y;

        float bestZNCC = -2.0f;
        float best_du = 0.0f, best_dv = 0.0f;

        // 计算极线
        float a = F[0]*u0 + F[1]*v0 + F[2];
        float b = F[3]*u0 + F[4]*v0 + F[5];
        float c = F[6]*u0 + F[7]*v0 + F[8];
        float norm = sqrtf(a*a + b*b);
        a /= norm; b /= norm; c /= norm;

        for (int d = -searchLength; d <= searchLength; ++d) {
            float ur = u0 - b * d;
            float vr = v0 + a * d;
            if (ur - subsetRadius < 1 || ur + subsetRadius >= width - 1 ||
                vr - subsetRadius < 1 || vr + subsetRadius >= height - 1)
                continue;

            float zncc = ComputeZNCC(refImg, tarImg, width, height,
                                        u0, v0, ur, vr, subsetRadius);

            if (zncc > bestZNCC) {
                bestZNCC = zncc;
                best_du = ur - u0;
                best_dv = vr - v0;
            }
        }

        poi.deformation.u = best_du;
        poi.deformation.v = best_dv;
        poi.result.zncc = bestZNCC;
    }



    EpipolarSearchGpu::EpipolarSearchGpu(int subsetRadius, int search_step, cv::Mat F):
        subset_radius(subsetRadius), search_step(search_step)
    {
        cv::Mat F32;
        F.convertTo(F32, CV_32F); // 转为 float32
        memcpy(h_F, F32.ptr<float>(), sizeof(float) * 9);
    }

    EpipolarSearchGpu::~EpipolarSearchGpu() 
    {
        release_cuda();
    }

    void EpipolarSearchGpu::prepare_cuda(const float* ref_image, const float* tar_image, int height, int width, cudaStream_t stream)
    {
        this->height = height;
        this->width = width;

        // 先释放之前分配的内存（防止内存泄漏）
        release_cuda();

        // 分配设备内存
        cudaMalloc(&d_ref_image, height * width * sizeof(float));
        cudaMalloc(&d_tar_image, height * width * sizeof(float));
        cudaMalloc(&d_F, 9 * sizeof(float)); // 基础矩阵是3x3的

        // 将图像数据从主机复制到设备
        cudaMemcpyAsync(d_ref_image, ref_image, height * width * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_tar_image, tar_image, height * width * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_F, h_F, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    void EpipolarSearchGpu::compute_batch_cuda(CudaPOI2D* pois, int N, cudaStream_t stream)
    {
        // 分配设备内存
        CudaPOI2D* d_pois;
        cudaMalloc(&d_pois, N * sizeof(CudaPOI2D));
        cudaMemcpyAsync(d_pois, pois, N * sizeof(CudaPOI2D), cudaMemcpyHostToDevice, stream);

        // 计算网格和块的维度
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;

        // 启动核函数
        zncc_epipolar_search_kernel<<<numBlocks, blockSize, 0, stream>>>(
            d_ref_image, d_tar_image,
            width, height,
            d_F,
            d_pois, subset_radius,
            search_step, N);

        // 将结果从设备复制回主机
        cudaMemcpyAsync(pois, d_pois, N * sizeof(CudaPOI2D), cudaMemcpyDeviceToHost, stream);

        // 释放本 batch 局部内存
        cudaFree(d_pois); // 只释放 d_pois，d_ref_image/d_tar_image/d_F 由类析构/prepare/release 统一管理
    }

    void EpipolarSearchGpu::release_cuda()
    {
        if (d_ref_image) {
            cudaFree(d_ref_image);
            d_ref_image = nullptr;
        }
        if (d_tar_image) {
            cudaFree(d_tar_image);
            d_tar_image = nullptr;
        }
        if (d_F) {
            cudaFree(d_F);
            d_F = nullptr;
        }
    }

}
