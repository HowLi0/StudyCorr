#include "sc_icgn.h"


namespace StudyCorr_GPU {

__global__ void compute_gradients_3d(const float* image, int depth, int height, int width,
                                     float* grad_x, float* grad_y, float* grad_z) {
    for (int z = 1; z < depth - 1; ++z)
        for (int y = 1; y < height - 1; ++y)
            for (int x = 1; x < width - 1; ++x) {
                int idx = z * height * width + y * width + x;
                grad_x[idx] =
                    -image[idx - width - height*width] - 2*image[idx - height*width] - image[idx + width - height*width]
                    +image[idx - width + height*width] + 2*image[idx + height*width] + image[idx + width + height*width];
                grad_x[idx] /= 8.0f;
                grad_y[idx] =
                    -image[idx - 1 - height*width] - 2*image[idx - height*width] - image[idx + 1 - height*width]
                    +image[idx - 1 + height*width] + 2*image[idx + height*width] + image[idx + 1 + height*width];
                grad_y[idx] /= 8.0f;
                grad_z[idx] =
                    -image[idx - width - 1] - 2*image[idx - 1] - image[idx + width - 1]
                    +image[idx - width + 1] + 2*image[idx + 1] + image[idx + width + 1];
                grad_z[idx] /= 8.0f;
            }
}
/****************************************** ICGN2D **************************************************/

__global__ void icgn2d_batch_kernel(
    const float* ref_image, const float* tar_image,
    int height, int width,
    CudaPOI2D* pois, int subsetRadius,
    double convergenceThreshold, int maxIterations,
    int N,float numParams)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    CudaPOI2D& poi = pois[idx];
    float center_y = poi.y;
    float center_x = poi.x;

    // 边界检测（更严格：整个子集必须都在图像内部）
    if (center_x - subsetRadius < 1 || center_x + subsetRadius >= width - 1 ||
        center_y - subsetRadius < 1 || center_y + subsetRadius >= height - 1) {
        poi.result.zncc = -4.f; // 无效
        poi.result.u0 = 0.f;
        poi.result.v0 = 0.f;
        poi.result.iteration = 0;
        poi.result.convergence = 0.f;
        // deformation参数清零
        poi.deformation.u = poi.deformation.ux = poi.deformation.uy = poi.deformation.v = poi.deformation.vx = poi.deformation.vy = 0.f;
        // 应变清零
        poi.strain.exx = poi.strain.eyy = poi.strain.exy = 0.f;
        return;
    }

    // 初始化形函数参数（从poi.deformation赋值）
    float warpParams[12] = {0.0f};
    poi.result.u0 = poi.deformation.u;
    poi.result.v0 = poi.deformation.v;
    if (numParams >= 6) 
    {
        warpParams[0] = poi.deformation.u; // u
        warpParams[1] = poi.deformation.v; // v
        warpParams[2] = poi.deformation.ux; // ux
        warpParams[3] = poi.deformation.uy; // uy
        warpParams[4] = poi.deformation.vx; // vx
        warpParams[5] = poi.deformation.vy; // vy
    }
    if (numParams >= 12)
    {
        warpParams[6] = poi.deformation.uxx; // uxx
        warpParams[7] = poi.deformation.uxy; // uxy
        warpParams[8] = poi.deformation.uyy; // uyy
        warpParams[9] = poi.deformation.vxx; // vxx
        warpParams[10] = poi.deformation.vxy; // vxy
        warpParams[11] = poi.deformation.vyy; // vyy
    }
    


    float hessian[144] = {0.0f};
    computehessian(ref_image, subsetRadius, height, width, center_y, center_x, numParams, hessian);

    // ICGN迭代优化，改进的收敛条件
    float prevZNCC = -4.0f;
    bool converged = false;
    int iter = 0;
    // 迭代优化
    for (iter = 0; iter < maxIterations && !converged; iter++) {
            // 合并计算ZNCC和误差向量
            ZNCCAndErrorResult znccError = computeZNCCAndError(
                ref_image, tar_image, center_y, center_x, warpParams,
                height, width, subsetRadius, numParams
            );
            float currentZNCC = znccError.zncc;
            float* errorVector = znccError.errorVector;

            bool znccConverged = (iter > 0) && (fabs(currentZNCC - prevZNCC) < convergenceThreshold);
            if (znccConverged) {
                converged = true;
                poi.result.zncc = currentZNCC;
                break;
            }
            prevZNCC = currentZNCC;

            float deltaP[12] = {0};
            // deltaP = H^{-1} * errorVector
            bool solved = solveLinearSystem(hessian, errorVector, deltaP, numParams); // Δp = H^{-1} * b
            if (!solved) {
                poi.result.zncc = currentZNCC;
                converged = (currentZNCC < 0.5);
                break;
            }
            for (int p = 0; p < numParams; p++) {
                warpParams[p] += deltaP[p];
            }// 更新形函数参数
            float deltaNorm = 0.0f;
            for (int p = 0; p < numParams; p++) {
                deltaNorm += deltaP[p] * deltaP[p];
            }// 计算参数更新的L2范数
            if (sqrt(deltaNorm) < convergenceThreshold * 10.0f) {
                converged = true;
                poi.result.zncc = currentZNCC;
            }// 如果参数更新小于阈值，则认为收敛
        }

        if (!converged && iter >= maxIterations) {
            ZNCCAndErrorResult znccError = computeZNCCAndError(
                ref_image, tar_image, center_y, center_x, warpParams,
                height, width, subsetRadius, numParams
            );
            poi.result.zncc = znccError.zncc;
            converged = (znccError.zncc > 0.8f); // 如果ZNCC大于0.8则认为收敛
        }

        poi.result.convergence = converged ? 1.0f : 0.0f;
        poi.result.iteration = iter;
        poi.deformation.u = warpParams[0];
        poi.deformation.v = warpParams[1];
        poi.deformation.ux = warpParams[2];
        poi.deformation.uy = warpParams[3];
        poi.deformation.vx = warpParams[4];
        poi.deformation.vy = warpParams[5];
        if (numParams >= 12) {
            poi.deformation.uxx = warpParams[6];
            poi.deformation.uxy = warpParams[7];
            poi.deformation.uyy = warpParams[8];
            poi.deformation.vxx = warpParams[9];
            poi.deformation.vxy = warpParams[10];
            poi.deformation.vyy = warpParams[11];
        }
}
    
/****************************************** ICGN3D1 **************************************************/

// ICGN3D1核
__global__ void icgn3d1_batch_kernel(
    const float* ref_image, const float* tar_image,
    int depth, int height, int width,
    const CudaPOI3D* pois, int subsetRadius,
    float convergenceThreshold, int maxIterations,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    CudaPOI3D poi = pois[idx];

    float center_z = poi.z;
    float center_y = poi.y;
    float center_x = poi.x;

    // 边界严格处理
    if (center_x - subsetRadius < 1 || center_x + subsetRadius >= width - 1 ||
        center_y - subsetRadius < 1 || center_y + subsetRadius >= height - 1 ||
        center_z - subsetRadius < 1 || center_z + subsetRadius >= depth - 1) {
        poi.result.zncc = -1.f;
        poi.result.u0 = 0.f;
        poi.result.v0 = 0.f;
        poi.result.w0 = 0.f;
        poi.result.iteration = 0;
        poi.result.convergence = 0;
        return;
    }

    // 初始化形函数参数
    float warpParams[12] = {0.0f};
    poi.result.u0 = poi.deformation.u;
    poi.result.v0 = poi.deformation.v;
    poi.result.w0 = poi.deformation.w;

    warpParams[0] = poi.deformation.u; // u
    warpParams[1] = poi.deformation.v; // v
    warpParams[2] = poi.deformation.w; // w
    warpParams[3] = poi.deformation.ux; // ux
    warpParams[4] = poi.deformation.uy; // uy
    warpParams[5] = poi.deformation.uz; // uz
    warpParams[6] = poi.deformation.vx; // vx
    warpParams[7] = poi.deformation.vy; // vy
    warpParams[8] = poi.deformation.vz; // vz
    warpParams[9] = poi.deformation.wx; // wx
    warpParams[10] = poi.deformation.wy; // wy
    warpParams[11] = poi.deformation.wz; // wz

    float hessian[144] = {0.0f};
    computehessian3d(ref_image,subsetRadius,width,height, depth,
    center_x, center_y, center_z,12, hessian);

    // ICGN迭代优化，改进的收敛条件
    float prevZNCC = -4.0f;
    bool converged = false;
    int iter = 0;

    for (iter = 0; iter < maxIterations && !converged; iter++) {
        // 计算当前的ZNCC
        ZNCCAndErrorResult znccError = computeZNCCAndError3D(
            ref_image, tar_image, center_z, center_y, center_x, warpParams,
            depth, height, width, subsetRadius);

        float currentZNCC = znccError.zncc;
        float* errorVector = znccError.errorVector;

        bool znccConverged = (iter > 0) && (fabs(currentZNCC - prevZNCC) < convergenceThreshold);
        if (znccConverged) {
            converged = true;
            poi.result.zncc = currentZNCC;
            break;
        }

        prevZNCC = currentZNCC;

        float deltaP[12] = {0};
        // 解线性方程组 H * Δp = b
        bool solved = solveLinearSystem(hessian, errorVector, deltaP, 12);
        if (!solved) {
            poi.result.zncc = currentZNCC;
            converged = (currentZNCC < 0.5);
            break;
        }
        for (int p = 0; p < 12; p++) {
            warpParams[p] += deltaP[p];
        } // 更新形函数参数
        float deltaNorm = 0.0f;
        for (int p = 0; p < 12; p++) {
            deltaNorm += deltaP[p] * deltaP[p];
        } // 计算参数更新的L2范数
        if (sqrt(deltaNorm) < convergenceThreshold * 10.0f) {
            converged = true;
            poi.result.zncc = currentZNCC;
        } // 如果参数更新小于阈值，则认为收敛

        if (!converged && iter >= maxIterations) {
            ZNCCAndErrorResult znccError = computeZNCCAndError3D(
                ref_image, tar_image, center_z, center_y, center_x, warpParams,
                depth, height, width, subsetRadius);
            poi.result.zncc = znccError.zncc;
            converged = (znccError.zncc > 0.8f); // 如果ZNCC大于0.8则认为收敛
        }

        poi.result.convergence = converged ? 1.0f : 0.0f;
        poi.result.iteration = iter;
        poi.deformation.u = warpParams[0];
        poi.deformation.v = warpParams[1];
        poi.deformation.w = warpParams[2];
        poi.deformation.ux = warpParams[3];
        poi.deformation.uy = warpParams[4];
        poi.deformation.uz = warpParams[5];
        poi.deformation.vx = warpParams[6];
        poi.deformation.vy = warpParams[7];
        poi.deformation.vz = warpParams[8];
        poi.deformation.wx = warpParams[9];
        poi.deformation.wy = warpParams[10];
        poi.deformation.wz = warpParams[11];
    }

}

/****************************************** C++接口实现 **************************************************/

ICGN2D1BatchGpu::ICGN2D1BatchGpu() : d_ref_image(nullptr), d_tar_image(nullptr) {}
ICGN2D1BatchGpu::~ICGN2D1BatchGpu() { release_cuda(); }

void ICGN2D1BatchGpu::prepare_cuda(const float* ref_image, const float* tar_image, int h, int w, const ICGNParam& param_, cudaStream_t stream) {
    height = h; width = w; param = param_;
    size_t bytes = h * w * sizeof(float);
    cudaMalloc(&d_ref_image, bytes);
    cudaMalloc(&d_tar_image, bytes);
    cudaMemcpy(d_ref_image, ref_image, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tar_image, tar_image, bytes, cudaMemcpyHostToDevice);
    cudaStreamSynchronize(stream);
}

void ICGN2D1BatchGpu::compute_batch_cuda(CudaPOI2D* pois, int N, cudaStream_t stream) {
    CudaPOI2D* d_pois;
    cudaMalloc(&d_pois, N * sizeof(CudaPOI2D));
    cudaMemcpyAsync(d_pois, pois, N * sizeof(CudaPOI2D), cudaMemcpyHostToDevice, stream);

    int block = 128, grid = (N + block - 1) / block;
    icgn2d_batch_kernel<<<grid, block, 0, stream>>>(
        d_ref_image, d_tar_image,
        height, width, d_pois,
        param.subsetRadius, param.convergenceThreshold, param.maxIterations, N, 6);

    cudaMemcpyAsync(pois, d_pois, N * sizeof(CudaPOI2D), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_pois);
}

void ICGN2D1BatchGpu::release_cuda() {
    if (d_ref_image) cudaFree(d_ref_image);
    if (d_tar_image) cudaFree(d_tar_image);
    d_ref_image = d_tar_image = nullptr;
}


ICGN2D2BatchGpu::ICGN2D2BatchGpu() : d_ref_image(nullptr), d_tar_image(nullptr) {}
ICGN2D2BatchGpu::~ICGN2D2BatchGpu() { release_cuda(); }

void ICGN2D2BatchGpu::prepare_cuda(const float* ref_image, const float* tar_image, int h, int w, const ICGNParam& param_, cudaStream_t stream) {
    height = h; width = w; param = param_;
    size_t bytes = h * w * sizeof(float);
    cudaMalloc(&d_ref_image, bytes);
    cudaMalloc(&d_tar_image, bytes);
    cudaMemcpy(d_ref_image, ref_image, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tar_image, tar_image, bytes, cudaMemcpyHostToDevice);
    cudaStreamSynchronize(stream);
}


void ICGN2D2BatchGpu::compute_batch_cuda(CudaPOI2D* pois, int N, cudaStream_t stream) {
    CudaPOI2D* d_pois;
    cudaMalloc(&d_pois, N * sizeof(CudaPOI2D));
    cudaMemcpyAsync(d_pois, pois, N * sizeof(CudaPOI2D), cudaMemcpyHostToDevice, stream);

    int block = 128, grid = (N + block - 1) / block;
    icgn2d_batch_kernel<<<grid, block, 0, stream>>>(
        d_ref_image, d_tar_image, 
        height, width,d_pois,
        param.subsetRadius, param.convergenceThreshold, param.maxIterations, N, 12);

    cudaMemcpyAsync(pois, d_pois, N * sizeof(CudaPOI2D), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_pois);
}
void ICGN2D2BatchGpu::release_cuda() {
    if (d_ref_image) cudaFree(d_ref_image);
    if (d_tar_image) cudaFree(d_tar_image);
    d_ref_image = d_tar_image = nullptr;
}

ICGN3D1BatchGpu::ICGN3D1BatchGpu() : d_ref_image(nullptr), d_tar_image(nullptr) {}
ICGN3D1BatchGpu::~ICGN3D1BatchGpu() { release_cuda(); }

void ICGN3D1BatchGpu::prepare_cuda(const float* ref_image, const float* tar_image, int dx, int dy, int dz, const ICGNParam& param_) {
    dim_x = dx; dim_y = dy; dim_z = dz; param = param_;
    size_t bytes = dx * dy * dz * sizeof(float);
    cudaMalloc(&d_ref_image, bytes);
    cudaMalloc(&d_tar_image, bytes);
    cudaMemcpy(d_ref_image, ref_image, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tar_image, tar_image, bytes, cudaMemcpyHostToDevice);
}
void ICGN3D1BatchGpu::compute_batch_cuda(CudaPOI3D* pois, int N, cudaStream_t stream) {
    CudaPOI3D* d_pois;
    cudaMalloc(&d_pois, N * sizeof(CudaPOI3D));
    cudaMemcpyAsync(d_pois, pois, N * sizeof(CudaPOI3D), cudaMemcpyHostToDevice, stream);

    int block = 128, grid = (N + block - 1) / block;
    icgn3d1_batch_kernel<<<grid, block, 0, stream>>>(
        d_ref_image, d_tar_image,
        dim_x, dim_y, dim_z,d_pois,
        param.subsetRadius, param.convergenceThreshold, param.maxIterations, N);

    cudaMemcpyAsync(pois, d_pois, N * sizeof(CudaPOI3D), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_pois);
}
void ICGN3D1BatchGpu::release_cuda() {
    if (d_ref_image) cudaFree(d_ref_image);
    if (d_tar_image) cudaFree(d_tar_image);
    d_ref_image = d_tar_image = nullptr;
}

} // namespace StudyCorr