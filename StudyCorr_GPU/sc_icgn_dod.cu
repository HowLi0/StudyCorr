/*
 * Data-Oriented Design (DOD) version of ICGN CUDA kernels
 * Uses Structure of Arrays (SoA) for better memory access patterns
 */

#include "sc_icgn_dod.h"
#include "sc_icgn.h"
#include "sc_icgn_device_function.cuh"
#include "sc_cuda_check.h"

namespace StudyCorr_GPU {

/****************************************** DOD ICGN2D Kernel **************************************************/

// DOD version of ICGN2D kernel using SoA layout
__global__ void icgn2d_batch_kernel_dod(
    const float* ref_image, const float* tar_image,
    int height, int width,
    // Position arrays
    const float* pos_x, const float* pos_y,
    // Deformation input arrays
    float* deform_u, float* deform_ux, float* deform_uy, 
    float* deform_uxx, float* deform_uxy, float* deform_uyy,
    float* deform_v, float* deform_vx, float* deform_vy, 
    float* deform_vxx, float* deform_vxy, float* deform_vyy,
    // Result output arrays
    float* result_u0, float* result_v0, float* result_zncc,
    float* result_iteration, float* result_convergence,
    // Strain output arrays  
    float* strain_exx, float* strain_eyy, float* strain_exy,
    // Parameters
    int subsetRadius, double convergenceThreshold, int maxIterations,
    int N, int numParams)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load position from SoA
    float center_x = pos_x[idx];
    float center_y = pos_y[idx];

    // Boundary check
    if (center_x - subsetRadius < 1 || center_x + subsetRadius >= width - 1 ||
        center_y - subsetRadius < 1 || center_y + subsetRadius >= height - 1) {
        result_zncc[idx] = -4.f; // Invalid
        result_u0[idx] = 0.f;
        result_v0[idx] = 0.f;
        result_iteration[idx] = 0;
        result_convergence[idx] = 0.f;
        // Clear deformation
        deform_u[idx] = deform_ux[idx] = deform_uy[idx] = 0.f;
        deform_v[idx] = deform_vx[idx] = deform_vy[idx] = 0.f;
        if (numParams >= 12) {
            deform_uxx[idx] = deform_uxy[idx] = deform_uyy[idx] = 0.f;
            deform_vxx[idx] = deform_vxy[idx] = deform_vyy[idx] = 0.f;
        }
        strain_exx[idx] = strain_eyy[idx] = strain_exy[idx] = 0.f;
        return;
    }

    // Initialize warp parameters from SoA deformation
    float warpParams[12] = {0.0f};
    result_u0[idx] = deform_u[idx];
    result_v0[idx] = deform_v[idx];
    
    if (numParams >= 6) {
        warpParams[0] = deform_u[idx];
        warpParams[1] = deform_v[idx];
        warpParams[2] = deform_ux[idx];
        warpParams[3] = deform_uy[idx];
        warpParams[4] = deform_vx[idx];
        warpParams[5] = deform_vy[idx];
    }
    if (numParams >= 12) {
        warpParams[6] = deform_uxx[idx];
        warpParams[7] = deform_uxy[idx];
        warpParams[8] = deform_uyy[idx];
        warpParams[9] = deform_vxx[idx];
        warpParams[10] = deform_vxy[idx];
        warpParams[11] = deform_vyy[idx];
    }

    // Compute Hessian matrix
    float hessian[144] = {0.0f};
    computehessian(ref_image, subsetRadius, height, width, center_y, center_x, numParams, hessian);

    // ICGN iterative optimization
    float prevZNCC = -4.0f;
    bool converged = false;
    int iter = 0;
    
    for (iter = 0; iter < maxIterations && !converged; iter++) {
        // Compute ZNCC and error vector
        ZNCCAndErrorResult znccError = computeZNCCAndError(
            ref_image, tar_image, center_y, center_x, warpParams,
            height, width, subsetRadius, numParams
        );
        float currentZNCC = znccError.zncc;
        float* errorVector = znccError.errorVector;

        // Use a separate threshold for ZNCC convergence (ZNCC is in range [-1, 1])
        const float znccConvergenceThreshold = 0.0001f;
        bool znccConverged = (iter > 0) && (fabs(currentZNCC - prevZNCC) < znccConvergenceThreshold);
        if (znccConverged) {
            converged = true;
            result_zncc[idx] = currentZNCC;
            break;
        }
        prevZNCC = currentZNCC;

        float deltaP[12] = {0};
        // deltaP = H^{-1} * errorVector
        bool solved = solveLinearSystem(hessian, errorVector, deltaP, numParams);
        if (!solved) {
            result_zncc[idx] = currentZNCC;
            converged = (currentZNCC < 0.5);
            break;
        }
        
        // Update warp parameters
        for (int p = 0; p < numParams; p++) {
            warpParams[p] += deltaP[p];
        }
        
        // Check parameter convergence
        float deltaNorm = 0.0f;
        for (int p = 0; p < numParams; p++) {
            deltaNorm += deltaP[p] * deltaP[p];
        }
        if (sqrt(deltaNorm) < convergenceThreshold * 10.0f) {
            converged = true;
            result_zncc[idx] = currentZNCC;
        }
    }

    // Final ZNCC computation if not converged
    if (!converged && iter >= maxIterations) {
        ZNCCAndErrorResult znccError = computeZNCCAndError(
            ref_image, tar_image, center_y, center_x, warpParams,
            height, width, subsetRadius, numParams
        );
        result_zncc[idx] = znccError.zncc;
        converged = (znccError.zncc > 0.8f);
    }

    // Write results back to SoA
    result_convergence[idx] = converged ? 1.0f : 0.0f;
    result_iteration[idx] = iter;
    
    deform_u[idx] = warpParams[0];
    deform_v[idx] = warpParams[1];
    deform_ux[idx] = warpParams[2];
    deform_uy[idx] = warpParams[3];
    deform_vx[idx] = warpParams[4];
    deform_vy[idx] = warpParams[5];
    
    if (numParams >= 12) {
        deform_uxx[idx] = warpParams[6];
        deform_uxy[idx] = warpParams[7];
        deform_uyy[idx] = warpParams[8];
        deform_vxx[idx] = warpParams[9];
        deform_vxy[idx] = warpParams[10];
        deform_vyy[idx] = warpParams[11];
    }
}

/****************************************** DOD ICGN3D Kernel **************************************************/

// DOD version of ICGN3D1 kernel using SoA layout
__global__ void icgn3d1_batch_kernel_dod(
    const float* ref_image, const float* tar_image,
    int depth, int height, int width,
    // Position arrays
    const float* pos_x, const float* pos_y, const float* pos_z,
    // Deformation input/output arrays
    float* deform_u, float* deform_ux, float* deform_uy, float* deform_uz,
    float* deform_v, float* deform_vx, float* deform_vy, float* deform_vz,
    float* deform_w, float* deform_wx, float* deform_wy, float* deform_wz,
    // Result output arrays
    float* result_u0, float* result_v0, float* result_w0,
    float* result_zncc, float* result_iteration, float* result_convergence,
    // Strain output arrays
    float* strain_exx, float* strain_eyy, float* strain_ezz,
    float* strain_exy, float* strain_eyz, float* strain_ezx,
    // Parameters
    int subsetRadius, float convergenceThreshold, int maxIterations, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load position from SoA
    float center_x = pos_x[idx];
    float center_y = pos_y[idx];
    float center_z = pos_z[idx];

    // Boundary check
    if (center_x - subsetRadius < 1 || center_x + subsetRadius >= width - 1 ||
        center_y - subsetRadius < 1 || center_y + subsetRadius >= height - 1 ||
        center_z - subsetRadius < 1 || center_z + subsetRadius >= depth - 1) {
        result_zncc[idx] = -1.f;
        result_u0[idx] = 0.f;
        result_v0[idx] = 0.f;
        result_w0[idx] = 0.f;
        result_iteration[idx] = 0;
        result_convergence[idx] = 0;
        return;
    }

    // Initialize warp parameters from SoA deformation
    float warpParams[12] = {0.0f};
    result_u0[idx] = deform_u[idx];
    result_v0[idx] = deform_v[idx];
    result_w0[idx] = deform_w[idx];

    warpParams[0] = deform_u[idx];
    warpParams[1] = deform_v[idx];
    warpParams[2] = deform_w[idx];
    warpParams[3] = deform_ux[idx];
    warpParams[4] = deform_uy[idx];
    warpParams[5] = deform_uz[idx];
    warpParams[6] = deform_vx[idx];
    warpParams[7] = deform_vy[idx];
    warpParams[8] = deform_vz[idx];
    warpParams[9] = deform_wx[idx];
    warpParams[10] = deform_wy[idx];
    warpParams[11] = deform_wz[idx];

    // TODO: Full 3D ICGN implementation needed:
    // - computeHessian3D() for 3D subset gradients
    // - computeZNCCAndError3D() for 3D correlation calculation
    // - solveLinearSystem() for 3D parameter updates
    // - 3D trilinear interpolation for warped coordinates
    // Currently returns placeholder values until implementation is complete
    
    result_zncc[idx] = 0.0f;
    result_iteration[idx] = 0;
    result_convergence[idx] = 0.0f;
}

/****************************************** ICGN2D1BatchGpuDOD Implementation **************************************************/

ICGN2D1BatchGpuDOD::ICGN2D1BatchGpuDOD() {}

ICGN2D1BatchGpuDOD::~ICGN2D1BatchGpuDOD() {
    release_cuda();
}

void ICGN2D1BatchGpuDOD::prepare_cuda(const float* ref_image, const float* tar_image,
                                     int height, int width, const ICGNParam& param, cudaStream_t stream) {
    this->height = height;
    this->width = width;
    this->param = param;

    size_t img_size = height * width * sizeof(float);
    cudaMalloc(&d_ref_image, img_size);
    cudaMalloc(&d_tar_image, img_size);

    cudaMemcpyAsync(d_ref_image, ref_image, img_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_tar_image, tar_image, img_size, cudaMemcpyHostToDevice, stream);
}

void ICGN2D1BatchGpuDOD::compute_batch_cuda(POI2D_SoA& pois_soa, cudaStream_t stream) {
    int N = pois_soa.count;
    if (N == 0) return;

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    icgn2d_batch_kernel_dod<<<numBlocks, blockSize, 0, stream>>>(
        d_ref_image, d_tar_image, height, width,
        // Position
        pois_soa.x, pois_soa.y,
        // Deformation
        pois_soa.u, pois_soa.ux, pois_soa.uy,
        pois_soa.uxx, pois_soa.uxy, pois_soa.uyy,
        pois_soa.v, pois_soa.vx, pois_soa.vy,
        pois_soa.vxx, pois_soa.vxy, pois_soa.vyy,
        // Results
        pois_soa.u0, pois_soa.v0, pois_soa.zncc,
        pois_soa.iteration, pois_soa.convergence,
        // Strain
        pois_soa.exx, pois_soa.eyy, pois_soa.exy,
        // Parameters
        param.subsetRadius, param.convergenceThreshold, param.maxIterations,
        N, 6 // numParams for 1st order
    );
}

void ICGN2D1BatchGpuDOD::release_cuda() {
    if (d_ref_image) {
        cudaFree(d_ref_image);
        d_ref_image = nullptr;
    }
    if (d_tar_image) {
        cudaFree(d_tar_image);
        d_tar_image = nullptr;
    }
}

/****************************************** ICGN2D2BatchGpuDOD Implementation **************************************************/

ICGN2D2BatchGpuDOD::ICGN2D2BatchGpuDOD() {}

ICGN2D2BatchGpuDOD::~ICGN2D2BatchGpuDOD() {
    release_cuda();
}

void ICGN2D2BatchGpuDOD::prepare_cuda(const float* ref_image, const float* tar_image,
                                     int height, int width, const ICGNParam& param, cudaStream_t stream) {
    this->height = height;
    this->width = width;
    this->param = param;

    size_t img_size = height * width * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_ref_image, img_size));
    CUDA_CHECK(cudaMalloc(&d_tar_image, img_size));

    CUDA_CHECK(cudaMemcpyAsync(d_ref_image, ref_image, img_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tar_image, tar_image, img_size, cudaMemcpyHostToDevice, stream));
}

void ICGN2D2BatchGpuDOD::compute_batch_cuda(POI2D_SoA& pois_soa, cudaStream_t stream) {
    int N = pois_soa.count;
    if (N == 0) return;

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    icgn2d_batch_kernel_dod<<<numBlocks, blockSize, 0, stream>>>(
        d_ref_image, d_tar_image, height, width,
        // Position
        pois_soa.x, pois_soa.y,
        // Deformation
        pois_soa.u, pois_soa.ux, pois_soa.uy,
        pois_soa.uxx, pois_soa.uxy, pois_soa.uyy,
        pois_soa.v, pois_soa.vx, pois_soa.vy,
        pois_soa.vxx, pois_soa.vxy, pois_soa.vyy,
        // Results
        pois_soa.u0, pois_soa.v0, pois_soa.zncc,
        pois_soa.iteration, pois_soa.convergence,
        // Strain
        pois_soa.exx, pois_soa.eyy, pois_soa.exy,
        // Parameters
        param.subsetRadius, param.convergenceThreshold, param.maxIterations,
        N, 12 // numParams for 2nd order
    );
}

void ICGN2D2BatchGpuDOD::release_cuda() {
    if (d_ref_image) {
        cudaFree(d_ref_image);
        d_ref_image = nullptr;
    }
    if (d_tar_image) {
        cudaFree(d_tar_image);
        d_tar_image = nullptr;
    }
}

/****************************************** ICGN3D1BatchGpuDOD Implementation **************************************************/

ICGN3D1BatchGpuDOD::ICGN3D1BatchGpuDOD() {}

ICGN3D1BatchGpuDOD::~ICGN3D1BatchGpuDOD() {
    release_cuda();
}

void ICGN3D1BatchGpuDOD::prepare_cuda(const float* ref_image, const float* tar_image,
                                     int dim_x, int dim_y, int dim_z, const ICGNParam& param) {
    this->dim_x = dim_x;
    this->dim_y = dim_y;
    this->dim_z = dim_z;
    this->param = param;

    size_t img_size = dim_x * dim_y * dim_z * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_ref_image, img_size));
    CUDA_CHECK(cudaMalloc(&d_tar_image, img_size));

    CUDA_CHECK(cudaMemcpy(d_ref_image, ref_image, img_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tar_image, tar_image, img_size, cudaMemcpyHostToDevice));
}

void ICGN3D1BatchGpuDOD::compute_batch_cuda(POI3D_SoA& pois_soa, cudaStream_t stream) {
    int N = pois_soa.count;
    if (N == 0) return;

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    icgn3d1_batch_kernel_dod<<<numBlocks, blockSize, 0, stream>>>(
        d_ref_image, d_tar_image, dim_z, dim_y, dim_x,
        // Position
        pois_soa.x, pois_soa.y, pois_soa.z,
        // Deformation
        pois_soa.u, pois_soa.ux, pois_soa.uy, pois_soa.uz,
        pois_soa.v, pois_soa.vx, pois_soa.vy, pois_soa.vz,
        pois_soa.w, pois_soa.wx, pois_soa.wy, pois_soa.wz,
        // Results
        pois_soa.u0, pois_soa.v0, pois_soa.w0,
        pois_soa.zncc, pois_soa.iteration, pois_soa.convergence,
        // Strain
        pois_soa.exx, pois_soa.eyy, pois_soa.ezz,
        pois_soa.exy, pois_soa.eyz, pois_soa.ezx,
        // Parameters
        param.subsetRadius, param.convergenceThreshold, param.maxIterations, N
    );
}

void ICGN3D1BatchGpuDOD::release_cuda() {
    if (d_ref_image) {
        cudaFree(d_ref_image);
        d_ref_image = nullptr;
    }
    if (d_tar_image) {
        cudaFree(d_tar_image);
        d_tar_image = nullptr;
    }
}

} // namespace StudyCorr_GPU
