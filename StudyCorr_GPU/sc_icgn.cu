#include "sc_icgn.h"
#include <cuda_runtime.h>
#include <cassert>
#include"sc_icgn_device_function.cuh"


namespace StudyCorr_GPU {

__global__ void sobel_gradient_kernel(const float* image, int height, int width, float* grad_x, float* grad_y) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;
int idx = y * width + x;
grad_x[idx] =
    -image[(y-1)*width + (x-1)] - 2*image[y*width + (x-1)] - image[(y+1)*width + (x-1)]
    +image[(y-1)*width + (x+1)] + 2*image[y*width + (x+1)] + image[(y+1)*width + (x+1)];
grad_x[idx] /= 8.0f;
grad_y[idx] =
    -image[(y-1)*width + (x-1)] - 2*image[(y-1)*width + x] - image[(y-1)*width + (x+1)]
    +image[(y+1)*width + (x-1)] + 2*image[(y+1)*width + x] + image[(y+1)*width + (x+1)];
grad_y[idx] /= 8.0f;
}

// Sobel梯度图
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
/****************************************** ICGN2D1 **************************************************/

__global__ void icgn2d1_batch_kernel(
    const float* ref_image, const float* tar_image,
    const float* grad_x_tar, const float* grad_y_tar, // 新增
    int height, int width,
    CudaPOI2D* pois, int subsetRadius,
    double convergenceThreshold, int maxIterations,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    CudaPOI2D& poi = pois[idx];
    int subset_w = 2 * subsetRadius + 1;
    int subset_h = 2 * subsetRadius + 1;
    int subset_rx = subsetRadius;
    int subset_ry = subsetRadius;
    int subset_size = subset_w * subset_h;

    float center_y = poi.y;
    float center_x = poi.x;

    // 边界检测（更严格：整个子集必须都在图像内部）
    if (center_x - subset_rx < 1 || center_x + subset_rx >= width - 1 ||
        center_y - subset_ry < 1 || center_y + subset_ry >= height - 1) {
        poi.result.zncc = -1.f; // 无效
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

    // 预计算参考子集/均值/方差
    float ref_subset[2601]; // 支持最大51*51子集
    float mean_ref = 0.f, std_ref = 0.f;
    int rx = subset_w / 2, ry = subset_h / 2;
    int idx_img = 0;
    for (int r = 0; r < subset_h; ++r) {
        for (int c = 0; c < subset_w; ++c, ++idx_img) {
            int img_y = int(center_y + r - ry);
            int img_x = int(center_x + c - rx);
            int img_idx = img_y * width + img_x;
            ref_subset[idx_img] = ref_image[img_idx];
            mean_ref += ref_subset[idx_img];
        }
    }
    mean_ref /= subset_size;
    for (int i = 0; i < subset_size; ++i)
        std_ref += (ref_subset[i] - mean_ref) * (ref_subset[i] - mean_ref);
    std_ref = sqrtf(std_ref / subset_size);

    // 初始化形函数参数（从poi.deformation赋值）
    ShapeParam2D1 p_current;
    p_current.u = poi.deformation.u;
    p_current.ux = poi.deformation.ux;
    p_current.uy = poi.deformation.uy;
    p_current.v = poi.deformation.v;
    p_current.vx = poi.deformation.vx;
    p_current.vy = poi.deformation.vy;

    double dp_norm_max = 1e10;
    int iter = 0;

    float tar_subset[2601]; // 支持最大51*51子集
    float mean_tar = 0.f, std_tar = 0.f;

    while (iter < maxIterations && dp_norm_max > convergenceThreshold) {
        double hessian[36] = {0.0};
        double numerator[6] = {0.0};
        double delta_p[6] = {0.0};
        mean_tar = 0.f; std_tar = 0.f;

        idx_img = 0;
        for (int r = 0; r < subset_h; ++r) {
            for (int c = 0; c < subset_w; ++c, ++idx_img) {
                double x_local = c - rx;
                double y_local = r - ry;
                // 形函数变换
                double u_warp, v_warp;
                warp2d1(p_current, x_local, y_local, u_warp, v_warp);
                float tar_y = center_y + v_warp;
                float tar_x = center_x + u_warp;
                // 子集采样
                float I_tar = bilinear_interpolate(tar_image, height, width, tar_y, tar_x);
                tar_subset[idx_img] = I_tar;
                mean_tar += I_tar;

                // 梯度亚像素插值
                float gx = bilinear_interpolate(grad_x_tar, height, width, tar_y, tar_x);
                float gy = bilinear_interpolate(grad_y_tar, height, width, tar_y, tar_x);

                double grad_u[6], grad_v[6];
                shape_gradient2d1(x_local, y_local, grad_u, grad_v);

                double sd[6];
                for (int i = 0; i < 6; ++i)
                    sd[i] = gx * grad_u[i] + gy * grad_v[i];

                for (int i = 0; i < 6; ++i)
                    for (int j = 0; j < 6; ++j)
                        hessian[i*6 + j] += sd[i] * sd[j];

                float error = I_tar - ref_subset[idx_img];
                for (int i = 0; i < 6; ++i)
                    numerator[i] += sd[i] * error;
            }
        }
        mean_tar /= subset_size;
        for (int i = 0; i < subset_size; ++i)
            std_tar += (tar_subset[i] - mean_tar) * (tar_subset[i] - mean_tar);
        std_tar = sqrtf(std_tar / subset_size);

        // Hessian逆
        double inv_hessian[36];
        bool ok = qr_inverse_6x6(hessian, inv_hessian);
        if (!ok) {
            poi.result.zncc = -1.f;
            poi.result.u0 = 0.f;
            poi.result.v0 = 0.f;
            poi.result.iteration = iter;
            poi.result.convergence = 0;
            // deformation参数清零
            poi.deformation.u = poi.deformation.ux = poi.deformation.uy = poi.deformation.v = poi.deformation.vx = poi.deformation.vy = 0.f;
            poi.strain.exx = poi.strain.eyy = poi.strain.exy = 0.f;
            return;
        }
        for (int i = 0; i < 6; ++i) {
            delta_p[i] = 0.0;
            for (int j = 0; j < 6; ++j)
                delta_p[i] += inv_hessian[i*6 + j] * numerator[j];
        }
        ShapeParam2D1 delta_param;
        set_deformation2d1(delta_p, delta_param);
        ShapeParam2D1 p_new;
        compose2d1(p_current, delta_param, p_new);
        dp_norm_max = delta_norm2d1(delta_param, subset_rx, subset_ry);
        p_current = p_new;
        iter++;
    }

    // 计算最终相关系数
    float zncc = compute_znssd(ref_subset, tar_subset, subset_size, mean_ref, std_ref, mean_tar, std_tar);

    // 结果全部写回POI2D各成员
    poi.result.zncc = zncc;
    poi.result.u0 = p_current.u;
    poi.result.v0 = p_current.v;
    poi.result.iteration = iter;
    poi.result.convergence = dp_norm_max;
    // deformation参数写回最新
    poi.deformation.u = p_current.u;
    poi.deformation.ux = p_current.ux;
    poi.deformation.uy = p_current.uy;
    poi.deformation.v = p_current.v;
    poi.deformation.vx = p_current.vx;
    poi.deformation.vy = p_current.vy;
}

/****************************************** ICGN2D2 **************************************************/

__global__ void icgn2d2_batch_kernel(
    const float* ref_image, const float* tar_image, int height, int width,
    const float* grad_x_tar, const float* grad_y_tar,
    const CudaPOI2D* pois, int subsetRadius,
    double convergenceThreshold, int maxIterations,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    CudaPOI2D poi = pois[idx];
    int subset_w = 2 * subsetRadius + 1;
    int subset_h = 2 * subsetRadius + 1;
    int subset_rx = subsetRadius;
    int subset_ry = subsetRadius;
    int subset_size = subset_w * subset_h;

    float center_y = poi.y;
    float center_x = poi.x;

    // 边界严格处理
    if (center_x - subset_rx < 1 || center_x + subset_rx >= width - 1 ||
        center_y - subset_ry < 1 || center_y + subset_ry >= height - 1) {
        poi.result.zncc = -1.f; // 无效
        poi.result.u0 = 0.f;
        poi.result.v0 = 0.f;
        poi.result.iteration = 0;
        poi.result.convergence = 0;
        return;
    }

    // 预计算参考子集/均值/方差
    float ref_subset[2601]; // 支持最大51*51子集
    float mean_ref = 0.f, std_ref = 0.f;
    int rx = subset_w / 2, ry = subset_h / 2;
    int idx_img = 0;
    for (int r = 0; r < subset_h; ++r) {
        for (int c = 0; c < subset_w; ++c, ++idx_img) {
            int img_y = int(center_y + r - ry);
            int img_x = int(center_x + c - rx);
            int img_idx = img_y * width + img_x;
            ref_subset[idx_img] = ref_image[img_idx];
            mean_ref += ref_subset[idx_img];
        }
    }
    mean_ref /= subset_size;
    for (int i = 0; i < subset_size; ++i)
        std_ref += (ref_subset[i] - mean_ref) * (ref_subset[i] - mean_ref);
    std_ref = sqrtf(std_ref / subset_size);

    // 初始化形函数参数
    ShapeParam2D2 p_current;
    p_current.u = poi.deformation.u;
    p_current.ux = poi.deformation.ux;
    p_current.uy = poi.deformation.uy;
    p_current.uxx = poi.deformation.uxx;
    p_current.uxy = poi.deformation.uxy;
    p_current.uyy = poi.deformation.uyy;
    p_current.v = poi.deformation.v;
    p_current.vx = poi.deformation.vx;
    p_current.vy = poi.deformation.vy;
    p_current.vxx = poi.deformation.vxx;
    p_current.vxy = poi.deformation.vxy;
    p_current.vyy = poi.deformation.vyy;

    double dp_norm_max = 1e10;
    int iter = 0;
    float tar_subset[2601]; // 支持最大51*51子集
    float mean_tar = 0.f, std_tar = 0.f;

    while (iter < maxIterations && dp_norm_max > convergenceThreshold) {
        double hessian[144] = {0.0};
        double numerator[12] = {0.0};
        double delta_p[12] = {0.0};
        mean_tar = 0.f; std_tar = 0.f;
        idx_img = 0;
        for (int r = 0; r < subset_h; ++r) {
            for (int c = 0; c < subset_w; ++c, ++idx_img) {
                double x_local = c - rx;
                double y_local = r - ry;
                double u_warp, v_warp;
                warp2d2(p_current, x_local, y_local, u_warp, v_warp);
                float tar_y = center_y + v_warp;
                float tar_x = center_x + u_warp;
                float I_tar = bilinear_interpolate(tar_image, height, width, tar_y, tar_x);
                tar_subset[idx_img] = I_tar;
                mean_tar += I_tar;
                // 梯度亚像素插值
                float gx = bilinear_interpolate(grad_x_tar, height, width, tar_y, tar_x);
                float gy = bilinear_interpolate(grad_y_tar, height, width, tar_y, tar_x);

                double grad_u[12], grad_v[12];
                shape_gradient2d2(x_local, y_local, grad_u, grad_v);

                double sd[12];
                for (int i = 0; i < 12; ++i)
                    sd[i] = gx * grad_u[i] + gy * grad_v[i];

                for (int i = 0; i < 12; ++i)
                    for (int j = 0; j < 12; ++j)
                        hessian[i*12 + j] += sd[i] * sd[j];

                float error = I_tar - ref_subset[idx_img];
                for (int i = 0; i < 12; ++i)
                    numerator[i] += sd[i] * error;
            }
        }
        mean_tar /= subset_size;
        for (int i = 0; i < subset_size; ++i)
            std_tar += (tar_subset[i] - mean_tar) * (tar_subset[i] - mean_tar);
        std_tar = sqrtf(std_tar / subset_size);

        double inv_hessian[144];
        bool ok = qr_inverse_12x12(hessian, inv_hessian);
        if (!ok) {
            poi.result.zncc = -1.f;
            poi.result.u0 = 0.f;
            poi.result.v0 = 0.f;
            poi.result.iteration = iter;
            poi.result.convergence = 0;
            return;
        }
        for (int i = 0; i < 12; ++i) {
            delta_p[i] = 0.0;
            for (int j = 0; j < 12; ++j)
                delta_p[i] += inv_hessian[i*12 + j] * numerator[j];
        }
        ShapeParam2D2 delta_param;
        set_deformation2d2(delta_p, delta_param);
        ShapeParam2D2 p_new;
        compose2d2(p_current, delta_param, p_new);
        dp_norm_max = delta_norm2d2(delta_param, subset_rx, subset_ry);
        p_current = p_new;
        iter++;
    }

    // 相关系数
    float zncc = compute_znssd(ref_subset, tar_subset, subset_size, mean_ref, std_ref, mean_tar, std_tar);

    poi.result.zncc = zncc;
    poi.result.u0 = p_current.u;
    poi.result.v0 = p_current.v;
    poi.result.iteration = iter;
    poi.result.convergence = dp_norm_max;
    // deformation参数写回最新
    poi.deformation.u = p_current.u;
    poi.deformation.v = p_current.v;
    poi.deformation.ux = p_current.ux;
    poi.deformation.uy = p_current.uy;
    poi.deformation.uxx = p_current.uxx;
    poi.deformation.uxy = p_current.uxy;
    poi.deformation.uyy = p_current.uyy;
    poi.deformation.vx = p_current.vx;
    poi.deformation.vy = p_current.vy;
    poi.deformation.vxx = p_current.vxx;
    poi.deformation.vxy = p_current.vxy;
    poi.deformation.vyy = p_current.vyy;

}

/****************************************** ICGN3D1 **************************************************/

// ICGN3D1核
__global__ void icgn3d1_batch_kernel(
    const float* ref_image, const float* tar_image,
    const float* grad_x_tar, const float* grad_y_tar, const float* grad_z_tar,
    int depth, int height, int width,
    const CudaPOI3D* pois, int subsetRadius,
    double convergenceThreshold, int maxIterations,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    CudaPOI3D poi = pois[idx];
    int subset_w = 2 * subsetRadius + 1;
    int subset_h = 2 * subsetRadius + 1;
    int subset_d = 2 * subsetRadius + 1;
    int subset_rx = subsetRadius;
    int subset_ry = subsetRadius;
    int subset_rz = subsetRadius;
    int subset_size = subset_w * subset_h * subset_d;

    float center_z = poi.z;
    float center_y = poi.y;
    float center_x = poi.x;

    // 边界严格处理
    if (center_x - subset_rx < 1 || center_x + subset_rx >= width - 1 ||
        center_y - subset_ry < 1 || center_y + subset_ry >= height - 1 ||
        center_z - subset_rz < 1 || center_z + subset_rz >= depth - 1) {
        poi.result.zncc = -1.f;
        poi.result.u0 = 0.f;
        poi.result.v0 = 0.f;
        poi.result.w0 = 0.f;
        poi.result.iteration = 0;
        poi.result.convergence = 0;
        return;
    }

    // 预计算参考子集/均值/方差
    float ref_subset[1331]; // subsetRadius<=5时安全
    float mean_ref = 0.f, std_ref = 0.f;
    int rx = subset_w / 2, ry = subset_h / 2, rz = subset_d / 2;
    int idx_img = 0;
    for (int d = 0; d < subset_d; ++d)
        for (int r = 0; r < subset_h; ++r)
            for (int c = 0; c < subset_w; ++c, ++idx_img) {
                int img_z = int(center_z + d - rz);
                int img_y = int(center_y + r - ry);
                int img_x = int(center_x + c - rx);
                int img_idx = img_z * height * width + img_y * width + img_x;
                ref_subset[idx_img] = ref_image[img_idx];
                mean_ref += ref_subset[idx_img];
            }
    mean_ref /= subset_size;
    for (int i = 0; i < subset_size; ++i)
        std_ref += (ref_subset[i] - mean_ref) * (ref_subset[i] - mean_ref);
    std_ref = sqrtf(std_ref / subset_size);

    // 初始化形函数参数
    ShapeParam3D1 p_current;
    p_current.u = poi.deformation.u;
    p_current.ux = poi.deformation.ux;
    p_current.uy = poi.deformation.uy;
    p_current.v = poi.deformation.v;
    p_current.vx = poi.deformation.vx;
    p_current.vy = poi.deformation.vy;
    p_current.w = poi.deformation.w;
    p_current.wx = poi.deformation.wx;
    p_current.wy = poi.deformation.wy;
    p_current.wz = poi.deformation.wz;

    double dp_norm_max = 1e10;
    int iter = 0;
    float tar_subset[1331];
    float mean_tar = 0.f, std_tar = 0.f;

    while (iter < maxIterations && dp_norm_max > convergenceThreshold) {
        double hessian[144] = {0.0};
        double numerator[12] = {0.0};
        double delta_p[12] = {0.0};
        mean_tar = 0.f; std_tar = 0.f;
        idx_img = 0;
        for (int d = 0; d < subset_d; ++d)
            for (int r = 0; r < subset_h; ++r)
                for (int c = 0; c < subset_w; ++c, ++idx_img) {
                    double x_local = c - rx;
                    double y_local = r - ry;
                    double z_local = d - rz;
                    double u_warp, v_warp, w_warp;
                    warp3d1(p_current, x_local, y_local, z_local, u_warp, v_warp, w_warp);
                    float tar_z = center_z + w_warp;
                    float tar_y = center_y + v_warp;
                    float tar_x = center_x + u_warp;
                    float I_tar = trilinear_interpolate(tar_image, depth, height, width, tar_z, tar_y, tar_x);
                    tar_subset[idx_img] = I_tar;
                    mean_tar += I_tar;
                    // 梯度亚像素插值
                    float gx = trilinear_interpolate(grad_x_tar, depth, height, width, tar_z, tar_y, tar_x);
                    float gy = trilinear_interpolate(grad_y_tar, depth, height, width, tar_z, tar_y, tar_x);
                    float gz = trilinear_interpolate(grad_z_tar, depth, height, width, tar_z, tar_y, tar_x);

                    double grad_u[12], grad_v[12], grad_w[12];
                    shape_gradient3d1(x_local, y_local, z_local, grad_u, grad_v, grad_w);

                    double sd[12];
                    for (int i = 0; i < 12; ++i)
                        sd[i] = gx * grad_u[i] + gy * grad_v[i] + gz * grad_w[i];

                    for (int i = 0; i < 12; ++i)
                        for (int j = 0; j < 12; ++j)
                            hessian[i*12 + j] += sd[i] * sd[j];

                    float error = I_tar - ref_subset[idx_img];
                    for (int i = 0; i < 12; ++i)
                        numerator[i] += sd[i] * error;
                }
        mean_tar /= subset_size;
        for (int i = 0; i < subset_size; ++i)
            std_tar += (tar_subset[i] - mean_tar) * (tar_subset[i] - mean_tar);
        std_tar = sqrtf(std_tar / subset_size);

        double inv_hessian[144];
        bool ok = qr_inverse_12x12(hessian, inv_hessian);
        if (!ok) {
            poi.result.zncc = -1.f;
            poi.result.u0 = 0.f;
            poi.result.v0 = 0.f;
            poi.result.w0 = 0.f;
            poi.result.iteration = iter;
            poi.result.convergence = 0;
            return;
        }
        for (int i = 0; i < 12; ++i) {
            delta_p[i] = 0.0;
            for (int j = 0; j < 12; ++j)
                delta_p[i] += inv_hessian[i*12 + j] * numerator[j];
        }
        ShapeParam3D1 delta_param;
        set_deformation3d1(delta_p, delta_param);
        ShapeParam3D1 p_new;
        compose3d1(p_current, delta_param, p_new);
        dp_norm_max = delta_norm3d1(delta_param, subset_rx, subset_ry, subset_rz);
        p_current = p_new;
        iter++;
    }

    // 相关系数
    float zncc = compute_znssd(ref_subset, tar_subset, subset_size, mean_ref, std_ref, mean_tar, std_tar);

    poi.result.zncc = zncc;
    poi.result.u0 = p_current.u;
    poi.result.v0 = p_current.v;
    poi.result.w0 = p_current.w;
    poi.result.iteration = iter;
    poi.result.convergence = dp_norm_max;
    // deformation参数写回最新
    poi.deformation.u = p_current.u;
    poi.deformation.v = p_current.v;
    poi.deformation.w = p_current.w;
    poi.deformation.ux = p_current.ux;
    poi.deformation.uy = p_current.uy;
    poi.deformation.uz = p_current.uz;
    poi.deformation.vx = p_current.vx;
    poi.deformation.vy = p_current.vy;
    poi.deformation.vz = p_current.vz;
    poi.deformation.wx = p_current.wx;
    poi.deformation.wy = p_current.wy;
    poi.deformation.wz = p_current.wz;
}

/****************************************** C++接口实现 **************************************************/

ICGN2D1BatchGpu::ICGN2D1BatchGpu() : d_ref_image(nullptr), d_tar_image(nullptr) {}
ICGN2D1BatchGpu::~ICGN2D1BatchGpu() { release_cuda(); }

void ICGN2D1BatchGpu::prepare_cuda(const float* ref_image, const float* tar_image, int h, int w, const ICGNParam& param_) {
    height = h; width = w; param = param_;
    size_t bytes = h * w * sizeof(float);
    cudaMalloc(&d_ref_image, bytes);
    cudaMalloc(&d_tar_image, bytes);
    cudaMemcpy(d_ref_image, ref_image, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tar_image, tar_image, bytes, cudaMemcpyHostToDevice);
}

void ICGN2D1BatchGpu::compute_batch_cuda(CudaPOI2D* pois, int N, cudaStream_t stream) {
    CudaPOI2D* d_pois;
    cudaMalloc(&d_pois, N * sizeof(CudaPOI2D));
    cudaMemcpyAsync(d_pois, pois, N * sizeof(CudaPOI2D), cudaMemcpyHostToDevice, stream);

    // ========== 新增全局梯度缓存 ==========
    float* d_grad_x_tar; float* d_grad_y_tar;
    size_t bytes = height * width * sizeof(float);
    cudaMalloc(&d_grad_x_tar, bytes);
    cudaMalloc(&d_grad_y_tar, bytes);

    dim3 blockDim(16, 16);
    dim3 gridDim((width+15)/16, (height+15)/16);
    sobel_gradient_kernel<<<gridDim, blockDim, 0, stream>>>(d_tar_image, height, width, d_grad_x_tar, d_grad_y_tar);
    cudaStreamSynchronize(stream);
    // ========== end ==========

    int block = 128, grid = (N + block - 1) / block;
    icgn2d1_batch_kernel<<<grid, block, 0, stream>>>(
        d_ref_image, d_tar_image, d_grad_x_tar, d_grad_y_tar,
        height, width, d_pois,
        param.subsetRadius, param.convergenceThreshold, param.maxIterations, N);

    cudaMemcpyAsync(pois, d_pois, N * sizeof(CudaPOI2D), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_grad_x_tar);
    cudaFree(d_grad_y_tar);
    cudaFree(d_pois);
}

void ICGN2D1BatchGpu::release_cuda() {
    if (d_ref_image) cudaFree(d_ref_image);
    if (d_tar_image) cudaFree(d_tar_image);
    d_ref_image = d_tar_image = nullptr;
}


ICGN2D2BatchGpu::ICGN2D2BatchGpu() : d_ref_image(nullptr), d_tar_image(nullptr) {}
ICGN2D2BatchGpu::~ICGN2D2BatchGpu() { release_cuda(); }

void ICGN2D2BatchGpu::prepare_cuda(const float* ref_image, const float* tar_image, int h, int w, const ICGNParam& param_) {
    height = h; width = w; param = param_;
    size_t bytes = h * w * sizeof(float);
    cudaMalloc(&d_ref_image, bytes);
    cudaMalloc(&d_tar_image, bytes);
    cudaMemcpy(d_ref_image, ref_image, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tar_image, tar_image, bytes, cudaMemcpyHostToDevice);
}


void ICGN2D2BatchGpu::compute_batch_cuda(CudaPOI2D* pois, int N, cudaStream_t stream) {
    CudaPOI2D* d_pois;
    cudaMalloc(&d_pois, N * sizeof(CudaPOI2D));
    cudaMemcpyAsync(d_pois, pois, N * sizeof(CudaPOI2D), cudaMemcpyHostToDevice, stream);

    // ========== 新增全局梯度缓存 ==========
    float* d_grad_x_tar; float* d_grad_y_tar;
    size_t bytes = height * width * sizeof(float);
    cudaMalloc(&d_grad_x_tar, bytes);
    cudaMalloc(&d_grad_y_tar, bytes);

    dim3 blockDim(16, 16);
    dim3 gridDim((width+15)/16, (height+15)/16);
    sobel_gradient_kernel<<<gridDim, blockDim, 0, stream>>>(d_tar_image, height, width, d_grad_x_tar, d_grad_y_tar);
    cudaStreamSynchronize(stream);
    // ========== end ==========

    int block = 128, grid = (N + block - 1) / block;
    icgn2d2_batch_kernel<<<grid, block, 0, stream>>>(
        d_ref_image, d_tar_image, height, width,
        d_grad_x_tar, d_grad_y_tar,
        d_pois,
        param.subsetRadius, param.convergenceThreshold, param.maxIterations, N);

    cudaMemcpyAsync(pois, d_pois, N * sizeof(CudaPOI2D), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_grad_x_tar);
    cudaFree(d_grad_y_tar);
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
    // ========== 新增全局梯度缓存 ==========
    // Sobel梯度图
    float *d_grad_x_tar, *d_grad_y_tar, *d_grad_z_tar;
    size_t bytes = dim_x * dim_y * dim_z * sizeof(float);
    cudaMalloc(&d_grad_x_tar, bytes);
    cudaMalloc(&d_grad_y_tar, bytes);
    cudaMalloc(&d_grad_z_tar, bytes);

    dim3 blockDim(8, 8, 8);
    dim3 gridDim((dim_x + 7) / 8, (dim_y + 7) / 8, (dim_z + 7) / 8);
    compute_gradients_3d<<<gridDim, blockDim, 0, stream>>>(d_tar_image, dim_x, dim_y, dim_z,
                                                            d_grad_x_tar, d_grad_y_tar, d_grad_z_tar);
    cudaStreamSynchronize(stream);
    // ========== end ==========

    int block = 128, grid = (N + block - 1) / block;
    icgn3d1_batch_kernel<<<grid, block, 0, stream>>>(
        d_ref_image, d_tar_image,
        d_grad_x_tar, d_grad_y_tar, d_grad_z_tar,
        dim_x, dim_y, dim_z,
        d_pois,
        param.subsetRadius, param.convergenceThreshold, param.maxIterations, N);

    cudaMemcpyAsync(pois, d_pois, N * sizeof(CudaPOI3D), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_grad_x_tar);
    cudaFree(d_grad_y_tar);
    cudaFree(d_grad_z_tar);
}
void ICGN3D1BatchGpu::release_cuda() {
    if (d_ref_image) cudaFree(d_ref_image);
    if (d_tar_image) cudaFree(d_tar_image);
    d_ref_image = d_tar_image = nullptr;
}

} // namespace StudyCorr