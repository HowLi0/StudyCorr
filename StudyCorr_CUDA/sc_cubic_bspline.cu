#include "sc_cubic_bspline.h"
#include <cuda_runtime.h>
#include <cmath>

// ====== CUDA常量表 ======
__constant__ float CONTROL_MATRIX[4][4];
__constant__ float FUNCTION_MATRIX[4][4];
__constant__ float BSPLINE_PREFILTER[8];

namespace StudyCorr {

// 构造器：记住图像指针、分配系数
BicubicBsplineInterp::BicubicBsplineInterp(CudaImage2D& img)
    : coeff(img.height, img.width, 4, 4)
{
    interp_img = &img;
}

// CUDA核：每像素系数构建
__global__ void kernel_bspline2d_prepare(
    const float* img, int h, int w, float* coeff)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < h && c < w && r >= 1 && c >= 1 && r < h-2 && c < w-2) {
        float g[4][4] = {0}, b[4][4] = {0};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                g[i][j] = img[(r-1+i)*w + (c-1+j)];
        for (int k = 0; k < 4; ++k)
        for (int l = 0; l < 4; ++l)
        for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n)
            b[k][l] += CONTROL_MATRIX[k][m] * CONTROL_MATRIX[l][n] * g[n][m];

        for (int k = 0; k < 4; ++k)
        for (int l = 0; l < 4; ++l) {
            float sum = 0;
            for (int m = 0; m < 4; ++m)
            for (int n = 0; n < 4; ++n)
                sum += FUNCTION_MATRIX[k][m] * FUNCTION_MATRIX[l][n] * b[n][m];
            // 对称交换
            int kk = k, ll = l;
            if (k < 2) { kk = 3 - k; ll = 3 - l; }
            int idx1 = ((r* w + c) * 4 + k)*4 + l;
            int idx2 = ((r* w + c) * 4 + kk)*4 + ll;
            float* coeff_data = coeff;
            if (k < 2) {
                float tmp = sum;
                sum = coeff_data[idx2];
                coeff_data[idx2] = tmp;
            }
            coeff_data[idx1] = sum;
        }
    }
}

void BicubicBsplineInterp::prepare()
{
    // 初始化常量表
    float h_CONTROL_MATRIX[4][4] = {
        { 71.0f / 56.0f, -19.0f / 56.0f, 5.0f / 56.0f, -1.0f / 56.0f },
        { -19.0f / 56.0f, 95.0f / 56.0f, -25.0f / 56.0f, 5.0f / 56.0f },
        { 5.0f / 56.0f, -25.0f / 56.0f, 95.0f / 56.0f, -19.0f / 56.0f },
        { -1.0f / 56.0f, 5.0f / 56.0f, -19.0f / 56.0f, 71.0f / 56.0f }
    };
    float h_FUNCTION_MATRIX[4][4] = {
        { -1.0f / 6.0f, 3.0f / 6.0f, -3.0f / 6.0f, 1.0f / 6.0f },
        { 3.0f / 6.0f, -6.0f / 6.0f, 3.0f / 6.0f, 0.0f },
        { -3.0f / 6.0f, 0.0f, 3.0f / 6.0f, 0.0f },
        { 1.0f / 6.0f, 4.0f / 6.0f, 1.0f / 6.0f, 0.0f }
    };
    cudaMemcpyToSymbol(CONTROL_MATRIX, h_CONTROL_MATRIX, sizeof(float)*16);
    cudaMemcpyToSymbol(FUNCTION_MATRIX, h_FUNCTION_MATRIX, sizeof(float)*16);

    int h = interp_img->height, w = interp_img->width;
    dim3 block(16, 16), grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    kernel_bspline2d_prepare<<<grid, block>>>(interp_img->data.get(), h, w, coeff.get());
    cudaDeviceSynchronize();
}

// GPU设备侧插值（可供kernel调用）
__device__ float BicubicBsplineInterp::interp_at(float x, float y) const
{
    int w = interp_img->width, h = interp_img->height;
    if (x < 0 || y < 0 || x >= w || y >= h || isnan(x) || isnan(y))
        return -1.f;
    int xi = (int)floorf(x), yi = (int)floorf(y);
    float xd = x - xi, yd = y - yi;
    float x2 = xd*xd, y2 = yd*yd, x3 = x2*xd, y3 = y2*yd;
    float val = 0;
    for (int k = 0; k < 4; ++k)
    for (int l = 0; l < 4; ++l) {
        float basis_k = (k==0?1: (k==1?yd: (k==2?y2:y3)));
        float basis_l = (l==0?1: (l==1?xd: (l==2?x2:x3)));
        val += coeff(yi, xi, k, l) * basis_k * basis_l;
    }
    return val;
}

// Host接口：单点插值
float BicubicBsplineInterp::compute(Point2D& location)
{
    float x = location.x, y = location.y;
    float result;
    // 用host侧Array1D做1个元素
    Array1D<float> xs(1), ys(1), outs(1);
    xs[0] = x;
    ys[0] = y;
    dim3 block(1), grid(1);
    bspline2d_interp_kernel<<<grid, block>>>(*this, xs.get(), ys.get(), outs.get(), 1);
    cudaDeviceSynchronize();
    result = outs[0];
    return result;
}

// 批量插值核
__global__ void bspline2d_interp_kernel(
    const BicubicBsplineInterp interp,
    const float* xs, const float* ys, float* outs, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        outs[idx] = interp.interp_at(xs[idx], ys[idx]);
}


// 构造器
TricubicBsplineInterp::TricubicBsplineInterp(CudaImage3D& img)
    : coeff(img.dim_z, img.dim_y, img.dim_x)
{
    interp_img = &img;
}

// CUDA核：一次沿x, y, z卷积生成系数
__global__ void kernel_bspline3d_prepare(
    const float* img, int d, int h, int w, float* coeff, const float* prefilter)
{
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= d || y >= h || x >= w) return;
    // 仅在有效范围卷积（边界可补全）
    float sum = prefilter[0] * img[(z*h + y)*w + x];
    for (int i = 1; i < 8; ++i) {
        int xm = max(x - i, 0), xp = min(x + i, w - 1);
        int ym = max(y - i, 0), yp = min(y + i, h - 1);
        int zm = max(z - i, 0), zp = min(z + i, d - 1);
        sum += prefilter[i] * (
            img[(z*h + y)*w + xm] + img[(z*h + y)*w + xp] +
            img[(z*h + ym)*w + x] + img[(z*h + yp)*w + x] +
            img[(zm*h + y)*w + x] + img[(zp*h + y)*w + x]
        );
    }
    coeff[(z*h + y)*w + x] = sum;
}

void TricubicBsplineInterp::prepare()
{
    // B样条预滤波常量
    float h_PREFILTER[8] = {
        1.732176555412860f,  //b0
        -0.464135309171000f, //b1
        0.124364681271139f,  //b2
        -0.033323415913556f, //b3
        0.008928982383084f,  //b4
        -0.002392513618779f, //b5
        0.000641072092032f,  //b6
        -0.000171774749350f  //b7
    };
    cudaMemcpyToSymbol(BSPLINE_PREFILTER, h_PREFILTER, sizeof(float)*8);

    int d = interp_img->dim_z, h = interp_img->dim_y, w = interp_img->dim_x;
    dim3 block(8, 8, 8), grid((w+block.x-1)/block.x, (h+block.y-1)/block.y, (d+block.z-1)/block.z);
    kernel_bspline3d_prepare<<<grid, block>>>(interp_img->data.get(), d, h, w, coeff.get(), BSPLINE_PREFILTER);
    cudaDeviceSynchronize();
}

// 设备端三次B样条基函数
__device__ float basis0(float u) { return (1.f/6.f)*(u*(u*(-u+3.f)-3.f)+1.f);}//(1/6)*(2-(x+1))^3 for x-(-1)
__device__ float basis1(float u) { return (1.f/6.f)*(u*u*(3.f*u-6.f)+4.f);}//(1/6)*(x+1)^3 for x-(-1)
__device__ float basis2(float u) { return (1.f/6.f)*(u*(u*(-3.f*u+3.f)+3.f)+1.f);}//(1/6)*(2-(x+1))^3 for x-0
__device__ float basis3(float u) { return (1.f/6.f)*(u*u*u);}//(1/6)*(x+1)^3 for x-0

__device__ float TricubicBsplineInterp::interp_at(float x, float y, float z) const
{
    int w = interp_img->dim_x, h = interp_img->dim_y, d = interp_img->dim_z;
    if (x < 1 || y < 1 || z < 1 || x >= w-2 || y >= h-2 || z >= d-2
        || isnan(x) || isnan(y) || isnan(z)) return -1.f;
    int xi = (int)floorf(x);
    int yi = (int)floorf(y);
    int zi = (int)floorf(z);
    float xd = x-xi, yd = y-yi, zd = z-zi;

    float bx[4] = {basis0(xd), basis1(xd), basis2(xd), basis3(xd)};
    float by[4] = {basis0(yd), basis1(yd), basis2(yd), basis3(yd)};
    float bz[4] = {basis0(zd), basis1(zd), basis2(zd), basis3(zd)};

    float sum_x[4], sum_y[4];
    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
            sum_x[j] = bx[0]*coeff((zi+i-1),(yi+j-1),(xi-1))
                     + bx[1]*coeff((zi+i-1),(yi+j-1),(xi))
                     + bx[2]*coeff((zi+i-1),(yi+j-1),(xi+1))
                     + bx[3]*coeff((zi+i-1),(yi+j-1),(xi+2));
        }
        sum_y[i] = by[0]*sum_x[0] + by[1]*sum_x[1] + by[2]*sum_x[2] + by[3]*sum_x[3];
    }
    float val = bz[0]*sum_y[0] + bz[1]*sum_y[1] + bz[2]*sum_y[2] + bz[3]*sum_y[3];
    return val;
}

// Host接口，自动调用device核
float TricubicBsplineInterp::compute(Point3D& location)
{
    float x = location.x, y = location.y, z = location.z;
    Array1D<float> xs(1), ys(1), zs(1), outs(1);
    xs[0] = x; ys[0] = y; zs[0] = z;
    dim3 block(1), grid(1);
    bspline3d_interp_kernel<<<grid, block>>>(*this, xs.get(), ys.get(), zs.get(), outs.get(), 1);
    cudaDeviceSynchronize();
    return outs[0];
}

// 批量插值核
__global__ void bspline3d_interp_kernel(
    const TricubicBsplineInterp interp,
    const float* xs, const float* ys, const float* zs, float* outs, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        outs[idx] = interp.interp_at(xs[idx], ys[idx], zs[idx]);
}

}