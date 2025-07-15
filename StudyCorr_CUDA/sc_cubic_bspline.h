#pragma once
#include "sc_image.h"
#include "sc_array.h"
#include "sc_interpolation.h"

/*
#include "sc_bspline.h"
using namespace StudyCorr;

CudaImage2D img("your_image.png"); // 加载图像
BicubicBsplineInterp interp(img);
interp.prepare(); // CUDA端并行prepare

Point2D pt{23.4f, 51.8f};
float val = interp.compute(pt); // host单点插值

// 批量插值
int N = ...;
Array1D<float> xs(N), ys(N), outs(N);
// 填充 xs, ys
dim3 block(128), grid((N+127)/128);
bspline2d_interp_kernel<<<grid, block>>>(interp, xs.get(), ys.get(), outs.get(), N);
cudaDeviceSynchronize();
*/

namespace StudyCorr
{

    // 2D三次B样条插值
    class BicubicBsplineInterp : public CudaInterpolation2D
    {
    public:
        Array4D<float> coeff;   // [height][width][4][4] 插值系数（CUDA managed）

        BicubicBsplineInterp(CudaImage2D& img);
        ~BicubicBsplineInterp() = default;

        void prepare() override;                        // CUDA上并行构建系数
        float compute(Point2D& location) override;      // host接口，自动调用device

        // device内核用
        __device__ float interp_at(float x, float y) const;
    };

    // 批量插值核
    __global__ void bspline2d_interp_kernel(
        const BicubicBsplineInterp interp,
        const float* xs, const float* ys, float* outs, int n);


    class TricubicBsplineInterp : public CudaInterpolation3D
    {
    public:
        Array3D<float> coeff;  // [z][y][x] 插值系数（CUDA managed）

        TricubicBsplineInterp(CudaImage3D& img);

        void prepare() override;                       // CUDA并行预处理
        float compute(Point3D& location) override;     // host接口，自动调用device

        // device内核用
        __device__ float interp_at(float x, float y, float z) const;
    };

    // 批量插值核
    __global__ void bspline3d_interp_kernel(
        const TricubicBsplineInterp interp,
        const float* xs, const float* ys, const float* zs, float* outs, int n);

    CUDA_HD int getLow(int x, int y)
    {
        return (x < y ? x : y);
    }
    CUDA_HD int getHigh(int x, int y)
    {
        return (x > y ? x : y);
    }
}
