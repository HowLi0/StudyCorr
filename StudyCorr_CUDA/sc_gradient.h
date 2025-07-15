#pragma once
#include "sc_array.h"
#include "sc_image.h"

#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif

namespace StudyCorr
{

    // 四阶中心差分精度的2D一阶梯度
    //grad = [ -f(c+2) + 8*f(c+1) - 8*f(c-1) + f(c-2) ] / 12
    class Gradient2D4
    {
    public:
        const CudaImage2D* image = nullptr;

        Array2D<float> grad_x, grad_y, grad_xy;

        CUDA_HD Gradient2D4(const CudaImage2D& img)
            : image(&img),
            grad_x(img.height, img.width),
            grad_y(img.height, img.width),
            grad_xy(img.height, img.width)
        {}

        CUDA_HD ~Gradient2D4();

        void getGradientX();
        void getGradientY();
        void getGradientXY();

    };


    // 四阶精度的3D一阶梯度
    class Gradient3D4
    {
    public:
        const CudaImage3D* image = nullptr;

        Array3D<float> grad_x, grad_y, grad_z;

        CUDA_HD Gradient3D4(const CudaImage3D& img)
            : image(&img),
            grad_x(img.dim_z, img.dim_y, img.dim_x),
            grad_y(img.dim_z, img.dim_y, img.dim_x),
            grad_z(img.dim_z, img.dim_y, img.dim_x)
        {}

        CUDA_HD ~Gradient3D4();

        // 计算X方向梯度
        void getGradientX();

        // 计算Y方向梯度
        void getGradientY();

        // 计算Z方向梯度
        void getGradientZ();
    };

} // namespace StudyCorr