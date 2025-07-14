#pragma once
#include <cmath>
#include <cassert>
#include <string>
#include "sc_array.h" 
#include "sc_point.h"    
#include "sc_image.h"  

namespace StudyCorr 
{

    class CudaSubset2D {
    public:
        Point2D center;
        int radius_x, radius_y;
        int width, height;
        Array2D<float> data; // CUDA可访问

        CudaSubset2D(Point2D center_, int radius_x_, int radius_y_)
            : center(center_), radius_x(radius_x_), radius_y(radius_y_),
            width(2*radius_x_+1), height(2*radius_y_+1), data(2*radius_y_+1, 2*radius_x_+1)
        {
            assert(radius_x_ >= 1 && radius_y_ >= 1);
        }

        // 从图像中提取子集
        void fill(const CudaImage2D& image);

        // 零均值归一化
        float zeroMeanNorm();
    };

    class CudaSubset3D {
    public:
        Point3D center;
        int radius_x, radius_y, radius_z;
        int dim_x, dim_y, dim_z;
        Array3D<float> data;

        CudaSubset3D(Point3D center_, int rx, int ry, int rz)
            : center(center_), radius_x(rx), radius_y(ry), radius_z(rz),
            dim_x(2*rx+1), dim_y(2*ry+1), dim_z(2*rz+1), data(2*rz+1, 2*ry+1, 2*rx+1)
        {
            assert(rx>=1 && ry>=1 && rz>=1);
        }

        void fill(const CudaImage3D& image);

        //     int x0 = center.x - radius_x;
        //     int y0 = center.y - radius_y;
        //     int z0 = center.z - radius_z;
        //     for(int z=0; z<dim_z; ++z)
        //         for(int y=0; y<dim_y; ++y)
        //             for(int x=0; x<dim_x; ++x) {
        //                 int gx = x0 + x, gy = y0 + y, gz = z0 + z;
        //                 if(gx>=0 && gx<image.dim_x && gy>=0 && gy<image.dim_y && gz>=0 && gz<image.dim_z)
        //                     data(z,y,x) = image.data(z,y,x);
        //                 else
        //                     data(z,y,x) = 0;
        //             }
        // 

        float zeroMeanNorm();
        // {
        //     float mean = 0.0f, cnt = float(dim_x*dim_y*dim_z);
        //     for(int z=0; z<dim_z; ++z)
        //         for(int y=0; y<dim_y; ++y)
        //             for(int x=0; x<dim_x; ++x)
        //                 mean += data(z,y,x);
        //     mean /= cnt;

        //     float sum = 0.0f;
        //     for(int z=0; z<dim_z; ++z)
        //         for(int y=0; y<dim_y; ++y)
        //             for(int x=0; x<dim_x; ++x) {
        //                 data(z,y,x) -= mean;
        //                 sum += data(z,y,x)*data(z,y,x);
        //             }
        //     return std::sqrt(sum);
        // }
    };

} // namespace StudyCorr