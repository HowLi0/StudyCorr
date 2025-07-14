#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include "sc_array.h"

namespace StudyCorr
{

    class CudaImage2D {
    public:
        int width = 0, height = 0;
        std::string file_path;

        Array2D<float> data; // CUDA可访问的图像数据

        CudaImage2D() = default;

        // 新建空图像
        CudaImage2D(int width_, int height_)
            : width(width_), height(height_), data(height_, width_) {}

        // 从文件加载（灰度图）
        CudaImage2D(const std::string& file_path_) : file_path(file_path_) {
            load(file_path_);
        }

        void load(const std::string& file_path_);
    };


    class CudaImage3D {
    public:
        int dim_x = 0, dim_y = 0, dim_z = 0;
        std::string file_path;

        Array3D<float> data; // CUDA可访问的体数据

        CudaImage3D() = default;

        CudaImage3D(int dim_x_, int dim_y_, int dim_z_)
            : dim_x(dim_x_), dim_y(dim_y_), dim_z(dim_z_), data(dim_z_, dim_y_, dim_x_) {}

        void load_from_bin(const std::string& file_path_);

        void load_from_tiff(const std::string& file_path_);
    };

} // namespace StudyCorr