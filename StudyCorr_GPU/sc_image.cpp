/*
 * This file is part of StudyCorr, based on OpenCorr functionality.
 * Compatible with both OpenCV and Qt point types.
 */

#include "sc_image.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

namespace StudyCorr_GPU
{
    // Image2D 类的实现
    Image2D::Image2D(int width, int height) : width(width), height(height)
    {
        cv_mat = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));//创建了一个单通道的灰度图像矩阵，矩阵中的每个元素都是一个无符号 8 位整数，取值范围从 0 到 255。
    }

    Image2D::Image2D(std::string file_path) : file_path(file_path)
    {
        load(file_path);
    }

    Image2D::~Image2D()
    {
        release();
        data_ptrs = nullptr; // 确保指针被清空
    }

    void Image2D::load(std::string file_path)
    {
        this->file_path = file_path;
        cv::Mat img = cv::imread(file_path, cv::IMREAD_UNCHANGED); // 保持原始通道数载入

        if (img.empty())
        {
            std::cerr << "Failed to load image: " << file_path << std::endl;
            return;
        }

        // 转成灰度图
        cv::Mat gray;
        if (img.channels() == 1)
            gray = img;
        else
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        height = gray.rows;
        width = gray.cols;

        gray.convertTo(cv_mat, CV_32F);
        float_data.resize(cv_mat.total());
        memcpy(float_data.data(), cv_mat.ptr<float>(0), sizeof(float)*cv_mat.total());
        data_ptrs = float_data.data();
    }

    void Image2D::release()
    {
        cv_mat.release();
    }

    // Image3D 类的实现
    Image3D::Image3D(int dim_x, int dim_y, int dim_z) : dim_x(dim_x), dim_y(dim_y), dim_z(dim_z)
    {
		vol_mat = new3D(dim_z, dim_y, dim_x);
		this->dim_x = dim_x;
		this->dim_y = dim_y;
		this->dim_z = dim_z;
    }

    Image3D::Image3D(std::string file_path) : file_path(file_path)
    {
        load(file_path);
    }

    Image3D::~Image3D()
    {
        if (vol_mat != nullptr)
        {
            delete3D(vol_mat);
        }
    }

    void Image3D::loadBin(std::string file_path)
    {
        // 释放之前的内存
        if (vol_mat != nullptr)
        {
			delete3D(vol_mat);
        }

        std::ifstream file_in(file_path, std::ios::binary);
        if (!file_in)
        {
            std::cerr << "Failed to open bin file: " << file_path << std::endl;
            return;
        }

        this->file_path = file_path;

        // 获取文件长度
        file_in.seekg(0, std::ios::end);
        std::streamsize file_length = file_in.tellg();
        file_in.seekg(0, std::ios::beg);
        std::streamsize data_length = file_length - sizeof(int) * 3;

        // 读取头部信息
        int img_dimension[3];
        file_in.read(reinterpret_cast<char*>(img_dimension), sizeof(int) * 3);
        dim_x = img_dimension[0];
        dim_y = img_dimension[1];
        dim_z = img_dimension[2];

        // 创建 3D 矩阵
        vol_mat = new3D(dim_z, dim_y, dim_x);

        // 读取数据
        file_in.read(reinterpret_cast<char*>(**vol_mat), sizeof(float) * dim_x * dim_y * dim_z);

        file_in.close();    
    }

    void Image3D::loadTiff(std::string file_path)
    {
        // 释放之前的内存
        if (vol_mat != nullptr)
        {
            delete3D(vol_mat);
        }

        // 读取多页 TIFF 图像
        std::vector<cv::Mat> tiff_mat;
        if (!cv::imreadmulti(file_path, tiff_mat, cv::IMREAD_GRAYSCALE))
        {
            std::cerr << "Fail to load multi-page tiff: " << file_path << std::endl;
            return;
        }

        // 获取 3D 图像的三个维度
        dim_x = tiff_mat[0].cols;
        dim_y = tiff_mat[0].rows;
        dim_z = static_cast<int>(tiff_mat.size());

        // 创建 3D 矩阵
        vol_mat = new3D(dim_z, dim_y, dim_x);

        // 填充数据
#pragma omp parallel for
        for (int i = 0; i < dim_z; ++i)
        {
            for (int j = 0; j < dim_y; ++j)
            {
                for (int k = 0; k < dim_x; ++k)
                {
                    vol_mat[i][j][k] = (float)tiff_mat[i].at<uchar>(j, k);
                }
            }
        }
    }

    void Image3D::load(std::string file_path)
    {
        // 检查文件扩展名
        size_t dot_pos = file_path.find_last_of('.');
        if (dot_pos == std::string::npos)
        {
            std::cerr << "Invalid file path: no file extension" << std::endl;
            return;
        }
        std::string file_ext = file_path.substr(dot_pos + 1);
        std::transform(file_ext.begin(), file_ext.end(), file_ext.begin(), ::tolower);

        if (file_ext == "bin")
        {
            loadBin(file_path);
        }
        else if (file_ext == "tif" || file_ext == "tiff")
        {
            loadTiff(file_path);
        }
        else
        {
            std::cerr << "Not binary file or multi-page tiff" << std::endl;
        }
    }

}//namespace StudyCorr