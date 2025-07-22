/*
 * This file is part of StudyCorr, based on OpenCorr functionality.
 * Compatible with both OpenCV and Qt point types.
 */

#pragma once

#ifndef _STUDYCORR_IMAGE_H_
#define _STUDYCORR_IMAGE_H_

#include <opencv2/opencv.hpp>
#include <string>
#include "sc_array.h"
#include "sc_point.h"

namespace StudyCorr_GPU
{
    class Image2D
    {
    public:
        int height, width;
        std::string file_path;

        cv::Mat cv_mat; 
        std::vector<float> float_data;  // 新增
        const float* data_ptrs = nullptr; // 用于存储图像数据指针

        Image2D(int width, int height);
        Image2D(std::string file_path);
        ~Image2D();

        void load(std::string file_path);
        void release();
    };

    class Image3D
    {
    public:
        int dim_x, dim_y, dim_z;
        std::string file_path;

        float*** vol_mat = nullptr;

        Image3D(int dim_x, int dim_y, int dim_z);
        Image3D(std::string file_path);
        ~Image3D();

        void loadBin(std::string file_path);
        void loadTiff(std::string file_path);
        void load(std::string file_path);
    };
}//namespace StudyCorr

#endif //_STUDYCORR_IMAGE_H_