/*
 * This file is part of StudyCorr, based on OpenCorr functionality.
 * Compatible with both OpenCV and Qt point types.
 */

#pragma once

#ifndef _STUDYCORR_SUBSET_H_
#define _STUDYCORR_SUBSET_H_

#include "sc_image.h"
#include "sc_point.h"

namespace StudyCorr_GPU
{
    class Subset2D
    {
    public:
        Point2D center;
        int radius_x, radius_y;
        int height, width;
        int size;

        cv::Mat subset_mat;  // OpenCV matrix

        Subset2D(Point2D center, int radius_x, int radius_y);
        ~Subset2D()=default;

        void fill(Image2D* image);
        float zeroMeanNorm();
    };

    class Subset3D
    {
    public:
        Point3D center;
        int radius_x, radius_y, radius_z;
        int dim_x, dim_y, dim_z;
        int size;

        float*** vol_mat = nullptr;  // 3D subset volume using OpenCV

        Subset3D(Point3D center, int radius_x, int radius_y, int radius_z);
        ~Subset3D();

        void fill(Image3D* image);
        float zeroMeanNorm();
    };

}//namespace StudyCorr

#endif //_STUDYCORR_SUBSET_H_