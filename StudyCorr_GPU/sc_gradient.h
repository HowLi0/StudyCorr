#pragma once

#ifndef _GRADIENT_H_
#define _GRADIENT_H_

#include "sc_array.h"
#include "sc_image.h"

namespace StudyCorr_GPU
{
    class Gradient2D4
    {
    protected:
        Image2D* grad_img = nullptr;

    public:
        cv::Mat gradient_x;
        cv::Mat gradient_y;
        cv::Mat gradient_xy;

        Gradient2D4(Image2D& image);
        ~Gradient2D4();

        void getGradientX(); //create an array of gradient_x
        void getGradientY(); //create an array of gradient_y
        void getGradientXY(); //create an array of gradient_xy
    };

    class Gradient3D4
    {
    protected:
        Image3D* grad_img = nullptr;

    public:
        float*** gradient_x = nullptr;
        float*** gradient_y = nullptr;
        float*** gradient_z = nullptr;

        Gradient3D4(Image3D& image);
        ~Gradient3D4();

        void getGradientX(); //create an array of gradient_x
        void getGradientY(); //create an array of gradient_y
        void getGradientZ(); //create an array of gradient_z
    };

}//namespace StudyCorr

#endif //_GRADIENT_H_
