/*
 * This file is part of StudyCorr_GPU, based on OpenCorr functionality.
 * Compatible with both OpenCV and Qt point types.
 */

#ifndef _STUDYCORR_DEFORMATION_H_
#define _STUDYCORR_DEFORMATION_H_

#include "sc_point.h" 
#include <opencv2/core.hpp>

namespace StudyCorr_GPU
{
    //2D deformation with the 1st order shape function
    class Deformation2D1
    {
    public:
        float u, ux, uy;
        float v, vx, vy;
        // Use OpenCV matrix instead of Eigen
        cv::Mat warp_matrix;

        Deformation2D1();
        Deformation2D1(float u, float ux, float uy, float v, float vx, float vy);
        Deformation2D1(float p[6]);
        ~Deformation2D1();

        void setDeformation(); // set deformation according to warp_matrix
        void setDeformation(float u, float ux, float uy, float v, float vx, float vy);
        void setDeformation(float p[6]);
        void setDeformation(Deformation2D1& another_deformation);

        void setWarp(); // update warp_matrix according to deformation
        Point2D warp(const Point2D& location) const;
    };

    //2D deformation with the 2nd order shape function
    class Deformation2D2
    {
    public:
        float u, ux, uy, uxx, uxy, uyy;
        float v, vx, vy, vxx, vxy, vyy;
        // Use OpenCV matrix instead of Eigen
        cv::Mat warp_matrix;

        Deformation2D2();
        Deformation2D2(float u, float ux, float uy, float uxx, float uxy, float uyy,
            float v, float vx, float vy, float vxx, float vxy, float vyy);
        Deformation2D2(float p[12]);
        ~Deformation2D2();

        void setDeformation(); // set deformation according to warp_matrix
        void setDeformation(float u, float ux, float uy, float uxx, float uxy, float uyy,
            float v, float vx, float vy, float vxx, float vxy, float vyy);
        void setDeformation(float p[12]);
        void setDeformation(Deformation2D2& another_deformation);
        void setDeformation(Deformation2D1& another_deformation);

        void setWarp(); // update warp_matrix according to deformation
        Point2D warp(const Point2D& location) const;
    };

    //3D deformation with the 1st order shape function
    class Deformation3D1
    {
    public:
        float u, ux, uy, uz;
        float v, vx, vy, vz;
        float w, wx, wy, wz;
        // Use OpenCV matrix instead of Eigen
        cv::Mat warp_matrix;

        Deformation3D1();
        Deformation3D1(float u, float ux, float uy, float uz,
            float v, float vx, float vy, float vz,
            float w, float wx, float wy, float wz);
        Deformation3D1(float p[12]);
        ~Deformation3D1();

        void setDeformation(); // set deformation according to warp_matrix
        void setDeformation(float u, float ux, float uy, float uz,
            float v, float vx, float vy, float vz,
            float w, float wx, float wy, float wz);
        void setDeformation(float p[12]);
        void setDeformation(Deformation3D1& another_deformation);

        void setWarp(); // update warp_matrix according to deformation
        Point3D warp(const Point3D& point) const;
    };

} // namespace StudyCorr_GPU

#endif // _STUDYCORR_DEFORMATION_H_
