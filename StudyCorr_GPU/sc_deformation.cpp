/*
 * This file is part of StudyCorr_GPU, based on OpenCorr functionality.
 * Compatible with both OpenCV and Qt point types.
 */

#include "sc_deformation.h"

namespace StudyCorr_GPU
{
    // Deformation2D1 implementation
    Deformation2D1::Deformation2D1()
    {
        u = 0.f;
        ux = 0.f;
        uy = 0.f;

        v = 0.f;
        vx = 0.f;
        vy = 0.f;

        // Initialize warp_matrix to identity matrix
        warp_matrix = cv::Mat::eye(3, 3, CV_32F);
    }

    Deformation2D1::Deformation2D1(float u, float ux, float uy, float v, float vx, float vy)
    {
        this->u = u;
        this->ux = ux;
        this->uy = uy;

        this->v = v;
        this->vx = vx;
        this->vy = vy;

        warp_matrix = cv::Mat::zeros(3, 3, CV_32F);
        setWarp();
    }

    Deformation2D1::Deformation2D1(float p[6])
    {
        u = p[0];
        ux = p[1];
        uy = p[2];

        v = p[3];
        vx = p[4];
        vy = p[5];

        warp_matrix = cv::Mat::zeros(3, 3, CV_32F);
        setWarp();
    }

    Deformation2D1::~Deformation2D1()
    {
        // Nothing to do
    }

    void Deformation2D1::setDeformation(float u, float ux, float uy, float v, float vx, float vy)
    {
        this->u = u;
        this->ux = ux;
        this->uy = uy;

        this->v = v;
        this->vx = vx;
        this->vy = vy;

        setWarp();
    }

    void Deformation2D1::setDeformation(float p[6])
    {
        u = p[0];
        ux = p[1];
        uy = p[2];

        v = p[3];
        vx = p[4];
        vy = p[5];

        setWarp();
    }

    void Deformation2D1::setDeformation(Deformation2D1& another_deformation)
    {
        u = another_deformation.u;
        ux = another_deformation.ux;
        uy = another_deformation.uy;

        v = another_deformation.v;
        vx = another_deformation.vx;
        vy = another_deformation.vy;

        setWarp();
    }

    void Deformation2D1::setDeformation()
    {
        u = warp_matrix.at<float>(0, 2);
        ux = warp_matrix.at<float>(0, 0) - 1.f;
        uy = warp_matrix.at<float>(0, 1);

        v = warp_matrix.at<float>(1, 2);
        vx = warp_matrix.at<float>(1, 0);
        vy = warp_matrix.at<float>(1, 1) - 1.f;
    }

    void Deformation2D1::setWarp()
    {
        // Fill the warp matrix based on deformation parameters
        warp_matrix.at<float>(0, 0) = 1.f + ux;
        warp_matrix.at<float>(0, 1) = uy;
        warp_matrix.at<float>(0, 2) = u;

        warp_matrix.at<float>(1, 0) = vx;
        warp_matrix.at<float>(1, 1) = 1.f + vy;
        warp_matrix.at<float>(1, 2) = v;

        warp_matrix.at<float>(2, 0) = 0.f;
        warp_matrix.at<float>(2, 1) = 0.f;
        warp_matrix.at<float>(2, 2) = 1.f;
    }

    Point2D Deformation2D1::warp(const Point2D& location) const
    {
        // Create a column vector for the point coordinates in homogeneous form
        cv::Mat point_vector(3, 1, CV_32F);
        point_vector.at<float>(0, 0) = location.x;
        point_vector.at<float>(1, 0) = location.y;
        point_vector.at<float>(2, 0) = 1.0f;

        // Perform matrix multiplication
        cv::Mat result = warp_matrix * point_vector;

        // Return the new point coordinates
        return Point2D(result.at<float>(0, 0), result.at<float>(1, 0));
    }

    // Deformation2D2 implementation
    Deformation2D2::Deformation2D2()
    {
        u = 0.f;
        ux = 0.f;
        uy = 0.f;
        uxx = 0.f;
        uxy = 0.f;
        uyy = 0.f;

        v = 0.f;
        vx = 0.f;
        vy = 0.f;
        vxx = 0.f;
        vxy = 0.f;
        vyy = 0.f;

        // Initialize warp_matrix
        warp_matrix = cv::Mat::zeros(6, 6, CV_32F);
        setWarp();
    }

    Deformation2D2::Deformation2D2(float u, float ux, float uy, float uxx, float uxy, float uyy,
        float v, float vx, float vy, float vxx, float vxy, float vyy)
    {
        this->u = u;
        this->ux = ux;
        this->uy = uy;
        this->uxx = uxx;
        this->uxy = uxy;
        this->uyy = uyy;

        this->v = v;
        this->vx = vx;
        this->vy = vy;
        this->vxx = vxx;
        this->vxy = vxy;
        this->vyy = vyy;

        warp_matrix = cv::Mat::zeros(6, 6, CV_32F);
        setWarp();
    }

    Deformation2D2::Deformation2D2(float p[12])
    {
        u = p[0];
        ux = p[1];
        uy = p[2];
        uxx = p[3];
        uxy = p[4];
        uyy = p[5];

        v = p[6];
        vx = p[7];
        vy = p[8];
        vxx = p[9];
        vxy = p[10];
        vyy = p[11];

        warp_matrix = cv::Mat::zeros(6, 6, CV_32F);
        setWarp();
    }

    Deformation2D2::~Deformation2D2() {}

    void Deformation2D2::setDeformation(float u, float ux, float uy, float uxx, float uxy, float uyy,
        float v, float vx, float vy, float vxx, float vxy, float vyy)
    {
        this->u = u;
        this->ux = ux;
        this->uy = uy;
        this->uxx = uxx;
        this->uxy = uxy;
        this->uyy = uyy;

        this->v = v;
        this->vx = vx;
        this->vy = vy;
        this->vxx = vxx;
        this->vxy = vxy;
        this->vyy = vyy;

        setWarp();
    }

    void Deformation2D2::setDeformation(float p[12])
    {
        u = p[0];
        ux = p[1];
        uy = p[2];
        uxx = p[3];
        uxy = p[4];
        uyy = p[5];

        v = p[6];
        vx = p[7];
        vy = p[8];
        vxx = p[9];
        vxy = p[10];
        vyy = p[11];

        setWarp();
    }

    void Deformation2D2::setDeformation(Deformation2D2& another_deformation)
    {
        u = another_deformation.u;
        ux = another_deformation.ux;
        uy = another_deformation.uy;
        uxx = another_deformation.uxx;
        uxy = another_deformation.uxy;
        uyy = another_deformation.uyy;

        v = another_deformation.v;
        vx = another_deformation.vx;
        vy = another_deformation.vy;
        vxx = another_deformation.vxx;
        vxy = another_deformation.vxy;
        vyy = another_deformation.vyy;

        setWarp();
    }

    void Deformation2D2::setDeformation(Deformation2D1& another_deformation)
    {
        u = another_deformation.u;
        ux = another_deformation.ux;
        uy = another_deformation.uy;
        uxx = 0.f;
        uxy = 0.f;
        uyy = 0.f;

        v = another_deformation.v;
        vx = another_deformation.vx;
        vy = another_deformation.vy;
        vxx = 0.f;
        vxy = 0.f;
        vyy = 0.f;

        setWarp();
    }

    void Deformation2D2::setDeformation()
    {
        u = warp_matrix.at<float>(3, 5);
        ux = warp_matrix.at<float>(3, 3) - 1.f;
        uy = warp_matrix.at<float>(3, 4);
        uxx = warp_matrix.at<float>(3, 0) * 2.f;
        uxy = warp_matrix.at<float>(3, 1);
        uyy = warp_matrix.at<float>(3, 2) * 2.f;

        v = warp_matrix.at<float>(4, 5);
        vx = warp_matrix.at<float>(4, 3);
        vy = warp_matrix.at<float>(4, 4) - 1.f;
        vxx = warp_matrix.at<float>(4, 0) * 2.f;
        vxy = warp_matrix.at<float>(4, 1);
        vyy = warp_matrix.at<float>(4, 2) * 2.f;
    }

    void Deformation2D2::setWarp()
    {
        // row 0, S1 - S6
        warp_matrix.at<float>(0, 0) = 1.f + 2.f * ux + ux * ux + u * uxx;
        warp_matrix.at<float>(0, 1) = 2.f * u * uxy + 2.f * (1.f + ux) * uy;
        warp_matrix.at<float>(0, 2) = uy * uy + u * uyy;
        warp_matrix.at<float>(0, 3) = 2.f * u * (1 + ux);
        warp_matrix.at<float>(0, 4) = 2.f * u * uy;
        warp_matrix.at<float>(0, 5) = u * u;

        // row 1, S7 - S12
        warp_matrix.at<float>(1, 0) = 0.5f * (v * uxx + 2.f * (1.f + ux) * vx + u * vxx);
        warp_matrix.at<float>(1, 1) = 1.f + uy * vx + ux * vy + v * uxy + u * vxy + vy + ux;
        warp_matrix.at<float>(1, 2) = 0.5f * (v * uyy + 2.f * uy * (1.f + vy) + u * vyy);
        warp_matrix.at<float>(1, 3) = v + v * ux + u * vx;
        warp_matrix.at<float>(1, 4) = u + v * uy + u * vy;
        warp_matrix.at<float>(1, 5) = u * v;

        // row 2, S12 - S18
        warp_matrix.at<float>(2, 0) = vx * vx + v * vxx;
        warp_matrix.at<float>(2, 1) = 2.f * v * vxy + 2.f * vx * (1.f + vy);
        warp_matrix.at<float>(2, 2) = 1.f + 2.f * vy + vy * vy + v * vyy;
        warp_matrix.at<float>(2, 3) = 2.f * v * vx;
        warp_matrix.at<float>(2, 4) = 2.f * v * (1.f + vy);
        warp_matrix.at<float>(2, 5) = v * v;

        // row 3
        warp_matrix.at<float>(3, 0) = 0.5f * uxx;
        warp_matrix.at<float>(3, 1) = uxy;
        warp_matrix.at<float>(3, 2) = 0.5f * uyy;
        warp_matrix.at<float>(3, 3) = 1.f + ux;
        warp_matrix.at<float>(3, 4) = uy;
        warp_matrix.at<float>(3, 5) = u;

        // row 4
        warp_matrix.at<float>(4, 0) = 0.5f * vxx;
        warp_matrix.at<float>(4, 1) = vxy;
        warp_matrix.at<float>(4, 2) = 0.5f * vyy;
        warp_matrix.at<float>(4, 3) = vx;
        warp_matrix.at<float>(4, 4) = 1.f + vy;
        warp_matrix.at<float>(4, 5) = v;

        // row 5
        warp_matrix.at<float>(5, 0) = 0.f;
        warp_matrix.at<float>(5, 1) = 0.f;
        warp_matrix.at<float>(5, 2) = 0.f;
        warp_matrix.at<float>(5, 3) = 0.f;
        warp_matrix.at<float>(5, 4) = 0.f;
        warp_matrix.at<float>(5, 5) = 1.f;
    }

    Point2D Deformation2D2::warp(const Point2D& location) const
    {
        // Create a vector for the point coordinates and higher order terms
        cv::Mat point_vector(6, 1, CV_32F);
        point_vector.at<float>(0, 0) = location.x * location.x;
        point_vector.at<float>(1, 0) = location.x * location.y;
        point_vector.at<float>(2, 0) = location.y * location.y;
        point_vector.at<float>(3, 0) = location.x;
        point_vector.at<float>(4, 0) = location.y;
        point_vector.at<float>(5, 0) = 1.f;

        // Multiply matrix by vector
        cv::Mat warped_vector = warp_matrix * point_vector;

        // Return the new point coordinates
        return Point2D(warped_vector.at<float>(3, 0), warped_vector.at<float>(4, 0));
    }

    // Deformation3D1 implementation
    Deformation3D1::Deformation3D1()
    {
        u = 0.f;
        ux = 0.f;
        uy = 0.f;
        uz = 0.f;

        v = 0.f;
        vx = 0.f;
        vy = 0.f;
        vz = 0.f;

        w = 0.f;
        wx = 0.f;
        wy = 0.f;
        wz = 0.f;

        warp_matrix = cv::Mat::eye(4, 4, CV_32F);
    }

    Deformation3D1::Deformation3D1(float u, float ux, float uy, float uz,
        float v, float vx, float vy, float vz,
        float w, float wx, float wy, float wz)
    {
        this->u = u;
        this->ux = ux;
        this->uy = uy;
        this->uz = uz;

        this->v = v;
        this->vx = vx;
        this->vy = vy;
        this->vz = vz;

        this->w = w;
        this->wx = wx;
        this->wy = wy;
        this->wz = wz;

        warp_matrix = cv::Mat::zeros(4, 4, CV_32F);
        setWarp();
    }

    Deformation3D1::Deformation3D1(float p[12])
    {
        u = p[0];
        ux = p[1];
        uy = p[2];
        uz = p[3];

        v = p[4];
        vx = p[5];
        vy = p[6];
        vz = p[7];

        w = p[8];
        wx = p[9];
        wy = p[10];
        wz = p[11];

        warp_matrix = cv::Mat::zeros(4, 4, CV_32F);
        setWarp();
    }

    Deformation3D1::~Deformation3D1() {}

    void Deformation3D1::setDeformation(float u, float ux, float uy, float uz,
        float v, float vx, float vy, float vz,
        float w, float wx, float wy, float wz)
    {
        this->u = u;
        this->ux = ux;
        this->uy = uy;
        this->uz = uz;

        this->v = v;
        this->vx = vx;
        this->vy = vy;
        this->vz = vz;

        this->w = w;
        this->wx = wx;
        this->wy = wy;
        this->wz = wz;

        setWarp();
    }

    void Deformation3D1::setDeformation(float p[12])
    {
        u = p[0];
        ux = p[1];
        uy = p[2];
        uz = p[3];

        v = p[4];
        vx = p[5];
        vy = p[6];
        vz = p[7];

        w = p[8];
        wx = p[9];
        wy = p[10];
        wz = p[11];

        setWarp();
    }

    void Deformation3D1::setDeformation(Deformation3D1& another_deformation)
    {
        u = another_deformation.u;
        ux = another_deformation.ux;
        uy = another_deformation.uy;
        uz = another_deformation.uz;

        v = another_deformation.v;
        vx = another_deformation.vx;
        vy = another_deformation.vy;
        vz = another_deformation.vz;

        w = another_deformation.w;
        wx = another_deformation.wx;
        wy = another_deformation.wy;
        wz = another_deformation.wz;

        setWarp();
    }

    void Deformation3D1::setDeformation()
    {
        u = warp_matrix.at<float>(0, 3);
        ux = warp_matrix.at<float>(0, 0) - 1.f;
        uy = warp_matrix.at<float>(0, 1);
        uz = warp_matrix.at<float>(0, 2);

        v = warp_matrix.at<float>(1, 3);
        vx = warp_matrix.at<float>(1, 0);
        vy = warp_matrix.at<float>(1, 1) - 1.f;
        vz = warp_matrix.at<float>(1, 2);

        w = warp_matrix.at<float>(2, 3);
        wx = warp_matrix.at<float>(2, 0);
        wy = warp_matrix.at<float>(2, 1);
        wz = warp_matrix.at<float>(2, 2) - 1.f;
    }

    void Deformation3D1::setWarp()
    {
        warp_matrix.at<float>(0, 0) = 1.f + ux;
        warp_matrix.at<float>(0, 1) = uy;
        warp_matrix.at<float>(0, 2) = uz;
        warp_matrix.at<float>(0, 3) = u;

        warp_matrix.at<float>(1, 0) = vx;
        warp_matrix.at<float>(1, 1) = 1.f + vy;
        warp_matrix.at<float>(1, 2) = vz;
        warp_matrix.at<float>(1, 3) = v;

        warp_matrix.at<float>(2, 0) = wx;
        warp_matrix.at<float>(2, 1) = wy;
        warp_matrix.at<float>(2, 2) = 1.f + wz;
        warp_matrix.at<float>(2, 3) = w;

        warp_matrix.at<float>(3, 0) = 0.f;
        warp_matrix.at<float>(3, 1) = 0.f;
        warp_matrix.at<float>(3, 2) = 0.f;
        warp_matrix.at<float>(3, 3) = 1.f;
    }

    Point3D Deformation3D1::warp(const Point3D& point) const
    {
        // Create a vector for the point coordinates in homogeneous form
        cv::Mat point_vector(4, 1, CV_32F);
        point_vector.at<float>(0, 0) = point.x;
        point_vector.at<float>(1, 0) = point.y;
        point_vector.at<float>(2, 0) = point.z;
        point_vector.at<float>(3, 0) = 1.0f;

        // Multiply matrix by vector
        cv::Mat warped_vector = warp_matrix * point_vector;

        // Return the new point coordinates
        return Point3D(warped_vector.at<float>(0, 0), warped_vector.at<float>(1, 0), warped_vector.at<float>(2, 0));
    }
} // namespace StudyCorr_GPU
