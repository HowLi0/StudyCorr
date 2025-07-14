#pragma once
#include "sc_point.h"

namespace StudyCorr
{

// ==================== 2D 一阶变形 ====================

class CudaDeformation2D1
{
public:
    float u, ux, uy;
    float v, vx, vy;

    CUDA_HD CudaDeformation2D1()
        : u(0), ux(0), uy(0), v(0), vx(0), vy(0) {}

    CUDA_HD CudaDeformation2D1(float u, float ux, float uy, float v, float vx, float vy)
        : u(u), ux(ux), uy(uy), v(v), vx(vx), vy(vy) {}

    CUDA_HD CudaDeformation2D1(const float p[6])
        : u(p[0]), ux(p[1]), uy(p[2]), v(p[3]), vx(p[4]), vy(p[5]) {}

    CUDA_HD void setDeformation(float u_, float ux_, float uy_, float v_, float vx_, float vy_) {
        u = u_; ux = ux_; uy = uy_; v = v_; vx = vx_; vy = vy_;
    }
    CUDA_HD void setDeformation(const float p[6]) {
        u = p[0]; ux = p[1]; uy = p[2]; v = p[3]; vx = p[4]; vy = p[5];
    }
    CUDA_HD void setDeformation(const CudaDeformation2D1& other) {
        u = other.u; ux = other.ux; uy = other.uy;
        v = other.v; vx = other.vx; vy = other.vy;
    }

    CUDA_HD Point2D warp(const Point2D& location) const {
        return Point2D(
            location.x + u + ux * location.x + uy * location.y,
            location.y + v + vx * location.x + vy * location.y
        );
    }
    // |x'|   |1+ux  uy   u|   |x|
    // |y'| = |vx   1+vy v| * |y|
    // |1 |   |0    0    1|   |1|

    // x' = (1+ux)*x + uy*y + u
    // y' = vx*x + (1+vy)*y + v
};

// ==================== 2D 二阶变形 ====================

class CudaDeformation2D2
{
public:
    float u, ux, uy, uxx, uxy, uyy;
    float v, vx, vy, vxx, vxy, vyy;

    CUDA_HD CudaDeformation2D2()
        : u(0), ux(0), uy(0), uxx(0), uxy(0), uyy(0),
          v(0), vx(0), vy(0), vxx(0), vxy(0), vyy(0) {}

    CUDA_HD CudaDeformation2D2(
        float u, float ux, float uy, float uxx, float uxy, float uyy,
        float v, float vx, float vy, float vxx, float vxy, float vyy)
        : u(u), ux(ux), uy(uy), uxx(uxx), uxy(uxy), uyy(uyy),
          v(v), vx(vx), vy(vy), vxx(vxx), vxy(vxy), vyy(vyy) {}

    CUDA_HD CudaDeformation2D2(const float p[12])
        : u(p[0]), ux(p[1]), uy(p[2]), uxx(p[3]), uxy(p[4]), uyy(p[5]),
          v(p[6]), vx(p[7]), vy(p[8]), vxx(p[9]), vxy(p[10]), vyy(p[11]) {}

    CUDA_HD void setDeformation(
        float u_, float ux_, float uy_, float uxx_, float uxy_, float uyy_,
        float v_, float vx_, float vy_, float vxx_, float vxy_, float vyy_)
    {
        u = u_; ux = ux_; uy = uy_; uxx = uxx_; uxy = uxy_; uyy = uyy_;
        v = v_; vx = vx_; vy = vy_; vxx = vxx_; vxy = vxy_; vyy = vyy_;
    }
    CUDA_HD void setDeformation(const float p[12]) {
        u = p[0]; ux = p[1]; uy = p[2]; uxx = p[3]; uxy = p[4]; uyy = p[5];
        v = p[6]; vx = p[7]; vy = p[8]; vxx = p[9]; vxy = p[10]; vyy = p[11];
    }
    CUDA_HD void setDeformation(const CudaDeformation2D2& other) {
        u = other.u; ux = other.ux; uy = other.uy; uxx = other.uxx; uxy = other.uxy; uyy = other.uyy;
        v = other.v; vx = other.vx; vy = other.vy; vxx = other.vxx; vxy = other.vxy; vyy = other.vyy;
    }

    CUDA_HD Point2D warp(const Point2D& location) const {
        float xx = location.x * location.x, yy = location.y * location.y, xy = location.x * location.y;
        return Point2D(
            location.x + u + ux * location.x + uy * location.y + 0.5f*uxx * xx + uxy * xy + 0.5f*uyy * yy,
            location.y + v + vx * location.x + vy * location.y + 0.5f*vxx * xx + vxy * xy + 0.5f*vyy * yy
        );
    }

    //opencorr在二阶项上添加了0.5的系数，即uxx → 0.5f*uxx, uyy → 0.5f*uyy.

    // |x'|   |1+ux  uy   u + 0.5*uxx*x + uxy*y + 0.5*uyy*y^2|   |x|
    // |y'| = |vx   1+vy v + 0.5*vxx*x + vxy*y + 0.5*vyy*y^2| * |y|
    // |1 |   |0    0    1 + 0*x + 0*y + 0*y^2|   |1|

    // x' = (1+ux)*x + uy*y + u + 0.5*uxx*x^2 + uxy*xy + 0.5*uyy*y^2
    // y' = vx*x + (1+vy)*y + v + 0.5*vxx*x^2 + vxy*xy + 0.5*vyy*y^2
};

// ==================== 3D 一阶变形 ===================

class CudaDeformation3D1
{
public:
    float u, ux, uy, uz;
    float v, vx, vy, vz;
    float w, wx, wy, wz;

    CUDA_HD CudaDeformation3D1()
        : u(0), ux(0), uy(0), uz(0),
          v(0), vx(0), vy(0), vz(0),
          w(0), wx(0), wy(0), wz(0) {}

    CUDA_HD CudaDeformation3D1(
        float u, float ux, float uy, float uz,
        float v, float vx, float vy, float vz,
        float w, float wx, float wy, float wz)
        : u(u), ux(ux), uy(uy), uz(uz),
          v(v), vx(vx), vy(vy), vz(vz),
          w(w), wx(wx), wy(wy), wz(wz) {}

    CUDA_HD CudaDeformation3D1(const float p[12])
        : u(p[0]), ux(p[1]), uy(p[2]), uz(p[3]),
          v(p[4]), vx(p[5]), vy(p[6]), vz(p[7]),
          w(p[8]), wx(p[9]), wy(p[10]), wz(p[11]) {}

    CUDA_HD void setDeformation(
        float u_, float ux_, float uy_, float uz_,
        float v_, float vx_, float vy_, float vz_,
        float w_, float wx_, float wy_, float wz_)
    {
        u = u_; ux = ux_; uy = uy_; uz = uz_;
        v = v_; vx = vx_; vy = vy_; vz = vz_;
        w = w_; wx = wx_; wy = wy_; wz = wz_;
    }
    CUDA_HD void setDeformation(const float p[12]) {
        u = p[0]; ux = p[1]; uy = p[2]; uz = p[3];
        v = p[4]; vx = p[5]; vy = p[6]; vz = p[7];
        w = p[8]; wx = p[9]; wy = p[10]; wz = p[11];
    }
    CUDA_HD void setDeformation(const CudaDeformation3D1& other) {
        u = other.u; ux = other.ux; uy = other.uy; uz = other.uz;
        v = other.v; vx = other.vx; vy = other.vy; vz = other.vz;
        w = other.w; wx = other.wx; wy = other.wy; wz = other.wz;
    }

    CUDA_HD Point3D warp(const Point3D& location) const {
        return Point3D(
            location.x + u + ux * location.x + uy * location.y + uz * location.z,
            location.y + v + vx * location.x + vy * location.y + vz * location.z,
            location.z + w + wx * location.x + wy * location.y + wz * location.z
        );
    }
    // |x'|   |1+ux  uy  uz   u|   |x|
    // |y'| = |vx   1+vy vz   v| * |y|
    // |z'|   |wx   wy  1+wz   w|   |z|
    // |1 |   |0    0    0    1|   |1|

    // x' = (1+ux)*x + uy*y + uz*z + u
    // y' = vx*x + (1+vy)*y + vz*z + v
    // z' = wx*x + wy*y + (1+wz)*z + w
};

// ==================== 批量warp核函数 ====================

    // 每点独立参数的一阶2D
    __global__ void warp_2d1_kernel(
        const CudaDeformation2D1* defs,
        const Point2D* pts_in,
        Point2D* pts_out, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            pts_out[idx] = defs[idx].warp(pts_in[idx]);
        }
    }

    // 每点独立参数的二阶2D
    __global__ void warp_2d2_kernel(
        const CudaDeformation2D2* defs,
        const Point2D* pts_in,
        Point2D* pts_out, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            pts_out[idx] = defs[idx].warp(pts_in[idx]);
        }
    }

    // 每点独立参数的一阶3D
    __global__ void warp_3d1_kernel(
        const CudaDeformation3D1* defs,
        const Point3D* pts_in,
        Point3D* pts_out, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            pts_out[idx] = defs[idx].warp(pts_in[idx]);
        }
    }
} // namespace StudyCorr
