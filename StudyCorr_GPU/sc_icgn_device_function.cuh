#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>


// 图像双线性插值（2D/2D二阶公用）
__inline__ __device__ float bilinear_interpolate(const float* image, int height, int width, float y, float x) {
    int x0 = floorf(x), x1 = x0 + 1;
    int y0 = floorf(y), y1 = y0 + 1;
    float dx = x - x0, dy = y - y0;
    x0 = max(0, min(x0, width - 1));
    x1 = max(0, min(x1, width - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));
    float I00 = image[y0 * width + x0];
    float I01 = image[y0 * width + x1];
    float I10 = image[y1 * width + x0];
    float I11 = image[y1 * width + x1];
    return (1 - dx) * (1 - dy) * I00
         + dx * (1 - dy) * I01
         + (1 - dx) * dy * I10
         + dx * dy * I11;
}

// Sobel梯度图（2D/2D二阶公用）
__inline__ __device__ void compute_gradients(const float* image, int height, int width, float* grad_x, float* grad_y) {
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            grad_x[idx] =
                -image[(y-1)*width + (x-1)] - 2*image[y*width + (x-1)] - image[(y+1)*width + (x-1)]
                +image[(y-1)*width + (x+1)] + 2*image[y*width + (x+1)] + image[(y+1)*width + (x+1)];
            grad_x[idx] /= 8.0f;
            grad_y[idx] =
                -image[(y-1)*width + (x-1)] - 2*image[(y-1)*width + x] - image[(y-1)*width + (x+1)]
                +image[(y+1)*width + (x-1)] + 2*image[(y+1)*width + x] + image[(y+1)*width + (x+1)];
            grad_y[idx] /= 8.0f;
        }
    }
}

// Sobel算子计算梯度
__inline__ __device__ float gradient_x(const float* image, int height, int width, float y, float x) {
    int ix = int(roundf(x));
    int iy = int(roundf(y));
    ix = max(1, min(ix, width - 2));
    iy = max(1, min(iy, height - 2));
    float gx =
        -image[(iy-1)*width + (ix-1)] - 2*image[iy*width + (ix-1)] - image[(iy+1)*width + (ix-1)]
        +image[(iy-1)*width + (ix+1)] + 2*image[iy*width + (ix+1)] + image[(iy+1)*width + (ix+1)];
    gx /= 8.0f;
    return gx;
}

__inline__ __device__ float gradient_y(const float* image, int height, int width, float y, float x) {
    int ix = int(roundf(x));
    int iy = int(roundf(y));
    ix = max(1, min(ix, width - 2));
    iy = max(1, min(iy, height - 2));
    float gy =
        -image[(iy-1)*width + (ix-1)] - 2*image[(iy-1)*width + ix] - image[(iy-1)*width + (ix+1)]
        +image[(iy+1)*width + (ix-1)] + 2*image[(iy+1)*width + ix] + image[(iy+1)*width + (ix+1)];
    gy /= 8.0f;
    return gy;
}

// device端6x6 Hessian逆矩阵实现
__inline__ __device__ bool qr_inverse_6x6(const double* A, double* invA) {
    const int N = 6;
    double Q[N][N] = {0};
    double R[N][N] = {0};
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < N; ++i)
            Q[i][k] = A[i*N + k];
        for (int j = 0; j < k; ++j) {
            double dot = 0.0;
            for (int i = 0; i < N; ++i)
                dot += Q[i][j] * Q[i][k];
            for (int i = 0; i < N; ++i)
                Q[i][k] -= dot * Q[i][j];
        }
        double norm = 0.0;
        for (int i = 0; i < N; ++i)
            norm += Q[i][k] * Q[i][k];
        norm = sqrt(norm);
        if (norm < 1e-12) return false;
        for (int i = 0; i < N; ++i)
            Q[i][k] /= norm;
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += Q[k][i] * A[k*N + j];
            R[i][j] = sum;
        }
    double invR[N][N] = {0};
    for (int i = N - 1; i >= 0; --i) {
        if (fabs(R[i][i]) < 1e-12) return false;
        invR[i][i] = 1.0 / R[i][i];
        for (int j = i + 1; j < N; ++j) {
            double sum = 0.0;
            for (int k = i + 1; k <= j; ++k)
                sum += R[i][k] * invR[k][j];
            invR[i][j] = -sum / R[i][i];
        }
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += invR[i][k] * Q[j][k];
            invA[i*N + j] = sum;
        }
    return true;
}

// QR分解逆，病态矩阵处理
__inline__ __device__ bool qr_inverse_12x12(const double* A, double* invA) {
    const int N = 12;
    double Q[N][N] = {0}; double R[N][N] = {0};
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < N; ++i) Q[i][k] = A[i*N + k];
        for (int j = 0; j < k; ++j) {
            double dot = 0.0;
            for (int i = 0; i < N; ++i) dot += Q[i][j] * Q[i][k];
            for (int i = 0; i < N; ++i) Q[i][k] -= dot * Q[i][j];
        }
        double norm = 0.0;
        for (int i = 0; i < N; ++i) norm += Q[i][k] * Q[i][k];
        norm = sqrt(norm);
        if (norm < 1e-12) return false;
        for (int i = 0; i < N; ++i) Q[i][k] /= norm;
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += Q[k][i] * A[k*N + j];
            R[i][j] = sum;
        }
    double invR[N][N] = {0};
    for (int i = N - 1; i >= 0; --i) {
        if (fabs(R[i][i]) < 1e-12) return false;
        invR[i][i] = 1.0 / R[i][i];
        for (int j = i + 1; j < N; ++j) {
            double sum = 0.0;
            for (int k = i + 1; k <= j; ++k)
                sum += R[i][k] * invR[k][j];
            invR[i][j] = -sum / R[i][i];
        }
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += invR[i][k] * Q[j][k];
            invA[i*N + j] = sum;
        }
    return true;
}

// ZNSSD相关系数
__inline__ __device__ float compute_znssd(const float* ref_img, const float* tar_img, int subset_size,
                               float mean_ref, float std_ref, float mean_tar, float std_tar) {
    float znssd = 0.f;
    for (int i = 0; i < subset_size; ++i) {
        float nr = (ref_img[i] - mean_ref) / (std_ref + 1e-12f);
        float nt = (tar_img[i] - mean_tar) / (std_tar + 1e-12f);
        znssd += (nr - nt) * (nr - nt);
    }
    znssd /= subset_size;
    float zncc = 0.5f * (2.f - znssd);
    return zncc;
}

/****************************************** 2D一阶形函数 device functions + Hessian逆/插值/梯度 ******************************************/

struct ShapeParam2D1 {
    double u, ux, uy, v, vx, vy;
};

__device__ void set_deformation2d1(const double* p, ShapeParam2D1& param) {
    param.u  = p[0];
    param.ux = p[1];
    param.uy = p[2];
    param.v  = p[3];
    param.vx = p[4];
    param.vy = p[5];
}

__device__ void warp2d1(const ShapeParam2D1& param, double x_local, double y_local, double& u_warp, double& v_warp) {
    u_warp = param.u + param.ux * x_local + param.uy * y_local;
    v_warp = param.v + param.vx * x_local + param.vy * y_local;
}

__device__ void shape_gradient2d1(double x_local, double y_local, double* grad_u, double* grad_v) {
    grad_u[0] = 1.0;    grad_u[1] = x_local; grad_u[2] = y_local;
    grad_u[3] = 0.0;    grad_u[4] = 0.0;     grad_u[5] = 0.0;
    grad_v[0] = 0.0;    grad_v[1] = 0.0;     grad_v[2] = 0.0;
    grad_v[3] = 1.0;    grad_v[4] = x_local; grad_v[5] = y_local;
}

__device__ void compose2d1(const ShapeParam2D1& p_old, const ShapeParam2D1& delta_p, ShapeParam2D1& p_new) {
    p_new.u  = p_old.u  + delta_p.u;
    p_new.ux = p_old.ux + delta_p.ux;
    p_new.uy = p_old.uy + delta_p.uy;
    p_new.v  = p_old.v  + delta_p.v;
    p_new.vx = p_old.vx + delta_p.vx;
    p_new.vy = p_old.vy + delta_p.vy;
}

__device__ double delta_norm2d1(const ShapeParam2D1& delta_p, int subset_rx, int subset_ry) {
    double norm = 0.0;
    norm += delta_p.u  * delta_p.u;
    norm += delta_p.ux * delta_p.ux * subset_rx * subset_rx;
    norm += delta_p.uy * delta_p.uy * subset_ry * subset_ry;
    norm += delta_p.v  * delta_p.v;
    norm += delta_p.vx * delta_p.vx * subset_rx * subset_rx;
    norm += delta_p.vy * delta_p.vy * subset_ry * subset_ry;
    return sqrt(norm);
}


/****************************************** 2D二阶形函数 device functions + Hessian逆 ******************************************/

struct ShapeParam2D2 {
    double u, ux, uy, uxx, uxy, uyy;
    double v, vx, vy, vxx, vxy, vyy;
};

__device__ void set_deformation2d2(const double* p, ShapeParam2D2& param) {
    param.u   = p[0];
    param.ux  = p[1];
    param.uy  = p[2];
    param.uxx = p[3];
    param.uxy = p[4];
    param.uyy = p[5];
    param.v   = p[6];
    param.vx  = p[7];
    param.vy  = p[8];
    param.vxx = p[9];
    param.vxy = p[10];
    param.vyy = p[11];
}

__device__ void warp2d2(const ShapeParam2D2& param, double x, double y, double& u_warp, double& v_warp) {
    u_warp = param.u + param.ux * x + param.uy * y
           + param.uxx * x * x + param.uxy * x * y + param.uyy * y * y;
    v_warp = param.v + param.vx * x + param.vy * y
           + param.vxx * x * x + param.vxy * x * y + param.vyy * y * y;
}

__device__ void shape_gradient2d2(double x, double y, double* grad_u, double* grad_v) {
    grad_u[0] = 1.0;
    grad_u[1] = x;
    grad_u[2] = y;
    grad_u[3] = x * x;
    grad_u[4] = x * y;
    grad_u[5] = y * y;
    grad_u[6] = 0.0; grad_u[7] = 0.0; grad_u[8] = 0.0; grad_u[9] = 0.0; grad_u[10] = 0.0; grad_u[11] = 0.0;
    grad_v[0] = 0.0; grad_v[1] = 0.0; grad_v[2] = 0.0; grad_v[3] = 0.0; grad_v[4] = 0.0; grad_v[5] = 0.0;
    grad_v[6] = 1.0;
    grad_v[7] = x;
    grad_v[8] = y;
    grad_v[9] = x * x;
    grad_v[10] = x * y;
    grad_v[11] = y * y;
}

__device__ void compose2d2(const ShapeParam2D2& p_old, const ShapeParam2D2& delta_p, ShapeParam2D2& p_new) {
    p_new.u   = p_old.u   + delta_p.u;
    p_new.ux  = p_old.ux  + delta_p.ux;
    p_new.uy  = p_old.uy  + delta_p.uy;
    p_new.uxx = p_old.uxx + delta_p.uxx;
    p_new.uxy = p_old.uxy + delta_p.uxy;
    p_new.uyy = p_old.uyy + delta_p.uyy;
    p_new.v   = p_old.v   + delta_p.v;
    p_new.vx  = p_old.vx  + delta_p.vx;
    p_new.vy  = p_old.vy  + delta_p.vy;
    p_new.vxx = p_old.vxx + delta_p.vxx;
    p_new.vxy = p_old.vxy + delta_p.vxy;
    p_new.vyy = p_old.vyy + delta_p.vyy;
}

__device__ double delta_norm2d2(const ShapeParam2D2& delta_p, int subset_rx, int subset_ry) {
    double norm = 0.0;
    norm += delta_p.u   * delta_p.u;
    norm += delta_p.ux  * delta_p.ux  * subset_rx * subset_rx;
    norm += delta_p.uy  * delta_p.uy  * subset_ry * subset_ry;
    norm += delta_p.uxx * delta_p.uxx * subset_rx * subset_rx * subset_rx * subset_rx;
    norm += delta_p.uxy * delta_p.uxy * subset_rx * subset_rx * subset_ry * subset_ry;
    norm += delta_p.uyy * delta_p.uyy * subset_ry * subset_ry * subset_ry * subset_ry;
    norm += delta_p.v   * delta_p.v;
    norm += delta_p.vx  * delta_p.vx  * subset_rx * subset_rx;
    norm += delta_p.vy  * delta_p.vy  * subset_ry * subset_ry;
    norm += delta_p.vxx * delta_p.vxx * subset_rx * subset_rx * subset_rx * subset_rx;
    norm += delta_p.vxy * delta_p.vxy * subset_rx * subset_rx * subset_ry * subset_ry;
    norm += delta_p.vyy * delta_p.vyy * subset_ry * subset_ry * subset_ry * subset_ry;
    return sqrt(norm);
}

/****************************************** 3D一阶形函数 device functions + Hessian逆/插值/梯度 ******************************************/

struct ShapeParam3D1 {
    double u, ux, uy, uz;
    double v, vx, vy, vz;
    double w, wx, wy, wz;
};

__device__ void set_deformation3d1(const double* p, ShapeParam3D1& param) {
    param.u  = p[0];
    param.ux = p[1];
    param.uy = p[2];
    param.uz = p[3];
    param.v  = p[4];
    param.vx = p[5];
    param.vy = p[6];
    param.vz = p[7];
    param.w  = p[8];
    param.wx = p[9];
    param.wy = p[10];
    param.wz = p[11];
}

__device__ void warp3d1(const ShapeParam3D1& param, double x, double y, double z,
                        double& u_warp, double& v_warp, double& w_warp) {
    u_warp = param.u  + param.ux * x + param.uy * y + param.uz * z;
    v_warp = param.v  + param.vx * x + param.vy * y + param.vz * z;
    w_warp = param.w  + param.wx * x + param.wy * y + param.wz * z;
}

__device__ void shape_gradient3d1(double x, double y, double z, double* grad_u, double* grad_v, double* grad_w) {
    grad_u[0] = 1.0;  grad_u[1] = x;  grad_u[2] = y;  grad_u[3] = z;
    grad_u[4] = grad_u[5] = grad_u[6] = grad_u[7] = grad_u[8] = grad_u[9] = grad_u[10] = grad_u[11] = 0.0;
    grad_v[0] = grad_v[1] = grad_v[2] = grad_v[3] = 0.0;
    grad_v[4] = 1.0;  grad_v[5] = x;  grad_v[6] = y;  grad_v[7] = z;
    grad_v[8] = grad_v[9] = grad_v[10] = grad_v[11] = 0.0;
    grad_w[0] = grad_w[1] = grad_w[2] = grad_w[3] = grad_w[4] = grad_w[5] = grad_w[6] = grad_w[7] = 0.0;
    grad_w[8] = 1.0;  grad_w[9] = x;  grad_w[10] = y;  grad_w[11] = z;
}

__device__ void compose3d1(const ShapeParam3D1& p_old, const ShapeParam3D1& delta_p, ShapeParam3D1& p_new) {
    for (int i = 0; i < 12; ++i) {
        ((double*)&p_new)[i] = ((double*)&p_old)[i] + ((double*)&delta_p)[i];
    }
}

__device__ double delta_norm3d1(const ShapeParam3D1& delta_p, int subset_rx, int subset_ry, int subset_rz) {
    double norm = 0.0;
    norm += delta_p.u  * delta_p.u;
    norm += delta_p.ux * delta_p.ux * subset_rx * subset_rx;
    norm += delta_p.uy * delta_p.uy * subset_ry * subset_ry;
    norm += delta_p.uz * delta_p.uz * subset_rz * subset_rz;
    norm += delta_p.v  * delta_p.v;
    norm += delta_p.vx * delta_p.vx * subset_rx * subset_rx;
    norm += delta_p.vy * delta_p.vy * subset_ry * subset_ry;
    norm += delta_p.vz * delta_p.vz * subset_rz * subset_rz;
    norm += delta_p.w  * delta_p.w;
    norm += delta_p.wx * delta_p.wx * subset_rx * subset_rx;
    norm += delta_p.wy * delta_p.wy * subset_ry * subset_ry;
    norm += delta_p.wz * delta_p.wz * subset_rz * subset_rz;
    return sqrt(norm);
}

// 三线性插值
__device__ float trilinear_interpolate(const float* image, int depth, int height, int width,
                                       float z, float y, float x) {
    int x0 = floorf(x), x1 = x0 + 1;
    int y0 = floorf(y), y1 = y0 + 1;
    int z0 = floorf(z), z1 = z0 + 1;
    float dx = x - x0, dy = y - y0, dz = z - z0;
    x0 = max(0, min(x0, width - 1));
    x1 = max(0, min(x1, width - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));
    z0 = max(0, min(z0, depth - 1));
    z1 = max(0, min(z1, depth - 1));
    #define IDX(z,y,x) ((z)*height*width + (y)*width + (x))
    float c000 = image[IDX(z0, y0, x0)];
    float c001 = image[IDX(z0, y0, x1)];
    float c010 = image[IDX(z0, y1, x0)];
    float c011 = image[IDX(z0, y1, x1)];
    float c100 = image[IDX(z1, y0, x0)];
    float c101 = image[IDX(z1, y0, x1)];
    float c110 = image[IDX(z1, y1, x0)];
    float c111 = image[IDX(z1, y1, x1)];
    float c00 = c000 * (1 - dx) + c001 * dx;
    float c01 = c010 * (1 - dx) + c011 * dx;
    float c10 = c100 * (1 - dx) + c101 * dx;
    float c11 = c110 * (1 - dx) + c111 * dx;
    float c0 = c00 * (1 - dy) + c01 * dy;
    float c1 = c10 * (1 - dy) + c11 * dy;
    return c0 * (1 - dz) + c1 * dz;
}

// Sobel梯度图
__device__ void compute_gradients_3d(const float* image, int depth, int height, int width,
                                     float* grad_x, float* grad_y, float* grad_z) {
    for (int z = 1; z < depth - 1; ++z)
        for (int y = 1; y < height - 1; ++y)
            for (int x = 1; x < width - 1; ++x) {
                int idx = z * height * width + y * width + x;
                grad_x[idx] =
                    -image[idx - width - height*width] - 2*image[idx - height*width] - image[idx + width - height*width]
                    +image[idx - width + height*width] + 2*image[idx + height*width] + image[idx + width + height*width];
                grad_x[idx] /= 8.0f;
                grad_y[idx] =
                    -image[idx - 1 - height*width] - 2*image[idx - height*width] - image[idx + 1 - height*width]
                    +image[idx - 1 + height*width] + 2*image[idx + height*width] + image[idx + 1 + height*width];
                grad_y[idx] /= 8.0f;
                grad_z[idx] =
                    -image[idx - width - 1] - 2*image[idx - 1] - image[idx + width - 1]
                    +image[idx - width + 1] + 2*image[idx + 1] + image[idx + width + 1];
                grad_z[idx] /= 8.0f;
            }
}