#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>



struct ZNCCAndErrorResult {
    float zncc;
    float errorVector[12];
    int count;
};

//******************************************************icgn2D******************************************************/

// 图像双线性插值（2D/2D二阶公用）
__device__ __forceinline__ float BilinearInterpolation(float x, float y, const float* image, int width, int height) {
    // 边界检查，与CPU版本完全一致
    if (x < 0.0f || x >= width - 1.0f || y < 0.0f || y >= height - 1.0f) {
        return 0.0f;
    }
    
    // 获取整数和小数部分，使用与CPU相同的方法                          
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float fx = x - static_cast<float>(x1);// x 方向的权重
    float fy = y - static_cast<float>(y1);// y 方向的权重

    // 双线性插值，与CPU版本完全一致的公式
    float val = (1.0f - fx) * (1.0f - fy) * image[y1 * width + x1] +
                fx * (1.0f - fy) * image[y1 * width + x2] +
                (1.0f - fx) * fy * image[y2 * width + x1] +
                fx * fy * image[y2 * width + x2];
    
    return val;
}

//Sobel梯度计算
__device__ __forceinline__ void computeSobelGradients(const float* image, int x, int y, int width, int height,
                                                      float& gradX, float& gradY) {
    gradX = 0.0f;
    gradY = 0.0f;

    // 边界检查
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // Sobel X核: [-1 0 1; -2 0 2; -1 0 1] / 8
        gradX = (-image[(y-1)*width + (x-1)] + image[(y-1)*width + (x+1)]
                -2.0f*image[y*width + (x-1)] + 2.0f*image[y*width + (x+1)]
                -image[(y+1)*width + (x-1)] + image[(y+1)*width + (x+1)]) / 8.0f;

        // Sobel Y核: [-1 -2 -1; 0 0 0; 1 2 1] / 8
        gradY = (-image[(y-1)*width + (x-1)] - 2.0f*image[(y-1)*width + x] - image[(y-1)*width + (x+1)]
                +image[(y+1)*width + (x-1)] + 2.0f*image[(y+1)*width + x] + image[(y+1)*width + (x+1)]) / 8.0f;
    }
}

__device__ __forceinline__ void WarpPoint(float x, float y, const float* warpParams, 
                                                   float& warpedX, float& warpedY, int numParams) {
    // 提取参数
    float u = warpParams[0];
    float v = warpParams[1];

    // 基础平移
    warpedX = x + u;
    warpedY = y + v;
    
    // 一阶形变参数（至少6个参数）
    if (numParams >= 6) {
        float dudx = warpParams[2];
        float dudy = warpParams[3];
        float dvdx = warpParams[4];
        float dvdy = warpParams[5];

        warpedX += dudx * x + dudy * y;
        warpedY += dvdx * x + dvdy * y;
    }
    
    // 二阶形变参数（12个参数）
    if (numParams >= 12) {
        float d2udx2 = warpParams[6];
        float d2udxdy = warpParams[7];
        float d2udy2 = warpParams[8];
        float d2vdx2 = warpParams[9];
        float d2vdxdy = warpParams[10];
        float d2vdy2 = warpParams[11];
        
        warpedX += 0.5 * d2udx2 * x * x + d2udxdy * x * y + 0.5 * d2udy2 * y * y;
        warpedY += 0.5 * d2vdx2 * x * x + d2vdxdy * x * y + 0.5 * d2vdy2 * y * y;
    }
}


// 高精度QR分解求解线性方程组 - 更稳定的数值方法
__device__ bool solveLinearSystemQR(const float* A, const float* b, float* x, int n) {
    // 复制矩阵到局部内存
    float Q[144]; // 最大12x12矩阵 - 正交矩阵
    float R[144]; // 最大12x12矩阵 - 上三角矩阵
    float bb[12]; // 最大12维向量

    // 初始化
    for (int i = 0; i < n * n; i++) {
        Q[i] = A[i]; // 初始将A复制到Q
        R[i] = 0.0f; // R矩阵初始化为0
    }
    for (int i = 0; i < n; i++) {
        bb[i] = b[i];
        x[i] = 0.0f;
    }
    
    // Modified Gram-Schmidt QR分解
    for (int j = 0; j < n; j++) {
        // 计算列向量的范数
        float norm = 0.0f;
        for (int i = 0; i < n; i++) {
            norm += Q[i * n + j] * Q[i * n + j];
        }
        norm = sqrt(norm);
        
        // 检查数值稳定性
        if (norm < 1e-14) {
            return false; // 矩阵奇异
        }
        
        R[j * n + j] = norm;
        
        // 归一化列向量
        for (int i = 0; i < n; i++) {
            Q[i * n + j] /= norm;
        }
        
        // 计算与后续列的内积并正交化
        for (int k = j + 1; k < n; k++) {
            float dot = 0.0f;
            for (int i = 0; i < n; i++) {
                dot += Q[i * n + j] * Q[i * n + k];
            }
            R[j * n + k] = dot;
            
            // 从后续列中减去投影
            for (int i = 0; i < n; i++) {
                Q[i * n + k] -= dot * Q[i * n + j];
            }
        }
    }
    
    // 计算 Q^T * b
    double QtB[12];
    for (int i = 0; i < n; i++) {
        QtB[i] = 0.0;
        for (int j = 0; j < n; j++) {
            QtB[i] += Q[j * n + i] * bb[j]; // Q^T[i][j] = Q[j][i]
        }
    }
    
    // 后向替换求解 R * x = Q^T * b
    for (int i = n - 1; i >= 0; i--) {
        float sum = 0.0f;
        for (int j = i + 1; j < n; j++) {
            sum += R[i * n + j] * x[j];
        }
        
        // 检查对角元素避免除零
        if (fabs(R[i * n + i]) < 1e-14) {
            return false;
        }
        
        x[i] = (QtB[i] - sum) / R[i * n + i];
    }
    
    return true;
}

// 添加正则化的QR分解以处理病态矩阵
__device__ bool solveLinearSystemRegularizedQR(const float* A, const float* b, float* x, int n) {
    // 复制矩阵并添加Tikhonov正则化
    float ARegularized[144];
    float regularization = 1e-8f; // 正则化参数

    for (int i = 0; i < n * n; i++) {
        ARegularized[i] = A[i];
    }
    
    // 添加对角正则化项
    for (int i = 0; i < n; i++) {
        ARegularized[i * n + i] += regularization;
    }
    
    // 使用正则化矩阵进行QR分解
    return solveLinearSystemQR(ARegularized, b, x, n);
}

// 主要的线性系统求解函数
__device__ bool solveLinearSystem(const float* A, const float* b, float* x, int n) {
    // 首先尝试标准QR分解
    if (solveLinearSystemQR(A, b, x, n)) {
        return true;
    }
    
    // 如果失败，尝试正则化QR分解
    return solveLinearSystemRegularizedQR(A, b, x, n);
}


// shape函数求导工具
__device__ float getShapeFn(int paramIdx, int numParams, float gradX, float gradY, float x, float y) {
    if (paramIdx < 0) return 0.0f;
    switch (paramIdx) {
        case 0: return gradX; // du
        case 1: return gradY; // dv
        // 仿射参数
        case 2: return (numParams >= 6) ? gradX * x : 0.0f; // du/dx
        case 3: return (numParams >= 6) ? gradX * y : 0.0f; // du/dy
        case 4: return (numParams >= 6) ? gradY * x : 0.0f; // dv/dx
        case 5: return (numParams >= 6) ? gradY * y : 0.0f; // dv/dy
        // 二阶参数
        case 6:  return (numParams >= 12) ? gradX * x * x * 0.5f : 0.0f; // d²u/dx²
        case 7:  return (numParams >= 12) ? gradX * x * y : 0.0f;         // d²u/dxdy
        case 8:  return (numParams >= 12) ? gradX * y * y * 0.5f : 0.0f; // d²u/dy²
        case 9:  return (numParams >= 12) ? gradY * x * x * 0.5f : 0.0f; // d²v/dx²
        case 10: return (numParams >= 12) ? gradY * x * y : 0.0f;        // d²v/dxdy
        case 11: return (numParams >= 12) ? gradY * y * y * 0.5f : 0.0f; // d²v/dy²
        default: return 0.0f;
    }
}

 // 预计算Hessian矩阵，与CPU版本完全一致
__device__ void computehessian(const float* ref_image, int subsetRadius, int imageHeight, int imageWidth,
                                      int centerY, int centerX, int numParams, float* hessian) 
 {
    for (int i = 0; i < numParams; i++) 
    {
        for (int j = i; j < numParams; j++) 
        {
            float sum = 0.0f;

            // 遍历子集计算Hessian元素
            for (int ly = -subsetRadius; ly <= subsetRadius; ly++) {
                for (int lx = -subsetRadius; lx <= subsetRadius; lx++) {
                    int refX = centerX + lx;
                    int refY = centerY + ly;

                    if (refX >= 1 && refX < imageWidth - 1 && refY >= 1 && refY < imageHeight - 1) {
                        // 计算Sobel梯度
                        float gradX, gradY;
                        computeSobelGradients(ref_image, refX, refY, imageWidth, imageHeight, gradX, gradY);
                        
                        // 计算shape function derivatives
                        float x = static_cast<float>(lx);
                        float y = static_cast<float>(ly);

                        float shapeFni, shapeFnj;
                        shapeFni = getShapeFn(i, numParams, gradX, gradY, x, y);
                        shapeFnj = getShapeFn(j, numParams, gradX, gradY, x, y);
                        sum += shapeFni * shapeFnj;
                    }
                }
            }
            // 存储Hessian矩阵元素
            hessian[i * numParams + j] = sum;
            hessian[j * numParams + i] = sum; // 对称矩阵
        }
    }
 }   


__device__ ZNCCAndErrorResult computeZNCCAndError(
    const float* refImage, const float* tarImage,
    float centerY, float centerX, const float* warpParams,
    int imageHeight, int imageWidth, int subsetRadius, int numParams)
{
    float sumRef = 0.0f, sumDef = 0.0f;
    float sumRefSq = 0.0f, sumDefSq = 0.0f;
    float sumRefDef = 0.0f;
    int count = 0;
    float errorVec[12] = {0};

    for (int ly = -subsetRadius; ly <= subsetRadius; ly++) 
    {
        for (int lx = -subsetRadius; lx <= subsetRadius; lx++) 
        {
            int refX = centerX + lx;
            int refY = centerY + ly;
            // 确保在图像边界内
            if (refX >= 1 && refX < imageWidth - 1 && refY >= 1 && refY < imageHeight - 1) {
                float refI = refImage[refY * imageWidth + refX];
                float warpedX, warpedY;
                WarpPoint(float(lx), float(ly), warpParams, warpedX, warpedY, numParams);
                float tarImgX = float(centerX) + warpedX;
                float tarImgY = float(centerY) + warpedY;
                if (tarImgX >= 0.0f && tarImgX < imageWidth - 1.0f && tarImgY >= 0.0f && tarImgY < imageHeight - 1.0f) {
                    float defI = BilinearInterpolation(tarImgX, tarImgY, tarImage, imageWidth, imageHeight);
                    float error = refI - defI;

                    // ZNCC统计量
                    sumRef += refI;
                    sumDef += defI;
                    sumRefSq += refI * refI;
                    sumDefSq += defI * defI;
                    sumRefDef += refI * defI;
                    count++;

                    // 误差向量
                    float gradX, gradY;
                    computeSobelGradients(refImage, refX, refY, imageWidth, imageHeight, gradX, gradY);
                    float x = float(lx), y = float(ly);

                    if (numParams >= 6) {
                        errorVec[0] += error * gradX;       // du
                        errorVec[1] += error * gradY;       // dv
                        errorVec[2] += error * gradX * x;   // du/dx
                        errorVec[3] += error * gradX * y;   // du/dy
                        errorVec[4] += error * gradY * x;   // dv/dx
                        errorVec[5] += error * gradY * y;   // dv/dy
                    }
                    if (numParams >= 12) {
                        errorVec[6]  += error * gradX * x * x * 0.5;    // d²u/dx²
                        errorVec[7]  += error * gradX * x * y;          // d²u/dxdy
                        errorVec[8]  += error * gradX * y * y * 0.5;    // d²u/dy²
                        errorVec[9]  += error * gradY * x * x * 0.5;    // d²v/dx²
                        errorVec[10] += error * gradY * x * y;          // d²v/dxdy
                        errorVec[11] += error * gradY * y * y * 0.5;    // d²v/dy²
                    }
                }
            }
        }
    }
    // 计算ZNCC
    float zncc = -4.0f; // 默认值
    if (count > 0) {
        float meanRef = sumRef / count, meanDef = sumDef / count;
        float varRef = sumRefSq / count - meanRef * meanRef;
        float varDef = sumDefSq / count - meanDef * meanDef;
        float covar = sumRefDef / count - meanRef * meanDef;
        if (varRef > 1e-10 && varDef > 1e-10)
            zncc = covar / sqrt(varRef * varDef);
    }
    ZNCCAndErrorResult result;
    result.zncc = zncc;
    for (int i = 0; i < 12; ++i) result.errorVector[i] = errorVec[i];
    result.count = count;
    return result;
}


//******************************************************icgn3D******************************************************/
__device__ void computeSobelGradients3D(const float* image, int x, int y, int z,
                                     int width, int height, int depth,
                                     float* grad_x, float* grad_y, float* grad_z)
{
    // Sobel核权重
    const int wx[3] = {-1, 0, 1};
    const int w[3]  = {1, 2, 1};

    // 边界检查
    if (x < 1 || x >= width-1 || y < 1 || y >= height-1 || z < 1 || z >= depth-1) {
        *grad_x = 0;
        *grad_y = 0;
        *grad_z = 0;
        return;
    }

    float gx = 0.0f, gy = 0.0f, gz = 0.0f;

    #pragma unroll
    for (int dz = -1; dz <= 1; ++dz) {
        int zpos = z + dz;
        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            int ypos = y + dy;
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                int xpos = x + dx;
                float val = image[zpos * height * width + ypos * width + xpos];
                gx += wx[dx+1] * w[dy+1] * w[dz+1] * val;
                gy += wx[dy+1] * w[dx+1] * w[dz+1] * val;
                gz += wx[dz+1] * w[dx+1] * w[dy+1] * val;
            }
        }
    }
    *grad_x = gx / 32.0f;
    *grad_y = gy / 32.0f;
    *grad_z = gz / 32.0f;
}


// 三维shape function查表
__device__ float getShapeFn3D(int paramIdx, int numParams, float gradX, float gradY, float gradZ, float x, float y, float z) {
    // 3D 12参数顺序: u, ux, uy, uz, v, vx, vy, vz, w, wx, wy, wz
    switch (paramIdx) {
        case 0:  return gradX;                // du
        case 1:  return gradX * x;            // du/dx
        case 2:  return gradX * y;            // du/dy
        case 3:  return gradX * z;            // du/dz
        case 4:  return gradY;                // dv
        case 5:  return gradY * x;            // dv/dx
        case 6:  return gradY * y;            // dv/dy
        case 7:  return gradY * z;            // dv/dz
        case 8:  return gradZ;                // dw
        case 9:  return gradZ * x;            // dw/dx
        case 10: return gradZ * y;            // dw/dy
        case 11: return gradZ * z;            // dw/dz
        default: return 0.0f;
    }
}

// 三维Hessian预计算
__device__ void computehessian3d(
    const float* ref_image,
    int subsetRadius,
    int imageWidth,
    int imageHeight,
    int imageDepth,
    int centerX,
    int centerY,
    int centerZ,
    int numParams,
    float* hessian)
{
    for (int i = 0; i < numParams; i++) {
        for (int j = i; j < numParams; j++) {
            float sum = 0.0f;

            // 遍历三维子集
            for (int lz = -subsetRadius; lz <= subsetRadius; lz++) {
                for (int ly = -subsetRadius; ly <= subsetRadius; ly++) {
                    for (int lx = -subsetRadius; lx <= subsetRadius; lx++) {
                        int refX = centerX + lx;
                        int refY = centerY + ly;
                        int refZ = centerZ + lz;

                        if (refX >= 1 && refX < imageWidth - 1 &&
                            refY >= 1 && refY < imageHeight - 1 &&
                            refZ >= 1 && refZ < imageDepth - 1) {

                            // 计算三维Sobel梯度
                            float gradX, gradY, gradZ;
                            computeSobelGradients3D(ref_image, refX, refY, refZ, imageWidth, imageHeight, imageDepth, &gradX, &gradY, &gradZ);

                            float x = static_cast<float>(lx);
                            float y = static_cast<float>(ly);
                            float z = static_cast<float>(lz);

                            float shapeFni = getShapeFn3D(i, numParams, gradX, gradY, gradZ, x, y, z);
                            float shapeFnj = getShapeFn3D(j, numParams, gradX, gradY, gradZ, x, y, z);

                            sum += shapeFni * shapeFnj;
                        }
                    }
                }
            }

            hessian[i * numParams + j] = sum;
            hessian[j * numParams + i] = sum; // 对称矩阵
        }
    }
}


// 三维二次形变仿射+二阶，参数顺序：u, ux, uy, uz, v, vx, vy, vz, w, wx, wy, wz
__device__ __forceinline__ void WarpPoint3D(
    float x, float y, float z, const float* warpParams,
    float& warpedX, float& warpedY, float& warpedZ)
{
    // 平移分量
    float u = warpParams[0];
    float v = warpParams[4];
    float w = warpParams[8];

    warpedX = x + u;
    warpedY = y + v;
    warpedZ = z + w;

    // du/dx, du/dy, du/dz
    float ux = warpParams[1];
    float uy = warpParams[2];
    float uz = warpParams[3];
    // dv/dx, dv/dy, dv/dz
    float vx = warpParams[5];
    float vy = warpParams[6];
    float vz = warpParams[7];
    // dw/dx, dw/dy, dw/dz
    float wx = warpParams[9];
    float wy = warpParams[10];
    float wz = warpParams[11];

    warpedX += ux * x + uy * y + uz * z;
    warpedY += vx * x + vy * y + vz * z;
    warpedZ += wx * x + wy * y + wz * z;
}


__device__ __forceinline__ float TrilinearInterpolation(
    float x, float y, float z,
    const float* image, int width, int height, int depth)
{
    // 边界检查
    if (x < 0.0f || x >= width - 1.0f ||
        y < 0.0f || y >= height - 1.0f ||
        z < 0.0f || z >= depth - 1.0f) {
        return 0.0f;
    }

    // 整数和小数部分
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int z1 = static_cast<int>(z);
    int x2 = x1 + 1;
    int y2 = y1 + 1;
    int z2 = z1 + 1;

    float fx = x - static_cast<float>(x1);
    float fy = y - static_cast<float>(y1);
    float fz = z - static_cast<float>(z1);

    // 边界保护
    x2 = x2 >= width  ? width  - 1 : x2;
    y2 = y2 >= height ? height - 1 : y2;
    z2 = z2 >= depth  ? depth  - 1 : z2;

    // 取8个角点
    float c000 = image[z1 * height * width + y1 * width + x1];
    float c001 = image[z1 * height * width + y1 * width + x2];
    float c010 = image[z1 * height * width + y2 * width + x1];
    float c011 = image[z1 * height * width + y2 * width + x2];
    float c100 = image[z2 * height * width + y1 * width + x1];
    float c101 = image[z2 * height * width + y1 * width + x2];
    float c110 = image[z2 * height * width + y2 * width + x1];
    float c111 = image[z2 * height * width + y2 * width + x2];

    // 3D插值
    float c00 = c000 * (1 - fx) + c001 * fx;
    float c01 = c010 * (1 - fx) + c011 * fx;
    float c10 = c100 * (1 - fx) + c101 * fx;
    float c11 = c110 * (1 - fx) + c111 * fx;

    float c0 = c00 * (1 - fy) + c01 * fy;
    float c1 = c10 * (1 - fy) + c11 * fy;

    float val = c0 * (1 - fz) + c1 * fz;

    return val;
}


__device__ ZNCCAndErrorResult computeZNCCAndError3D(
    const float* refImage, const float* tarImage,
    float centerZ, float centerY, float centerX, const float* warpParams,
    int depth, int height, int width, int subsetRadius)
{
    float sumRef = 0.0f, sumDef = 0.0f;
    float sumRefSq = 0.0f, sumDefSq = 0.0f;
    float sumRefDef = 0.0f;
    int count = 0;
    float errorVec[12] = {0};

    for (int lz = -subsetRadius; lz <= subsetRadius; lz++) {
        for (int ly = -subsetRadius; ly <= subsetRadius; ly++) {
            for (int lx = -subsetRadius; lx <= subsetRadius; lx++) {
                int refX = int(centerX) + lx;
                int refY = int(centerY) + ly;
                int refZ = int(centerZ) + lz;
                // 确保在图像边界内
                if (refX >= 1 && refX < width - 1 &&
                    refY >= 1 && refY < height - 1 &&
                    refZ >= 1 && refZ < depth - 1) {
                    float refI = refImage[refZ * height * width + refY * width + refX];
                    float warpedX, warpedY, warpedZ;
                    WarpPoint3D(float(lx), float(ly), float(lz), warpParams, warpedX, warpedY, warpedZ);
                    float tarImgX = centerX + warpedX;
                    float tarImgY = centerY + warpedY;
                    float tarImgZ = centerZ + warpedZ;
                    if (tarImgX >= 0.0f && tarImgX < width - 1.0f &&
                        tarImgY >= 0.0f && tarImgY < height - 1.0f &&
                        tarImgZ >= 0.0f && tarImgZ < depth - 1.0f) {
                        float defI = TrilinearInterpolation(tarImgX, tarImgY, tarImgZ, tarImage, width, height, depth);
                        float error = refI - defI;

                        // ZNCC统计量
                        sumRef += refI;
                        sumDef += defI;
                        sumRefSq += refI * refI;
                        sumDefSq += defI * defI;
                        sumRefDef += refI * defI;
                        count++;

                        // 误差向量（3D 12参数：u, ux, uy, uz, v, vx, vy, vz, w, wx, wy, wz）
                        float gradX, gradY, gradZ;
                        computeSobelGradients3D(refImage, refX, refY, refZ, width, height, depth, &gradX, &gradY, &gradZ);
                        float x = float(lx), y = float(ly), z = float(lz);

                        errorVec[0]  += error * gradX;           // u
                        errorVec[1]  += error * gradX * x;       // ux
                        errorVec[2]  += error * gradX * y;       // uy
                        errorVec[3]  += error * gradX * z;       // uz

                        errorVec[4]  += error * gradY;           // v
                        errorVec[5]  += error * gradY * x;       // vx
                        errorVec[6]  += error * gradY * y;       // vy
                        errorVec[7]  += error * gradY * z;       // vz

                        errorVec[8]  += error * gradZ;           // w
                        errorVec[9]  += error * gradZ * x;       // wx
                        errorVec[10] += error * gradZ * y;       // wy
                        errorVec[11] += error * gradZ * z;       // wz
                    }
                }
            }
        }
    }
    // 计算ZNCC
    float zncc = -4.0f; // 默认值
    if (count > 0) {
        float meanRef = sumRef / count, meanDef = sumDef / count;
        float varRef = sumRefSq / count - meanRef * meanRef;
        float varDef = sumDefSq / count - meanDef * meanDef;
        float covar = sumRefDef / count - meanRef * meanDef;
        if (varRef > 1e-10f && varDef > 1e-10f)
            zncc = covar / sqrtf(varRef * varDef);
    }
    ZNCCAndErrorResult result;
    result.zncc = zncc;
    for (int i = 0; i < 12; ++i) result.errorVector[i] = errorVec[i];
    result.count = count;
    return result;
}

