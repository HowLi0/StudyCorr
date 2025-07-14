#include "sc_gradient.h"
#include <cuda_runtime.h>

namespace StudyCorr
{

// ------------------ CUDA kernel for 2D ------------------
__global__ void kernel_grad2d4_x(const float* img, int w, int h, float* grad)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < h && c >= 2 && c < w - 2) {
        grad[r * w + c] =
            - img[r * w + (c + 2)] / 12.0f
            + img[r * w + (c + 1)] * (2.0f / 3.0f)
            - img[r * w + (c - 1)] * (2.0f / 3.0f)
            + img[r * w + (c - 2)] / 12.0f;
    }
}

__global__ void kernel_grad2d4_y(const float* img, int w, int h, float* grad)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < w && r >= 2 && r < h - 2) {
        grad[r * w + c] =
            - img[(r + 2) * w + c] / 12.0f
            + img[(r + 1) * w + c] * (2.0f / 3.0f)
            - img[(r - 1) * w + c] * (2.0f / 3.0f)
            + img[(r - 2) * w + c] / 12.0f;
    }
}

// 3. 混合导数（先对X，再对Y）
// grad_xy = d/dy(grad_x)
// 可直接传 grad_x 给 kernel
__global__ void kernel_grad2d4_xy(const float* gradx, int w, int h, float* gradxy)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < w && r >= 2 && r < h - 2) {
        gradxy[r * w + c] =
            - gradx[(r + 2) * w + c] / 12.0f
            + gradx[(r + 1) * w + c] * (2.0f / 3.0f)
            - gradx[(r - 1) * w + c] * (2.0f / 3.0f)
            + gradx[(r - 2) * w + c] / 12.0f;
    }
}

// ------------------ CUDA kernel for 3D ------------------
__global__ void kernel_grad3d4_x(const float* img, int D, int H, int W, float* grad)
{
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (z < D && y < H && x >= 2 && x < W - 2) {
        int idx = (z * H + y) * W + x;
        grad[idx] =
            - img[(z * H + y) * W + (x + 2)] / 12.0f
            + img[(z * H + y) * W + (x + 1)] * (2.0f / 3.0f)
            - img[(z * H + y) * W + (x - 1)] * (2.0f / 3.0f)
            + img[(z * H + y) * W + (x - 2)] / 12.0f;
    }
}

__global__ void kernel_grad3d4_y(const float* img, int D, int H, int W, float* grad)
{
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (z < D && x < W && y >= 2 && y < H - 2) {
        int idx = (z * H + y) * W + x;
        grad[idx] =
            - img[(z * H + (y + 2)) * W + x] / 12.0f
            + img[(z * H + (y + 1)) * W + x] * (2.0f / 3.0f)
            - img[(z * H + (y - 1)) * W + x] * (2.0f / 3.0f)
            + img[(z * H + (y - 2)) * W + x] / 12.0f;
    }
}

__global__ void kernel_grad3d4_z(const float* img, int D, int H, int W, float* grad)
{
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < H && x < W && z >= 2 && z < D - 2) {
        int idx = (z * H + y) * W + x;
        grad[idx] =
            - img[((z + 2) * H + y) * W + x] / 12.0f
            + img[((z + 1) * H + y) * W + x] * (2.0f / 3.0f)
            - img[((z - 1) * H + y) * W + x] * (2.0f / 3.0f)
            + img[((z - 2) * H + y) * W + x] / 12.0f;
    }
}

// ========== Gradient2D4 method implementation ==========
void Gradient2D4::getGradientX()
{
    dim3 block(32, 8), grid((image->width + block.x - 1) / block.x, (image->height + block.y - 1) / block.y);
    kernel_grad2d4_x<<<grid, block>>>(image->data.get(), image->width, image->height, grad_x.get());
    cudaDeviceSynchronize();
}

void Gradient2D4::getGradientY()
{
    dim3 block(32, 8), grid((image->width + block.x - 1) / block.x, (image->height + block.y - 1) / block.y);
    kernel_grad2d4_y<<<grid, block>>>(image->data.get(), image->width, image->height, grad_y.get());
    cudaDeviceSynchronize();
}

void Gradient2D4::getGradientXY()
{
    dim3 block(32, 8), grid((image->width + block.x - 1) / block.x, (image->height + block.y - 1) / block.y);
    kernel_grad2d4_xy<<<grid, block>>>(grad_x.get(), image->width, image->height, grad_xy.get());
    cudaDeviceSynchronize();
}

// ========== Gradient3D4 method implementation ==========
void Gradient3D4::getGradientX()
{
    dim3 block(8, 8, 8), grid(
        (image->dim_x + block.x - 1) / block.x,
        (image->dim_y + block.y - 1) / block.y,
        (image->dim_z + block.z - 1) / block.z);
    kernel_grad3d4_x<<<grid, block>>>(image->data.get(), image->dim_z, image->dim_y, image->dim_x, grad_x.get());
    cudaDeviceSynchronize();
}

void Gradient3D4::getGradientY()
{
    dim3 block(8, 8, 8), grid(
        (image->dim_x + block.x - 1) / block.x,
        (image->dim_y + block.y - 1) / block.y,
        (image->dim_z + block.z - 1) / block.z);
    kernel_grad3d4_y<<<grid, block>>>(image->data.get(), image->dim_z, image->dim_y, image->dim_x, grad_y.get());
    cudaDeviceSynchronize();
}

void Gradient3D4::getGradientZ()
{
    dim3 block(8, 8, 8), grid(
        (image->dim_x + block.x - 1) / block.x,
        (image->dim_y + block.y - 1) / block.y,
        (image->dim_z + block.z - 1) / block.z);
    kernel_grad3d4_z<<<grid, block>>>(image->data.get(), image->dim_z, image->dim_y, image->dim_x, grad_z.get());
    cudaDeviceSynchronize();
}

} // namespace StudyCorr