
#include "sc_icgn.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cmath>
#include <vector>

namespace StudyCorr {

//****************************************ICGN2D1****************************************

ICGN2D1::ICGN2D1(const ICGN2DConfig& config) : cfg(config) {}
ICGN2D1::~ICGN2D1() {
    delete ref_gradient;
    delete tar_interp;
    if (cusolverH) cusolverDnDestroy(cusolverH);
}

void ICGN2D1::setImages(CudaImage2D* ref, CudaImage2D* tar) {
    ref_img = ref;
    tar_img = tar;
}
void ICGN2D1::prepare() {
    if (ref_gradient) delete ref_gradient;
    ref_gradient = new Gradient2D4(*ref_img);
    ref_gradient->getGradientX();
    ref_gradient->getGradientY();

    if (tar_interp) delete tar_interp;
    tar_interp = new BicubicBsplineInterp(*tar_img);
    tar_interp->prepare();

    if (!cusolverH) cusolverDnCreate(&cusolverH);
    build_hessian_and_inverse();
}

void ICGN2D1::build_hessian_and_inverse() {
    // 计算Steepest Descent图像并积累Hessian（host侧）
    int rx = cfg.subset_radius_x, ry = cfg.subset_radius_y;
    int w = 2 * rx + 1, h = 2 * ry + 1, area = w * h;
    memset(hessian, 0, sizeof(hessian));
    for(int r=0;r<h;++r) for(int c=0;c<w;++c) {
        int x = c - rx, y = r - ry;
        int gx = rx + x, gy = ry + y; // 参考点位置
        float gradx = ref_gradient->grad_x(gy, gx);
        float grady = ref_gradient->grad_y(gy, gx);
        float sd[6] = {
            gradx,
            gradx * x,
            gradx * y,
            grady,
            grady * x,
            grady * y
        };
        for(int i=0;i<6;++i)
            for(int j=0;j<=i;++j)
                hessian[i*6 + j] += sd[i]*sd[j];
    }
    for(int i=0;i<6;++i) for(int j=0;j<i;++j)
        hessian[j*6+i] = hessian[i*6+j];

    // 用cuSolver求逆
    float *d_A, *d_I;
    int n = 6, lda = 6, info;
    cudaMalloc(&d_A, sizeof(float)*36);
    cudaMalloc(&d_I, sizeof(float)*36);
    cudaMemcpy(d_A, hessian, sizeof(float)*36, cudaMemcpyHostToDevice);
    std::vector<float> hI(36,0.f);
    for(int i=0;i<6;++i) hI[i*6+i]=1.f;
    cudaMemcpy(d_I, hI.data(), sizeof(float)*36, cudaMemcpyHostToDevice);
    int lwork=0, *devInfo;
    float *d_work;
    cudaMalloc(&devInfo, sizeof(int));
    cusolverDnSgetrf_bufferSize(cusolverH, n, n, d_A, lda, &lwork);
    cudaMalloc(&d_work, sizeof(float)*lwork);
    int* pivots; cudaMalloc(&pivots, sizeof(int)*n);
    cusolverDnSgetrf(cusolverH, n, n, d_A, lda, d_work, pivots, devInfo);
    cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, n, n, d_A, lda, pivots, d_I, lda, devInfo);
    cudaMemcpy(inv_hessian, d_I, sizeof(float)*36, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_I); cudaFree(d_work); cudaFree(pivots); cudaFree(devInfo);
}

// 放在constant，所有线程共享
__constant__ float d_inv_hessian[36];

// CUDA kernel
__global__ void icgn2d1_kernel(
    const float* ref_data, int ref_w, int ref_h,
    const float* tar_data, int tar_w, int tar_h,
    const float* grad_x, const float* grad_y,
    CudaPOI2D* pois, int n_poi,
    int subset_rx, int subset_ry,
    float conv_criterion, int max_iter,
    BicubicBsplineInterp tar_interp
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_poi) return;
    CudaPOI2D& poi = pois[idx];

    int cx = static_cast<int>(poi.location.x), cy = static_cast<int>(poi.location.y);
    int subset_w = 2*subset_rx+1, subset_h = 2*subset_ry+1, area = subset_w*subset_h;
    if (cx - subset_rx < 0 || cy - subset_ry < 0 ||
        cx + subset_rx >= ref_w || cy + subset_ry >= ref_h) {
        poi.result.zncc = -3.f; return;
    }
    float ref_subset[961], tar_subset[961], error_img[961], sd[6];
    // 1. 拷贝ref subset
    for(int r=0;r<subset_h;++r) for(int c=0;c<subset_w;++c)
        ref_subset[r*subset_w+c] = ref_data[(cy+r-subset_ry)*ref_w + (cx+c-subset_rx)];
    float mean_ref=0; for(int i=0;i<area;++i) mean_ref += ref_subset[i];
    mean_ref /= area;
    float norm_ref=0; for(int i=0;i<area;++i) {ref_subset[i]-=mean_ref; norm_ref+=ref_subset[i]*ref_subset[i];}
    norm_ref = sqrtf(norm_ref);

    // 2. SD图像（每次迭代可复用）
    float sd_img[961][6];
    for(int r=0;r<subset_h;++r) for(int c=0;c<subset_w;++c) {
        int x = c - subset_rx, y = r - subset_ry;
        float gx = grad_x[(cy+y)*ref_w + (cx+x)], gy = grad_y[(cy+y)*ref_w + (cx+x)];
        sd_img[r*subset_w+c][0]=gx; sd_img[r*subset_w+c][1]=gx*x; sd_img[r*subset_w+c][2]=gx*y;
        sd_img[r*subset_w+c][3]=gy; sd_img[r*subset_w+c][4]=gy*x; sd_img[r*subset_w+c][5]=gy*y;
    }

    float p[6] = {poi.deformation.u, poi.deformation.ux, poi.deformation.uy,
                  poi.deformation.v, poi.deformation.vx, poi.deformation.vy};
    float znssd=0, dp_norm=1e8; int iter=0;
    float numerator[6], dp[6];
    do {
        float mean_tar=0;
        for(int r=0;r<subset_h;++r) for(int c=0;c<subset_w;++c) {
            int x_local = c - subset_rx, y_local = r - subset_ry;
            float xw = cx + x_local + p[0] + p[1]*x_local + p[2]*y_local;
            float yw = cy + y_local + p[3] + p[4]*x_local + p[5]*y_local;
            float val = (xw>=0 && xw<tar_w-1 && yw>=0 && yw<tar_h-1) ? tar_interp.interp_at(xw, yw) : 0;
            tar_subset[r*subset_w+c]=val; mean_tar+=val;
        }
        mean_tar/=area;
        float norm_tar=0;
        for(int i=0;i<area;++i) { tar_subset[i]-=mean_tar; norm_tar+=tar_subset[i]*tar_subset[i]; }
        norm_tar=sqrtf(norm_tar);

        for(int i=0;i<area;++i)
            error_img[i]=tar_subset[i]*(norm_ref/norm_tar) - ref_subset[i];

        float error_sum=0;
        for(int i=0;i<area;++i) error_sum+=error_img[i]*error_img[i];
        znssd=error_sum/(norm_ref*norm_ref);

        for(int i=0;i<6;++i) {numerator[i]=0; for(int k=0;k<area;++k) numerator[i]+=sd_img[k][i]*error_img[k];}
        for(int i=0;i<6;++i) {dp[i]=0; for(int j=0;j<6;++j) dp[i]+=d_inv_hessian[i*6+j]*numerator[j];}
        for(int i=0;i<6;++i) p[i]-=dp[i];
        dp_norm=0; for(int i=0;i<6;++i) dp_norm+=dp[i]*dp[i]; dp_norm=sqrtf(dp_norm);
        ++iter;
        if(dp_norm<conv_criterion) break;
    } while(iter<max_iter);

    poi.deformation.u  = p[0];
    poi.deformation.ux = p[1];
    poi.deformation.uy = p[2];
    poi.deformation.v  = p[3];
    poi.deformation.vx = p[4];
    poi.deformation.vy = p[5];
    poi.result.zncc = 0.5f * (2 - znssd);
    poi.result.iteration = iter;
    poi.result.convergence = dp_norm;
}

void ICGN2D1::compute(Array2D<CudaPOI2D>& poi_queue) {
    // 把inv_hessian上传到constant内存
    cudaMemcpyToSymbol(d_inv_hessian, inv_hessian, sizeof(float)*36);
    int n_poi = poi_queue.dim1() * poi_queue.dim2();
    dim3 block(128), grid((n_poi+block.x-1)/block.x);
    icgn2d1_kernel<<<grid,block>>>(
        ref_img->data.get(), ref_img->width, ref_img->height,
        tar_img->data.get(), tar_img->width, tar_img->height,
        ref_gradient->grad_x.get(), ref_gradient->grad_y.get(),
        poi_queue.get(), n_poi,
        cfg.subset_radius_x, cfg.subset_radius_y,
        cfg.conv_criterion, cfg.max_iterations,
        *tar_interp
    );
    cudaDeviceSynchronize();
}

//****************************************ICGN2D2****************************************

ICGN2D2::ICGN2D2(const ICGN2DConfig& config) : cfg(config) {}
ICGN2D2::~ICGN2D2() {
    delete ref_gradient;
    delete tar_interp;
    if (cusolverH) cusolverDnDestroy(cusolverH);
}

void ICGN2D2::setImages(CudaImage2D* ref, CudaImage2D* tar) {
    ref_img = ref;
    tar_img = tar;
}
void ICGN2D2::prepare() {
    if (ref_gradient) delete ref_gradient;
    ref_gradient = new Gradient2D4(*ref_img);
    ref_gradient->getGradientX();
    ref_gradient->getGradientY();

    if (tar_interp) delete tar_interp;
    tar_interp = new BicubicBsplineInterp(*tar_img);
    tar_interp->prepare();

    if (!cusolverH) cusolverDnCreate(&cusolverH);
    build_hessian_and_inverse();
}

void ICGN2D2::build_hessian_and_inverse() {
    // 计算Steepest Descent图像并积累Hessian（host侧）
    int rx = cfg.subset_radius_x, ry = cfg.subset_radius_y;
    int w = 2 * rx + 1, h = 2 * ry + 1, area = w * h;
    memset(hessian, 0, sizeof(hessian));
    for(int r=0;r<h;++r) for(int c=0;c<w;++c) {
        int x = c - rx, y = r - ry;
        float xx = 0.5f * x * x;
        float xy = float(x * y);
        float yy = 0.5f * y * y;
        int gx = rx + x, gy = ry + y; // 参考点位置
        float gradx = ref_gradient->grad_x(gy, gx);
        float grady = ref_gradient->grad_y(gy, gx);
        float sd[12] = {
            gradx,
            gradx * x,
            gradx * y,
            gradx * xx,
            gradx * xy,
            gradx * yy,
            grady,
            grady * x,
            grady * y,
            grady * xx,
            grady * xy,
            grady * yy
        };
        for(int i=0;i<12;++i)
            for(int j=0;j<=i;++j)
                hessian[i*12 + j] += sd[i]*sd[j];
    }
    for(int i=0;i<12;++i) for(int j=0;j<i;++j)
        hessian[j*12+i] = hessian[i*12+j];

    // 用cuSolver求逆
    float *d_A, *d_I;
    int n = 12, lda = 12;
    cudaMalloc(&d_A, sizeof(float)*144);
    cudaMalloc(&d_I, sizeof(float)*144);
    cudaMemcpy(d_A, hessian, sizeof(float)*144, cudaMemcpyHostToDevice);
    std::vector<float> hI(144,0.f);
    for(int i=0;i<12;++i) hI[i*12+i]=1.f;
    cudaMemcpy(d_I, hI.data(), sizeof(float)*144, cudaMemcpyHostToDevice);
    int lwork=0, *devInfo;
    float *d_work;
    cudaMalloc(&devInfo, sizeof(int));
    cusolverDnSgetrf_bufferSize(cusolverH, n, n, d_A, lda, &lwork);
    cudaMalloc(&d_work, sizeof(float)*lwork);
    int* pivots; cudaMalloc(&pivots, sizeof(int)*n);
    cusolverDnSgetrf(cusolverH, n, n, d_A, lda, d_work, pivots, devInfo);
    cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, n, n, d_A, lda, pivots, d_I, lda, devInfo);
    cudaMemcpy(inv_hessian, d_I, sizeof(float)*144, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_I); cudaFree(d_work); cudaFree(pivots); cudaFree(devInfo);
}

// constant内存共享inv_hessian
__constant__ float d_inv_hessian2d2[144];

__global__ void icgn2d2_kernel(
    const float* ref_data, int ref_w, int ref_h,
    const float* tar_data, int tar_w, int tar_h,
    const float* grad_x, const float* grad_y,
    CudaPOI2D* pois, int n_poi,
    int subset_rx, int subset_ry,
    float conv_criterion, int max_iter,
    BicubicBsplineInterp tar_interp
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_poi) return;
    CudaPOI2D& poi = pois[idx];

    int cx = static_cast<int>(poi.location.x), cy = static_cast<int>(poi.location.y);
    int subset_w = 2*subset_rx+1, subset_h = 2*subset_ry+1, area = subset_w*subset_h;
    if (cx - subset_rx < 0 || cy - subset_ry < 0 ||
        cx + subset_rx >= ref_w || cy + subset_ry >= ref_h) {
        poi.result.zncc = -3.f; return;
    }
    float ref_subset[961], tar_subset[961], error_img[961];
    float sd_img[961][12];
    // 1. 拷贝ref subset
    for(int r=0;r<subset_h;++r) for(int c=0;c<subset_w;++c)
        ref_subset[r*subset_w+c] = ref_data[(cy+r-subset_ry)*ref_w + (cx+c-subset_rx)];
    float mean_ref=0; for(int i=0;i<area;++i) mean_ref += ref_subset[i];
    mean_ref /= area;
    float norm_ref=0; for(int i=0;i<area;++i) {ref_subset[i]-=mean_ref; norm_ref+=ref_subset[i]*ref_subset[i];}
    norm_ref = sqrtf(norm_ref);

    // 2. SD图像
    for(int r=0;r<subset_h;++r) for(int c=0;c<subset_w;++c) {
        int x = c - subset_rx, y = r - subset_ry;
        float xx = 0.5f * x * x, xy = float(x*y), yy = 0.5f * y * y;
        float gx = grad_x[(cy+y)*ref_w + (cx+x)], gy = grad_y[(cy+y)*ref_w + (cx+x)];
        sd_img[r*subset_w+c][0]=gx;      sd_img[r*subset_w+c][1]=gx*x;      sd_img[r*subset_w+c][2]=gx*y;
        sd_img[r*subset_w+c][3]=gx*xx;   sd_img[r*subset_w+c][4]=gx*xy;     sd_img[r*subset_w+c][5]=gx*yy;
        sd_img[r*subset_w+c][6]=gy;      sd_img[r*subset_w+c][7]=gy*x;      sd_img[r*subset_w+c][8]=gy*y;
        sd_img[r*subset_w+c][9]=gy*xx;   sd_img[r*subset_w+c][10]=gy*xy;    sd_img[r*subset_w+c][11]=gy*yy;
    }

    float p[12] = {poi.deformation.u, poi.deformation.ux, poi.deformation.uy,
        poi.deformation.uxx, poi.deformation.uxy, poi.deformation.uyy,
        poi.deformation.v, poi.deformation.vx, poi.deformation.vy,
        poi.deformation.vxx, poi.deformation.vxy, poi.deformation.vyy};
    float znssd=0, dp_norm=1e8; int iter=0;
    float numerator[12], dp[12];
    do {
        float mean_tar=0;
        for(int r=0;r<subset_h;++r) for(int c=0;c<subset_w;++c) {
            int x_local = c - subset_rx, y_local = r - subset_ry;
            float xx = 0.5f * x_local * x_local;
            float xy = float(x_local * y_local);
            float yy = 0.5f * y_local * y_local;
            float xw = cx + x_local + p[0] + p[1]*x_local + p[2]*y_local + p[3]*xx + p[4]*xy + p[5]*yy;
            float yw = cy + y_local + p[6] + p[7]*x_local + p[8]*y_local + p[9]*xx + p[10]*xy + p[11]*yy;
            float val = (xw>=0 && xw<tar_w-1 && yw>=0 && yw<tar_h-1) ? tar_interp.interp_at(xw, yw) : 0;
            tar_subset[r*subset_w+c]=val; mean_tar+=val;
        }
        mean_tar/=area;
        float norm_tar=0;
        for(int i=0;i<area;++i) { tar_subset[i]-=mean_tar; norm_tar+=tar_subset[i]*tar_subset[i]; }
        norm_tar=sqrtf(norm_tar);

        for(int i=0;i<area;++i)
            error_img[i]=tar_subset[i]*(norm_ref/norm_tar) - ref_subset[i];

        float error_sum=0;
        for(int i=0;i<area;++i) error_sum+=error_img[i]*error_img[i];
        znssd=error_sum/(norm_ref*norm_ref);

        for(int i=0;i<12;++i) {numerator[i]=0; for(int k=0;k<area;++k) numerator[i]+=sd_img[k][i]*error_img[k];}
        for(int i=0;i<12;++i) {dp[i]=0; for(int j=0;j<12;++j) dp[i]+=d_inv_hessian2d2[i*12+j]*numerator[j];}
        for(int i=0;i<12;++i) p[i]-=dp[i];
        dp_norm=0; for(int i=0;i<12;++i) dp_norm+=dp[i]*dp[i]; dp_norm=sqrtf(dp_norm);
        ++iter;
        if(dp_norm<conv_criterion) break;
    } while(iter<max_iter);

    poi.deformation.u  = p[0];
    poi.deformation.ux = p[1];
    poi.deformation.uy = p[2];
    poi.deformation.uxx = p[3];
    poi.deformation.uxy = p[4];
    poi.deformation.uyy = p[5];
    poi.deformation.v  = p[6];
    poi.deformation.vx = p[7];
    poi.deformation.vy = p[8];
    poi.deformation.vxx = p[9];
    poi.deformation.vxy = p[10];
    poi.deformation.vyy = p[11];
    poi.result.zncc = 0.5f * (2 - znssd);
    poi.result.iteration = iter;
    poi.result.convergence = dp_norm;
}

void ICGN2D2::compute(Array2D<CudaPOI2D>& poi_queue) {
    cudaMemcpyToSymbol(d_inv_hessian2d2, inv_hessian, sizeof(float)*144);
    int n_poi = poi_queue.dim1() * poi_queue.dim2();
    dim3 block(128), grid((n_poi+block.x-1)/block.x);
    icgn2d2_kernel<<<grid,block>>>(
        ref_img->data.get(), ref_img->width, ref_img->height,
        tar_img->data.get(), tar_img->width, tar_img->height,
        ref_gradient->grad_x.get(), ref_gradient->grad_y.get(),
        poi_queue.get(), n_poi,
        cfg.subset_radius_x, cfg.subset_radius_y,
        cfg.conv_criterion, cfg.max_iterations,
        *tar_interp
    );
    cudaDeviceSynchronize();
}


} // namespace StudyCorr