#pragma once
#include "sc_array.h"
#include "sc_poi.h"
#include "sc_image.h"
#include "sc_gradient.h"
#include "sc_cubic_bspline.h"
#include <cusolverDn.h>

namespace StudyCorr {

struct ICGN2DConfig {
    int subset_radius_x = 15;
    int subset_radius_y = 15;
    float conv_criterion = 1e-3f;
    int max_iterations = 10;
};

class ICGN2D1 {
public:
    CudaImage2D* ref_img = nullptr;
    CudaImage2D* tar_img = nullptr;
    Gradient2D4* ref_gradient = nullptr;
    BicubicBsplineInterp* tar_interp = nullptr;
    ICGN2DConfig cfg;

    float hessian[36];
    float inv_hessian[36];

    ICGN2D1(const ICGN2DConfig& config = ICGN2DConfig());
    ~ICGN2D1();

    void setImages(CudaImage2D* ref, CudaImage2D* tar);
    void prepare(); // 计算梯度/插值和Hessian/逆
    void compute(Array2D<CudaPOI2D>& poi_queue);

private:
    void build_hessian_and_inverse();
    cusolverDnHandle_t cusolverH = nullptr;
};

class ICGN2D2 {
public:
    CudaImage2D* ref_img = nullptr;
    CudaImage2D* tar_img = nullptr;
    Gradient2D4* ref_gradient = nullptr;
    BicubicBsplineInterp* tar_interp = nullptr;
    ICGN2DConfig cfg;

    float hessian[144];
    float inv_hessian[144];

    ICGN2D2(const ICGN2DConfig& config = ICGN2DConfig());
    ~ICGN2D2();

    void setImages(CudaImage2D* ref, CudaImage2D* tar);
    void prepare();
    void compute(Array2D<CudaPOI2D>& poi_queue);

private:
    void build_hessian_and_inverse();
    cusolverDnHandle_t cusolverH = nullptr;
};

} // namespace StudyCorr