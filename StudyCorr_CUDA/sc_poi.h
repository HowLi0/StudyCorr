#pragma once
#include "sc_point.h"

namespace StudyCorr
{

// 2D Deformation, Strain, Result
union DeformationVector2D
{
    struct { float u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy; };
    float p[12];
    CUDA_HD DeformationVector2D() { for(int i=0;i<12;++i) p[i]=0.f; }
};

union StrainVector2D
{
    struct { float exx, eyy, exy; };
    float e[3];
    CUDA_HD StrainVector2D() { for(int i=0;i<3;++i) e[i]=0.f; }
};

union Result2D
{
    struct { float u0, v0, zncc, iteration, convergence, feature; };
    float r[6];
    CUDA_HD Result2D() { for(int i=0;i<6;++i) r[i]=0.f; }
};

// 2DS (stereo) Result, Displacement, Strain
union DisplacementVector3D
{
    struct { float u, v, w; };
    float p[3];
    CUDA_HD DisplacementVector3D() { for(int i=0;i<3;++i) p[i]=0.f; }
};

union Result2DS
{
    struct { float r1r2_zncc, r1t1_zncc, r1t2_zncc, r2_x, r2_y, t1_x, t1_y, t2_x, t2_y; };
    float r[9];
    CUDA_HD Result2DS() { for(int i=0;i<9;++i) r[i]=0.f; }
};

union StrainVector3D
{
    struct { float exx, eyy, ezz, exy, eyz, ezx; };
    float e[6];
    CUDA_HD StrainVector3D() { for(int i=0;i<6;++i) e[i]=0.f; }
};

// 3D Deformation, Result
union DeformationVector3D
{
    struct { float u, ux, uy, uz, v, vx, vy, vz, w, wx, wy, wz; };
    float p[12];
    CUDA_HD DeformationVector3D() { for(int i=0;i<12;++i) p[i]=0.f; }
};

union Result3D
{
    struct { float u0, v0, w0, zncc, iteration, convergence, feature; };
    float r[7];
    CUDA_HD Result3D() { for(int i=0;i<7;++i) r[i]=0.f; }
};

// =============================================
// CUDA-accessible POI types

struct CudaPOI2D
{
    Point2D location;
    DeformationVector2D deformation;
    Result2D result;
    StrainVector2D strain;
    Point2D subset_radius;

    CUDA_HD CudaPOI2D() : location(), deformation(), result(), strain(), subset_radius() {}
    CUDA_HD CudaPOI2D(Point2D loc) : location(loc), deformation(), result(), strain(), subset_radius() {}

    CUDA_HD void clear() {
        for(int i=0;i<12;++i) deformation.p[i]=0.f;
        for(int i=0;i<6;++i) result.r[i]=0.f;
        for(int i=0;i<3;++i) strain.e[i]=0.f;
        subset_radius.x=0.f; subset_radius.y=0.f;
    }
};

struct CudaPOI2DS
{
    Point2D location;
    DisplacementVector3D deformation;
    Result2DS result;
    Point3D ref_coor, tar_coor;
    StrainVector3D strain;
    Point2D subset_radius;

    CUDA_HD CudaPOI2DS() : location(), deformation(), result(), ref_coor(), tar_coor(), strain(), subset_radius() {}
    CUDA_HD CudaPOI2DS(Point2D loc) : location(loc), deformation(), result(), ref_coor(), tar_coor(), strain(), subset_radius() {}

    CUDA_HD void clear() {
        for(int i=0;i<3;++i) deformation.p[i]=0.f;
        for(int i=0;i<9;++i) result.r[i]=0.f;
        ref_coor = Point3D();
        tar_coor = Point3D();
        for(int i=0;i<6;++i) strain.e[i]=0.f;
        subset_radius.x=0.f; subset_radius.y=0.f;
    }
};

struct CudaPOI3D
{
    Point3D location;
    DeformationVector3D deformation;
    Result3D result;
    StrainVector3D strain;
    Point3D subset_radius;

    CUDA_HD CudaPOI3D() : location(), deformation(), result(), strain(), subset_radius() {}
    CUDA_HD CudaPOI3D(Point3D loc) : location(loc), deformation(), result(), strain(), subset_radius() {}

    CUDA_HD void clear() {
        for(int i=0;i<12;++i) deformation.p[i]=0.f;
        for(int i=0;i<7;++i) result.r[i]=0.f;
        for(int i=0;i<6;++i) strain.e[i]=0.f;
        subset_radius.x=0.f; subset_radius.y=0.f; subset_radius.z=0.f;
    }
};

} // namespace StudyCorr