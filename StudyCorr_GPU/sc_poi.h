#pragma once
#include "sc_point.h"
#include "sc_deformation.h"

namespace StudyCorr_GPU
{
    //structures included in POI
	union DeformationVector2D
	{
		struct
		{
			float u, ux, uy, uxx, uxy, uyy;
			float v, vx, vy, vxx, vxy, vyy;
		};
		float p[12]; //order: u ux uy uxx uxy uyy v vx vy vxx vxy vyy
	};

	union StrainVector2D
	{
		struct
		{
			float exx, eyy, exy;
		};
		float e[3]; //order: exx, eyy, exy
	};

	union Result2D
	{
		struct
		{
			float u0, v0, zncc, iteration, convergence, feature;
		};
		float r[6];
	};

	union Result2DS
	{
		struct
		{
			float r1r2_zncc, r1t1_zncc, r1t2_zncc, r2_x, r2_y, t1_x, t1_y, t2_x, t2_y;
		};
		float r[9];
	};

	union DeformationVector3D
	{
		struct
		{
			float u, ux, uy, uz;
			float v, vx, vy, vz;
			float w, wx, wy, wz;
		};
		float p[12]; //order: u ux uy uz v vx vy vz w wx wy wz
	};

	union DisplacementVector3D
	{
		struct
		{
			float u, v, w;
		};
		float p[3];
	};

	union StrainVector3D
	{
		struct
		{
			float exx, eyy, ezz;
			float exy, eyz, ezx;
		};
		float e[6]; //order: exx, eyy, ezz, exy, eyz, ezx
	};

	union Result3D
	{
		struct
		{
			float u0, v0, w0, zncc, iteration, convergence, feature;
		};
		float r[7];
	};

    struct CudaPOI2D
    {
        float x, y;
        DeformationVector2D deformation;
        Result2D result;
        StrainVector2D strain;
        float subset_radius_x, subset_radius_y;
    };
   

    class POI2D : public Point2D
    {
    public:
        DeformationVector2D deformation;
        Result2D result;
        StrainVector2D strain;
        Point2D subset_radius;

        __host__ __device__ POI2D(int x, int y);
        __host__ __device__ POI2D(float x, float y);
        __host__ __device__ POI2D(Point2D location);
        __host__ __device__ ~POI2D();

        // 重置除位置外的数据
        void clear();
    };

	struct CudaPOI2DS
    {
		Point2D left_coor, right_coor;
		Point3D coor3D;
        DeformationVector3D deformation;
        Result2DS result;
        StrainVector3D strain;
        float subset_radius_x, subset_radius_y;
    };

    // 立体 DIC 的 POI 类
    class POI2DS : public Point2D
    {
    public:
        DisplacementVector3D deformation;
        Result2DS result;
        Point3D ref_coor, tar_coor;
        StrainVector3D strain;
        Point2D subset_radius;

        __host__ __device__ POI2DS(int x, int y);
        __host__ __device__ POI2DS(float x, float y);
        __host__ __device__ POI2DS(Point2D location);
        __host__ __device__ ~POI2DS();

        // 重置除位置外的数据
        void clear();
    };

    struct CudaPOI3D
    {
        float x, y, z;
        DeformationVector3D deformation;
        Result3D result;
        StrainVector3D strain;
        float subset_radius_x, subset_radius_y, subset_radius_z;
    };

    // DVC 的 POI 类
    class POI3D : public Point3D
    {
    public:
        DeformationVector3D deformation;
        Result3D result;
        StrainVector3D strain;
        Point3D subset_radius;

        __host__ __device__ POI3D(int x, int y, int z);
        __host__ __device__ POI3D(float x, float y, float z);
        __host__ __device__ POI3D(Point3D location);
        __host__ __device__ ~POI3D();

        // 重置除位置外的数据
        void clear();
    };
}

