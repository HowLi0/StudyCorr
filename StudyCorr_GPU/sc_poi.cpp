#include "sc_poi.h"

namespace StudyCorr_GPU
{
    // POI2D 构造函数
    POI2D::POI2D(int x, int y) : Point2D(x, y)
    {
        clear();
    }

    POI2D::POI2D(float x, float y) : Point2D(x, y)
    {
        clear();
    }

    POI2D::POI2D(Point2D location) : Point2D(location)
    {
        clear();
    }

    POI2D::~POI2D() {}

    // 重置 POI2D 的数据
    void POI2D::clear()
    {
        deformation = DeformationVector2D();
        result = Result2D();
        strain = StrainVector2D();
        subset_radius.x = 0.f;
        subset_radius.y = 0.f;
    }

    // POI2DS 构造函数
    POI2DS::POI2DS(int x, int y) : Point2D(x, y)
    {
        clear();
    }

    POI2DS::POI2DS(float x, float y) : Point2D(x, y)
    {
        clear();
    }

    POI2DS::POI2DS(Point2D location) : Point2D(location)
    {
        clear();
    }

    POI2DS::~POI2DS() {}

    // 重置 POI2DS 的数据
    void POI2DS::clear()
    {
        deformation = DisplacementVector3D();
        result = Result2DS();
        ref_coor = Point3D();
        tar_coor = Point3D();
        strain = StrainVector3D();
        subset_radius.x = 0.f;
        subset_radius.y = 0.f;
    }

    // POI3D 构造函数
    POI3D::POI3D(int x, int y, int z) : Point3D(x, y, z)
    {
        clear();
    }

    POI3D::POI3D(float x, float y, float z) : Point3D(x, y, z)
    {
        clear();
    }

    POI3D::POI3D(Point3D location) : Point3D(location)
    {
        clear();
    }

    POI3D::~POI3D() {}

    // 重置 POI3D 的数据
    void POI3D::clear()
    {
        deformation = DeformationVector3D();
        result = Result3D();
        strain = StrainVector3D();
        subset_radius.x = 0.f;
        subset_radius.y = 0.f;
        subset_radius.z = 0.f;
    }
}//namespace StudyCorr
