#pragma once
#include "sc_point.h"
#include "sc_array.h"

namespace StudyCorr
{

class NearestNeighbor
{
public:
    NearestNeighbor() = default;
    ~NearestNeighbor() = default;

    // 2D半径搜索
    static void radiusSearch2D(
        const Array1D<Point2D>& ref_pts,
        const Array1D<Point2D>& queries,
        float radius,
        Array1D<int>& out_counts,
        Array1D<int>& out_indices,
        int max_neighbors);

    // 3D半径搜索
    static void radiusSearch3D(
        const Array1D<Point3D>& ref_pts,
        const Array1D<Point3D>& queries,
        float radius,
        Array1D<int>& out_counts,
        Array1D<int>& out_indices,
        int max_neighbors);

    // 2D KNN搜索
    static void knnSearch2D(
        const Array1D<Point2D>& ref_pts,
        const Array1D<Point2D>& queries,
        int K,
        Array1D<int>& out_indices);

    // 3D KNN搜索
    static void knnSearch3D(
        const Array1D<Point3D>& ref_pts,
        const Array1D<Point3D>& queries,
        int K,
        Array1D<int>& out_indices);
};

} // namespace StudyCorr