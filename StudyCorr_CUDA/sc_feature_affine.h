#pragma once
#include "sc_dic.h"
#include "sc_array.h"
#include "sc_point.h"
#include "sc_poi.h"
#include "sc_nearest_neighbor.h"

namespace StudyCorr
{

struct AffineRansacConfig {
    int ransac_trials = 20;        // oc默认20
    float inlier_thresh = 1.5f;    // oc默认1.5f
    int max_neighbors = 32;
    float search_radius = 21.0f;
    int min_neighbor_num = 7;      // oc默认7
};

class FeatureAffine2D : public CudaDIC
{
public:
    FeatureAffine2D(int subset_radius_x, int subset_radius_y)
        : CudaDIC()
    {
        setSubsetRadius(subset_radius_x, subset_radius_y);
        cfg_.search_radius = sqrtf(
            subset_radius_x * subset_radius_x +
            subset_radius_y * subset_radius_y);
    }

    void setKeypoints(const Array1D<Point2D>& ref, const Array1D<Point2D>& tar);
    void setAffineConfig(const AffineRansacConfig& cfg);
    void setSearchParameters(float radius, int min_neighbor_num);

    void prepare() override {}

    void compute(CudaPOI2D* poi) override {}

    void compute(Array2D<CudaPOI2D>& poi_queue) override;

private:
    Array1D<Point2D> ref_pts_, tar_pts_;
    int n_ref_ = 0;
    AffineRansacConfig cfg_;

    void cuda_affine_ransac(
        const Array1D<Point2D>& ref_pts,
        const Array1D<Point2D>& tar_pts,
        const Array1D<int>& nb_counts,
        const Array1D<int>& nb_indices,
        CudaPOI2D* pois, int n_poi, int max_neighbors,
        int ransac_trials, float inlier_thresh, int min_neighbor_num);
};

} // namespace StudyCorr