#include "sc_sift.h"
#include "sc_sift_affine.h"
#include "sc_icgn.h"
#include <vector>
#include <cuda_runtime.h>

/*
// 假定有N组图像和POI，每组对应一组参考图、目标图和POI列表
const int N = ...; // 批量数量
std::vector<const float*> ref_imgs(N);   // 每组参考图像指针
std::vector<const float*> tar_imgs(N);   // 每组目标图像指针
int width = ...;
int height = ...;
std::vector<std::vector<POI2D>> poi_lists(N); // 每组POI列表

cudaStream_t stream;
cudaStreamCreate(&stream);

for (int i = 0; i < N; ++i) {
    // --- SIFT批量检测与匹配 ---
    StudyCorr_GPU::SiftFeatureBatchGpu sift_batch;
    sift_batch.prepare_cuda(ref_imgs[i], tar_imgs[i], width, height, stream);
    sift_batch.compute_match_batch_cuda(stream);

    // 匹配结果（每组一一配对）
    const SiftFeature2D* match_kp_ref = sift_batch.match_kp_ref.data();
    const SiftFeature2D* match_kp_tar = sift_batch.match_kp_tar.data();
    int num_match = sift_batch.num_match;

    // --- 仿射初始化批量流 ---
    StudyCorr_GPU::SiftAffineBatchGpu affine_batch;
    affine_batch.set_poi_list(poi_lists[i]);
    affine_batch.prepare_cuda(match_kp_ref, match_kp_tar, num_match);

    affine_batch.compute_batch_cuda(poi_lists[i].data(), int(poi_lists[i].size()), stream);

    // --- ICGN批量流 ---
    StudyCorr_GPU::ICGN2D1BatchGpu icgn_batch;
    StudyCorr_GPU::ICGNParam icgn_param;
    icgn_batch.prepare_cuda(ref_imgs[i], tar_imgs[i], height, width, icgn_param);

    // 此时POI deformation已被仿射初始化
    icgn_batch.compute_batch_cuda(poi_lists[i].data(), int(poi_lists[i].size()), stream);

    // 此组结果已更新在 poi_lists[i] 中
}

cudaStreamDestroy(stream);

*/