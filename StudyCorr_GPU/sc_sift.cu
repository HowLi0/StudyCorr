#include "sc_sift.h"
#include <cudaSift.h>
#include <cudaImage.h>
#include <cstring>
#include "sc_epipolar_search.h"

namespace StudyCorr_GPU {

SiftFeatureBatchGpu::SiftFeatureBatchGpu() {}
SiftFeatureBatchGpu::~SiftFeatureBatchGpu() { release(); }

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
#ifdef MANAGEDMEM
  SiftPoint *mpts = data.m_data;
#else
  if (data.h_data==NULL)
    return 0;
  SiftPoint *mpts = data.h_data;
#endif
  float limit = thresh*thresh;
  int numPts = data.numPts;
  cv::Mat M(8, 8, CV_64FC1);
  cv::Mat A(8, 1, CV_64FC1), X(8, 1, CV_64FC1);
  double Y[8];
  for (int i=0;i<8;i++) 
    A.at<double>(i, 0) = homography[i] / homography[8];
  for (int loop=0;loop<numLoops;loop++) {
    M = cv::Scalar(0.0);
    X = cv::Scalar(0.0);
    for (int i=0;i<numPts;i++) {
      SiftPoint &pt = mpts[i];
      if (pt.score<minScore || pt.ambiguity>maxAmbiguity)
	continue;
      float den = A.at<double>(6)*pt.xpos + A.at<double>(7)*pt.ypos + 1.0f;
      float dx = (A.at<double>(0)*pt.xpos + A.at<double>(1)*pt.ypos + A.at<double>(2)) / den - pt.match_xpos;
      float dy = (A.at<double>(3)*pt.xpos + A.at<double>(4)*pt.ypos + A.at<double>(5)) / den - pt.match_ypos;
      float err = dx*dx + dy*dy;
      float wei = (err<limit ? 1.0f : 0.0f); //limit / (err + limit);
      Y[0] = pt.xpos;
      Y[1] = pt.ypos;
      Y[2] = 1.0;
      Y[3] = Y[4] = Y[5] = 0.0;
      Y[6] = - pt.xpos * pt.match_xpos;
      Y[7] = - pt.ypos * pt.match_xpos;
      for (int c=0;c<8;c++) 
        for (int r=0;r<8;r++) 
          M.at<double>(r,c) += (Y[c] * Y[r] * wei);
      X += (cv::Mat(8,1,CV_64FC1,Y) * pt.match_xpos * wei);
      Y[0] = Y[1] = Y[2] = 0.0;
      Y[3] = pt.xpos;
      Y[4] = pt.ypos; 
      Y[5] = 1.0;
      Y[6] = - pt.xpos * pt.match_ypos;
      Y[7] = - pt.ypos * pt.match_ypos;
      for (int c=0;c<8;c++) 
        for (int r=0;r<8;r++) 
          M.at<double>(r,c) += (Y[c] * Y[r] * wei);
      X += (cv::Mat(8,1,CV_64FC1,Y) * pt.match_ypos * wei);
    }
    cv::solve(M, X, A, cv::DECOMP_CHOLESKY);
  }
  int numfit = 0;
  for (int i=0;i<numPts;i++) {
    SiftPoint &pt = mpts[i];
    float den = A.at<double>(6)*pt.xpos + A.at<double>(7)*pt.ypos + 1.0;
    float dx = (A.at<double>(0)*pt.xpos + A.at<double>(1)*pt.ypos + A.at<double>(2)) / den - pt.match_xpos;
    float dy = (A.at<double>(3)*pt.xpos + A.at<double>(4)*pt.ypos + A.at<double>(5)) / den - pt.match_ypos;
    float err = dx*dx + dy*dy;
    if (err<limit) 
      numfit++;
    pt.match_error = sqrt(err);
  }
  for (int i=0;i<8;i++) 
    homography[i] = A.at<double>(i);
  homography[8] = 1.0f;
  return numfit;
}

void SiftFeatureBatchGpu::release() {
    if (d_ref_img_) { cudaFree(d_ref_img_); d_ref_img_ = nullptr; }
    if (d_tar_img_) { cudaFree(d_tar_img_); d_tar_img_ = nullptr; }
    match_kp_ref.clear();
    match_kp_tar.clear();
    num_match = 0;
    width_ = height_ = 0;
}

void SiftFeatureBatchGpu::prepare_cuda(const float *ref_img, const float *tar_img, int height, int width, cudaStream_t stream)
{
    release();
    width_ = width;
    height_ = height;
    size_t img_bytes = width_ * height_ * sizeof(float);

    cudaMalloc(&d_ref_img_, img_bytes);
    cudaMemcpyAsync(d_ref_img_, ref_img, img_bytes, cudaMemcpyHostToDevice, stream);

    cudaMalloc(&d_tar_img_, img_bytes);
    cudaMemcpyAsync(d_tar_img_, tar_img, img_bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
}

void SiftFeatureBatchGpu::compute_match_batch_cuda(cudaStream_t stream) {
    // --- 检测参考图特征点 ---
    SiftData ref_data;
    InitSiftData(ref_data, max_feat_, true, true);
    CudaImage ref_img;
    ref_img.Allocate(width_, height_, iAlignUp(width_, 128), false, d_ref_img_);
    InitCuda(0);
    ExtractSift(ref_data, ref_img, 5, 1.0, 2.0f, 0.0f, false, nullptr);
    cudaStreamSynchronize(stream);

    // --- 检测目标图特征点 ---
    SiftData tar_data;
    InitSiftData(tar_data, max_feat_, true, true);
    CudaImage tar_img;
    tar_img.Allocate(width_, height_, iAlignUp(width_, 128), false, d_tar_img_);
    ExtractSift(tar_data, tar_img, 5, 1.0, 2.0f, 0.0f, false, nullptr);
    cudaStreamSynchronize(stream);

    // --- CUDA SIFT批量匹配 ---
    MatchSiftData(ref_data, tar_data);
    float homography[9];
    int numMatches;
    FindHomography(ref_data, homography, &numMatches, 10000, 0.00f, 0.90f, 8.0);
    int numFit = ImproveHomography(ref_data, homography, 5, 0.00f, 0.90f, 8.0);
    std::cout << "Number of original features: " <<  ref_data.numPts << " " << tar_data.numPts << std::endl;
    std::cout << "Number of matching features: " << numFit << " " << numMatches << " " << 100.0f*numFit/std::min(ref_data.numPts, tar_data.numPts) << "% " << 1 << " " << 3.0 << std::endl;
    
    num_match = 0;
    match_kp_ref.clear();
    match_kp_tar.clear();
    for (int i = 0; i < ref_data.numPts; ++i) {
        const SiftPoint& pt = ref_data.h_data[i];
        if (pt.match >= 0 && pt.match_error < 8.0) { // thresh与ImproveHomography一致
            SiftFeature2D ref, tar;
            ref.x = pt.xpos;
            ref.y = pt.ypos;
            tar.x = pt.match_xpos;
            tar.y = pt.match_ypos;
            match_kp_ref.push_back(ref);
            match_kp_tar.push_back(tar);
        }
    }
    num_match = int(match_kp_ref.size());

    FreeSiftData(ref_data);
    FreeSiftData(tar_data);
}
} // namespace StudyCorr