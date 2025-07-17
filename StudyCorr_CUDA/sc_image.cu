#include "sc_image.h"
#include <fstream>

namespace StudyCorr
{
    void CudaImage2D::load(const std::string& file_path_) 
    {
        file_path = file_path_;
        cv::Mat img = cv::imread(file_path_, cv::IMREAD_GRAYSCALE);
        if (!img.data) throw std::runtime_error("Failed to load file: " + file_path_);

        width = img.cols;
        height = img.rows;
        data = Array2D<float>(height, width);

        // 拷贝到CUDA数组
        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
                data(i, j) = static_cast<float>(img.at<uchar>(i, j));
    }

    void CudaImage3D::load_from_bin(const std::string& file_path_) 
    {
        file_path = file_path_;
        std::ifstream fin(file_path_, std::ios::binary);
        if (!fin) throw std::runtime_error("Failed to open BIN file: " + file_path_);

        int dims[3];
        fin.read(reinterpret_cast<char*>(dims), sizeof(int) * 3);
        dim_x = dims[0];
        dim_y = dims[1];
        dim_z = dims[2];

        data = Array3D<float>(dim_z, dim_y, dim_x);
        fin.read(reinterpret_cast<char*>(data.get()), sizeof(float) * dim_x * dim_y * dim_z);
        fin.close();
    }

    void CudaImage3D::load_from_tiff(const std::string& file_path_)
    {
        file_path = file_path_;
        std::vector<cv::Mat> slices;
        if (!cv::imreadmulti(file_path_, slices, cv::IMREAD_GRAYSCALE))
            throw std::runtime_error("Failed to load multi-page TIFF: " + file_path_);

        dim_z = static_cast<int>(slices.size());
        dim_y = slices[0].rows;
        dim_x = slices[0].cols;
        data = Array3D<float>(dim_z, dim_y, dim_x);

        for (int z = 0; z < dim_z; ++z)
            for (int y = 0; y < dim_y; ++y)
                for (int x = 0; x < dim_x; ++x)
                    data(z, y, x) = static_cast<float>(slices[z].at<uchar>(y, x));
    }
}