/*
 * This file is part of StudyCorr, based on OpenCorr functionality.
 * Compatible with both OpenCV and Qt point types.
 */

#include "sc_subset.h"

namespace StudyCorr_GPU
{
    // 2D subset
    Subset2D::Subset2D(Point2D center, int radius_x, int radius_y)
    {
        if (radius_x < 1 || radius_y < 1)
        {
            std::cerr << "Too small radius:" << radius_x << ", " << radius_y << std::endl;
        }

        this->center = center;
        this->radius_x = radius_x;
        this->radius_y = radius_y;
        width = radius_x * 2 + 1;   // 计算子区域宽度
        height = radius_y * 2 + 1;  // 计算子区域高度
        size = height * width;      // 计算子区域总像素数

        // 初始化子区域矩阵
		subset_mat = cv::Mat::zeros(height, width, CV_32FC1);
    }

    void Subset2D::fill(Image2D* image)
    {
        // 计算子区域左上角点
		Point2D topleft_point(center.x - radius_x, center.y - radius_y);  

        // 从 image 对象的 cv_mat 矩阵中提取子集的数据，填充到 Subset2D 对象的 subset_mat 矩阵中
        subset_mat = image->cv_mat(cv::Rect(static_cast<int>(topleft_point.y), static_cast<int>(topleft_point.x), height, width)).clone();
    }

    float Subset2D::zeroMeanNorm()  
    {  
       // 计算子区域的均值  
       cv::Scalar subset_mean = cv::mean(subset_mat);  

       // 将所有元素减去均值（零均值化）  
       subset_mat = subset_mat - subset_mean[0];  

       // 计算零均值后的模长（L2范数）  
       return static_cast<float>(cv::norm(subset_mat));  
    }

    // 3D subset
    Subset3D::Subset3D(Point3D center, int radius_x, int radius_y, int radius_z)
    {
        if (radius_x < 1 || radius_y < 1 || radius_z < 1)
        {
            std::cerr << "Too small radius:" << radius_x << ", " << radius_y << ", " << radius_z << std::endl;
        }

        this->center = center;
        this->radius_x = radius_x;
        this->radius_y = radius_y;
        this->radius_z = radius_z;
        dim_x = radius_x * 2 + 1;  // 计算子体积X尺寸
        dim_y = radius_y * 2 + 1;  // 计算子体积Y尺寸
        dim_z = radius_z * 2 + 1;  // 计算子体积Z尺寸
        size = dim_z * dim_y * dim_x;  // 计算子体积总体素数

        // 初始化3D体积
        vol_mat = new3D(dim_z, dim_y, dim_x);
    }

    Subset3D::~Subset3D()
    {
		delete3D(vol_mat);
    }

    void Subset3D::fill(Image3D* image)
    {
        // 计算子体积开始点
        Point3D start_point(center.x - radius_x, center.y - radius_y, center.z - radius_z);

        for (int i = 0; i < dim_z; i++)
        {
            for (int j = 0; j < dim_y; j++)
            {
                for (int k = 0; k < dim_x; k++)
                {
                    vol_mat[i][j][k] = image->vol_mat[int(start_point.z + i)][int(start_point.y + j)][int(start_point.x + k)];
                }
            }
        }
    }

    float Subset3D::zeroMeanNorm()
    {
        //calculate the mean of gray-scale values
        float mean_value = 0;
        for (int i = 0; i < dim_z; i++)
        {
            for (int j = 0; j < dim_y; j++)
            {
                for (int k = 0; k < dim_x; k++)
                {
                    mean_value += vol_mat[i][j][k];
                }
            }
        }
        mean_value /= (dim_x * dim_y * dim_z);

        //make the distribution of gray-scale values zero-mean
        float subset_sum = 0;
        for (int i = 0; i < dim_z; i++)
        {
            for (int j = 0; j < dim_y; j++)
            {
                for (int k = 0; k < dim_x; k++)
                {
                    vol_mat[i][j][k] -= mean_value;
                    subset_sum += (vol_mat[i][j][k] * vol_mat[i][j][k]);
                }
            }
        }

        return sqrt(subset_sum);
    }
}//namespace StudyCorr
