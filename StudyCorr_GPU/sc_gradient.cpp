#include "sc_gradient.h"

namespace StudyCorr_GPU
{
    Gradient2D4::Gradient2D4(Image2D& image)
    {
        grad_img = &image;
    }

    Gradient2D4::~Gradient2D4() {}

    void Gradient2D4::getGradientX()
    {
        int height = grad_img->height;
        int width = grad_img->width;

        gradient_x = cv::Mat::zeros(height, width, CV_32F);

#pragma omp parallel for
        for (int r = 0; r < height; r++)
        {
            for (int c = 2; c < width - 2; c++)
            {
                float result = 0.0f;
                result -= grad_img->cv_mat.at<float>(r, c + 2) / 12.f;
                result += grad_img->cv_mat.at<float>(r, c + 1) * (2.f / 3.f);
                result -= grad_img->cv_mat.at<float>(r, c - 1) * (2.f / 3.f);
                result += grad_img->cv_mat.at<float>(r, c - 2) / 12.f;
                gradient_x.at<float>(r, c) = result;
            }
        }
    }

    void Gradient2D4::getGradientY()
    {
        int height = grad_img->cv_mat.rows;
        int width = grad_img->cv_mat.cols;

        gradient_y = cv::Mat::zeros(height, width, CV_32F);

#pragma omp parallel for
        for (int r = 2; r < height - 2; r++)
        {
            for (int c = 0; c < width; c++)
            {
                float result = 0.0f;
                result -= grad_img->cv_mat.at<float>(r + 2, c) / 12.f;
                result += grad_img->cv_mat.at<float>(r + 1, c) * (2.f / 3.f);
                result -= grad_img->cv_mat.at<float>(r - 1, c) * (2.f / 3.f);
                result += grad_img->cv_mat.at<float>(r - 2, c) / 12.f;
                gradient_y.at<float>(r, c) = result;
            }
        }
    }

    void Gradient2D4::getGradientXY()
    {
        int height = grad_img->cv_mat.rows;
        int width = grad_img->cv_mat.cols;

        gradient_xy = cv::Mat::zeros(height, width, CV_32F);

        if (gradient_x.rows != height || gradient_x.cols != width)
        {
            getGradientX();
        }

#pragma omp parallel for
        for (int r = 2; r < height - 2; r++)
        {
            for (int c = 0; c < width; c++)
            {
                float result = 0.0f;
                result -= gradient_x.at<float>(r + 2, c) / 12.f;
                result += gradient_x.at<float>(r + 1, c) * (2.f / 3.f);
                result -= gradient_x.at<float>(r - 1, c) * (2.f / 3.f);
                result += gradient_x.at<float>(r - 2, c) / 12.f;
                gradient_xy.at<float>(r, c) = result;
            }
        }
    }

    Gradient3D4::Gradient3D4(Image3D& image)
    {
        grad_img = &image;
    }

    Gradient3D4::~Gradient3D4() {}

    void Gradient3D4::getGradientX()
    {
        int dim_x = grad_img->dim_x;
        int dim_y = grad_img->dim_y;
        int dim_z = grad_img->dim_z;

        gradient_x = new3D(dim_z, dim_y, dim_x);

#pragma omp parallel for
        for (int i = 0; i < dim_z; i++)
        {
            for (int j = 0; j < dim_y; j++)
            {
                for (int k = 2; k < dim_x - 2; k++)
                {
                    float result = 0.0f;
                    result -= grad_img->vol_mat[i][j][k + 2] / 12.f;
                    result += grad_img->vol_mat[i][j][k + 1] * (2.f / 3.f);
                    result -= grad_img->vol_mat[i][j][k - 1] * (2.f / 3.f);
                    result += grad_img->vol_mat[i][j][k - 2] / 12.f;
                    gradient_x[i][j][k] = result;
                }
            }
        }
    }

    void Gradient3D4::getGradientY()
    {
        int dim_x = grad_img->dim_x;
        int dim_y = grad_img->dim_y;
        int dim_z = grad_img->dim_z;

        gradient_y = new3D(dim_z, dim_y, dim_x);

#pragma omp parallel for
        for (int k = 0; k < dim_x; k++)
        {
            for (int i = 0; i < dim_z; i++)
            {
                for (int j = 2; j < dim_y - 2; j++)
                {
                    float result = 0.0f;
                    result -= grad_img->vol_mat[i][j + 2][k] / 12.f;
                    result += grad_img->vol_mat[i][j + 1][k] * (2.f / 3.f);
                    result -= grad_img->vol_mat[i][j - 1][k] * (2.f / 3.f);
                    result += grad_img->vol_mat[i][j - 2][k] / 12.f;
                    gradient_y[i][j][k] = result;
                }
            }
        }
    }

    void Gradient3D4::getGradientZ()
    {
        int dim_x = grad_img->dim_x;
        int dim_y = grad_img->dim_y;
        int dim_z = grad_img->dim_z;

        gradient_z = new3D(dim_z, dim_y, dim_x);

#pragma omp parallel for
        for (int j = 0; j < dim_y; j++)
        {
            for (int k = 0; k < dim_x; k++)
            {
                for (int i = 2; i < dim_z - 2; i++)
                {
                    float result = 0.0f;
                    result -= grad_img->vol_mat[i + 2][j][k] / 12.f;
                    result += grad_img->vol_mat[i + 1][j][k] * (2.f / 3.f);
                    result -= grad_img->vol_mat[i - 1][j][k] * (2.f / 3.f);
                    result += grad_img->vol_mat[i - 2][j][k] / 12.f;
                    gradient_z[i][j][k] = result;
                }
            }
        }
    }

} //namespace opencorr