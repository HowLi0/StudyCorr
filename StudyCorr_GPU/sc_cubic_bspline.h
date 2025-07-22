/*
 * This file is part of StudyCorr_GPU, based on OpenCorr functionality.
 * Compatible with both OpenCV and Qt point types.
 */

#pragma once

#ifndef  _CUBIC_BSPLINE_H_
#define  _CUBIC_BSPLINE_H_

#include "sc_interpolation.h"
#include <cuda_runtime.h>

namespace StudyCorr_GPU
{
	//the 2D part of module is the implementation of
	//Z. Pan et al, Theoretical and Applied Mechanics Letters (2016) 6(3): 126-130.
	//https://doi.org/10.1016/j.taml.2016.04.003

	class BicubicBspline : public Interpolation2D
	{
	public:
		BicubicBspline(Image2D& image);
		~BicubicBspline();

		void prepare();
		float compute(Point2D& location);

	private:

		float**** interp_coefficient = nullptr;

		const float CONTROL_MATRIX[4][4] =
		{
			{ 71.0f / 56.0f, -19.0f / 56.0f, 5 / 56.0f, -1.0f / 56.0f },
			{ -19.0f / 56.0f, 95.0f / 56.0f, -25 / 56.0f, 5.0f / 56.0f },
			{ 5.0f / 56.0f, -25.0f / 56.0f, 95 / 56.0f, -19.0f / 56.0f },
			{ -1.0f / 56.0f, 5.0f / 56.0f, -19 / 56.0f, 71.0f / 56.0f }
		};

		const float FUNCTION_MATRIX[4][4] =
		{
			{ -1.0f / 6.0f, 3.0f / 6.0f, -3.0f / 6.0f, 1.0f / 6.0f },
			{ 3.0f / 6.0f, -6.0f / 6.0f, 3.0f / 6.0f, 0.0f },
			{ -3.0f / 6.0f, 0.0f, 3.0f / 6.0f, 0.0f },
			{ 1.0f / 6.0f, 4.0f / 6.0f, 1.0f / 6.0f, 0.0f }
		};
	};

	//the 3D part of module is the implementation of
	//J. Yang et al, Optics and Lasers in Engineering (2021) 136: 106323.
	//https://doi.org/10.1016/j.optlaseng.2020.106323

	class TricubicBspline : public Interpolation3D
	{
	public:
		TricubicBspline(Image3D& image);
		~TricubicBspline();

		void prepare();
		float compute(Point3D& location);

	private:
		float*** interp_coefficient = nullptr;

		//B-spline prefilter
		const float BSPLINE_PREFILTER[8] =
		{
			1.732176555412860f,  //b0
			-0.464135309171000f, //b1
			0.124364681271139f,  //b2
			-0.033323415913556f, //b3
			0.008928982383084f,  //b4
			-0.002392513618779f, //b5
			0.000641072092032f,  //b6
			-0.000171774749350f, //b7
		};

	};

	//Four cubic B-spline basis functions when input falls in different range
	inline float basis0(float coor_decimal)
	{
		return (1.f / 6.f) * (coor_decimal * (coor_decimal * (-coor_decimal + 3.f) - 3.f) + 1.f); //(1/6)*(2-(x+1))^3 for x-(-1)
	}

	inline float basis1(float coor_decimal)
	{
		return (1.f / 6.f) * (coor_decimal * coor_decimal * (3.f * coor_decimal - 6.f) + 4.f); //(2/3)-(1/2)*(2-x)*x^2 for x-0
	}

	inline float basis2(float coor_decimal)
	{
		return (1.f / 6.f) * (coor_decimal * (coor_decimal * (-3.f * coor_decimal + 3.f) + 3.f) + 1.f); //(2/3)-(1/2)*(2-(1-x))*(1-x)^2 for x-1
	}

	inline float basis3(float coor_decimal)
	{
		return (1.f / 6.f) * (coor_decimal * coor_decimal * coor_decimal); //(1/6)*(2-(2-x))^3 for x-2
	}

	//return the lower value of the two inputs
	int getLow(int x, int y);

	//retrun the higher value of the two inputs
	int getHigh(int x, int y);



	//*******************************************************************CUDA******************************************************************/
	/*
				// 用户侧示例
			std::vector<cudaStream_t> streams(num_batches);
			for (int i = 0; i < num_batches; ++i)
				cudaStreamCreate(&streams[i]);

			for (int i = 0; i < num_batches; ++i)
				bspline.compute_batch_cuda(x_array[i], y_array[i], N_array[i], result_array[i], streams[i]);

			for (int i = 0; i < num_batches; ++i)
				cudaStreamSynchronize(streams[i]);
	*/
	class BicubicBsplineCuda {
	public:
		void prepare_cuda(const float* image, int height, int width);
		void compute_batch_cuda(const float* loc_x, const float* loc_y, int N, float* results, cudaStream_t stream = 0);
		void release_cuda();

	private:
		float* d_image = nullptr;
		float* d_coeffs = nullptr;
		int height, width;
	};

}//namespace StudyCorr_GPU

#endif //_CUBIC_BSPLINE_H_

