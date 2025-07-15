#pragma once

#include "sc_array.h"
#include "sc_image.h"
#include "sc_point.h"

namespace StudyCorr
{
	class CudaInterpolation2D
	{
	protected:
		CudaImage2D* interp_img = nullptr;

	public:
		virtual ~CudaInterpolation2D() = default;

		virtual void prepare() = 0;
		virtual float compute(Point2D& location) = 0;
	};

	class CudaInterpolation3D
	{
	protected:
		CudaImage3D* interp_img = nullptr;

	public:
		virtual ~CudaInterpolation3D() = default;

		virtual void prepare() = 0;
		virtual float compute(Point3D& location) = 0;
	};

}//namespace StudyCorr