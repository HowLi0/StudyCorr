/*
 * This file is part of StudyCorr_GPU, based on OpenCorr functionality.
 * Compatible with both OpenCV and Qt point types.
 */
#pragma once

#ifndef _INTERPOLATION_H_
#define _INTERPOLATION_H_

#include "sc_array.h"
#include "sc_image.h"
#include "sc_point.h"

namespace StudyCorr_GPU
{
	class Interpolation2D
	{
	protected:
		Image2D* interp_img = nullptr;

	public:
		virtual ~Interpolation2D() = default;

		virtual void prepare() = 0;
		virtual float compute(Point2D& location) = 0;
	};

	class Interpolation3D
	{
	protected:
		Image3D* interp_img = nullptr;

	public:
		virtual ~Interpolation3D() = default;

		virtual void prepare() = 0;
		virtual float compute(Point3D& location) = 0;
	};

}//namespace StudyCorr_GPU

#endif //_INTERPOLATION_H_

