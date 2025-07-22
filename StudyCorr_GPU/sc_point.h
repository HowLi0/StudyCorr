/*
 * This file is part of StudyCorr, based on OpenCorr functionality.
 * Compatible with both OpenCV and Qt point types.
 */

#pragma once
#include "cuda_runtime.h"

#include <iostream>

namespace StudyCorr_GPU
{
	//Point in 2D plane
	class Point2D
	{
	public:
		float x, y;

		__host__ __device__ Point2D():	x(0.f), y(0.f) {};
		__host__ __device__ Point2D(float x, float y):x(x), y(y) {};
		__host__ __device__ Point2D(int x, int y):x((float)x), y((float)y) {};
		__host__ __device__ ~Point2D(){};

		__host__ __device__ float vectorNorm() const
		{
			return sqrt(x * x + y * y);
		};

		friend std::ostream& operator<<(std::ostream& output, const Point2D& point)
		{
			output << point.x << "," << point.y;
			return output;
		};
	};

	//reload basic operators
	inline Point2D operator+(Point2D point, Point2D offset)
	{
		return Point2D(point.x + offset.x, point.y + offset.y);
	}

	inline Point2D operator-(Point2D point, Point2D offset)
	{
		return point + Point2D(-offset.x, -offset.y);
	}

	inline Point2D operator*(float factor, Point2D point)
	{
		return Point2D(factor * point.x, factor * point.y);
	}

	inline Point2D operator*(int factor, Point2D point)
	{
		return float(factor) * point;
	}

	inline Point2D operator*(Point2D point, float factor)
	{
		return factor * point;
	}

	inline Point2D operator*(Point2D point, int factor)
	{
		return float(factor) * point;
	}

	//dot product
	inline float operator*(Point2D point1, Point2D point2)
	{
		return (point1.x * point2.x + point1.y * point2.y);
	}

	inline Point2D operator/(Point2D point, float factor)
	{
		return Point2D(point.x / factor, point.y / factor);
	}

	inline Point2D operator/(Point2D point, int factor)
	{
		return point / float(factor);
	}

	//cross product
	inline float operator/(Point2D point1, Point2D point2)
	{
		return (point1.x * point2.y - point1.y * point2.x);
	}

	// Point in 3D space
	class Point3D
	{
	public:
		float x, y, z;

		__host__ __device__ Point3D(): x(0.f), y(0.f), z(0.f) {};
		__host__ __device__ Point3D(float x, float y, float z):x(x), y(y), z(z) {};
		__host__ __device__ Point3D(int x, int y, int z):x((float)x), y((float)y), z((float)z) {};
		__host__ __device__ ~Point3D(){};

		__host__ __device__ float vectorNorm() const
		{
			return sqrt(x * x + y * y + z * z);
		};

		friend std::ostream& operator<<(std::ostream& output, const Point3D& point)
		{
			output << point.x << "," << point.y << "," << point.z;
			return output;
		};
	};

	//reload basic operators
	inline Point3D operator+(Point3D point, Point3D offset)
	{
		return Point3D(point.x + offset.x, point.y + offset.y, point.z + offset.z);
	}

	inline Point3D operator-(Point3D point, Point3D offset)
	{
		return point + Point3D(-offset.x, -offset.y, -offset.z);
	}

	inline Point3D operator*(float factor, Point3D point)
	{
		return Point3D(factor * point.x, factor * point.y, factor * point.z);
	}

	inline Point3D operator*(int factor, Point3D point)
	{
		return float(factor) * point;
	}

	inline Point3D operator*(Point3D point, float factor)
	{
		return factor * point;
	}

	inline Point3D operator*(Point3D point, int factor)
	{
		return float(factor) * point;
	}

	//dot product
	inline float operator*(Point3D point1, Point3D point2)
	{
		return (point1.x * point2.x + point1.y * point2.y + point1.z * point2.z);
	}

	inline Point3D operator/(Point3D point, float factor)
	{
		return Point3D(point.x / factor, point.y / factor, point.z / factor);
	}

	inline Point3D operator/(Point3D point, int factor)
	{
		return point / float(factor);
	}

	//cross product
	inline Point3D operator/(Point3D point1, Point3D point2)
	{
		return Point3D((point1.y * point2.z - point1.z * point2.y),
			(point1.z * point2.x - point1.x * point2.z),
			(point1.x * point2.y - point1.y * point2.x));
	}
}//namespace StudyCorr