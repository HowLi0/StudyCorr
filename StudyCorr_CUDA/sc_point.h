
#pragma once
#include <iostream>
#include"sc_array.h"

#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif

namespace StudyCorr
{
	//Point in 2D plane
	class Point2D
	{
	public:
		float x, y;

		CUDA_HD Point2D() : x(0), y(0) {}
		CUDA_HD Point2D(float x, float y) : x(x), y(y) {}
		CUDA_HD Point2D(int x, int y) : x((float)x), y((float)y) {}
		CUDA_HD ~Point2D() {}

		CUDA_HD float vectorNorm() const { return sqrtf(x * x + y * y); }

		friend std::ostream& operator<<(std::ostream& output, const Point2D& point);
	};

	//reload basic operators
	CUDA_HD inline Point2D operator+(Point2D point, Point2D offset)
	{
		return Point2D(point.x + offset.x, point.y + offset.y);
	}

	CUDA_HD inline Point2D operator-(Point2D point, Point2D offset)
	{
		return point + Point2D(-offset.x, -offset.y);
	}

	CUDA_HD inline Point2D operator*(float factor, Point2D point)
	{
		return Point2D(factor * point.x, factor * point.y);
	}

	CUDA_HD inline Point2D operator*(int factor, Point2D point)
	{
		return float(factor) * point;
	}

	CUDA_HD inline Point2D operator*(Point2D point, float factor)
	{
		return factor * point;
	}

	CUDA_HD inline Point2D operator*(Point2D point, int factor)
	{
		return float(factor) * point;
	}

	//dot product
	CUDA_HD inline float operator*(Point2D point1, Point2D point2)
	{
		return (point1.x * point2.x + point1.y * point2.y);
	}

	CUDA_HD inline Point2D operator/(Point2D point, float factor)
	{
		return Point2D(point.x / factor, point.y / factor);
	}

	CUDA_HD inline Point2D operator/(Point2D point, int factor)
	{
		return point / float(factor);
	}

	//cross product
	CUDA_HD inline float operator/(Point2D point1, Point2D point2)
	{
		return (point1.x * point2.y - point1.y * point2.x);
	}

	// Point in 3D space
	class Point3D
	{
	public:
		float x, y, z;

		CUDA_HD Point3D() : x(0), y(0), z(0) {}
		CUDA_HD Point3D(float x, float y, float z) : x(x), y(y), z(z) {}
		CUDA_HD Point3D(int x, int y, int z) : x((float)x), y((float)y), z((float)z) {}
		CUDA_HD ~Point3D() {}

		CUDA_HD float vectorNorm() const { return sqrtf(x * x + y * y + z * z); }

		friend std::ostream& operator<<(std::ostream& output, const Point3D& point);
	};

	//reload basic operators
	CUDA_HD inline Point3D operator+(Point3D point, Point3D offset)
	{
		return Point3D(point.x + offset.x, point.y + offset.y, point.z + offset.z);
	}

	CUDA_HD inline Point3D operator-(Point3D point, Point3D offset)
	{
		return point + Point3D(-offset.x, -offset.y, -offset.z);
	}

	CUDA_HD inline Point3D operator*(float factor, Point3D point)
	{
		return Point3D(factor * point.x, factor * point.y, factor * point.z);
	}

	CUDA_HD inline Point3D operator*(int factor, Point3D point)
	{
		return float(factor) * point;
	}

	CUDA_HD inline Point3D operator*(Point3D point, float factor)
	{
		return factor * point;
	}

	CUDA_HD inline Point3D operator*(Point3D point, int factor)
	{
		return float(factor) * point;
	}

	//dot product
	CUDA_HD inline float operator*(Point3D point1, Point3D point2)
	{
		return (point1.x * point2.x + point1.y * point2.y + point1.z * point2.z);
	}

	CUDA_HD inline Point3D operator/(Point3D point, float factor)
	{
		return Point3D(point.x / factor, point.y / factor, point.z / factor);
	}

	CUDA_HD inline Point3D operator/(Point3D point, int factor)
	{
		return point / float(factor);
	}

	//cross product
	CUDA_HD inline Point3D operator/(Point3D point1, Point3D point2)
	{
		return Point3D((point1.y * point2.z - point1.z * point2.y),
			(point1.z * point2.x - point1.x * point2.z),
			(point1.x * point2.y - point1.y * point2.x));
	}

}//namespace StudyCorr