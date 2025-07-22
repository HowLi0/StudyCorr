#include "sc_array.h"

namespace StudyCorr_GPU
{
float** new2D(int dimension1, int dimension2)
	{
		float** ptr = nullptr;
		createPtr(ptr, dimension1, dimension2);
		return ptr;
	}

	void delete2D(float**& ptr)
	{
		if (ptr == nullptr) return;
		destroyPtr(ptr);
	}

	float*** new3D(int dimension1, int dimension2, int dimension3)
	{
		float*** ptr = nullptr;
		createPtr(ptr, dimension1, dimension2, dimension3);
		return ptr;
	}

	void delete3D(float***& ptr)
	{
		if (ptr == nullptr) return;
		destroyPtr(ptr);
	}

	float**** new4D(int dimension1, int dimension2, int dimension3, int dimension4)
	{
		float**** ptr = nullptr;
		createPtr(ptr, dimension1, dimension2, dimension3, dimension4);
		return ptr;
	}

	void delete4D(float****& ptr)
	{
		if (ptr == nullptr) return;
		destroyPtr(ptr);
	}
}