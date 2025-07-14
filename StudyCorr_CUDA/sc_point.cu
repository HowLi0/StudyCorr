#include "sc_point.h"

namespace StudyCorr
{
    std::ostream& operator<<(std::ostream& output, const Point2D& point)
	{
		output << point.x << "," << point.y;
		return output;
	}

    std::ostream& operator<<(std::ostream& output, const Point3D& point)
	{
		output << point.x << "," << point.y << "," << point.z;
		return output;
	}
}