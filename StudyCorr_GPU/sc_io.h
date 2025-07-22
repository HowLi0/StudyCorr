
#pragma once
#include "sc_poi.h"
#include "sc_array.h"
#include "sc_calibration.h"

namespace StudyCorr_GPU
{
    //output variable for DIC
    //u, v, w are displacement in x, y, z direction
    //e_xx, e_yy, e_zz are strain in x, y, z direction
    //zncc is the normalized cross-correlation value
    //deformation_increment is the convergence of deformation
    //iteration_step is the iteration step number
    //feature_nearby is the number of SIFT features nearby
	enum OutputVariable
	{
		u = 1, //displacement
		v = 2,
		w = 3,
		e_xx = 4, //strain
		e_yy = 5,
		e_zz = 6,
		e_xy = 7,
		e_yz = 8,
		e_zx = 9,
		zncc = 10, //r1t1
		zncc_r1r2 = 11,
		zncc_r1t2 = 12,
		deformation_increment = 13, //convergence
		iteration_step = 14, //stop
		feature_nearby = 15, //SIFT feature
		u_x = 16, //displacement gradient
		u_y = 17,
		u_z = 18,
		v_x = 19,
		v_y = 20,
		v_z = 21,
		w_x = 22,
		w_y = 23,
		w_z = 24,
	};

	//this module is made to input data from csv table and output data to csv data
	class IO2D
	{
	private:
		std::string file_path;
		std::string delimiter;
		int width, height;

	public:
		IO2D();
		~IO2D();

		OutputVariable out_var;

		std::string getPath() const;
		std::string getDelimiter() const;
		int getWidth() const;
		int getHeight() const;
		void setPath(std::string file_path);
		void setDelimiter(std::string delimiter);
		void setWidth(int width);
		void setHeight(int height);

		//load locations of POIs from csv table
		std::vector<Point2D> loadPoint2D(std::string file_path);

		//save locations of POIs to csv table
		void savePoint2D(std::vector<Point2D> point_queue, std::string file_path);

		//load camera calibration for stereovision
		//void loadCalibration(Calibration& calibration_cam1, Calibration& calibration_cam2, std::string file_path);

		//load deformation of POIs from csv table
		std::vector<POI2D> loadTable2D();

		//save results of DIC to csv table
		void saveTable2D(std::vector<POI2D>& poi_queue);
        void saveTableCuda2D(std::vector<CudaPOI2D>& poi_queue);
		void saveDeformationTable2D(std::vector<POI2D>& poi_queue);

		//variable could be referred to enum OutputVariable
		void saveMap2D(std::vector<POI2D>& poi_queue, OutputVariable variable);

		//load deformation of POIs from csv table
		std::vector<POI2DS> loadTable2DS();

		//save results of stereo DIC to csv table
		void saveTable2DS(std::vector<POI2DS>& poi_queue);

		//variable could be referred to enum OutputVariable
		void saveMap2DS(std::vector<POI2DS>& poi_queue, OutputVariable variable);
	};

	class IO3D
	{
	private:
		std::string file_path;
		std::string delimiter;
		int dim_x, dim_y, dim_z;

	public:
		IO3D();
		~IO3D();

		std::string getPath() const;
		std::string getDelimiter() const;
		void setPath(std::string file_path);
		void setDelimiter(std::string delimiter);

		int getDimX();
		int getDimY();
		int getDimZ();
		void setDimX(int dim_x);
		void setDimY(int dim_y);
		void setDimZ(int dim_z);

		//load locations of POIs from csv table
		std::vector<Point3D> loadPoint3D(std::string file_path);

		//save locations of POIs to csv table
		void savePoint3D(std::vector<Point3D> poi_queue, std::string file_path);

		//load deformation of POIs from saved date table
		std::vector<POI3D> loadTable3D();

		//save results of DVC to csv table
		void saveTable3D(std::vector<POI3D>& poi_queue);

		//variable could be referred to enum OutputVariable
		void saveMap3D(std::vector<POI3D>& poi_queue, OutputVariable variable);

		//save and load deformation of POIs into a binary matrix
		void saveMatrixBin(std::vector<POI3D>& poi_queue);
		std::vector<POI3D> loadMatrixBin();
	};

}//namespace StudyCorr_GPU
