#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include<QStringList>
#include<QDebug>

enum class CalibrationModel
{
    Chessboard,
    Circle
};


class ChessCalibration
{
public:
    ChessCalibration(int rows,int cols,int squareSize, QStringList chessPath);
    ChessCalibration(int rows, int cols, int squareSize, QStringList chessPathLeft, QStringList chessPathRight);
    ~ChessCalibration();
	bool prefareMonocularCalibration();//准备单目标定
	void startMonocularCompute();//开始单目标定
    cv::Mat computeProjMatrix(const cv::Mat R, const cv::Mat T, const cv::Mat cameraMatrix) const;//重投影矩阵
	bool prefareStereoCalibration();//准备双目标定
	void startStereoCalibration();//开始双目标定
    void drawCornersAndAxes(cv::Mat& img, const std::vector<cv::Point2f>& corners, cv::Size boardSize, bool found);
    std::vector<cv::Mat> img1_frames;//用于储存drawCornersAndAxes绘制的图片
    std::vector<cv::Mat> img2_frames;
    cv::Mat cameraMatrix1, cameraMatrix2;
    cv::Mat distCoeffs1, distCoeffs2;
    std::vector<cv::Mat> rvecs1, tvecs1, rvecs2, tvecs2;
    cv::Mat R, T, E, F;
private:
    cv::Size boardSize;
    float squareSize;
    std::vector<cv::Point3f> objectPoints;
    std::vector<std::vector<cv::Point3f>> objectPointsVec;
    std::vector<std::vector<cv::Point2f>> imagePointsVec1, imagePointsVec2;
    std::vector< cv::Mat> projMatrixVec1, projMatrixVec2;
    std::vector< std::string> chessPath, chessPathLeft, chessPathRight;
};

class CircleCalibration
{
public:
    CircleCalibration(int rows, int cols, int squareSize, QStringList circlePath);
    CircleCalibration(int rows, int cols, int squareSize, QStringList circlePathLeft, QStringList circlePathRight);
    ~CircleCalibration();
    bool prefareMonocularCalibration();//准备单目标定
    void startMonocularCompute();//开始单目标定
    bool prefareStereoCalibration();//准备双目标定
    void startStereoCalibration();//开始双目标定
    cv::Mat computeProjMatrix(const cv::Mat R, const cv::Mat T, const cv::Mat cameraMatrix) const;//重投影矩阵
    void drawCornersAndAxes(cv::Mat& img, const std::vector<cv::Point2f>& corners, cv::Size boardSize, bool found);
    std::vector<cv::Mat> img1_frames;//用于储存drawCornersAndAxes绘制的图片
    std::vector<cv::Mat> img2_frames;
    cv::Mat cameraMatrix1;
    cv::Mat cameraMatrix2;
    cv::Mat distCoeffs1;
    cv::Mat distCoeffs2;
    std::vector<cv::Mat> rvecs1, tvecs1;
    std::vector<cv::Mat> rvecs2, tvecs2;
    cv::Mat R, T, E, F;
private:
    cv::Size boardSize;
    float squareSize;
    std::vector<cv::Point3f> objectPoints;
    std::vector<std::vector<cv::Point3f>> objectPointsVec;
    std::vector<std::vector<cv::Point2f>> imagePointsVec1, imagePointsVec2;
    std::vector<cv::Mat> projMatrixVec1, projMatrixVec2;
    std::vector<std::string> circlePath,circlePathLeft,circlePathRight;
};
