#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include<QStringList>
#include<QDebug>


class Calibration
{
public:
    Calibration() = default;
    ~Calibration()= default;
    virtual bool prefareMonocularCalibration() = 0; // 准备单目标定
    virtual void startMonocularCompute() = 0; // 开始单目标定
    virtual bool prefareStereoCalibration() = 0; // 准备双目标定
    virtual void startStereoCalibration() = 0; // 开始双目标定
    virtual cv::Mat computeProjMatrix(const cv::Mat R, const cv::Mat T, const cv::Mat cameraMatrix) const = 0; // 重投影矩阵
    virtual void drawCornersAndAxes(cv::Mat& img, const std::vector<cv::Point2f>& corners, cv::Size boardSize, bool found) = 0; // 绘制角点和坐标轴
};


class ChessCalibration: public Calibration
{
public:
    ChessCalibration(int rows,int cols,int squareSize, QStringList chessPath);
    ChessCalibration(int rows, int cols, int squareSize, QStringList chessPathLeft, QStringList chessPathRight);
    ~ChessCalibration();
	bool prefareMonocularCalibration() override;//准备单目标定
	void startMonocularCompute() override;//开始单目标定
	bool prefareStereoCalibration() override;//准备双目标定
	void startStereoCalibration() override;//开始双目标定
    cv::Mat computeProjMatrix(const cv::Mat R, const cv::Mat T, const cv::Mat cameraMatrix) const override; // 重投影矩阵
    void drawCornersAndAxes(cv::Mat& img, const std::vector<cv::Point2f>& corners, cv::Size boardSize, bool found) override;
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

class CircleCalibration: public Calibration
{
public:
    CircleCalibration(int rows, int cols, int squareSize, QStringList circlePath);
    CircleCalibration(int rows, int cols, int squareSize, QStringList circlePathLeft, QStringList circlePathRight);
    ~CircleCalibration();
    bool prefareMonocularCalibration() override;//准备单目标定
    void startMonocularCompute() override;//开始单目标定
    bool prefareStereoCalibration() override;//准备双目标定
    void startStereoCalibration() override;//开始双目标定
    cv::Mat computeProjMatrix(const cv::Mat R, const cv::Mat T, const cv::Mat cameraMatrix) const override;//重投影矩阵
    void drawCornersAndAxes(cv::Mat& img, const std::vector<cv::Point2f>& corners, cv::Size boardSize, bool found) override; // 绘制角点和坐标轴
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
