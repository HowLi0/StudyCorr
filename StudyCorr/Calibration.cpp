#include "Calibration.h"

    ChessCalibration::ChessCalibration(int rows, int cols, int squareSize, QStringList chessPath)
    {
        this->boardSize = cv::Size(rows,cols);
        this->squareSize = squareSize;
        // 将 QStringList 转换为 std::vector<std::string>
        for (const QString& path : chessPath) {
            this->chessPath.push_back(path.toStdString());
        }
    }

    ChessCalibration::ChessCalibration(int rows, int cols, int squareSize, QStringList chessPathLeft, QStringList chessPathRight)
    {
        this->boardSize = cv::Size(rows, cols);
        this->squareSize = squareSize;
        // 将 QStringList 转换为 std::vector<std::string>
        for (const QString& path : chessPathLeft) {
            this->chessPathLeft.push_back(path.toUtf8().constData());
           // std::cout << path.toUtf8().constData() << std::endl;
        }
        for (const QString& path : chessPathRight) {
            this->chessPathRight.push_back(path.toUtf8().constData());
        }
    }

    ChessCalibration::~ChessCalibration()
    {
    }

    bool ChessCalibration::prefareMonocularCalibration()
    {
        if (chessPath.size()==0)
        {
            std::cerr << "There is nothing for calibration." << std::endl;
            return false;
        }

        for (int i = 0; i < boardSize.height; ++i)
        {
            for (int j = 0; j < boardSize.width; ++j)
            { 
                this->objectPoints.emplace_back(j * squareSize, i * squareSize, 0.00f);
            }
        }
        this->objectPointsVec = std::vector<std::vector<cv::Point3f>> (chessPath.size(), objectPoints);
        for (int i = 0; i < chessPath.size(); ++i)
        {
            cv::Mat img1 = cv::imread(chessPath[i]);
            cv::Mat gray1;
            cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point2f> corners1;
            bool found = cv::findChessboardCorners(gray1, boardSize, corners1);

            if (found)
            {
                cv::cornerSubPix(gray1, corners1, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
                this->imagePointsVec1.push_back(corners1);
                drawCornersAndAxes(img1, corners1, boardSize, found);
                this->img1_frames.push_back(img1);
            }
            else
            {
                if (!found)
                {
                    std::cerr << "Chessboard corners not found in image: " << chessPath[i] << std::endl;
                }
            }
        }
        return true;
    }

    void ChessCalibration::startMonocularCompute()
    {
        if (imagePointsVec1.empty())
        {
            std::cerr << "No valid chessboard corners were found in the provided images." << std::endl;
            exit(1);
        }

        this->cameraMatrix1 = cv::Mat::eye(3, 3, CV_64F);

        // Camera calibration for both cameras
        double rms = cv::calibrateCamera(objectPointsVec, imagePointsVec1, cv::Size(img1_frames[0].cols, img1_frames[0].rows), cameraMatrix1, distCoeffs1, rvecs1, tvecs1);
        std::cout << "RMS error of camera： " << rms << std::endl;

        // 输出相机内参、畸变系数、外参
        std::cout << "Camera Matrix：" << std::endl << cameraMatrix1 << std::endl;
        std::cout << "Distortion Coefficients：" << std::endl << distCoeffs1 << std::endl;
        for (int i = 0; i < chessPath.size(); i++)
        {
            // 将旋转向量转换为旋转矩阵
            cv::Mat R ,T;
            cv::Rodrigues(rvecs1[i], R);
            T = tvecs1[i];
            std::cout << "Rotation Matrix: " << std::endl << R << std::endl;
            std::cout << "Translation Vector: " << std::endl << tvecs1 [i]<< std::endl;
            cv::Mat projMatrix=computeProjMatrix(R, T, cameraMatrix1);
            this->projMatrixVec1.push_back(projMatrix);
        }
        std::cout << "compute is over"<< std::endl;
    }

    cv::Mat ChessCalibration::computeProjMatrix(const cv::Mat R, const cv::Mat T, const cv::Mat cameraMatrix) const
    {
        // Construct projection matrix for Camera : P = K * [R | T]
        cv::Mat projMatrix = cv::Mat::zeros(3, 4, CV_64F);
        cv::Mat RT;
        cv::hconcat(R, T, RT);  // Combine R and T into a 3x4 matrix
        projMatrix = cameraMatrix * RT;
        return projMatrix;
    }


    bool ChessCalibration::prefareStereoCalibration()
    {
        if (chessPathLeft.size() != chessPathRight.size()) {
            std::cerr << "The number of images from Camera1 and Camera2 should be the same." << std::endl;
            return false;
        }

        for (int i = 0; i < boardSize.height; ++i) {
            for (int j = 0; j < boardSize.width; ++j) {
                this->objectPoints.emplace_back(j * squareSize, i * squareSize, 0.00f);
            }
        }

        this->objectPointsVec = std::vector<std::vector<cv::Point3f>>(chessPathLeft.size(), objectPoints);

        for (int i = 0; i < chessPathLeft.size(); ++i) {
            cv::Mat img1 = cv::imread(chessPathLeft[i]);
            cv::Mat img2 = cv::imread(chessPathRight[i]);

            if (img1.empty() || img2.empty()) {
                std::cerr << "Failed to load images: " << chessPathLeft[i] << ", " << chessPathRight[i] << std::endl;
                return false;
            }

            cv::Mat gray1, gray2;
            cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point2f> corners1, corners2;
            bool found1 = cv::findChessboardCorners(gray1, boardSize, corners1,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            bool found2 = cv::findChessboardCorners(gray2, boardSize, corners2,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

            if (found1 && found2) {
                cv::cornerSubPix(gray1, corners1, cv::Size(11, 11), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
                cv::cornerSubPix(gray2, corners2, cv::Size(11, 11), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));

                this->imagePointsVec1.push_back(corners1);
                this->imagePointsVec2.push_back(corners2);

                drawCornersAndAxes(img1, corners1, boardSize, found1);
                drawCornersAndAxes(img2, corners2, boardSize, found2);

                this->img1_frames.push_back(img1);
                this->img2_frames.push_back(img2);
            }
        }
        return true;
    }

    void ChessCalibration::startStereoCalibration()
    {
        if (imagePointsVec1.empty() || imagePointsVec2.empty()) {
            std::cerr << "No valid chessboard corners were found in the provided images." << std::endl;
            return;
        }

        // 1. 预筛选：对每一帧单独进行单目标定，剔除重投影误差大的帧
        std::vector<int> validIndices;
        const double MAX_REPROJ_ERROR = 0.1; // 重投影误差阈值
        
        // 存储有效帧的角点数据
        std::vector<std::vector<cv::Point3f>> validObjectPoints;
        std::vector<std::vector<cv::Point2f>> validImagePoints1;
        std::vector<std::vector<cv::Point2f>> validImagePoints2;
        
        std::cout << "Performing pre-screening of frames..." << std::endl;
        for (int i = 0; i < imagePointsVec1.size(); ++i) {
            // 当前帧的角点数据
            std::vector<std::vector<cv::Point3f>> objPts = { objectPointsVec[i] };
            std::vector<std::vector<cv::Point2f>> imgPts1 = { imagePointsVec1[i] };
            std::vector<std::vector<cv::Point2f>> imgPts2 = { imagePointsVec2[i] };
            
            // 对左相机进行单帧标定
            cv::Mat cameraMatrix1_temp = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat distCoeffs1_temp;
            std::vector<cv::Mat> rvecs1, tvecs1;
            double rms1 = cv::calibrateCamera(objPts, imgPts1, img1_frames[i].size(), 
                                            cameraMatrix1_temp, distCoeffs1_temp, 
                                            rvecs1, tvecs1);
            
            // 对右相机进行单帧标定
            cv::Mat cameraMatrix2_temp = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat distCoeffs2_temp;
            std::vector<cv::Mat> rvecs2, tvecs2;
            double rms2 = cv::calibrateCamera(objPts, imgPts2, img2_frames[i].size(), 
                                            cameraMatrix2_temp, distCoeffs2_temp, 
                                            rvecs2, tvecs2);
            
            // 检查重投影误差
            if (rms1 <= MAX_REPROJ_ERROR && rms2 <= MAX_REPROJ_ERROR) {
                validIndices.push_back(i);
                validObjectPoints.push_back(objectPointsVec[i]);
                validImagePoints1.push_back(imagePointsVec1[i]);
                validImagePoints2.push_back(imagePointsVec2[i]);
                
                std::cout << "Frame " << i << " passed screening: "
                        << "L-RMS = " << rms1 << ", R-RMS = " << rms2 << std::endl;
            } else {
                std::cout << "Frame " << i << " rejected: "
                        << "L-RMS = " << rms1 << ", R-RMS = " << rms2 << std::endl;
            }
        }
        
        if (validObjectPoints.empty()) {
            std::cerr << "No valid frames after screening. Try increasing MAX_REPROJ_ERROR." << std::endl;
            return;
        }
        std::cout << "Selected " << validObjectPoints.size() << " valid frames." << std::endl;

        // 2. 用所有有效帧重新进行单目标定，得到最终内参
        cv::Mat cameraMatrix1_final = cv::Mat::eye(3, 3, CV_64F), distCoeffs1_final;
        cv::Mat cameraMatrix2_final = cv::Mat::eye(3, 3, CV_64F), distCoeffs2_final;
        std::vector<cv::Mat> rvecs1_final, tvecs1_final, rvecs2_final, tvecs2_final;
        
        double rms1_final = cv::calibrateCamera(validObjectPoints, validImagePoints1, 
                                            img1_frames[0].size(), cameraMatrix1_final, 
                                            distCoeffs1_final, rvecs1_final, tvecs1_final);
        
        double rms2_final = cv::calibrateCamera(validObjectPoints, validImagePoints2, 
                                            img2_frames[0].size(), cameraMatrix2_final, 
                                            distCoeffs2_final, rvecs2_final, tvecs2_final);
        
        std::cout << "\nFinal single-camera calibration:" << std::endl;
        std::cout << "Left camera RMS: " << rms1_final << std::endl;
        std::cout << "Right camera RMS: " << rms2_final << std::endl;

        // 3. 在有效帧中寻找最优帧（左右误差和最小且平衡）
        int bestIdx = -1;
        double minErrorSum = std::numeric_limits<double>::max();
        double minErrorDiff = std::numeric_limits<double>::max();
        double best_rms1 = 0, best_rms2 = 0;
        
        auto computeReprojectionError = [](const std::vector<cv::Point3f>& objectPoints,
                                        const std::vector<cv::Point2f>& imagePoints,
                                        const cv::Mat& cameraMatrix,
                                        const cv::Mat& distCoeffs) -> double {
            cv::Mat rvec, tvec;
            if (!cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec)) {
                return std::numeric_limits<double>::max();
            }
            
            std::vector<cv::Point2f> projectedPoints;
            cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
            
            double totalError = 0.0;
            for (size_t i = 0; i < imagePoints.size(); ++i) {
                double err = cv::norm(imagePoints[i] - projectedPoints[i]);
                totalError += err * err;
            }
            return std::sqrt(totalError / imagePoints.size());
        };

        std::cout << "\nSelecting best frame for stereo calibration..." << std::endl;
        for (int idx = 0; idx < validIndices.size(); ++idx) 
        {
            int frameIdx = validIndices[idx];
            
            double rms1 = computeReprojectionError(validObjectPoints[idx], validImagePoints1[idx], 
                                                cameraMatrix1_final, distCoeffs1_final);
            
            double rms2 = computeReprojectionError(validObjectPoints[idx], validImagePoints2[idx], 
                                                cameraMatrix2_final, distCoeffs2_final);
            
            if (rms1 >= std::numeric_limits<double>::max() || 
                rms2 >= std::numeric_limits<double>::max()) {
                continue;
            }
            
            double errorSum = rms1 + rms2;
            double errorDiff = std::abs(rms1 - rms2);
            
            std::cout << "Frame " << frameIdx << ": "
                    << "L-RMS = " << rms1 << ", R-RMS = " << rms2
                    << ", Sum = " << errorSum << ", Diff = " << errorDiff << std::endl;
            
            // 选择误差和小且平衡的帧
            if (errorSum < minErrorSum || (std::abs(errorSum - minErrorSum) < 0.05 && errorDiff < minErrorDiff))
            {
                minErrorSum = errorSum;
                minErrorDiff = errorDiff;
                bestIdx = frameIdx;
                best_rms1 = rms1;
                best_rms2 = rms2;
            }
        }
        if (bestIdx == -1) {
            std::cerr << "No valid frame found for stereo calibration." << std::endl;
            return;
        }
        std::cout << "\nSelected best frame: " << bestIdx << std::endl;

        // 4. 使用最优帧进行双目标定
        cv::Mat R, T, E, F;
        std::vector<std::vector<cv::Point3f>> objPts = { objectPointsVec[bestIdx] };
        std::vector<std::vector<cv::Point2f>> imgPts1 = { imagePointsVec1[bestIdx] };
        std::vector<std::vector<cv::Point2f>> imgPts2 = { imagePointsVec2[bestIdx] };
        
        double stereo_rms = cv::stereoCalibrate(
            objPts, imgPts1, imgPts2,
            cameraMatrix1_final, distCoeffs1_final,
            cameraMatrix2_final, distCoeffs2_final,
            img1_frames[bestIdx].size(), R, T, E, F,
            cv::CALIB_FIX_INTRINSIC
        );

        // 保存结果
        this->cameraMatrix1 = cameraMatrix1_final;
        this->cameraMatrix2 = cameraMatrix2_final;
        this->distCoeffs1 = distCoeffs1_final;
        this->distCoeffs2 = distCoeffs2_final;
        this->R = R;
        this->T = T;
        this->E = E;
        this->F = F;
        // 输出结果
        std::cout << "\n======= Final Stereo Calibration Results =======" << std::endl;
        std::cout << "Best frame index: " << bestIdx << std::endl;
        std::cout << "Left camera RMS: " << best_rms1 << " pixels" << std::endl;
        std::cout << "Right camera RMS: " << best_rms2 << " pixels" << std::endl;
        std::cout << "Stereo calibration RMS: " << stereo_rms << std::endl;
        
        std::cout << "\nLeft Camera Matrix:" << std::endl << cameraMatrix1_final << std::endl;
        std::cout << "Left Distortion Coefficients:" << std::endl << distCoeffs1_final.t() << std::endl;
        std::cout << "Right Camera Matrix:" << std::endl << cameraMatrix2_final << std::endl;
        std::cout << "Right Distortion Coefficients:" << std::endl << distCoeffs2_final.t() << std::endl;
        std::cout << "Rotation Matrix:" << std::endl << R << std::endl;
        std::cout << "Translation Vector:" << std::endl << T.t() << std::endl;
        
        // 计算重投影矩阵
        cv::Mat projMatrix1 = computeProjMatrix(R, T, cameraMatrix1_final);
        cv::Mat projMatrix2 = computeProjMatrix(R, T, cameraMatrix2_final);
        this->projMatrixVec1.push_back(projMatrix1);
        this->projMatrixVec2.push_back(projMatrix2);
        std::cout << "Left Projection Matrix:" << std::endl << projMatrix1 << std::endl;
        std::cout << "Right Projection Matrix:" << std::endl << projMatrix2 << std::endl;
    }


    void ChessCalibration::drawCornersAndAxes(cv::Mat& img, const std::vector<cv::Point2f>& corners, cv::Size boardSize, bool found)
    {
        if (found) {
            // 在角点处画空心圆
            for (const auto& corner : corners) {
                cv::circle(img, corner, 15, cv::Scalar(0, 0, 255), 2); // 红色空心圆圈，半径为15，线宽为2
            }

            // 计算第一个角点的位置，作为原点
            cv::Point2f origin = corners[0];

            // 计算X轴和Y轴的方向
            cv::Point2f xAxis = corners[3] - corners[0];
            cv::Point2f yAxis = corners[3 * boardSize.width] - corners[0];

            // 绘制X轴（蓝色）并添加箭头和标签
            cv::arrowedLine(img, origin, origin + xAxis, cv::Scalar(255, 0, 0), 4);
            cv::putText(img, "X", origin + xAxis * 1.1, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 4);

            // 绘制Y轴（绿色）并添加箭头和标签
            cv::arrowedLine(img, origin, origin + yAxis, cv::Scalar(0, 255, 0), 8);
            cv::putText(img, "Y", origin + yAxis * 1.1, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 4);

            // 绘制原点（白色空心圆）
            cv::circle(img, origin, 15, cv::Scalar(255, 255, 255), 2); // 白色空心圆圈，半径为15，线宽为2
        }
    }


CircleCalibration::CircleCalibration(int rows, int cols, int squareSize, QStringList circlePath)
{
    this->boardSize = cv::Size(cols, rows);
    this->squareSize = squareSize;
    for (const QString& path : circlePath) {
        this->circlePath.push_back(path.toStdString());
    }
}

CircleCalibration::CircleCalibration(int rows, int cols, int squareSize, QStringList circlePathLeft, QStringList circlePathRight)
{
    this->boardSize = cv::Size(cols, rows);
    this->squareSize = squareSize;
    for (const QString& path : circlePathLeft) {
        this->circlePathLeft.push_back(path.toStdString());
    }
    for (const QString& path : circlePathRight) {
        this->circlePathRight.push_back(path.toStdString());
    }
}

CircleCalibration::~CircleCalibration()
{
}

bool CircleCalibration::prefareMonocularCalibration()
{
    if (circlePath.size() == 0)
    {
        std::cerr << "There is nothing for calibration." << std::endl;
        return false;
    }

    for (int i = 0; i < boardSize.height; ++i)
    {
        for (int j = 0; j < boardSize.width; ++j)
        {
            this->objectPoints.emplace_back(j * squareSize, i * squareSize, 0.00f);
        }
    }
    this->objectPointsVec = std::vector<std::vector<cv::Point3f>>(circlePath.size(), objectPoints);
    for (int i = 0; i < circlePath.size(); ++i)
    {
        cv::Mat img1 = cv::imread(circlePath[i]);
        cv::Mat gray1;
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners1;
        bool found = cv::findCirclesGrid(gray1, boardSize, corners1, cv::CALIB_CB_SYMMETRIC_GRID);

        if (found)
        {
            cv::cornerSubPix(gray1, corners1, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            this->imagePointsVec1.push_back(corners1);
            drawCornersAndAxes(img1, corners1, boardSize, found);
            this->img1_frames.push_back(img1);
        }
        else
        {
            if (!found)
            {
                std::cerr << "Circle grid not found in image: " << circlePath[i] << std::endl;
            }
        }
    }
    return true;
}

void CircleCalibration::startMonocularCompute()
{
    if (imagePointsVec1.empty())
    {
        std::cerr << "No valid circle grid corners were found in the provided images." << std::endl;
        exit(1);
    }

    this->cameraMatrix1 = cv::Mat::eye(3, 3, CV_64F);

    // Camera calibration for both cameras
    double rms = cv::calibrateCamera(objectPointsVec, imagePointsVec1, cv::Size(img1_frames[0].cols, img1_frames[0].rows), cameraMatrix1, distCoeffs1, rvecs1, tvecs1);
    std::cout << "RMS error of camera： " << rms << std::endl;

    // 输出相机内参、畸变系数、外参
    std::cout << "Camera Matrix：" << std::endl << cameraMatrix1 << std::endl;
    std::cout << "Distortion Coefficients：" << std::endl << distCoeffs1 << std::endl;
    for (int i = 0; i < circlePath.size(); i++)
    {
        // 将旋转向量转换为旋转矩阵
        cv::Mat R, T;
        cv::Rodrigues(rvecs1[i], R);
        T = tvecs1[i];
        std::cout << "Rotation Matrix: " << std::endl << R << std::endl;
        std::cout << "Translation Vector: " << std::endl << tvecs1[i] << std::endl;
        cv::Mat projMatrix = computeProjMatrix(R, T, cameraMatrix1);
        this->projMatrixVec1.push_back(projMatrix);
    }
    std::cout << "compute is over" << std::endl;
}

cv::Mat CircleCalibration::computeProjMatrix(const cv::Mat R, const cv::Mat T, const cv::Mat cameraMatrix) const
{
    // Construct projection matrix for Camera : P = K * [R | T]
    cv::Mat projMatrix = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat RT;
    cv::hconcat(R, T, RT);  // Combine R and T into a 3x4 matrix
    projMatrix = cameraMatrix * RT;
    return projMatrix;
}

void CircleCalibration::drawCornersAndAxes(cv::Mat& img, const std::vector<cv::Point2f>& corners, cv::Size boardSize, bool found)
{
    if (found) {
        // 在角点处画空心圆
        for (const auto& corner : corners) {
            cv::circle(img, corner, 15, cv::Scalar(0, 0, 255), 2); // 红色空心圆圈，半径为15，线宽为2
        }

        // 计算第一个角点的位置，作为原点
        cv::Point2f origin = corners[0];

        // 计算X轴和Y轴的方向
        cv::Point2f xAxis = corners[1] - corners[0];
        cv::Point2f yAxis = corners[boardSize.width] - corners[0];

        // 绘制X轴（蓝色）并添加箭头和标签
        cv::arrowedLine(img, origin, origin + xAxis, cv::Scalar(255, 0, 0), 4);
        cv::putText(img, "X", origin + xAxis * 1.1, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 4);

        // 绘制Y轴（绿色）并添加箭头和标签
        cv::arrowedLine(img, origin, origin + yAxis, cv::Scalar(0, 255, 0), 8);
        cv::putText(img, "Y", origin + yAxis * 1.1, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 4);

        // 绘制原点（白色空心圆）
        cv::circle(img, origin, 15, cv::Scalar(255, 255, 255), 2); // 白色空心圆圈，半径为15，线宽为2
    }
}

bool CircleCalibration::prefareStereoCalibration()
{
    if (circlePathLeft.size() != circlePathRight.size()) {
        std::cerr << "The number of images from Camera1 and Camera2 should be the same." << std::endl;
        return false;
    }

    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            this->objectPoints.emplace_back(j * squareSize, i * squareSize, 0.00f);
        }
    }

    this->objectPointsVec = std::vector<std::vector<cv::Point3f>>(circlePathLeft.size(), objectPoints);

    for (int i = 0; i < circlePathLeft.size(); ++i) {
        cv::Mat img1 = cv::imread(circlePathLeft[i]);
        cv::Mat img2 = cv::imread(circlePathRight[i]);

        if (img1.empty() || img2.empty()) {
            std::cerr << "Failed to load images: " << circlePathLeft[i] << ", " << circlePathRight[i] << std::endl;
            return false;
        }

        cv::Mat gray1, gray2;
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners1, corners2;
        bool found1 = cv::findCirclesGrid(gray1, boardSize, corners1, cv::CALIB_CB_SYMMETRIC_GRID);
        bool found2 = cv::findCirclesGrid(gray2, boardSize, corners2, cv::CALIB_CB_SYMMETRIC_GRID);

        if (found1 && found2) {
            cv::cornerSubPix(gray1, corners1, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            cv::cornerSubPix(gray2, corners2, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));

            this->imagePointsVec1.push_back(corners1);
            this->imagePointsVec2.push_back(corners2);
        }
    }

    return true;
}


void CircleCalibration::startStereoCalibration()
{
    if (imagePointsVec1.empty() || imagePointsVec2.empty()) {
        std::cerr << "No valid circle grid corners were found in the provided images." << std::endl;
        return;
    }

    // 1. 预筛选：对每一帧单独进行单目标定，剔除重投影误差大的帧
    std::vector<int> validIndices;
    const double MAX_REPROJ_ERROR = 0.1; // 重投影误差阈值
    
    // 存储有效帧的角点数据
    std::vector<std::vector<cv::Point3f>> validObjectPoints;
    std::vector<std::vector<cv::Point2f>> validImagePoints1;
    std::vector<std::vector<cv::Point2f>> validImagePoints2;
    
    std::cout << "Performing pre-screening of frames..." << std::endl;
    for (int i = 0; i < imagePointsVec1.size(); ++i) {
        // 当前帧的角点数据
        std::vector<std::vector<cv::Point3f>> objPts = { objectPointsVec[i] };
        std::vector<std::vector<cv::Point2f>> imgPts1 = { imagePointsVec1[i] };
        std::vector<std::vector<cv::Point2f>> imgPts2 = { imagePointsVec2[i] };
        
        // 对左相机进行单帧标定
        cv::Mat cameraMatrix1_temp = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat distCoeffs1_temp;
        std::vector<cv::Mat> rvecs1, tvecs1;
        double rms1 = cv::calibrateCamera(objPts, imgPts1, img1_frames[i].size(), 
                                        cameraMatrix1_temp, distCoeffs1_temp, 
                                        rvecs1, tvecs1);
        
        // 对右相机进行单帧标定
        cv::Mat cameraMatrix2_temp = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat distCoeffs2_temp;
        std::vector<cv::Mat> rvecs2, tvecs2;
        double rms2 = cv::calibrateCamera(objPts, imgPts2, img2_frames[i].size(), 
                                        cameraMatrix2_temp, distCoeffs2_temp, 
                                        rvecs2, tvecs2);
        // 检查重投影误差
        if (rms1 <= MAX_REPROJ_ERROR && rms2 <= MAX_REPROJ_ERROR) {
            validIndices.push_back(i);
            validObjectPoints.push_back(objectPointsVec[i]);
            validImagePoints1.push_back(imagePointsVec1[i]);
            validImagePoints2.push_back(imagePointsVec2[i]);
            
            std::cout << "Frame " << i << " passed screening: "
                    << "L-RMS = " << rms1 << ", R-RMS = " << rms2 << std::endl;
        } else {
            std::cout << "Frame " << i << " rejected: "
                    << "L-RMS = " << rms1 << ", R-RMS = " << rms2 << std::endl;
        }
    }
    if (validObjectPoints.empty()) {
        std::cerr << "No valid frames after screening. Try increasing MAX_REPROJ_ERROR." << std::endl;
        return;
    }
    std::cout << "Selected " << validObjectPoints.size() << " valid frames." << std::endl;
    // 2. 用所有有效帧重新进行单目标定，得到最终内参
    cv::Mat cameraMatrix1_final = cv::Mat::eye(3, 3, CV_64F), distCoeffs1_final;
    cv::Mat cameraMatrix2_final = cv::Mat::eye(3, 3, CV_64F), distCoeffs2_final;
    std::vector<cv::Mat> rvecs1_final, tvecs1_final;
    std::vector<cv::Mat> rvecs2_final, tvecs2_final;
    double rms1_final = cv::calibrateCamera(validObjectPoints, validImagePoints1, 
                                        img1_frames[0].size(), cameraMatrix1_final, 
                                        distCoeffs1_final, rvecs1_final, tvecs1_final);
    double rms2_final = cv::calibrateCamera(validObjectPoints, validImagePoints2, 
                                        img2_frames[0].size(), cameraMatrix2_final,
                                        distCoeffs2_final, rvecs2_final, tvecs2_final);
    std::cout << "\nFinal single-camera calibration:" << std::endl;
    std::cout << "Left camera RMS: " << rms1_final << std::endl 
              << "Right camera RMS: " << rms2_final << std::endl;
    // 3. 在有效帧中寻找最优帧（左右误差和最小且平衡）
    int bestIdx = -1;
    double bestScore = std::numeric_limits<double>::max();
    for (int i = 0; i < validObjectPoints.size(); ++i) {
        double score = std::abs(rvecs1_final[i].at<double>(0) - rvecs2_final[i].at<double>(0)) +
                       std::abs(tvecs1_final[i].at<double>(0) - tvecs2_final[i].at<double>(0));
        if (score < bestScore) {
            bestScore = score;
            bestIdx = i;
        }
    }
    if (bestIdx == -1) {
        std::cerr << "No valid frame found for stereo calibration." << std::endl;
        return;
    }
    std::cout << "Selected best frame for stereo calibration: " << bestIdx << std::endl;
    // 4. 使用最优帧进行双目标定
    cv::Mat R, T, E, F;
    cv::stereoCalibrate(validObjectPoints, validImagePoints1, validImagePoints2,
                       cameraMatrix1_final, distCoeffs1_final,
                       cameraMatrix2_final, distCoeffs2_final,
                       img1_frames[0].size(), R, T, E, F);
    // 保存结果
    this->cameraMatrix1 = cameraMatrix1_final;
    this->distCoeffs1 = distCoeffs1_final;
    this->cameraMatrix2 = cameraMatrix2_final;
    this->distCoeffs2 = distCoeffs2_final;
    this->R = R;
    this->T = T;
    this->E = E;
    this->F = F;
    std::cout << "Stereo calibration completed successfully." << std::endl;
    // 输出结果
    std::cout << "\n======= Final Stereo Calibration Results =======" << std::endl;
    std::cout << "Left Camera Matrix:\n" << this->cameraMatrix1 << std::endl;
    std::cout << "Left Distortion Coefficients:\n" << this->distCoeffs1 << std::endl;
    std::cout << "Right Camera Matrix:\n" << this->cameraMatrix2 << std::endl;
    std::cout << "Right Distortion Coefficients:\n" << this->distCoeffs2 << std::endl;
    std::cout << "Rotation Matrix:\n" << this->R << std::endl;
    std::cout << "Translation Vector:\n" << this->T << std::endl;
    std::cout << "Essential Matrix:\n" << this->E << std::endl;
    std::cout << "Fundamental Matrix:\n" << this->F << std::endl;
    // 计算重投影矩阵
    cv::Mat projMatrix1 = computeProjMatrix(R, T, cameraMatrix1_final);
    cv::Mat projMatrix2 = computeProjMatrix(R, T, cameraMatrix2_final);
    this->projMatrixVec1.push_back(projMatrix1);
    this->projMatrixVec2.push_back(projMatrix2);
    std::cout << "Left Projection Matrix:\n" << projMatrix1 << std::endl;
    std::cout << "Right Projection Matrix:\n" << projMatrix2 << std::endl;
}
