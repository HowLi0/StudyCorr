# StudyCorr —— 3D-DIC数字图像相关库

## 项目概述
StudyCorr 是一个专注于三维数字图像相关（3D-DIC, Digital Image Correlation）的高性能库，基于 C++ 和 CUDA 实现。它主要用于图像配准、三维变形场计算、SIFT 特征提取、相机机标定等任务。项目集成了多种第三方库（如 cuSIFT、OpenCV、nanoflann），支持 GPU 加速，适用于材料力学、实验力学、计算机视觉等领域的 3D-DIC 研究与应用。

## 主要功能
- **图像配准与变形场计算**：支持基于相关性的方法进行图像配准，计算三维变形场。
- **SIFT 特征提取与匹配**：集成 cuSIFT，支持 GPU 加速的 3D SIFT 特征提取与匹配。
- **相机标定与校准**：提供多种 3D 标定工具，支持三维标定数据的加载与处理。
- **高性能计算**：利用 CUDA 和多线程技术加速大规模三维图像处理任务。
- **可视化与交互**：包含 Qt GUI，支持三维图像、三维变形场、三维特征点等结果的可视化。

## 目录结构
- `API/`：第三方库和自定义算法模块（如 cuSIFT、nanoflann、opencv 等）。
- `StudyCorr/`：主程序源码，包括三维标定、三维配准、界面等核心功能。
- `StudyCorr_GPU/`：GPU 加速相关模块，包含三维 CUDA 核心算法实现。
- `calibration_images/`：三维标定相关的图片和数据。
- `build/`：编译生成的中间文件和可执行文件。
- `cmake/`：CMake 配置脚本。

## 快速开始
1. **依赖安装**：确保已安装 CUDA、CMake、Visual Studio 2022、Qt6、OpenCV等依赖。
2. **编译项目**：
   - 使用 CMake 配置并生成工程：
     ```powershell
     cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64
     ```
   - 编译 Release 版本：
     ```powershell
     cmake --build build --config Release
     ```
3. **运行程序**：
   - 执行主程序：
     ```powershell
     .\build\bin\Release\StudyCorr.exe
     ```

## 主要依赖
- CUDA
- CMake
- Visual Studio 2022
- Qt6
- OpenCV
- nanoflann
- cuSIFT

## 联系与贡献
欢迎提交 issue 或 pull request 以改进本项目。如有关于 3D-DIC 或本库的疑问请联系项目维护者。
