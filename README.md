# StudyCorr

cd E:\code_C++\StudyCorr_GPU\build\bin\Release
compute-sanitizer ./StudyCorr.exe

基于 Qt6 的 C++ 图像处理项目，支持 VS Code 跨平台开发，集成 OpenCV、FFTW3、Eigen3、nanoflann，opencorr等常用库。

## 环境要求

- Visual Studio Code
- CMake ≥ 3.16
- Visual Studio 2022（MSVC 编译器）
- Qt6（推荐 6.7.0+，需完整安装 MSVC 组件）
- OpenCV（已集成于 API 目录）

## 推荐 VS Code 扩展

- ms-vscode.cpptools（C/C++ 支持）
- ms-vscode.cmake-tools（CMake 集成）
- tonka3000.qtvsctools（Qt 支持）

## 目录结构

```
StudyCorr/
├── .vscode/         # VS Code 配置
├── cmake/           # CMake 模块
├── API/             # 第三方库（OpenCV/FFTW3/Eigen3/nanoflann）
├── StudyCorr/       # 源码
├── build/           # 构建输出
├── CMakeLists.txt   # CMake 构建脚本
└── build_release.bat# 一键 Release 构建脚本
```

## 快速上手

### 1. 打开项目

```bash
cd StudyCorr
code .
```

### 2. 配置 Qt 路径

如 Qt 安装路径非默认，请编辑 `cmake/FindQt6Custom.cmake`，添加你的 Qt 安装路径。

### 3. 构建项目

#### 方式一：VS Code 任务

- `Ctrl+Shift+P` → “Tasks: Run Task” → 选择 “Build Release” 或 “Build Debug”

#### 方式二：CMake 扩展

- `Ctrl+Shift+P` → “CMake: Configure” → “CMake: Build”

#### 方式三：命令行

```powershell
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

#### 方式四：脚本

```powershell
./build_release.bat
```

### 4. 运行项目

- VS Code 任务：“Run Application”
- 或直接运行：`build\bin\Release\StudyCorr.exe`

## 常见问题

- **Qt 找不到**：检查 Qt 是否安装、`cmake/FindQt6Custom.cmake` 路径、环境变量 `Qt6_DIR`
- **OpenCV 链接错误**：确认 `API/opencv/lib` 路径和库文件名
- **编译失败**：确保 Visual Studio 2022、C++17、所有 include 路径正确

## 依赖说明

- Qt6: Core, Widgets, Gui, OpenGL, Network, Help 等
- OpenCV: 图像处理
- FFTW3: 傅里叶变换
- Eigen3: 线性代数
- nanoflann: KD-tree
- OpenCorrGPU: GPU 计算库

## 开发建议

- 已配置 IntelliSense，支持断点调试
- 推荐使用预定义任务构建/运行
- 建议安装代码格式化扩展

---

如遇问题，请查看 VS Code 输出面板和终端日志，或查阅 `cmake/FindQt6Custom.cmake` 及相关配置文件。
