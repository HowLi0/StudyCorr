@echo off
chcp 65001 >nul
echo ======================================
echo 构建StudyCorr (仅Release版本)
echo ======================================

REM 创建构建目录（如不存在）
if not exist "build" (
    echo 创建构建目录...
    mkdir build
)

REM 配置CMake (Release)
echo 配置CMake (Release)...
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022"

if %ERRORLEVEL% NEQ 0 (
    echo CMake配置失败！
    pause
    exit /b 1
)

REM 构建项目 (Release)
echo 构建项目 (Release)...
cmake --build build --config Release

if %ERRORLEVEL% NEQ 0 (
    echo 构建失败！
    pause
    exit /b 1
)

echo.
echo ======================================
echo 构建成功！
echo ======================================
echo 可执行文件位于: build\bin\Release\StudyCorr.exe
echo.

REM 检查可执行文件
if exist "build\bin\Release\StudyCorr.exe" (
    echo 文件信息：
    dir "build\bin\Release\StudyCorr.exe" | findstr StudyCorr
    echo.
    start "" "build\bin\Release\StudyCorr.exe"
    echo 应用程序已启动
) 

echo.
echo 按任意键退出...
pause >nul
