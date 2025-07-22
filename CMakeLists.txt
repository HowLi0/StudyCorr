cmake_minimum_required(VERSION 3.16)
project(StudyCorr VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_BUILD_TYPE Release)

# 确保MSVC使用正确的C++17标志和Release优化
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus /permissive-")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")
    add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add custom cmake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Include custom Qt6 finder (this will also enable AUTOMOC, AUTOUIC, AUTORCC)
include(FindQt6Custom)

# Find OpenCV (optional - use local version if not found)
find_package(OpenCV QUIET)

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/API/OpenCorr/src
    ${CMAKE_CURRENT_SOURCE_DIR}/API/opencv/include
    ${CMAKE_CURRENT_SOURCE_DIR}/API/opencv/include/opencv2
    ${CMAKE_CURRENT_SOURCE_DIR}/API/fftw3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/API/Eigen3
    ${CMAKE_CURRENT_SOURCE_DIR}/API/Eigen3/Eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/API/nanoflann
    ${CMAKE_CURRENT_SOURCE_DIR}/API/cuSIFT
    ${CMAKE_CURRENT_SOURCE_DIR}/StudyCorr_GPU
)

# Set library directories
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/API/OpenCorr/gpu_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/API/opencv/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/API/fftw3/lib
)

# Source files
file(GLOB STUDYCORR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/StudyCorr/*.cpp
)
file(GLOB OPENCORR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/API/OpenCorr/src/*.cpp
)
file(GLOB CUSIFT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/API/cuSIFT/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/API/cuSIFT/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/API/cuSIFT/*.cuh
)

file(GLOB STUDYCORR_GPU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/StudyCorr_GPU/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StudyCorr_GPU/*.cu
)
set(SOURCES ${STUDYCORR_SOURCES} ${OPENCORR_SOURCES} ${CUSIFT_SOURCES}  ${STUDYCORR_GPU_SOURCES})

# Header files that need MOC processing
set(MOC_HEADERS
    StudyCorr/StudyCorr.h
    StudyCorr/CalibrationLoadDialog.h
    StudyCorr/ComputeLoadDialog.h
    StudyCorr/Drawable.h
)

# Other header files

file(GLOB STUDYCORR_GPU_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/StudyCorr_GPU/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/StudyCorr_GPU/*.cuh
)

set(HEADERS
    StudyCorr/Calibration.h
    StudyCorr/customPixmapItem.h
    StudyCorr/Shape.h
    ${STUDYCORR_GPU_HEADERS}
)

# UI files
set(UI_FILES
    StudyCorr/StudyCorr.ui
)

# Resource files
set(RESOURCE_FILES
    StudyCorr/StudyCorr.qrc
)

# Process Qt files (this is now optional with AUTOMOC/AUTOUIC/AUTORCC enabled)
# qt6_wrap_cpp(MOC_SOURCES ${MOC_HEADERS})
# qt6_wrap_ui(UI_HEADERS ${UI_FILES})
# qt6_add_resources(RESOURCE_SOURCES ${RESOURCE_FILES})

# Create executable
add_executable(StudyCorr
    ${SOURCES}
    ${HEADERS}
    ${MOC_HEADERS}
    ${UI_FILES}
    ${RESOURCE_FILES}
)

# Link libraries
find_package(CUDAToolkit REQUIRED)
target_link_libraries(StudyCorr PRIVATE CUDA::cusolver)

# 统一为 keyword 语法
 target_link_libraries(StudyCorr PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    Qt6::Network
    Qt6::NetworkAuth
    Qt6::Help
)

# Add OpenCV libraries if found
if(OpenCV_FOUND)
    target_link_libraries(StudyCorr PRIVATE ${OpenCV_LIBS})
endif()

# Link libraries (Release版本)
target_link_libraries(StudyCorr PRIVATE
    opencv_world4100
    libfftw3-3
    libfftw3f-3
    libfftw3l-3
    OpenCorrGPU
)

# Set target properties
set_target_properties(StudyCorr PROPERTIES
    WIN32_EXECUTABLE False
    MACOSX_BUNDLE TRUE
    OUTPUT_NAME "StudyCorr"
)

# Copy necessary DLLs to output directory (for runtime)
if(WIN32)
    # Copy OpenCV DLLs
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/API/opencv/bin")
        add_custom_command(TARGET StudyCorr POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/API/opencv/bin"
            "$<TARGET_FILE_DIR:StudyCorr>"
            COMMENT "Copying OpenCV DLLs to output directory"
        )
    endif()
    
    # Copy FFTW DLLs
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/API/fftw3/dll")
        add_custom_command(TARGET StudyCorr POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/API/fftw3/dll"
            "$<TARGET_FILE_DIR:StudyCorr>"
            COMMENT "Copying FFTW DLLs to output directory"
        )
    endif()
    
    # Deploy Qt6 (if windeployqt is available)
    find_program(QT_DEPLOY_EXECUTABLE windeployqt HINTS ${Qt6_DIR}/../../../bin)
    if(QT_DEPLOY_EXECUTABLE)
        add_custom_command(TARGET StudyCorr POST_BUILD
            COMMAND ${QT_DEPLOY_EXECUTABLE} "$<TARGET_FILE:StudyCorr>"
            COMMENT "Deploying Qt6 libraries"
        )
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(StudyCorr PRIVATE OpenMP::OpenMP_CXX)
endif()
