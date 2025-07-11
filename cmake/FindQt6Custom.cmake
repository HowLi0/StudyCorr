# FindQt6Custom.cmake - Custom Qt6 finder for StudyCorr project

# Try to find Qt6 in common installation paths
set(QT6_SEARCH_PATHS
    "D:/QT/QTcreator/6.7.0/msvc2019_64"
    $ENV{Qt6_DIR}
    $ENV{QT_DIR}
    $ENV{QT6_DIR}
)

foreach(QT_PATH ${QT6_SEARCH_PATHS})
    if(EXISTS "${QT_PATH}")
        set(CMAKE_PREFIX_PATH ${QT_PATH} ${CMAKE_PREFIX_PATH})
        message(STATUS "Found Qt6 installation at: ${QT_PATH}")
        break()
    endif()
endforeach()

# Standard Qt6 find package
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui OpenGL OpenGLWidgets Network NetworkAuth Help)

if(Qt6_FOUND)
    message(STATUS "Qt6 found successfully")
    message(STATUS "Qt6 version: ${Qt6_VERSION}")
    message(STATUS "Qt6 installation path: ${Qt6_DIR}")
    
    # Enable automatic MOC, UIC, and RCC for Qt6
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)
    
    # Qt6 is ready to use - keywords enabled
else()
    message(FATAL_ERROR "Qt6 not found. Please install Qt6 or set Qt6_DIR environment variable.")
endif()
