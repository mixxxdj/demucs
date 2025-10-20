cmake_minimum_required(VERSION 3.0)

project(demucs.onnx)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DEIGEN_INTERNAL_DEBUGGING")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -flto -fno-signed-zeros -fassociative-math -freciprocal-math -fno-math-errno -fno-rounding-math -funsafe-math-optimizations -fno-trapping-math -fno-rtti -DNDEBUG")

# Set your onnxruntime installation path here
set(ONNXRUNTIME_HOME /opt/homebrew/Cellar/onnxruntime/1.21.0)
include_directories(${ONNXRUNTIME_HOME}/include)
link_directories(${ONNXRUNTIME_HOME}/lib)

set(LIBNYQUIST_BUILD_EXAMPLE OFF CACHE BOOL "Disable libnyquist example")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/libnyquist ${CMAKE_CURRENT_BINARY_DIR}/libnyquist_build)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/libnyquist/include)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../src_cli/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../onnx-models/model/*.ort.c")
add_executable(demucs ${SOURCES})

target_link_libraries(demucs onnxruntime libnyquist)
