cmake_minimum_required(VERSION 3.18)
project(HyperCudaX LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 编译优化选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC --use_fast_math")

option(ENABLE_BENCHMARK "Enable benchmark tests" OFF)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include_directories(${PROJECT_SOURCE_DIR}/include)

# 添加核心库
set(SRC_FILES src/matmul.cu src/matmul_fp16.cu)


if(BUILD_SHARED_LIBS)
    add_library(hypercudax SHARED ${SRC_FILES})
else()
    add_library(hypercudax STATIC ${SRC_FILES})
endif()


if(ENABLE_BENCHMARK)
    add_subdirectory(benchmark)
endif()
