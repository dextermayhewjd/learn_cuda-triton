cmake_minimum_required(VERSION 3.18)
project(gemm_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 开启优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# 手动添加 CUDA include 和 lib64 路径 (WSL 必须加)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# 项目自己的头文件
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(prepare_data src/prepare_data.cpp)
add_executable(bench_gemm src/bench_gemm.cu)

# 链接 CUDA 运行库
target_link_libraries(bench_gemm cudart)

set(CMAKE_BUILD_TYPE Release)

# 方便 profiler 显示源码行
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")