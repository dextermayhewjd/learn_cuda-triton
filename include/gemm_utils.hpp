// include/gemm_utils.hpp
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

#include <cuda_runtime.h>

// 简单错误检查宏
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// 随机初始化矩阵 [0, 9]
inline void init_matrix(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = static_cast<float>(rand() % 10);
}

// CPU GEMM：C = A(MxK) * B(KxN)
inline void cpu_gemm(const float* A, const float* B, float* C,
                     int M, int K, int N) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[r * K + i] * B[i * N + c];
            }
            C[r * N + c] = sum;
        }
    }
}

// 把矩阵保存为二进制：rows(int), cols(int), data(float)
inline void save_matrix_bin(const std::string& filename,
                            const float* data,
                            int rows, int cols) {
    FILE* fp = std::fopen(filename.c_str(), "wb");
    if (!fp) {
        throw std::runtime_error("Cannot open file for write: " + filename);
    }
    std::fwrite(&rows, sizeof(int), 1, fp);
    std::fwrite(&cols, sizeof(int), 1, fp);
    std::fwrite(data, sizeof(float), rows * cols, fp);
    std::fclose(fp);
}

// 从二进制读取矩阵
inline void load_matrix_bin(const std::string& filename,
                            std::vector<float>& data,
                            int& rows, int& cols) {
    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("Cannot open file for read: " + filename);
    }
    std::fread(&rows, sizeof(int), 1, fp);
    std::fread(&cols, sizeof(int), 1, fp);
    data.resize(rows * cols);
    std::fread(data.data(), sizeof(float), rows * cols, fp);
    std::fclose(fp);
}

// 验证 GPU 结果与 gold 结果
inline bool verify_result(const float* gpu, const float* gold,
                          int size, float tol = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(gpu[i] - gold[i]);
        if (diff > tol) {
            std::printf("Mismatch at %d: gpu=%f, gold=%f, diff=%f\n",
                        i, gpu[i], gold[i], diff);
            return false;
        }
    }
    return true;
}

// 简单 CUDA 计时器
struct CudaTimer {
    cudaEvent_t start_, stop_;
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop() { // 返回毫秒
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};
