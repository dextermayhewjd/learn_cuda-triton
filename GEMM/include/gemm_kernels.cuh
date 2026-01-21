// include/gemm_kernels.cuh
#pragma once

#include <cuda_runtime.h>
#include "gemm_utils.hpp"

enum class GemmKernelType
{
  Naive,
  Tiled16
};

// ==================== Naive GEMM ====================
// C = A(MxK) * B(KxN), 一线程算 C(row,col) 一个元素
__global__ void gemm_naive_kernel(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C,
                                  int M, int K, int N)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N)
  {
    if (row == 0 && col == 0)
    {
      printf("Running NAIVE kernel, M=%d, K=%d, N=%d\n", M, K, N);
    }
    float sum = 0.f;
    for (int i = 0; i < K; ++i)
    {
      sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
  }
}

// ==================== 16x16 tiled shared memory GEMM ====================
constexpr int TILE = 16;

__global__ void gemm_tiled16_kernel(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C,
                                    int M, int K, int N)
{
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0f;

  // 需要多少个 tile 覆盖 K 维度
  int numTiles = (K + TILE - 1) / TILE;

  for (int t = 0; t < numTiles; ++t)
  {
    int tiledColA = t * TILE + threadIdx.x; // A 的列
    int tiledRowB = t * TILE + threadIdx.y; // B 的行

    // load A tile
    if (row < M && tiledColA < K)
    {
      As[threadIdx.y][threadIdx.x] = A[row * K + tiledColA];
    }
    else
    {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // load B tile
    if (tiledRowB < K && col < N)
    {
      Bs[threadIdx.y][threadIdx.x] = B[tiledRowB * N + col];
    }
    else
    {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // 当前 tile 上做 16 次 MAC
    for (int i = 0; i < TILE; ++i)
    {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N)
  {
    C[row * N + col] = sum;
  }
}

// ==================== 调用封装 ====================
inline void launch_gemm_kernel(GemmKernelType type,
                               const float *dA,
                               const float *dB,
                               float *dC,
                               int M, int K, int N,
                               cudaStream_t stream = nullptr)
{
  dim3 block, grid;
  switch (type)
  {
  case GemmKernelType::Naive:
  case GemmKernelType::Tiled16:
    block = dim3(16, 16);
    grid = dim3((N + block.x - 1) / block.x,
                (M + block.y - 1) / block.y);
    break;
  default:
    throw std::runtime_error("Unknown kernel type");
  }

  switch (type)
  {
  case GemmKernelType::Naive:
    gemm_naive_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, K, N);
    break;
  case GemmKernelType::Tiled16:
    gemm_tiled16_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, K, N);
    break;
  }
}
