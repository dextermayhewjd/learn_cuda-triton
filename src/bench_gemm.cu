// src/bench_gemm.cu
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include "gemm_utils.hpp"
#include "gemm_kernels.cuh"

GemmKernelType parse_kernel_type(const std::string &s)
{
  if (s == "naive")
    return GemmKernelType::Naive;
  if (s == "tiled16")
    return GemmKernelType::Tiled16;
  throw std::runtime_error("Unknown kernel name: " + s);
}

const char *kernel_name(GemmKernelType t)
{
  switch (t)
  {
  case GemmKernelType::Naive:
    return "naive";
  case GemmKernelType::Tiled16:
    return "tiled16";
  default:
    return "unknown";
  }
}

int main(int argc, char **argv)
{
  std::string kernel = "naive"; // 默认 naive
  int repeat = 10;              // 重复次数

  if (argc >= 2)
    kernel = argv[1];
  if (argc >= 3)
    repeat = std::max(1, std::atoi(argv[2]));

  GemmKernelType ktype = parse_kernel_type(kernel);
  std::printf("Benchmark kernel: %s, repeat=%d\n",
              kernel_name(ktype), repeat);

  // 载入 A, B, C_gold
  std::vector<float> hA, hB, hC_gold;
  int M, K1, K2, N;
  int M2, N2;

  load_matrix_bin("A.bin", hA, M, K1);
  load_matrix_bin("B.bin", hB, K2, N);
  load_matrix_bin("C_gold.bin", hC_gold, M2, N2);

  if (K1 != K2 || M != M2 || N != N2)
  {
    std::cerr << "Matrix dimension mismatch between A/B/C_gold\n";
    return -1;
  }

  std::printf("Loaded matrices: M=%d, K=%d, N=%d\n", M, K1, N);

  size_t sizeA = static_cast<size_t>(M) * K1 * sizeof(float);
  size_t sizeB = static_cast<size_t>(K1) * N * sizeof(float);
  size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, sizeA));
  CUDA_CHECK(cudaMalloc(&dB, sizeB));
  CUDA_CHECK(cudaMalloc(&dC, sizeC));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeB, cudaMemcpyHostToDevice));

  // 暖机一次
  launch_gemm_kernel(ktype, dA, dB, dC, M, K1, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 正式计时
  CudaTimer timer;
  float total_ms = 0.0f;

  for (int i = 0; i < repeat; ++i)
  {
    timer.start();
    launch_gemm_kernel(ktype, dA, dB, dC, M, K1, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = timer.stop();
    total_ms += ms;
  }

  float avg_ms = total_ms / repeat;
  double flops = 2.0 * (double)M * (double)K1 * (double)N;
  double gflops = (flops / 1e9) / (avg_ms / 1e3);

  std::printf("Avg time: %.3f ms, Perf: %.3f GFLOP/s\n", avg_ms, gflops);

  // 验证结果
  std::vector<float> hC(M * N);
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeC, cudaMemcpyDeviceToHost));

  bool ok = verify_result(hC.data(), hC_gold.data(), M * N);
  std::printf("Verification: %s\n", ok ? "PASSED" : "FAILED");

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));

  return ok ? 0 : -1;
}
