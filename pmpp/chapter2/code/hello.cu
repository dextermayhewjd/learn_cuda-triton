#include <stdio.h>

__global__ void helloFromGPU()
{
  printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main()
{
  // 启动 1 个 block，5 个线程
  helloFromGPU<<<1, 5>>>();
  cudaDeviceSynchronize(); // 等待 GPU 执行完毕
  return 0;
}