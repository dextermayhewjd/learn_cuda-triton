#include <stdio.h>

__global__ void add(float *A, float *B, float *C, int N)
{

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
  {
    C[i] = A[i] + B[i];
  }
}

int main()
{
  const int N = 5;
  float *d_A, *d_B, *d_C;

  // 构造
  float h_A[N] = {1.0, 2.0, 3.0, 4.0, 5.0};
  float h_B[N] = {1.0, 2.0, 3.0, 4.0, 5.0};
  float h_C[N];

  int size = N * sizeof(float);

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

  add<<<1, N>>>(d_A, d_B, d_C, N);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++)
  {
    printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
  }
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}