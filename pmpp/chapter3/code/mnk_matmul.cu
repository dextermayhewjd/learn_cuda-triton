__global__ void mnklMatmulKernel(float *M_mat, float *N_mat, float *P_mat,
                                 int M, int K, int N)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  if ((row < M) && (col < N))
  {
    float pvalue = 0;
    for (int i = 0; i < K; i++)
    {
      pvalue += M_mat[i + row * K] * N_mat[col + i * N];
    }
    P_mat[row * N + col] = pvalue;
  }
}