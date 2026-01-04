__global__ void one_thread_row(float *M_mat, float *N_mat, float *P_mat,
                               int Width)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  // int i = threadIdx.x + blockDim.x * blockIdx.x; // 可以考虑为生成了 是在y排布

  if (row < Width)
  {
    // 处理一个行
    for (int w = 0; w < Width; w++)
    {
      int p_value = 0;
      // 内积
      for (int k = 0; k < Width; k++)
      {
        p_value += M_mat[row * Width + k] * N_mat[k * Width + w];
      }
      P_mat[row * Width + w] = p_value;
    }
  }
}