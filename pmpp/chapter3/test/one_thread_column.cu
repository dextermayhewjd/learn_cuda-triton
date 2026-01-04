__global__ void one_thread_row(float *M_mat, float *N_mat, float *P_mat,
                               int Width)
{
  // int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x; // 可以考虑为生成了 是在x排布

  // 每个thread 管一个column
  // M 矩阵上下左右都要 N矩阵只需要对应的一个col就行
  if (col < Width)
  {
    // 第一个for loop是用 M 矩阵的上下的
    for (int k = 0; k < Width; k++)
    {
      int p_value = 0;
      // M 矩阵 左右 N矩阵上下
      for (int i = 0; i < Width; i++)
      {
        p_value += M_mat[i + k * Width] * N_mat[i * Width + col];
      }
      P_mat[col + k * Width] = p_value;
    }
  }
}

// 脑袋里得有一张图 并且要写一下 for loop每个i在做什么
