__global__ void mmm_matmul(float *M_mat, float *N_mat, float *P_mat,
                           int Width)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  if (row < Width && col < Width)
  {
    float p_value = 0;
    for (int k = 0; k < Width; k++)
    {
      p_value += M_mat[row * Width + k] * N_mat[k * Width + col];
    }

    P_mat[row * Width + col] = p_value;
  }
}