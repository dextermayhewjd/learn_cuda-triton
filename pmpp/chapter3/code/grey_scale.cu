#include <stdio.h>
__global__ void picture_kernel(float *out, float *in, int width, int height)
{
  // 1. 计算当前线程对应的图像坐标
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  // 2. 边界检查（防止溢出图像范围）
  if (col < width && row < height)
  {
    // 3. 计算一维线性索引
    int idx = row * width + col;

    int greyOffSet = row * width + col;
    int rgbOffSet = greyOffSet * 3;

    unsigned char r = in[rgbOffSet];
    unsigned char g = in[rgbOffSet + 1];
    unsigned char b = in[rgbOffSet + 2];

    // 执行操作（例如灰度化或赋值）
    out[idx] = 0.21 * r + 0.71f * g + 0.07f * b;
  }
}

/*
我觉得这里的重点是 每个thread能够通过自身的col 和row来确定两个东西

pout的作为thread 作用的pixel
和pin的 作为thread 应该读取的pixel
idx = row * width + col永远应该是出图
*/