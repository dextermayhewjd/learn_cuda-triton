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
我觉得这里的重点是
# 每个thread能够通过自身的col 和row来确定两个东西
1. output中自己负责的pixel 或者2d图中的位置
2. 同时 要清楚在input中读取值 的位置 如何对应上

idx = row * width + col永远应该是出图

# 同时 传入的width和height能确定的点
1. 因为block会多传一些thread
要确定工作的thread的row和 column是在范围内的
  if (col < width && row < height)
2. 能够使用预先设定的width和 height来做 对应的内存访问位置的计算
int greyOffSet = row * width + col;

两者相互确保output的每个pixel的内存地址都有一个thread来管
*/