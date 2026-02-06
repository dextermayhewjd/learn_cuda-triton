#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>


__global__ void matrixMulKernel(float* M, float* N, float* P, int Width)
{
    int row = threadIdx.y + blockDim.y+blockIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (row < Width && col < Width)
    {
        float p_value = 0;
        for(int k = 0; k<Width;++k)
        {
            p_value = M[row * Width +k] * N[col + Width*k];
        }

        P[row * Width + col] = p_value;
    }
}
/*
这里其实隐含了一个点就是 thread分配的是数量是
由block size（多少个thread） 
和grid size共同决定的
但是世界active的是根据具体矩阵的大小来定义的
*/ 

#define TILE_WIDTH 16
__global__ void tiled_matmul(float *M,float* N,float* P,int Width)
{
    /*
    这个输入的第一个假设就是 两个各矩阵都是W*W 的矩阵
    */
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * Width + ty;
    int Col = bx * Width + tx;

    float p_value = 0;
    for (int ph = 0; ph< Width/TILE_WIDTH; ++ph)
    {
        Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[Col + Width*ty + ph*Width*TILE_WIDTH];
        __syncthreads__();

        for(int k = 0;k< TILE_WIDTH;++k)
        {
            p_value += Mds[ty][k] * Nds[k][tx];
        }
        __stncthreads__();
    }
    P[Row *Width + Col] = p_value;
}

/*
这里同样有默认

因为shared memory 是在block内共享
tile的 size是16 * 16

可以看到原本计算单元里 col 和 row是通过 
    int row = threadIdx.y + blockDim.y+blockIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
来计算的， 但是这里因为默认一个共享内存里面的放的平铺尺寸是16
所以block的dim 这里默认是 width*width 了 也就是256

因为平铺的思想 把原本从global memory读取的操作对应的方式
共享内存的ty tx是一个block里面就这么多
如何把原本在M N 全局内存对应呢
通过Width/ tile width来确定phase的次数

Row*Width 是在M全局内存中 上一行所有元素的个数
ph*TILE_WIDTH 是开始这个phase的starting point
tx 才是具体搬运第几个

两个sync 展现了两种数据依赖
第一种是 写-> 读
读取数据之前 先写入的依赖
如果不加，在做计算的时候，数据都没准备好

第二种是 读后写 读->写
写之前 必须先让所有数据都读完
如果不加，在写入新的数据的时候，之前旧数据必须全部使用读取完了


前者为真依赖，因为读取线程的数据需要真的写入了的数据，依赖源是数据本身

后者为假依赖，因为写的线程并不需要读线程的任何数据，依赖关系关系因为重复用了相同的位置
*/