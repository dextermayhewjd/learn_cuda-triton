# chapter2

## 2.4 设备全局储存 和传输
整体就几个点
### 三个函数
#### 1.cudaMalloc()
  两个参数  
  1.
    是void**
    是分配对象的指针的地址 & d_A
    cudaMalloc 会在 device 上申请内存，并把“device 地址”写入指针 d_A 中
  2.
    size 记得sizeof（int） * N
  

#### 2.cudaMemcpy()
四个参数
    1. 是目标地址 
    2. 是原地址  
    3. size
    4. cudaMemcpyHostToDevice / cudaMemcpyDeviceToHost
```c++
cudaMemcpy(destination, source, size, direction);
```
#### 3.cudaFree()
 一个参数 只要是 device上的地址就行 也就是 给 d_A

## 2.5 核函数 和 线程
### 限定词
```c++
__host__
__global__
__device__
```
这三种放在 正常函数前

### 块和线程大小
1和1 可以替换为gird 以及block
都可以三维的 书这里还没有提及
```c++
function_1<<< 1,1 >>>

n = 1000
int blocks = (N + blockSize - 1) / blockSize;
vecAddKernel<<<ceil(n,256.0),256>>>
```
此处是 4个块 每个块里256个thread

用于获取当前thread的i
```c++
int i = blockIdx.x * blockDim.x + threadIdx.x;

if(i<n){
  c[i] = a[i] + b[i]
}
```
此处限制执行 只有n以内的执行 也就是执行n词

其次 每个块的thread 大小有上线
配合ceil 1000个thread按照这样子必须要有4个block 每个block里256个thread


# chapter2 code
很容易写错的是 c++的数组 
传进去的是device_a传一个指针 
但是本地的H_a 需要的数组声明
```c++
int h_A[N];   // Host 数组（在 CPU 内存上）
int *d_A;     // 指针（指向 GPU 内存）
```
