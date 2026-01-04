# 🚀 CUDA GEMM 学习项目

本项目实现并 benchmark 经典矩阵乘法（GEMM）：
目标是帮助你：

- 理解 GEMM 的计算过程
- 学会编写基础 CUDA Kernel
- 比较 naive / tiled / Tensor Core 性能差异
- 熟悉 GPU 调试、验证与 benchmark

---

## 📦 1. 环境要求

### 硬件
NVIDIA 显卡（建议 RTX 20 系及以上）

### 软件

| 工具 | 版本 |
|------|------|
CUDA Toolkit | 11.8+（本项目测试于 13.x）
NVIDIA 驱动 | 必须 **>= CUDA 版本**
CMake | 3.18+
C++17 | GCC / Clang

检查：

```bash
nvidia-smi
nvcc --version
```


##  2. 编译
```bash
git clone <repo>
cd GEMM

mkdir build
cd build
cmake ..
make -j
```

生成：

`prepare_data`  
`bench_gemm`

⚠️ 可执行文件都在 build/ 目录中。

##  3. 生成测试数据
```bash
cd build
./prepare_data
```

默认尺寸：

M = K = N = 1024


自定义：

```bash
./prepare_data M K N
```

# 例如：
```bash
./prepare_data 2048 2048 2048
```
##  4. 运行 Benchmark
naive kernel
```bash
./bench_gemm naive
```

tiled kernel（使用 shared memory）

```bash 
./bench_gemm tiled16
```


可选参数：重复次数

```bash
./bench_gemm naive 20
```


示例输出：  
`此为3090 native cuda13.1 1024 1024 1024 十次下的跑分`
```bash 
Benchmark kernel: naive, repeat=10
Loaded matrices: M=1024, K=1024, N=1024
Avg time: 1.10 ms, Perf: 1937.8 GFLOP/s
Verification: PASSED
```