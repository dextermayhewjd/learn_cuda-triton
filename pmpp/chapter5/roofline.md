一、GPU 优化的总纲（先给结论）
GPU 优化 = 找瓶颈 → 针对瓶颈那一层“对症下药”

## 第一层：内存访问优化（Memory-bound 时）

这是你提到的重点，也是最常见瓶颈。

### 1️⃣ Coalesced 访问（这是底线，不是优化）
核心规则（一定要会背）

一个 warp（32 threads）
访问连续、对齐的内存地址

```c++
// 好
addr = base + threadIdx.x
// 坏
addr = base + threadIdx.x * stride
```

判断方法  
一个 warp 是否能用 1–2 次 memory transaction  
而不是 32 次  
📌 不 coalesced，后面一切优化白搭  

### 2️⃣ Cache 利用（L2 / L1 / Read-only cache）
你不一定总用 shared memory。  
常见手段  
数据复用  
同一个数据被多个 warp / 多次使用  
blocking / tiling  
__ldg()（老架构）或只读 cache 路径  
合理的数据 layout（AoS → SoA）  
📌 很多 kernel：  
不是 bandwidth 不够，而是 cache miss 太多  
### 3️⃣ Shared Memory（你提到的）
shared memory 的正确定位是：  
用空间换带宽，用同步换复用  

#### 什么时候用？

同一数据被多个 thread 多次用  
global → shared → 多次 compute  
#### 常见用途  
GEMM tiling  
attention 中的 K/V tile  
stencil / convolution  
#### 但注意两个坑  
❌ Bank conflict  
连续 thread 访问同一个 bank → 串行  
❌ 用了 shared 但 occupancy 掉光  
shared 用太多 → block 数下降  
👉 shared 是手术刀，不是大锤  
### 4️⃣ 预取（Latency hiding 的关键）

```c++
// pseudo  
load next_tile  
compute current_tile  
software pipelining  
double buffer  
cp.async（Ampere+）  
```

📌 GPU 不怕 latency，只怕没事干

## 第二层：并行度 & Occupancy（Memory / Compute 都需要）

即使 memory-bound，你也需要足够多的 warp。

### 1️⃣ Occupancy 不是越高越好，但不能太低

决定因素  
registers / thread  
shared memory / block  
block size  
常见误区  
“occupancy 100% 一定最快” ❌  
现实是：  
50–70% 常记已经够隐藏 latency  

### 2️⃣ Warp-level 思维（非常重要）

你要开始以 warp 为单位思考  
warp divergence（if/else）  
warp-level primitive（__shfl, __syncwarp）  
warp-reduction 代替 block-reduction  
📌 很多 kernel：
慢在 warp divergence，不是 FLOP  
## 第三层：指令 & 计算优化（Compute-bound 时）

当你进入 roofline 右边，就轮到这些。

### 1️⃣ 用到“对的计算单元”

这是 AI infra 最核心的一点。

写法	用的硬件
普通 FP32	CUDA core  
FP16/BF16 Tensor Core	HMMA  
INT8	IMMA  

👉 Roofline 的 peak GFLOPS
通常假设你在用 Tensor Core

### 2️⃣ 提高 ILP（Instruction-Level Parallelism） 

让 GPU scheduler 有东西可选：  
unroll（适度）   
减少 dependency chain    
多 accumulator  
acc0 += a0 * b0;  
acc1 += a1 * b1;  
📌 不要让指令一条条“排队等前一个”

### 3️⃣ 减少“非 FLOP 指令”

Roofline 只数 FLOP，但 GPU 不这么想。

要警惕：

address calculation

integer math

type cast

atomic

sync

👉 有时你 FLOP 很高，但：

每个 FLOP 都夹着一堆杂活

## 第四层：Kernel 形态 & 粒度
### 1️⃣ Kernel 太小 = GPU 没热起来

问题表现：

launch overhead 显著

SM 利用率低

解决：

fuse kernels

batch

persistent kernel

### 2️⃣ Kernel fusion（AI infra 非常重要）

例如：

bias + activation

attention 的多个 stage 合并

📌 减少 global memory round trip = 提高计算强度

六、把这些映射回 Roofline（非常关键）

你可以这样做 诊断 → 行动：

🔍 情况 A：点在斜线下方（memory-bound）

问自己：

coalesced 了吗？

cache hit 高吗？

shared / tiling 有意义吗？

warp 数够吗？

👉 行动：访存 & 并发优化

🔍 情况 B：点在右边但离屋顶远（compute-bound 但不满）

问自己：

用 Tensor Core 了吗？

精度对了吗？

指令 dependency 多吗？

kernel 太小了吗？

👉 行动：算力路径 & ILP

七、给你一个“GPU 优化流程表”（你可以照着用）
1. 画 roofline / 算 intensity  
2. 定位 memory-bound 还是 compute-bound
3. memory-bound:
   - coalescing
   - cache
   - shared / tiling
   - occupancy
4. compute-bound:
   - Tensor Core
   - ILP
   - precision
5. 看 kernel 粒度 & fusion