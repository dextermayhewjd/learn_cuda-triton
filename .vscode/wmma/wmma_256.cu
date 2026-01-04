// nvcc -arch=sm_70 wmma_256.cu -o wmma_256
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// 一个 block 里放多少个 warp（2x2 = 4 warps）
constexpr int WARPS_PER_BLOCK_M = 2;
constexpr int WARPS_PER_BLOCK_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N; // 4
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;                // 128

// A: row_major, B: col_major, C: row_major
__global__ void wmma_gemm_256(const half *__restrict__ A,
                              const half *__restrict__ B,
                              float *__restrict__ C,
                              int M, int N, int K,
                              int lda, int ldb, int ldc)
{
  // ---------------------------
  // 1) 找到 “我这个线程属于哪个 warp”
  // ---------------------------
  int tid = threadIdx.x; // 0..127
  int warpId = tid / 32; // 0..3
  int laneId = tid % 32; // 0..31  (通常不用手动用 laneId)

  // 这个 warp 在 block 内的 (warp_row, warp_col)
  int warp_row = warpId / WARPS_PER_BLOCK_N; // 0..1
  int warp_col = warpId % WARPS_PER_BLOCK_N; // 0..1

  // ---------------------------
  // 2) 这个 warp 负责的 C tile 坐标（以 16 为单位）
  // ---------------------------
  // blockIdx.y 控制 M 方向（行块），blockIdx.x 控制 N 方向（列块）
  int tile_m = (blockIdx.y * WARPS_PER_BLOCK_M + warp_row); // 0..15
  int tile_n = (blockIdx.x * WARPS_PER_BLOCK_N + warp_col); // 0..15

  int row = tile_m * WMMA_M; // C 子块左上角 row
  int col = tile_n * WMMA_N; // C 子块左上角 col

  // 越界保护（虽然 256x256 正好整除，这里写出来更通用）
  if (row >= M || col >= N)
    return;

  // ---------------------------
  // 3) 声明 fragments（注意：fragment 的“数据分布在整个 warp 的 32 个线程里”）
  // ---------------------------
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  // ---------------------------
  // 4) 沿 K 维度循环：每次处理 K=16 的一段
  //    C_tile += A_tile(16x16) * B_tile(16x16)
  // ---------------------------
  // A_tile 起点：A[row, k0]
  // B_tile 起点：B[k0, col]
  for (int k0 = 0; k0 < K; k0 += WMMA_K)
  {

    const half *A_tile_ptr = A + row * lda + k0; // row_major: row*lda + col
    const half *B_tile_ptr = B + k0 * ldb + col; // col_major: row*ldb + col (这里 row=k0)

    // ★关键点：load_matrix_sync 是 “warp collective”
    // 也就是：warp 内 32 线程一起执行这条指令，
    // 硬件会让每个 lane 去加载自己负责的那一小部分数据，最后拼成 fragment。
    wmma::load_matrix_sync(a_frag, A_tile_ptr, lda);
    wmma::load_matrix_sync(b_frag, B_tile_ptr, ldb);

    // ★关键点：mma_sync 也是 warp collective
    // warp 内一起在 Tensor Core 上做：c_frag += a_frag * b_frag
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // ---------------------------
  // 5) 写回 C：store_matrix_sync 也是 warp collective
  // ---------------------------
  float *C_tile_ptr = C + row * ldc + col;
  wmma::store_matrix_sync(C_tile_ptr, c_frag, ldc, wmma::mem_row_major);
}
