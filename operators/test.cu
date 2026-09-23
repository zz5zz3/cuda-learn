#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "cuda_check.cuh"
#include "cuda_device.cuh"
#include "cuda_timer.cuh"

using namespace nvcuda;


// ============================================================
// WMMA 的基本 MMA Tile 大小
//
// 一次 WMMA MMA 从编程视角计算：
//
// A: 16 × 16
// B: 16 × 16
//
// 得到：
//
// C: 16 × 16
//
// 即：
//
// [16×16] × [16×16] → [16×16]
//
// M/N/K 这里不是整个矩阵的 M/N/K，
// 而是一次 WMMA 操作处理的 Tile 尺寸。
// ============================================================

constexpr int WMMA_M = 16;   // 一个 warp 在 M 方向计算 16 行
constexpr int WMMA_N = 16;   // 一个 warp 在 N 方向计算 16 列
constexpr int WMMA_K = 16;   // 一次 MMA 沿 K 方向计算 16 个元素


// ============================================================
// Block Tile 大小
//
// 一个 CUDA Block 负责整个 C 矩阵中的：
//
// 32 × 32
//
// 由于一个 warp 负责 16×16，
//
// 所以：
//
// 32 / 16 = 2
//
// M方向需要2个warp
// N方向需要2个warp
//
// 总共：
//
// 2 × 2 = 4 warps
//
// 4 × 32 = 128 threads
// ============================================================

constexpr int BM = 32;   // 一个 Block 在 C 的 M 方向负责 32 行
constexpr int BN = 32;   // 一个 Block 在 C 的 N 方向负责 32 列


// ============================================================
// K方向的分块大小
//
// 整个 K 可能非常大，例如：
//
// K = 1024
//
// 不可能一次全部放入 Shared Memory。
//
// 所以每次只取：
//
// BK = 16
//
// 然后沿 K 方向不断循环：
//
// 0~15
// 16~31
// 32~47
// ...
//
// 每轮都执行一次 MMA 并累加到 c_frag。
// ============================================================

constexpr int BK = 16;


// ============================================================
// WMMA GEMM V4
//
// 矩阵：
//
// A: M × K     row-major
// B: K × N     col-major
// C: M × N     row-major
//
// 计算：
//
// C = A × B
//
// 推荐 launch：
//
// block = 128 threads
//
// grid.x = ceil(N / BN)
// grid.y = ceil(M / BM)
// ============================================================

__global__ void WMMA_GEMM_V4(
    const half* A,     // 输入矩阵 A，shape = [M,K]，row-major
    const half* B,     // 输入矩阵 B，shape = [K,N]，col-major
    float* C,          // 输出矩阵 C，shape = [M,N]，row-major
    int M,             // A/C 的行数
    int N,             // B/C 的列数
    int K)             // A的列数 = B的行数，也是 GEMM 的规约维度
{

    // ========================================================
    // 1. 当前 Block 负责 C 的哪个 32×32 Tile？
    //
    // blockIdx.x / blockIdx.y 是整个 Grid 中 Block 的二维坐标。
    //
    // 例如：
    //
    // blockIdx = (0,0)
    //      ↓
    // C[0:32][0:32]
    //
    // blockIdx = (1,0)
    //      ↓
    // C[0:32][32:64]
    //
    // blockIdx = (0,1)
    //      ↓
    // C[32:64][0:32]
    //
    // 所以：
    //
    // block_row = 当前 Block Tile 的起始行
    // block_col = 当前 Block Tile 的起始列
    // ========================================================

    int block_row = blockIdx.y * BM;
    // 当前 Block 在整个 C 中的起始行
    // 例如 blockIdx.y=2：
    // block_row = 2×32 = 64

    int block_col = blockIdx.x * BN;
    // 当前 Block 在整个 C 中的起始列
    // 例如 blockIdx.x=3：
    // block_col = 3×32 = 96



    // ========================================================
    // 2. 当前 Thread 属于哪个 Warp？
    //
    // 一个 Block = 128 threads
    //
    // thread 0~31    → warp0
    // thread 32~63   → warp1
    // thread 64~95   → warp2
    // thread 96~127  → warp3
    // ========================================================

    int tid = threadIdx.x;
    // 当前线程在 Block 内的一维线程编号
    // 范围：0~127

    int warp_id = tid / 32;
    // 当前线程属于哪个 warp
    //
    // tid 0~31   → 0
    // tid 32~63  → 1
    // tid 64~95  → 2
    // tid 96~127 → 3



    // ========================================================
    // 3. 把一维 warp_id 转换成二维 Warp Tile 坐标
    //
    // 当前一个 Block Tile = 32×32
    //
    // 一个 Warp Tile = 16×16
    //
    // 所以：
    //
    //             N方向
    //
    //            0       1
    //        ┌───────┬───────┐
    //   0    │warp0  │warp1  │
    // M方向  ├───────┼───────┤
    //   1    │warp2  │warp3  │
    //        └───────┴───────┘
    //
    // 一行有：
    //
    // BN / WMMA_N
    // = 32 / 16
    // = 2
    //
    // 个 Warp Tile。
    // ========================================================

    constexpr int tiles_num_row = BN / WMMA_N;
    // 当前 Block Tile 中，一行有多少个 Warp Tile
    // 当前 = 32/16 = 2

    int warp_m = warp_id / tiles_num_row;
    // 当前 warp 位于 Block Tile 的第几行
    //
    // warp0 → 0
    // warp1 → 0
    // warp2 → 1
    // warp3 → 1

    int warp_n = warp_id % tiles_num_row;
    // 当前 warp 位于 Block Tile 的第几列
    //
    // warp0 → 0
    // warp1 → 1
    // warp2 → 0
    // warp3 → 1



    // ========================================================
    // 4. Shared Memory
    //
    // 注意：
    //
    // Shared Memory 属于整个 Block。
    //
    // 128 个线程都可以访问同一份 As / Bs。
    //
    //
    // 每轮 K-loop 需要：
    //
    // A Tile：
    //
    //      BM × BK
    //      32 × 16
    //
    //
    // B Tile：
    //
    //      BK × BN
    //      16 × 32
    //
    //
    // 但是为了让 B 保持 col-major：
    //
    // Bs[col][k]
    //
    // 所以物理声明为：
    //
    // Bs[32][16]
    // ========================================================

    __shared__ half As[BM][BK];
    // Block 共享的 A Tile
    //
    // shape = [32][16]
    //
    // 第一维：M方向
    // 第二维：K方向

    __shared__ half Bs[BN][BK];
    // Block 共享的 B Tile
    //
    // shape = [32][16]
    //
    // 第一维：N方向，也就是 B 的列
    // 第二维：K方向
    //
    // 相当于按 col-major 方式组织 B



    // ========================================================
    // 5. 创建 WMMA Fragment
    //
    // Fragment 可以理解成：
    //
    // 一个 warp 共同持有的、分布在32个线程寄存器中的矩阵 Tile。
    // ========================================================

    wmma::fragment<
        wmma::matrix_a,     // 这是 MMA 左边的矩阵 A
        WMMA_M,             // M = 16
        WMMA_N,             // N = 16
        WMMA_K,             // K = 16
        half,               // A 使用 FP16
        wmma::row_major     // A 按 row-major 解释
    > a_frag;
    // 当前 warp 使用的 A fragment
    // 逻辑上对应一个 16×16 A Tile


    wmma::fragment<
        wmma::matrix_b,     // 这是 MMA 右边的矩阵 B
        WMMA_M,             // M = 16
        WMMA_N,             // N = 16
        WMMA_K,             // K = 16
        half,               // B 使用 FP16
        wmma::col_major     // B 按 col-major 解释
    > b_frag;
    // 当前 warp 使用的 B fragment
    // 逻辑上对应一个 16×16 B Tile


    wmma::fragment<
        wmma::accumulator,  // MMA 的累加器 C
        WMMA_M,             // 输出 M = 16
        WMMA_N,             // 输出 N = 16
        WMMA_K,             // MMA 的 K = 16
        float               // 使用 FP32 累加
    > c_frag;
    // 当前 warp 的结果 fragment
    //
    // 逻辑 shape = 16×16
    //
    // K-loop 的每一轮都会继续往这里累加



    wmma::fill_fragment(c_frag, 0.0f);
    // 把当前 warp 的整个 c_frag 初始化成 0
    //
    // 后面实现：
    //
    // c_frag += A_frag × B_frag



    // ========================================================
    // 6. K Loop
    //
    // 这是 GEMM 最核心的循环。
    //
    // 假设：
    //
    // K = 64
    //
    // BK = 16
    //
    // 那么：
    //
    // k0 = 0
    // k0 = 16
    // k0 = 32
    // k0 = 48
    //
    // 总共循环 4 次。
    //
    //
    // 每一轮：
    //
    // Global
    //    ↓
    // Shared
    //    ↓
    // Fragment
    //    ↓
    // Tensor Core MMA
    //    ↓
    // c_frag 累加
    //
    // ========================================================

    for (int k0 = 0; k0 < K; k0 += BK)
    // k0：
    // 当前正在处理的 K Tile 起始位置
    //
    // 每轮向前移动 BK=16
    {

        // ====================================================
        // 7. A：Global Memory → Shared Memory
        //
        // 当前 Block 需要：
        //
        // A[
        //   block_row : block_row + 32,
        //   k0        : k0 + 16
        // ]
        //
        // 即：
        //
        // 32 × 16 = 512 个 half
        //
        // 当前 Block 有128个线程。
        //
        // 所以平均：
        //
        // 512 / 128 = 4
        //
        // 每个线程搬4个元素。
        // ====================================================

        for (
            int i = tid;             // 每个 thread 从自己的 tid 开始
            i < BM * BK;             // 总共需要搬 32×16=512 个元素
            i += blockDim.x)         // 每次跨过整个 Block 的线程数=128
        //
        // 例如 thread0：
        //
        // i = 0
        // i = 128
        // i = 256
        // i = 384
        //
        // thread1：
        //
        // i = 1
        // i = 129
        // i = 257
        // i = 385
        //
        // ...
        //
        // 128个线程共同覆盖全部512个元素
        {
            int local_row = i / BK;
            // 把一维 i 转换成 As 的行号
            //
            // 范围：
            // 0~31

            int local_k = i % BK;
            // 把一维 i 转换成 As 的列号/K方向位置
            //
            // 范围：
            // 0~15


            int global_row = block_row + local_row;
            // 当前 As 元素对应到整个 A 的哪一行

            int global_k = k0 + local_k;
            // 当前 As 元素对应到整个 A 的哪个 K 位置


            if (global_row < M && global_k < K)
            // 防止矩阵边界越界
            {
                As[local_row][local_k] =
                    A[global_row * K + global_k];
                // A 是 row-major
                //
                // A[row][col]
                // =
                // A[row*K + col]
            }
            else
            {
                As[local_row][local_k] =
                    __float2half(0.0f);
                // 超出矩阵边界的部分补0
            }
        }



        // ====================================================
        // 8. B：Global Memory → Shared Memory
        //
        // 当前 Block 需要：
        //
        // B[
        //   k0        : k0+16,
        //   block_col : block_col+32
        // ]
        //
        // 数学 shape：
        //
        // 16 × 32
        //
        // 仍然一共：
        //
        // 512 个 half
        //
        // 128 threads
        //
        // 每个 thread 搬4个。
        // ====================================================

        for (
            int i = tid;             // 每个线程从自己的 tid 开始
            i < BN * BK;             // 32×16 = 512 个元素
            i += blockDim.x)         // 每次跨128
        {
            int local_col = i / BK;
            // 当前元素属于 B Tile 的哪一列
            //
            // 范围：
            // 0~31

            int local_k = i % BK;
            // 当前元素在 K 方向的位置
            //
            // 范围：
            // 0~15


            int global_col = block_col + local_col;
            // 映射到整个 B 的列

            int global_k = k0 + local_k;
            // 映射到整个 B 的行/K维度


            if (global_col < N && global_k < K)
            {
                Bs[local_col][local_k] =
                    B[global_col * K + global_k];

                // B 是 col-major
                //
                // B[row][col]
                // =
                // B[col*K + row]
                //
                // 这里：
                //
                // row = global_k
                // col = global_col
            }
            else
            {
                Bs[local_col][local_k] =
                    __float2half(0.0f);
                // 超出边界补0
            }
        }



        // ====================================================
        // 9. Block同步
        //
        // 前面的 Global → Shared 是128个线程共同完成的。
        //
        // 必须保证：
        //
        // As / Bs 全部搬完
        //
        // 才允许任何 warp 开始读取。
        // ====================================================

        __syncthreads();



        // ====================================================
        // 10. 每个 Warp 从 Shared Memory 中选择自己的 A Tile
        //
        //
        // As：
        //
        //        K=16
        //      ┌─────────┐
        //      │         │
        // 16   │   A0    │ ← warp0 / warp1 使用
        //      │         │
        //      ├─────────┤
        //      │         │
        // 16   │   A1    │ ← warp2 / warp3 使用
        //      │         │
        //      └─────────┘
        //
        // ====================================================

        const half* A_tile =
            &As[warp_m * WMMA_M][0];

        // warp_m=0：
        //
        // A_tile = &As[0][0]
        //
        // warp0 / warp1 使用 As 上半部分
        //
        //
        // warp_m=1：
        //
        // A_tile = &As[16][0]
        //
        // warp2 / warp3 使用 As 下半部分



        // ====================================================
        // 每个 Warp 从 Shared Memory 中选择自己的 B Tile
        //
        //
        // B：
        //
        //         N方向
        //
        //       B0        B1
        //
        //     16列      16列
        //
        // warp0/2     warp1/3
        //
        // ====================================================

        const half* B_tile =
            &Bs[warp_n * WMMA_N][0];

        // warp_n=0：
        //
        // B_tile = &Bs[0][0]
        //
        // warp0 / warp2 使用
        //
        //
        // warp_n=1：
        //
        // B_tile = &Bs[16][0]
        //
        // warp1 / warp3 使用



        // ====================================================
        // 11. Shared Memory → A Fragment
        //
        // A_tile：
        //
        // Shared Memory 中的起始地址
        //
        // BK=16：
        //
        // Shared Memory 中相邻两行之间间隔16个half
        //
        // 因此 leading dimension = 16
        // ====================================================

        wmma::load_matrix_sync(
            a_frag,       // 目标：A fragment
            A_tile,       // 来源：Shared Memory
            BK            // leading dimension = 16
        );



        // ====================================================
        // 12. Shared Memory → B Fragment
        //
        // Bs[col][k]
        //
        // 每一列包含16个连续的 K 元素
        //
        // 所以 leading dimension 同样 = 16
        // ====================================================

        wmma::load_matrix_sync(
            b_frag,       // 目标：B fragment
            B_tile,       // 来源：Shared Memory
            BK            // leading dimension = 16
        );



        // ====================================================
        // 13. Tensor Core MMA
        //
        // 当前 Warp 共同执行：
        //
        // c_frag =
        //
        //      a_frag × b_frag
        //          +
        //      c_frag
        //
        //
        // 第一次 K-loop：
        //
        // C = A0B0
        //
        // 第二次：
        //
        // C = A0B0 + A1B1
        //
        // 第三次：
        //
        // C = A0B0 + A1B1 + A2B2
        //
        // ...
        //
        // ====================================================

        wmma::mma_sync(
            c_frag,       // 输出
            a_frag,       // A
            b_frag,       // B
            c_frag        // 原来的累加结果
        );



        // ====================================================
        // 14. 第二次同步
        //
        // 当前 As/Bs 马上要在下一轮 k0 中被覆盖。
        //
        // 所以必须保证：
        //
        // warp0
        // warp1
        // warp2
        // warp3
        //
        // 全部已经使用完当前 As/Bs。
        //
        // 才能允许下一轮 Global → Shared 覆盖数据。
        // ====================================================

        __syncthreads();
    }



    // ========================================================
    // 15. K-loop 全部完成
    //
    // 此时每个 warp 的 c_frag 中已经保存：
    //
    // 完整的 16×16 C Warp Tile。
    //
    // 接下来计算这个 Tile 在整个 C 中的位置。
    // ========================================================

    int c_row =
        block_row + warp_m * WMMA_M;

    // 当前 Warp Tile 在整个 C 中的起始行
    //
    // =
    //
    // 当前 Block 起始行
    // +
    // 当前 warp 在 Block 内的行偏移



    int c_col =
        block_col + warp_n * WMMA_N;

    // 当前 Warp Tile 在整个 C 中的起始列
    //
    // =
    //
    // 当前 Block 起始列
    // +
    // 当前 warp 在 Block 内的列偏移



    // ========================================================
    // 16. 边界检查
    //
    // 当前版本只在完整 16×16 Tile 能放进 C 时 store。
    //
    // 例如：
    //
    // c_row=48
    // c_col=32
    //
    // 那么要写：
    //
    // C[48:64][32:48]
    //
    // 必须保证没有超出 M/N。
    // ========================================================

    if (
        c_row + WMMA_M <= M &&
        c_col + WMMA_N <= N
    )
    {

        float* C_tile =
            C + c_row * N + c_col;

        // C 是 row-major
        //
        // C[row][col]
        // =
        // C[row*N + col]
        //
        // C_tile 指向当前 Warp Tile 的左上角



        // ====================================================
        // 17. Fragment → Global Memory
        //
        // 把当前 warp 的 16×16 c_frag
        //
        // 写入：
        //
        // C[c_row:c_row+16]
        //  [c_col:c_col+16]
        //
        // C 是 row-major
        //
        // 相邻两行间隔 N 个 float
        // ====================================================

        wmma::store_matrix_sync(
            C_tile,                // 写到哪里
            c_frag,                // 写什么
            N,                     // C 的 leading dimension
            wmma::mem_row_major    // C 按 row-major 写入
        );
    }
}