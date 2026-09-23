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




__global__ void WMMA_GEMM( const half* A, const half* B, float* C,int M, int N,int K)
{

    int tid = threadIdx.x;

    int warp_id = threadIdx.x / 32;
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    
    constexpr int WMMA_M = 16;    // 一个 warp 在 M 方向计算 16 行
    constexpr int WMMA_N = 16;    // 一个 warp 在 N 方向计算 16 列
    constexpr int WMMA_K = 16;    // 一次 MMA 沿 K 方向计算 16 个元素

    int BM = 32;    // 一个 Block 在 C 的 M 方向负责 32 行
    int BN = 32;    // 一个 Block 在 C 的 N 方向负责 32 列


    int BK = 16;    // K 每轮 搬运 16


    int tiles_num_row = BN / WMMA_N;  //// 当前 Block Tile 中，一行有多少个 Warp Tile


    int warp_m = warp_id / tiles_num_row;// 当前 warp 位于 Block Tile 的第几行
    int warp_n = warp_id % tiles_num_row;// 当前 warp 位于 Block Tile 的第几列


    __shared__ half As[32][16];
    __shared__ half Bs[32][16];
    
    int row = id / N;
    int col = id % N;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,half,  wmma::row_major > a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,half,  wmma::col_major > b_frag;
    wmma::fragment<wmma::accumulator,WMMA_M, WMMA_N, WMMA_K,float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int block_row = blockIdx.y * BM;  // 当前 Block 在整个 C 中的起始行
    int block_col = blockIdx.x * BN;  // 当前 Block 在整个 C 中的起始列


    for(int k = 0;k<K;k=+BK){
        for(int i = tid ;i<BM * BK ; i+=blockDim.x){  //每个线程搬运多个呗
            int local_row = i / BK; // 把一维 i 转换成 As 的行号
            int local_k = i % BK;   // 把一维 i 转换成 As 的列号/K方向位置     
            int global_row = block_row + local_row; // 当前 As 元素对应到整个 A 的哪一行
            int global_k = k + local_k;          // 当前 As 元素对应到整个 A 的哪个 K 位置
    
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

            int global_k = k + local_k;
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
        __syncthreads();

        const half* A_tile =
            &As[warp_m * WMMA_M][0];

        const half* B_tile =
            &Bs[warp_n * WMMA_N][0];


        wmma::load_matrix_sync(
            a_frag,       // 目标：A fragment
            A_tile,       // 来源：Shared Memory
            BK            // leading dimension = 16
        );
        wmma::load_matrix_sync(
            b_frag,       // 目标：B fragment
            B_tile,       // 来源：Shared Memory
            BK            // leading dimension = 16
        );
        wmma::mma_sync(
            c_frag,       // 输出
            a_frag,       // A
            b_frag,       // B
            c_frag        // 原来的累加结果
        );
        __syncthreads();
    int c_row =
        block_row + warp_m * WMMA_M;

    int c_col =
        block_col + warp_n * WMMA_N;

    if (c_row + WMMA_M <= M && c_col + WMMA_N <= N  )
    {
        float* C_tile =
            C + c_row * N + c_col;
    
        wmma::store_matrix_sync(
            C_tile,                // 写到哪里
            c_frag,                // 写什么
            N,                     // C 的 leading dimension
            wmma::mem_row_major    // C 按 row-major 写入
        );

    }
    }
}
// ============================================================
// CPU Reference GEMM
//
// A : row-major
// B : col-major
// C : row-major
// ============================================================

void cpu_gemm(const half* A,const half* B,float* C,int M,int N,int K)
{
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                float a =
                    __half2float(A[row * K + k]);

                // B 是 col-major
                float b =
                    __half2float(B[col * K + k]);

                sum += a * b;
            }

            C[row * N + col] = sum;
        }
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 64;

    constexpr int MMA_K = 16;


    // ========================================================
    // 1. Host Memory
    // ========================================================

    half* h_A = new half[M * K];
    half* h_B = new half[K * N];

    float* h_C     = new float[M * N];
    float* h_C_ref = new float[M * N];


    // ========================================================
    // 2. 初始化 A
    //
    // A = [ I | I | I | I ]
    //
    // shape:
    //
    //              K = 64
    //
    //       16      16      16      16
    //
    //     ┌──────┬──────┬──────┬──────┐
    // A = │  I   │  I   │  I   │  I   │
    //     └──────┴──────┴──────┴──────┘
    //
    //              M = 16
    //
    //
    // 这样可以专门验证：
    //
    // C = B0 + B1 + B2 + B3
    //
    // 而不是只有第一次 MMA 有效。
    // ========================================================

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < K; ++col)
        {
            float value =
                (row == (col % MMA_K))
                ? 1.0f
                : 0.0f;

            h_A[row * K + col] =
                __float2half(value);
        }
    }


    // ========================================================
    // 3. 初始化 B
    //
    // 数学上的 B：
    //
    // 1      2      3      ...   16
    // 17     18     19     ...   32
    // 33     34     35     ...   48
    // ...
    //
    // shape:
    //
    //      64 x 16
    //
    //
    // 但 WMMA 的 b_frag 声明为：
    //
    //      wmma::col_major
    //
    // 所以内存必须按照 column-major 保存：
    //
    //      B[row,col] -> B[col*K + row]
    //
    // ========================================================

    for (int row = 0; row < K; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float value =
                static_cast<float>(row * N + col + 1);

            h_B[col * K + row] =
                __float2half(value);
        }
    }


    // ========================================================
    // 4. CPU Reference
    // ========================================================

    cpu_gemm(
        h_A,
        h_B,
        h_C_ref,
        M,
        N,
        K
    );


    // ========================================================
    // 5. Device Memory
    // ========================================================

    half* d_A = nullptr;
    half* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            M * K * sizeof(half)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            K * N * sizeof(half)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            M * N * sizeof(float)
        )
    );


    // ========================================================
    // 6. Host -> Device
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A,
            M * K * sizeof(half),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B,
            K * N * sizeof(half),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // 7. Launch WMMA Kernel
    //
    // 一个 block
    // 一个 warp
    //
    // 32 threads = 1 warp
    //
    // 这个 warp：
    //
    //   负责一个 16x16 C tile
    //
    // 并沿 K：
    //
    //   执行 4 次 MMA
    //
    // ========================================================

    WMMA_GEMM<<<1, 32>>>(
        d_A,
        d_B,
        d_C,
        M,N,K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // ========================================================
    // 8. Device -> Host
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            h_C,
            d_C,
            M * N * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // 9. Correctness Check
    // ========================================================

    float max_error = 0.0f;

    for (int i = 0; i < M * N; ++i)
    {
        float error =
            std::fabs(h_C[i] - h_C_ref[i]);

        if (error > max_error)
        {
            max_error = error;
        }
    }


    printf("\n");
    printf("========================================\n");
    printf("WMMA GEMM\n");
    printf("========================================\n");

    printf(
        "Matrix Shape : (%d x %d) x (%d x %d)\n",
        M, K,
        K, N
    );

    printf(
        "MMA Shape    : 16 x 16 x 16\n"
    );

    printf(
        "K Tiles      : %d\n",
        K / MMA_K
    );

    printf(
        "Max Error    : %.8f\n",
        max_error
    );

    if (max_error < 1e-3f)
    {
        printf("Result       : PASSED\n");
    }
    else
    {
        printf("Result       : FAILED\n");
    }

    printf("========================================\n\n");


    // ========================================================
    // 10. 打印 GPU C
    // ========================================================

    printf("GPU Result C:\n\n");

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            printf(
                "%8.1f ",
                h_C[row * N + col]
            );
        }

        printf("\n");
    }


    // ========================================================
    // 11. Release
    // ========================================================

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return 0;
}