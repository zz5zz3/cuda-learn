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

    int warp_id = threadIdx.x / 32;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    int tiles_num_row = N / WMMA_N;
    int tile_m = warp_id / tiles_num_row;
    int tile_n = warp_id % tiles_num_row;

    
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,half,  wmma::row_major > a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,half,  wmma::col_major > b_frag;
    wmma::fragment<wmma::accumulator,WMMA_M, WMMA_N, WMMA_K,float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K)
    {
        const half* A_tile = A + tile_m * WMMA_M * K +k;
        const half* B_tile = B + tile_n * WMMA_N * K +k; 
        wmma::load_matrix_sync(a_frag,A_tile, K );
        wmma::load_matrix_sync(b_frag,B_tile, K );


        // ----------------------------------------------------
        // c_frag = a_frag * b_frag + c_frag
        //
        // 每轮都会继续累加到同一个 accumulator。
        // ----------------------------------------------------

        wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);
    

    // ========================================================
    // Store C
    //
    // C shape = 16 x 16
    // C layout = row-major
    //
    // leading dimension = N = 16
    //
    // 注意：
    // 虽然 K=64，
    // 但 C 的一行仍然只有 16 个元素。
    // ========================================================
    float* C_tile = C + tile_m * N * WMMA_N + tile_n * WMMA_M;
    wmma::store_matrix_sync(C_tile, c_frag,N, wmma::mem_row_major );
    }
}


// ============================================================
// CPU Reference GEMM
//
// A : row-major
// B : col-major
// C : row-major
// ============================================================

void cpu_gemm(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K)
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