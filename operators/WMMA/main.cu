#include<stdio.h>
#include"cuda_check.cuh"
#include"cuda_device.cuh"
#include"cuda_timer.cuh"
#include <time.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;


__global__ void WMMA_GEMM(half* A,half* B,float* C){
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,float> c_frag;
    wmma::fill_fragment(c_frag,0.0f);
    wmma::load_matrix_sync(a_frag,A,16);
    wmma::load_matrix_sync(b_frag,B,16);
    wmma::mma_sync(c_frag, a_frag, b_frag,c_frag);  //dabc
    wmma::store_matrix_sync(C,c_frag,16,wmma::mem_row_major);

}

void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

int main()
{
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;


    // ========================================================
    // 1. Host Memory
    // ========================================================

    half* h_A = new half[M * K];
    half* h_B = new half[K * N];
    float* h_C = new float[M * N];


    // ========================================================
    // 2. 初始化 A
    //
    // A = 单位矩阵 I
    //
    // 1 0 0 ...
    // 0 1 0 ...
    // 0 0 1 ...
    // ...
    //
    // A 是 row-major
    // ========================================================

    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < K; col++)
        {
            float value = (row == col) ? 1.0f : 0.0f;

            h_A[row * K + col] = __float2half(value);
        }
    }


    // ========================================================
    // 3. 初始化 B
    //
    // 数学上的 B：
    //
    //  1   2   3  ...  16
    // 17  18  19  ...  32
    // 33  34  35  ...  48
    // ...
    //
    // 但是 b_frag 声明的是 col_major
    //
    // 所以内存索引：
    //
    // B[col * K + row]
    // ========================================================

    for (int row = 0; row < K; row++)
    {
        for (int col = 0; col < N; col++)
        {
            float value = row * N + col + 1;

            h_B[col * K + row] = __float2half(value);
        }
    }


    // ========================================================
    // 4. Device Memory
    // ========================================================

    half* d_A;
    half* d_B;
    float* d_C;

    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));


    // ========================================================
    // 5. Host -> Device
    // ========================================================

    cudaMemcpy(
        d_A,
        h_A,
        M * K * sizeof(half),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B,
        K * N * sizeof(half),
        cudaMemcpyHostToDevice
    );


    // ========================================================
    // 6. 启动 WMMA Kernel
    //
    // 1 block
    // 32 threads
    // = 1 warp
    //
    // 一个 warp 完成一个 16x16x16 MMA
    // ========================================================

    WMMA_GEMM<<<1, 32>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();


    // ========================================================
    // 7. Device -> Host
    // ========================================================

    cudaMemcpy(
        h_C,
        d_C,
        M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    );


    // ========================================================
    // 8. 打印 C
    //
    // 因为：
    //
    // C = A * B
    // A = I
    //
    // 所以：
    //
    // C = B
    // ========================================================

    printf("WMMA Result C:\n\n");

    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            printf("%6.1f ", h_C[row * N + col]);
        }

        printf("\n");
    }


    // ========================================================
    // 9. 释放内存
    // ========================================================

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}