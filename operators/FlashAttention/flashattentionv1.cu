#include<stdio.h>
#include"cuda_check.cuh"
#include"cuda_device.cuh"
#include"cuda_timer.cuh"
#include <time.h>



__device__ float ReduceSum(float val){
    uint mask = 0xFFFFFFFF;
    for(int offset=16;offset>0;offset/=2){
        val += __shfl_down_sync(mask,val,offset);
    }
    return val;
}

__device__ float ReduceMax(float val){
    uint mask = 0xFFFFFFFF;
    for(int offset=16;offset>0;offset/=2){
        val = fmaxf(__shfl_down_sync(mask,val,offset),val);
    }
    return val;
}

__global__ void flashattention(float *Q,float *K,float* V,float* O,int N,int d){
    extern __shared__ float shared[];
    // 教学版：
    // 一个 block 负责一个 Query row
    // 一个 warp 处理一个 K/V tile
    // 因此这里建议 blockDim.x = 32

    int row = blockIdx.x;
    int lane = threadIdx.x;
    int BK =32;
    
    int warpnum = (blockDim.x +31)/32;

    float m = -INFINITY;  //   当前已经处理过的所有 score 的最大值
    float l = 0.0f;    //   当前已经处理过的 softmax 分母

    constexpr  int thread_max = 32;
    float acc[thread_max];
    int count = 0;
    //acc acc就相当于O的分子
    for(int dim = lane; dim<d&& count < thread_max;dim +=32){
        acc[count ++] = 0.0f; 
    }

    for(int tile_start = 0;tile_start < N;tile_start += BK){
        int j = tile_start + lane;
        float score = -INFINITY;
        if(j<N){
            score  = 0.0f;
            for(int k = 0 ; k <d ; ++k){
                score +=Q[row *d + k] * K[j *d +k];
            }
        }
        score = score / sqrtf(d);

        float tile_max = ReduceMax(score);
        tile_max = __shfl_sync(0xffffffff , tile_max , 0);

        float m_new = fmaxf(m,tile_max);
        float alpha = expf(m - m_new);

        float p = 0.0f;
        if(j < N){
            p  = expf(score - m_new);   //p是要加进来的当前尺度的分子
        }

        float tile_sum = ReduceSum(p);
        tile_sum = __shfl_sync(0xffffffff , tile_sum , 0); 
        
        
        float l_new = l * alpha +tile_sum; //是我之前的l乘上现在的尺度变换成现在的尺度，然后加上现在尺度的sum

        int idx = 0;
        for(int dim =lane ; dim<d  && idx <count ; dim +=32){
            float new_acc =acc[idx] *alpha;
            for(int t = 0;t<BK;t++){
                int key_index = tile_start + t;
                if(key_index < N){
                    float s =0.0f;
                    for(int k = 0;k<d;k++){
                        s +=Q[row * d + k] *K[key_index * d + k];
                    }
                    s /=sqrtf(d);

                    float prob = expf(s -m_new);
                    new_acc += prob * V[key_index * d +dim];
                    }
                }
                acc[idx] = new_acc;
                ++idx;
            }
            m = m_new;
            l = l_new;
        }
        int idx = 0;

        for(int dim =lane; dim<d && idx<count ;dim+=32){
            O[row *d +dim] = acc[idx] /l;
            ++idx;
        }
    }

void cpuAttention(
    float* Q,
    float* K,
    float* V,
    float* O,
    int N,
    int d)
{
    float* S =
        (float*)malloc(
            N * N * sizeof(float));

    float* P =
        (float*)malloc(
            N * N * sizeof(float));


    // --------------------------------
    // Q K^T / sqrt(d)
    // --------------------------------

    float scale =
        1.0f / sqrtf((float)d);

    for(int row = 0; row < N; row++){

        for(int col = 0; col < N; col++){

            float sum = 0.0f;

            for(int k = 0; k < d; k++){

                sum +=
                    Q[row*d+k]
                    * K[col*d+k];
            }

            S[row*N+col]
                = sum * scale;
        }
    }
    // --------------------------------
    // Softmax
    // --------------------------------

    for(int row = 0; row < N; row++){

        float max_val = -INFINITY;

        for(int col = 0; col < N; col++){

            max_val =
                fmaxf(
                    max_val,
                    S[row*N+col]);
        }


        float sum = 0.0f;

        for(int col = 0; col < N; col++){

            float e =
                expf(
                    S[row*N+col]
                    - max_val);

            P[row*N+col] = e;

            sum += e;
        }


        for(int col = 0; col < N; col++){

            P[row*N+col] /= sum;
        }
    }


    // --------------------------------
    // P V
    // --------------------------------

    for(int row = 0; row < N; row++){

        for(int col = 0; col < d; col++){

            float sum = 0.0f;

            for(int k = 0; k < N; k++){

                sum +=
                    P[row*N+k]
                    * V[k*d+col];
            }

            O[row*d+col] = sum;
        }
    }


    free(S);
    free(P);
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
    set_cuda_device(0);


    // ========================================================
    // 0. 参数
    // ========================================================

    int N = 256;
    int d = 64;


    size_t qkv_bytes =
        N * d * sizeof(float);

    size_t sp_bytes =
        N * N * sizeof(float);

    size_t o_bytes =
        N * d * sizeof(float);


    // ========================================================
    // 1. Host
    // ========================================================

    float* hostQ =
        (float*)malloc(qkv_bytes);

    float* hostK =
        (float*)malloc(qkv_bytes);

    float* hostV =
        (float*)malloc(qkv_bytes);

    float* hostO =
        (float*)malloc(o_bytes);

    float* hostRef =
        (float*)malloc(o_bytes);


    initialData(hostQ, N*d);
    initialData(hostK, N*d);
    initialData(hostV, N*d);

    memset(hostO, 0, o_bytes);
    memset(hostRef, 0, o_bytes);


    // ========================================================
    // 2. Device
    // ========================================================

    float* cudaQ;
    float* cudaK;
    float* cudaV;

    float* cudaS;
    float* cudaP;

    float* cudaO;


    CUDA_CHECK(
        cudaMalloc(&cudaQ, qkv_bytes));

    CUDA_CHECK(
        cudaMalloc(&cudaK, qkv_bytes));

    CUDA_CHECK(
        cudaMalloc(&cudaV, qkv_bytes));

    CUDA_CHECK(
        cudaMalloc(&cudaS, sp_bytes));

    CUDA_CHECK(
        cudaMalloc(&cudaP, sp_bytes));

    CUDA_CHECK(
        cudaMalloc(&cudaO, o_bytes));


    CUDA_CHECK(
        cudaMemcpy(
            cudaQ,
            hostQ,
            qkv_bytes,
            cudaMemcpyHostToDevice));

    CUDA_CHECK(
        cudaMemcpy(
            cudaK,
            hostK,
            qkv_bytes,
            cudaMemcpyHostToDevice));

    CUDA_CHECK(
        cudaMemcpy(
            cudaV,
            hostV,
            qkv_bytes,
            cudaMemcpyHostToDevice));


    // ========================================================
    // 3. GPU Attention
    // ========================================================

    CudaTimer cudatimer;


    // ---------------- QK ----------------

    dim3 blockQK(16, 16);

    dim3 gridQK(
        (N + blockQK.x - 1)
            / blockQK.x,

        (N + blockQK.y - 1)
            / blockQK.y);


    // ---------------- Softmax ----------------
    //
    // 当前版本直接一个线程处理一个元素
    //

    dim3 blockSoftmax(N);
    dim3 gridSoftmax(N);


    // ---------------- PV ----------------

    dim3 blockPV(16, 16);

    dim3 gridPV(
        (d + blockPV.x - 1)
            / blockPV.x,

        (N + blockPV.y - 1)
            / blockPV.y);


    cudatimer.start();


dim3 block(32);
dim3 grid(N);

flashattention<<<grid, block>>>(
    cudaQ,
    cudaK,
    cudaV,
    cudaO,
    N,
    d
);

    float elapsed_ms =
        cudatimer.stop();


    CUDA_CHECK(
        cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaGetLastError());


    CUDA_CHECK(
        cudaMemcpy(
            hostO,
            cudaO,
            o_bytes,
            cudaMemcpyDeviceToHost));


    // ========================================================
    // 4. CPU Reference
    // ========================================================

    clock_t start = clock();


    cpuAttention(
        hostQ,
        hostK,
        hostV,
        hostRef,
        N,
        d);


    clock_t stop = clock();


    double cpu_time =
        (double)(stop-start)
        * 1000.0
        / CLOCKS_PER_SEC;


    // ========================================================
    // 5. Verify
    // ========================================================

    float max_error = 0.0f;

    for(int i = 0; i < N*d; i++){

        float error =
            fabsf(
                hostO[i]
                - hostRef[i]);

        max_error =
            fmaxf(
                max_error,
                error);
    }


    printf(
        "Max Error: %.8f\n",
        max_error);


    if(max_error < 1e-4f){

        printf("PASSED\n");
    }
    else{

        printf("FAILED\n");
    }


    printf("\nFirst 10 outputs:\n");

    for(int i = 0; i < 10; i++){

        printf(
            "GPU: %f   CPU: %f\n",
            hostO[i],
            hostRef[i]);
    }


    printf(
        "\nGPU time: %.4f ms\n",
        elapsed_ms);

    printf(
        "CPU time: %.4f ms\n",
        cpu_time);


    // ========================================================
    // 6. Free
    // ========================================================

    CUDA_CHECK(cudaFree(cudaQ));
    CUDA_CHECK(cudaFree(cudaK));
    CUDA_CHECK(cudaFree(cudaV));

    CUDA_CHECK(cudaFree(cudaS));
    CUDA_CHECK(cudaFree(cudaP));
    CUDA_CHECK(cudaFree(cudaO));


    free(hostQ);
    free(hostK);
    free(hostV);

    free(hostO);
    free(hostRef);


    return 0;
}