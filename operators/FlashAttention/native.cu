#include<stdio.h>
#include"cuda_check.cuh"
#include"cuda_device.cuh"
#include"cuda_timer.cuh"
#include <time.h>



__device__ float ReduceSum(float val){
    uint mask = 0xFFFFFFFF;
    for(int offset=16;offset>0;offset/=2){
        val = __shfl_down_sync(mask,val,offset);
    }
    return val;
}

__device__ float ReduceMax(float val){
    uint mask = 0xFFFFFFFF;
    for(int offset=16;offset>0;offset/=2){
        val += max(__shfl_down_sync(mask,val,offset),val);
    }
    return val;
}


__global__ void QK(float *Q,float *K,float *S,int N,int d){
    int row = blockIdx.x *blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < N && col <N){
        float sum=0;
        for(int i=0;i<d;i++){
            sum += Q[row * d + i] * K[col *d +i];     //S的形状是row*col，Q的row和K的col，row是Q的行，col是K的行，也是K^T的列
        }
        float scale = sqrtf(d);
        S[row*N+col]=sum/scale;

    }
}

__global__ void softmax(float* S,float* P,int N){
    __shared__ float shared[32];
    int row = blockIdx.x;
    int col = threadIdx.x;
    int lane = threadIdx.x%32;
    int warpid = threadIdx.x /32;
    int warpnum =( blockDim.x +31)/32;

    float max=-INFINITY;;
    float row_max=0;
    float warp_max=0;
    float sum=0;
    //float row_sum=0;
    float warp_sum=0;

    warp_max=ReduceMax(S[row*N+col]);
    if(lane ==0){
        shared[warpid] = warp_max;
    }    
    __syncthreads();
    
    if(warpid == 0){
        max = shared[lane];
        if(lane >= warpnum){
            shared[lane]= -INFINITY;;
        }
        max = ReduceMax(max);
        if(lane == 0){
            shared[0] = max;
        }
    }
    __syncthreads();

    max = shared[0];

    warp_sum = ReduceSum(exp(S[row * N + col]-max));
    if(lane == 0){
        shared[warpid] = warp_sum;
    }
    __syncthreads();
    if(warpid == 0){
        sum = shared[lane];
        if(lane>= warpnum){
            sum =0;
        }
        sum = ReduceSum(sum);
        if(lane ==0){
            shared[0] = sum;
        }
    }
    __syncthreads();
    sum = shared[0];
    P[row * N + col] = exp(S[row * N + col]- max)/sum;
}

__global__ void PV(float* P,float* V,float* O,int M,int N,int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum=0;
    if(row<M && col<N){
        for(int i = 0;i<K;i++){
            sum += P[row * K + i] * V[i * N + col]; 
        }
        O[row * N + col]=sum;

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

                sum +=  Q[row*d+k]  * K[col*d+k];
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


    QK<<<gridQK, blockQK>>>(
        cudaQ,
        cudaK,
        cudaS,
        N,
        d);


    softmax<<<gridSoftmax, blockSoftmax>>>(
        cudaS,
        cudaP,
        N);


    PV<<<gridPV, blockPV>>>(
        cudaP,
        cudaV,
        cudaO,
        N,       // M
        d,       // N
        N);      // K


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