#include<stdio.h>
#include"cuda_check.cuh"
#include"cuda_device.cuh"
#include"cuda_timer.cuh"
#include <time.h>

__global__ void GEMM(float* A,float* B,float* C ,int M,int K,int N){
    int x   = blockDim.x*blockIdx.x + threadIdx.x;
    int y   = blockDim.y*blockIdx.y + threadIdx.y;
    float sum=0;
    if(x<N && y<M){
        for(int i=0;i<K;i++){
            sum+=A[y*K+i] * B[i * N+ x];
        }
        C[y*N +x]=sum;
    }
}
/*__global__ void GEMM(float* A,float* B,float* C,int M,int K,int N){
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    if(x<N && y<M)
    {
        float sum=0;
        for(int k=0;k<K;k++)
        {
            sum += A[y*K+k] *  B[k*N+x];
        }
        C[y*N+x]=sum;
    }
}*/

void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

int main(){
/*0 初始化区*/
    set_cuda_device(0);
    int M=512;
    int N=512;
    int K=512;
    int elem=M*N;

    int sizeA=M*K *sizeof(float);
    int sizeB=K*N*sizeof(float);
    int sizeC=elem*sizeof(float);
//    int byte=2048;
    int byte = elem*sizeof(float);
/*1 主机准备区*/

    float* hostA;
    float* hostB;
    float* hostC;
    float* hostRef;
    hostA=(float*)malloc(sizeA);
    hostB=(float*)malloc(sizeB);
    hostC=(float*)malloc(sizeC);
    hostRef=(float*)malloc(sizeC);

    memset(hostA,0,sizeA);
    memset(hostB,0,sizeB);
    memset(hostC,0,sizeC);
    memset(hostRef,0,sizeC);

    initialData(hostA,M*K);
    initialData(hostB,K*N);
/*2 设备准备区*/

    float* cudaA;
    float* cudaB;
    float* cudaC;
    CudaTimer cudatimer;

    CUDA_CHECK(cudaMalloc((float**)&cudaA,sizeA));
    CUDA_CHECK(cudaMalloc(&cudaB,sizeB));
    CUDA_CHECK(cudaMalloc(&cudaC,sizeC));
    CUDA_CHECK(cudaMemset(cudaA,0,sizeA));
    CUDA_CHECK(cudaMemset(cudaB,0,sizeB));
    CUDA_CHECK(cudaMemset(cudaC,0,sizeC));

    CUDA_CHECK(cudaMemcpy(cudaA,hostA,sizeA,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cudaB,hostB,sizeB,cudaMemcpyHostToDevice));

/*3 GPU计算区*/
    dim3 block(32,32);
    dim3 grid((M + block.x - 1) / block.x , (N + block.x - 1) / block.x);
    cudatimer.start();
    GEMM<<<grid,block>>>(cudaA,cudaB,cudaC ,M,K,N);
    float elapsed_ms=cudatimer.stop();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hostC,cudaC,byte,cudaMemcpyDeviceToHost));

/*4 CPU计算区*/
    clock_t  start=clock();
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<K;k++){
                hostRef[i*N+j]+=hostA[i*K+k] * hostB[k*N+j];
            }
        }
    }
    clock_t  stop=clock();
    double time=(double)(stop-start)*1000/CLOCKS_PER_SEC;
/* 5 结果验证区 */


for(int i=0;i<10;i++){
    printf("cuda:%0.2f+%0.2f=%0.2f | %0.2f \n",hostA[i],hostB[i],hostC[i],hostRef[i]);
}

printf("cudatime:%0.2fms,cputime:%0.2fms\n",elapsed_ms,time);

double total_bytes =
    3.0 * elem * sizeof(float);

double bandwidth =
    total_bytes /
    (elapsed_ms / 1000.0) /
    1e9;

printf("Memory Bandwidth: %.2f GB/s\n", bandwidth);

/* 6 资源释放区 */

    CUDA_CHECK(cudaFree(cudaA));
    CUDA_CHECK(cudaFree(cudaB));
    CUDA_CHECK(cudaFree(cudaC));

    free(hostA);
    free(hostB);
    free(hostC);
    free(hostRef);

    return 0;
}