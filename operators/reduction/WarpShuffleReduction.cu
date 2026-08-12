#include<stdio.h>
#include"cuda_check.cuh"
#include"cuda_device.cuh"
#include"cuda_timer.cuh"
#include <time.h>

__device__ float WarpReduceSum(float val){
    
    for(int offset=16;offset>0;offset/=2){
        val+=__shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}

__global__ void WarpShuffleReduction(float *a,float *b,float *c,int n){
    extern __shared__ float shared_memory[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = bid*blockDim.x + tid;
    int warp_num = (blockDim.x+31)/32;
    int warp_id = tid / 32;
    int lane    = tid % 32;

    float val=0;
    if(id<n){
        val=a[id];
    }
    val=WarpReduceSum(val);

    if(lane==0){
    shared_memory[warp_id]=val;
    }


    __syncthreads();

    if(warp_id==0){
        if(lane<warp_num){
            val=shared_memory[lane];
        }
        else{
            val=0.0f;
        }
        val=WarpReduceSum(val);
        if(lane == 0){
            c[bid] = val;
        }
    }
    
}

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
    int elem=1 << 24;
//    int byte=2048;
    int byte = elem*sizeof(float);
/*1 主机准备区*/

    float* hostA;
    float* hostB;
    float* hostC;
    float* hostRef;
    hostA=(float*)malloc(byte);
    hostB=(float*)malloc(byte);
    hostC=(float*)malloc(byte);
    hostRef=(float*)malloc(byte);

    memset(hostA,0,byte);
    memset(hostB,0,byte);
    memset(hostC,0,byte);
    memset(hostRef,0,byte);

    initialData(hostA,elem);
    initialData(hostB,elem);
/*2 设备准备区*/

    float* cudaA;
    float* cudaB;
    float* cudaC;
    
    CudaTimer cudatimer;
    
    CUDA_CHECK(cudaMalloc((float**)&cudaA,byte));
    CUDA_CHECK(cudaMalloc(&cudaB,byte));
    CUDA_CHECK(cudaMalloc(&cudaC,byte));
    CUDA_CHECK(cudaMemset(cudaA,0,byte));
    CUDA_CHECK(cudaMemset(cudaB,0,byte));
    CUDA_CHECK(cudaMemset(cudaC,0,byte));

    CUDA_CHECK(cudaMemcpy(cudaA,hostA,byte,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cudaB,hostB,byte,cudaMemcpyHostToDevice));

/*3 GPU计算区*/
    dim3 block(64);
    dim3 grid((elem + block.x - 1) / block.x);
    int memory_size=block.x*sizeof(float);

    cudatimer.start();
    WarpShuffleReduction<<<grid,block,memory_size>>>(cudaA,cudaB,cudaC,elem);
    float elapsed_ms=cudatimer.stop();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hostC,cudaC,byte,cudaMemcpyDeviceToHost));

/*4 CPU计算区*/
    clock_t  start=clock();
    for(int i=0;i<elem/64;i++){
        for(int j=0;j<64;j++){
        hostRef[i]+=hostA[i*64+j];
        }
    }
    clock_t  stop=clock();
    double time=(stop-start)*1000/CLOCKS_PER_SEC;
/* 5 结果验证区 */

printf("gpu:%0.2f,cpu:%0.2f,PASSED?",hostC[0],hostRef[0]);

for(int i=0;i<10;i++){
    printf("cuda:%0.2f+%0.2f=%0.2f\n",hostA[i],hostB[i],hostC[i]);
}
printf("cudatime:%0.2fms,cputime:%0.2fms\n",elapsed_ms,time);

double total_bytes =
    (double) (elem * sizeof(float)+grid.x* sizeof(float));

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