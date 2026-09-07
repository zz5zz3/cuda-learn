#include<stdio.h>
#include"cuda_check.cuh"
#include"cuda_device.cuh"
#include"cuda_timer.cuh"
#include <time.h>

__device__ float WarpSum(float val){
    uint mask = 0xFFFFFFFF;
    for(int offset=16;offset>0;offset/=2){
        val += __shfl_down_sync(mask,val,offset);
    }
    return val;
}

__global__ void layernorm(float* input,float* output,int rows,int cols){
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int col = threadIdx.x;
    int lane = col %32;
    int warpid = col/32;
    int warpnum = (blockDim.x + 31) / 32;
    float local_sum=0;

    if(row >= rows) return;
    for(int stride=col;stride<cols;stride+=blockDim.x){
        local_sum += input[row * cols + stride];
    }

    local_sum=WarpSum(local_sum);
    if(lane==0){
        shared[warpid] = local_sum; 
    }
    __syncthreads();
    float global_sum=0;
    if(warpid == 0){
        if(lane<warpnum){
            global_sum=shared[lane];
        }
        global_sum = WarpSum(global_sum);
        if(lane == 0){
            shared[0] = global_sum;
        }
    }
    __syncthreads();
    global_sum= shared[0];
    float mean=global_sum/cols;

    float local_variance=0;
    for(int stride=col;stride<cols;stride+=blockDim.x){
        local_variance += (input[row*cols+stride]-mean)*(input[row*cols+stride]-mean);
    }
    local_variance=WarpSum(local_variance);
    float global_varience=0;
    if(lane==0){
        shared[warpid]=local_variance;
    }
    __syncthreads();
    if(warpid==0){
        if(lane<warpnum){
            global_varience=shared[lane];
        }
        global_varience=WarpSum(global_varience);
    
    if(lane==0){
        shared[0]=global_varience;
        }
    }
    __syncthreads();

    global_varience = shared[0];
    float variance = global_varience / cols;

    for(int stride = col; stride < cols; stride += blockDim.x){
        float x = input[row * cols + stride];
        output[row * cols + stride] = (x - mean) / sqrt(variance+ 1e-5f);
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
    int rows = 4096;
    int cols = 1024;
    int elem=rows *cols;

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
    dim3 block(256);
    dim3 grid(rows);
    int shared_memory_bytes = block.x*sizeof(float);

    cudatimer.start();

    layernorm<<<grid,block,shared_memory_bytes>>>(cudaA,cudaC,rows,cols);
    float elapsed_ms=cudatimer.stop();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hostC,cudaC,byte,cudaMemcpyDeviceToHost));

/*4 CPU计算区*/
    clock_t  start=clock();
for(int row = 0; row < rows; row++){

    float row_max = -__FLT_MAX__;

    for(int col = 0; col < cols; col++){
        row_max = fmaxf(
            row_max,
            hostA[row * cols + col]
        );
    }

    float row_sum = 0.0f;

    for(int col = 0; col < cols; col++){
        row_sum += expf(
            hostA[row * cols + col] - row_max
        );
    }

    for(int col = 0; col < cols; col++){
        hostRef[row * cols + col] =
            expf(hostA[row * cols + col] - row_max)
            / row_sum;
    }
}
    clock_t  stop=clock();
    double time=(stop-start)*1000/CLOCKS_PER_SEC;
/* 5 结果验证区 */

if(hostC[2]==hostRef[2])
printf("PASSED");

for(int i=0;i<10;i++){
    printf("cuda:%0.2f+%0.2f=%0.8f\n",hostA[i],hostB[i],hostC[i]);
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