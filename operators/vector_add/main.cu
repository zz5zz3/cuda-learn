#include<stdio.h>
#include"cuda_check.cuh"
#include"cuda_device.cuh"
#include"cuda_timer.cuh"

__global__ void vector_add(float *a,float *b,float *c,int n){
    const int bid=blockIdx.x;
    const int tid=threadIdx.x;
    int id=bid*blockDim.x+tid;
    if(id<n){
    c[id]=a[id]+b[id];}

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

    set_cuda_device(0);
    int elem=512;
//    int byte=2048;
    int byte = elem*sizeof(float);
    float* hostA;
    float* hostB;
    float* hostC;
    hostA=(float*)malloc(byte);
    hostB=(float*)malloc(byte);
    hostC=(float*)malloc(byte);
    memset(hostA,0,byte);
    memset(hostB,0,byte);
    memset(hostC,0,byte);
    initialData(hostA,elem);
    initialData(hostB,elem);

    float* cudaA;
    float* cudaB;
    float* cudaC;
    CUDA_CHECK(cudaMalloc((float**)&cudaA,byte));
    cudaMalloc(&cudaB,byte);
    cudaMalloc(&cudaC,byte);
    cudaMemset(cudaA,0,byte);
    cudaMemset(cudaB,0,byte);
    cudaMemset(cudaC,0,byte);

    cudaMemcpy(cudaA,hostA,byte,cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB,hostB,byte,cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(elem/32);
    vector_add<<<grid,block>>>(cudaA,cudaB,cudaC);
    cudaDeviceSynchronize();
    cudaMemcpy(hostC,cudaC,byte,cudaMemcpyDeviceToHost);
    
    for(int i=0;i<5;i++){
        printf("%0.2f+%0.2f=%0.2f\n",hostA[i],hostB[i],hostC[i]);
    }

}