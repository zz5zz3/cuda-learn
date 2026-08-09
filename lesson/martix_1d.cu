#include<iostream>
#include"../tools/common.cuh"

using namespace std;

__global__ void addformgpu(float* A,float* B,float* C,const int N){
    const int bid=blockIdx.x;
    const int tid=threadIdx.x;
    const int id=bid*blockDim.x+tid;
    C[id]=A[id]+B[id];
}

int initialdata(float* addr,int element){
    for(int i=0;i<element;i++){
        addr[i]= (float)(rand()&0xFF)/10.f;
    }
    return 0;
}

int main(){
    //1设置gpu
    Setgpu();
    //2分配内存然后初始化
    int iElemcount = 512;
    size_t stBytescount = iElemcount * sizeof(float);
    //3分配主机内存
    float *fphostA,*fphostB,*fphostC;
    fphostA = (float* )malloc(stBytescount);
    fphostB = (float* )malloc(stBytescount);
    fphostC = (float* )malloc(stBytescount);
    if(fphostA!=NULL&&fphostB!=NULL &&fphostC!=NULL){
        memset(fphostA,0,stBytescount);
        memset(fphostB,0,stBytescount);
        memset(fphostC,0,stBytescount);
    }
    else  
    {
        printf("cant alloat");
        exit(-1);
    }
  //分配设备内存
    float *fpdeviceA,*fpdeviceB,*fpdeviceC;
    cudaMalloc((float** )&fpdeviceA,stBytescount);
    cudaMalloc((float** )&fpdeviceB,stBytescount);
    cudaMalloc((float** )&fpdeviceC,stBytescount);
    cudaMemset(fpdeviceA,0,stBytescount);
    cudaMemset(fpdeviceB,0,stBytescount);
    cudaMemset(fpdeviceC,0,stBytescount);

    //4初始化数据
    srand(666);
    initialdata(fphostA,iElemcount);
    initialdata(fphostB,iElemcount);
    cudaMemcpy(fpdeviceA,fphostA,stBytescount,cudaMemcpyHostToDevice);
    cudaMemcpy(fpdeviceB,fphostB,stBytescount,cudaMemcpyHostToDevice);
    cudaMemcpy(fpdeviceC,fphostC,stBytescount,cudaMemcpyHostToDevice);

    //5调用核函数
    dim3 block(32);
    dim3 grid(iElemcount / block.x);
    addformgpu<<<grid,block>>>(fpdeviceA,fpdeviceB,fpdeviceC,iElemcount);
    cudaDeviceSynchronize();
    cudaMemcpy(fphostC,fpdeviceC,stBytescount,cudaMemcpyDeviceToHost);

    //6查看结果
    for(int i=0;i<iElemcount;i++){
        printf("idx=%2d,A=%.2f,B=%.2f,C=%.2f\n",i,fphostA[i],fphostB[i],fphostC[i]);
    }

    //7释放内存
    free(fphostA);
    free(fphostB);
    free(fphostC);
    cudaFree(fpdeviceA);
    cudaFree(fpdeviceB);
    cudaFree(fpdeviceC);

    cudaDeviceReset();
    return 0;
}