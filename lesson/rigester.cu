#include<stdio.h>
#include"tools/common.cuh"
using namespace std;

__global__ void addformgpu(float *A,float *B,float *C,const int nx,const int ny){
    const int idx=blockDim.x*blockIdx.x+threadIdx.x;
    const int idy=blockDim.y*blockIdx.y+threadIdx.y;
    int id=idy*nx+idx;

    if(idx<nx&&idy<ny)
    C[id]=A[id]+B[id];
}

int main(){
    Setgpu();

float A[3][5] = {
    {1,2,3,4,5},
    {6,7,8,9,10},
    {11,12,13,14,15}
};

float B[3][5] = {
    {11,12,23,24,35},
    {56,47,68,29,110},
    {11,12,13,14,15}
};

float C[3][5];

printf("%zu\n",sizeof(B));
size_t byte=sizeof(B);


float* cudaA;
float* cudaB;
float* cudaC;
cudaMalloc(&cudaA, byte);
cudaMalloc(&cudaB, byte);
cudaMalloc(&cudaC, byte);

cudaMemcpy(cudaA,A,byte,cudaMemcpyHostToDevice);
cudaMemcpy(cudaB,B,byte,cudaMemcpyHostToDevice);

dim3 block(5,3);
dim3 grid(1,1);

addformgpu<<<grid,block>>>(cudaA,cudaB,cudaC,5,3);

cudaDeviceSynchronize();

cudaMemcpy(C,cudaC,byte,cudaMemcpyDeviceToHost);


for(int i=0;i<3;i++){
    for(int j=0;j<5;j++){
        printf("%d,%d=%0.f\n",i,j,C[i][j]);
    }
}

cudaFree(cudaA);
cudaFree(cudaB);
cudaFree(cudaC);

return 0;

}