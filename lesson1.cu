#include<stdio.h>
using namespace std;

__global__ void hellofromgpu(){

    const int blc=blockIdx.x;
    const int thr=threadIdx.x;
    const int id=threadIdx.x + blockIdx.x * blockDim.x;

    printf("hello %d %d %d\n",blc,thr,id);


}

int main(){
    hellofromgpu<<<2,4>>>();
    cudaDeviceSynchronize();
    return 0;
}