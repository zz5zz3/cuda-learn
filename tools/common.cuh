#include<stdlib.h>
#include<stdio.h>

void Setgpu(){
    int ideviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&ideviceCount);
    printf("deviceCount:%d\n",ideviceCount);
    int idev=0;
    error = cudaSetDevice(idev);
    printf("set gpu:%d\n",idev);


}